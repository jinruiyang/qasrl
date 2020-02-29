import argparse
import math
import time
import sys
import os
from typing import List

import data
import model
import numpy as np
import torch
from utils import batchify, get_batch, repackage_hidden
# from parallel import DataParallelModel, DataParallelCriterion


parser = argparse.ArgumentParser(description='Plan and write language model')
parser.add_argument('--train-data', type=str,
                    default='data/rocstory/keyword_experiments/ROCStories_all_merge_tokenize.titlesepkeysepstory.train.5.2.1',
                    help='location of the training data corpus')
parser.add_argument('--valid-data', type=str,
                    default='data/rocstory/keyword_experiments/ROCStories_all_merge_tokenize.titlesepkeysepstory.dev.5.2.1',
                    help='location of the valid data corpus')
parser.add_argument('--test-data', type=str,
                    default='data/rocstory/keyword_experiments/ROCStories_all_merge_tokenize.titlesepkeysepstory.test.5.2.1',
                    help='location of the test data corpus')
parser.add_argument('--cached-data', type=str, default=None,
                    help='a torch pre-cached corpus, for resuming training or faster init')
parser.add_argument('--vocab-file', type=str, default='',
                    help='filename to save the vocabulary pickle to')
parser.add_argument('--pretrained-name', type=str, default=None, choices=['glove', 'fasttext', 'elmo_top', 'elmo_avg'],
                    help='type of pretrained embeddings to use')
parser.add_argument('--no-grad-mask', action='store_true', help='stops grad mask from being used on pretrained weights')
parser.add_argument('--seq2seq', action='store_true',
                    help='train in a seq2seq manner')
parser.add_argument('--delimiter', type=str,
                    help='delimiting character between source and target, required for seq2seq')
parser.add_argument('--keep-delimiter', action='store_true',
                    help='whether to store the delimiter used for splitting source and target')
parser.add_argument('--attn', type=str, choices=['none', 'general', 'dot', 'mlp'],
                    default='none', help='attention type between encoder-decoder')
parser.add_argument('--max-tgt-len', type=int, default=None,
                    help='maximum length of target for decoding. Truncates data to fit length')
parser.add_argument('--bptt', type=int, default=70,
                    help='backpropagation through time - applies only to decoder in seq2seq case')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=2.0,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--not-tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default=randomhash + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--anneal', action="store_true", help="divides learning rate when ppl not improving")
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 2 - accepts multiple')
parser.add_argument('--gpu-id', type=str, default="0,1,2,3",
                    help='gpu id for multi gpu using')
parser.add_argument('--multi-gpu', action='store_true',
                    help=' multi gpu')

#### Misc functions until I refactor this mess
def static_ppl(losses: List[float], num_epochs=5, sim_threshold=0.005) -> bool:
    last_x_epochs = losses[-num_epochs+1:-1]
    if len(losses) < num_epochs+1:
        return False
    min_loss = min(last_x_epochs)
    if math.isclose(losses[-1], min_loss, rel_tol=sim_threshold) or losses[-1] > (min_loss + 0.01):
        print("{} is equal to or more than minimum loss of the last 5. min: {} last 5: {}".format(losses[-1], min_loss, last_x_epochs))
        return True
    else:
        return False


args = parser.parse_args()
tie_weights = not args.not_tied
grad_mask = not args.no_grad_mask

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if torch.cuda.device_count() > 1 and args.multi_gpu:
    num_devices = torch.cuda.device_count()
    assert int(args.batch_size) % int(num_devices) == 0, "Tried to use {} GPUs with batch size {}. When using multi-gpu, batch must be divisible by num GPUs".format(num_devices, args.batch_size)
    print("Let's use", num_devices, "GPUs!")
###############################################################################
# Load data
###############################################################################

def model_save(fn):    
    with open(fn, 'wb') as f:
        if args.multi_gpu:
            torch.save([model.module, criterion, optimizer], f)
        else:
            torch.save([model, criterion, optimizer], f)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


import os
import hashlib

if args.seq2seq and not args.cached_data:
    assert args.delimiter is not None, "Need to provide a delimiter between source and target to use seq2seq"

fn = 'corpus.{}.data'.format("_".join(args.vocab_file.split("/")))

if args.cached_data:
    if args.cached_data != fn:
        print(
            'Warning: name of corpus to load: {} does not match vocab file convention: {}'.format(args.cached_data, fn))
    print('Loading cached dataset (a new vocab file will not be created)...')
    corpus = torch.load(args.cached_data)
else:
    # save corpus for resuming later
    print('Producing dataset...')
    corpus = data.Corpus(train_path=args.train_data, dev_path=args.valid_data,
                         test_path=args.test_data, dict_path=args.vocab_file,
                         delimiter=args.delimiter, keep_delimiter=args.keep_delimiter)
    torch.save(corpus, fn)

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
test_data = batchify(corpus.test, args.batch_size)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss

criterion = None

ntokens = len(corpus.dictionary)
print('Vocab size: ', ntokens)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth,
                       args.dropouti, args.dropoute, args.wdrop, tie_weights, args.seq2seq, args.attn,
                       pretrained_model=args.pretrained_name, grad_mask=grad_mask,
                       corpus_vocab=corpus.dictionary.word2idx.keys())
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, \
                                                                   args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop

        for rnn in model.rnns:
            if type(rnn) == WeightDrop:
                rnn.dropout = args.wdrop
            elif rnn.zoneout > 0:
                rnn.zoneout = args.wdrop
###
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###
device_ids = 0 #for if using cpu
if args.cuda:
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device_ids = range(torch.cuda.device_count())
    model = model.cuda()
    criterion = criterion.cuda()

    if len(device_ids) > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)
        # criterion = DataParallelCriterion(criterion, device_ids=[0, 1, 2, 3])

###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN':
        model.reset()
    total_loss = 0
    with torch.no_grad():
        for i in range(0, len(data_source)):    
            # source, targets = get_batch(data_source, i, args.max_tgt_len)
            source, targets = get_batch(data_source, i)
            if args.cuda:
                source, targets = source.cuda(), targets.cuda()

                if args.seq2seq:
                    hidden = None
                else:
                    hidden = model.init_hidden(source.size(0))

                bptt_cont = False
                if device_ids:
                    if targets.size(0) % len(device_ids) != 0:
                        continue
                for j in range(0, targets.size(1) - 1, args.bptt):

                    # Create truncated target.
                    tgt = targets[:, j: j + args.bptt]
                    tgt_len = tgt.size(1)

                    if args.seq2seq:
                        input = source, tgt
                    else:
                        input = tgt
                    try:
                        if len(input) == 0:
                            print("No input given", file=sys.stderr)
                            continue
                    except:
                        print("input tensor is zero-D", file=sys.stderr)
                        continue
                    output, hidden = model(input, hidden, bptt_continuation=bptt_cont)
                    output = output[:, :-1].contiguous().view(-1, output.size(2))
                    tgt = tgt[:, 1:].contiguous().view(-1)

                    bptt_cont = True

                    # Compute loss.
                    if args.multi_gpu:
                        total_loss += criterion(model.module.decoder.weight, model.module.decoder.bias, output, tgt).data * (tgt_len / targets.size(1))
                    else:
                        total_loss += criterion(model.decoder.weight, model.decoder.bias, output, tgt).data * (tgt_len / targets.size(1))
                    # Detach the history from hidden nodes
                    hidden = repackage_hidden(hidden)
    # print(type(total_loss))
    return total_loss.item() / len(data_source)
    # return total_loss / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    seq_len = max(5, int(np.random.normal(args.bptt, 5)))

    batch = 0
    while batch < len(train_data):
        #for big models, save at each 1/2 the way through an epoch
        if len(train_data) // (batch+1) == 2:
            model_save('{}.half_epoch'.format(args.save))
        model.train()

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt

        # if args.max_tgt_len is not None:
        #     source, targets = get_batch(train_data, batch, args.max_tgt_len)
        # else:
        source, targets = get_batch(train_data, batch)
        if args.cuda:
            source, targets = source.cuda(), targets.cuda()

        optimizer.zero_grad()

        if args.seq2seq:
            hidden = None
        else:
            hidden = model.init_hidden(source.size(0))

        # BPTT continuation flag , so that we do not init decoder hidden state with encoder last state when we just
        # want to continue with truncated batches of same sequence
        bptt_cont = False

        if device_ids: # makes sure multi-gpu doesn't try to split remainder batches that are invalid sizes for splitting
            if targets.size(0) % len(device_ids) != 0:
                continue

        for j in range(0, targets.size(1) - 1, args.bptt):


            # Create truncated target.
            tgt = targets[:, j: j + args.bptt]
            tgt_len = tgt.size(1)

            if args.seq2seq:
                input = source, tgt
            else:
                input = tgt


            # print("HIDDEN", hidden.size())
            # print(input[0].size())
            output, hidden, rnn_hs, dropped_rnn_hs = model(input, hidden, return_h=True, bptt_continuation=bptt_cont)
            output = output[:, :-1].contiguous().view(-1, output.size(2))
            tgt = tgt[:, 1:].contiguous().view(-1)

            bptt_cont = True

            # Compute loss.
            if args.multi_gpu:
                raw_loss = criterion(model.module.decoder.weight, model.module.decoder.bias, output, tgt)
            else:
                raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, tgt)

            loss = raw_loss
            # Activation Regularization
            if args.alpha:
                loss = loss + sum(
                    args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])

            # Temporal Activation Regularization (slowness)
            if args.beta:
                loss = loss + sum(args.beta * (rnn_h[:, 1:] - rnn_h[:, :-1]).pow(2).mean()
                                  for rnn_h in rnn_hs[-1:])

            if loss is not None:
                loss.backward()

            total_loss += raw_loss.data * (tgt_len / targets.size(1))

            # If truncated, don't backprop fully. Detach the history from hidden nodes
            hidden = repackage_hidden(hidden)
        # if batch == len(train_data) - 1:
        # print("INPUT",input[0].size())
        # output, hidden, rnn_hs, dropped_rnn_hs = model(input, hidden, return_h=True)
        # output = output[:, :-1].contiguous().view(-1, output.size(2))
        # targets = targets[:, 1:].contiguous().view(-1)
        # raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        # loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        optimizer.param_groups[0]['lr'] = lr2

        # total_loss += raw_loss.data
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data), optimizer.param_groups[0]['lr'], elapsed * 1000 / args.log_interval,
                cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1


# Loop over epochs.
lr = args.lr
best_val_loss, recent_val_losses = [], []
stored_loss = 100000000

# At any point you can hit Ctrl + C to break out of training early.


try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer  = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]: # This activates ASGD
            tmp = {}
            for name, prm in model.named_parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                if prm in tmp:
                    prm.data = tmp[prm].clone()

            recent_val_losses.append(val_loss2)

        else:
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                    len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            best_val_loss.append(val_loss)
            recent_val_losses.append(val_loss)

        # Check for annealing learning rate, hardcoded or automatic
        if (static_ppl(recent_val_losses) and args.anneal) or epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save('{}.e{}'.format(args.save, epoch))
            print('Dividing learning rate by 2')
            optimizer.param_groups[0]['lr'] /= 2
            recent_val_losses = [] #reset
        sys.stdout.flush()

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(args.save)

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
