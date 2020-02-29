"""
A script that generates based on a pretrained language model.

--data data/ROCStories_all_merge_tokenize.titlesepkey.test.5.1.1
--lm language-model/models/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_311_lr10.pt
--vocab language-model/models/ROC_title_key_story_e1000_h1500_edr0.2_hdr0.1_311_lr10.pkl
--print
--max_lines 10
--beam_size 4
"""

import sys, argparse, pickle, os
import torch
import numpy as np
import data
from importlib import import_module
from decoder.decoders import BeamRerankDecoder, BeamSearchDecoder, SamplingDecoder
from utils import update_embedding_layer, process_new_tokens


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None,
                    help='text file containing initial input strings to condition generation')
parser.add_argument('--skip', type=int, default=0,
                    help='number of lines to skip in data file before beginning')
parser.add_argument('--out', type=str, default='output.txt',
                    help='text file to write generations to')
parser.add_argument('--lm', type=str, default=None,
                    help='lm to use for decoding')
parser.add_argument('--vocab', type=str, default=None,
                    help='vocab file to use for lm')
parser.add_argument('--print', action='store_true',
                    help='whether to print output to stdout (in addition to writing it to a file)')
parser.add_argument('--both', action='store_true',
                    help='also include pure LM output as well as discriminator output')
parser.add_argument('--print_cond_data', action='store_true',
                    help='whether to print conditional data used for generation')
parser.add_argument('--cond_sep_char', type=str, default='',
                    help='char to separate input data & generated data (to make reading easier)')
parser.add_argument('--apply_disc', action='store_true',
                    help='whether to apply discriminators to output')
parser.add_argument('--decode_type', default='sample', choices=['beam', 'sample'],
                    help='whether to use beam search or sampling')
parser.add_argument('--dedup', action='store_true',
                    help='whether to enforce uniqueness of generations (for keyword generation)')
parser.add_argument('--split_on', type=str, default=None,
                    help='the character to split the input on into initial and continuation text'
                         'only the first one will be split on. If none, no split.')
parser.add_argument('--keep_split', action='store_true', help='whether to keep the split on char')
parser.add_argument('--max_lines', type=int, default=None,
                    help='maximum lines to generate')
parser.add_argument('--max_story_tokens', type=int, default=1000,
                    help='maximum tokens per story')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--verbosity', action='store_true',
                    help='whether to print out steps as decode')
## Decoding Stuff
parser.add_argument('--beam_size', type=int, default=10,
                    help='number of candidates in beam at every step')
parser.add_argument('--norm', action='store_true',
                    help='normalize beamsearch. usually a good idea but not always')
parser.add_argument('--end', type=str, default='<eos>',
                    help='what string to use as the end token')
parser.add_argument('--sep', type=str, default='</s>',
                    help='what string to use as the sentence seperator token')
parser.add_argument('--temp', type=float, default=None,
                    help='temperature, if using stochastic decoding')
parser.add_argument('--human_readable', action='store_true',
                    help='make readable to a layperson, forbid generation of special characters')
parser.add_argument('--nucleus', action='store_true',
                    help='apply nucleus sampling')
parser.add_argument('--top_k', type=int, default=0,
                    help='keep only top k tokens with highest probability (top-k filtering)')
parser.add_argument('--top_p', type=float, default=0.0,
                    help='keep the top tokens with cumulative probability >= top_p (nucleus filtering)')
# Scorers
parser.add_argument('--scorers', type=str, default=None,
                    help='tsv with scorer information')

# Scorer Learning
parser.add_argument('--learn', action='store_true')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--save_every', type=int, default=1)
parser.add_argument('--ranking_loss', action='store_true',
                    help='metaweight learning ranking loss')
parser.add_argument('--epochs', type=int, default=1,
                    help='how many times to go through the input file')

args = parser.parse_args()
print("Args: ", args, file=sys.stderr)

np.random.seed(args.seed)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print("Load model", file=sys.stderr)
with open(args.lm, 'rb') as model_file:
    # if train on GPU load on cpu
    device = torch.device('cpu')
    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, map_location=device))
    # model, criterion, optimizer = torch.load(model_file, map_location=lambda storage, loc: storage)
    model, criterion, optimizer = torch.load(model_file, map_location=device)
    # if isinstance(model, DataParallel):
    # model = model.module
    # model = model.to(device)

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

model.eval()

### Validation
# Backwards compatibility
if not hasattr(model, "seq2seq"):
    model.seq2seq = False

if not hasattr(model, "embedder"):
    assert not model.seq2seq, "model should have separate embedder and encoder if seq2seq"
    model.embedder = model.encoder

if args.learn:
    assert args.split_on, "cannot learn coefficients without splitting into intial and true continuation"


corpus = data.Corpus(applyDict=True, dict_path=args.vocab)
dictionary = corpus.dictionary

scorer_config, scorers, coefs = [], [], [] # so that the variables will exist regardless and make init easier
if args.apply_disc:
    print("Creating scorers", file=sys.stderr)
    if not args.scorers:
        sys.exit("Need to provide a .tsv file with the "
                 "--scorers arg in order to apply discriminators")
    with open(args.scorers) as scorer_file:
        for line in scorer_file:
            fields = line.strip().split('\t')
            scorer_config.append(fields)
            weight, module_path, classname  = fields[:3]
            weight = float(weight)
            model_location = fields[3]
            module = import_module(module_path)
            constructor = getattr(module, classname)
            scorer =  constructor(model_location, args.cuda) # this calls a scorer init, but a scorer init expects a pretrained model
            scorers.append(scorer)
            coefs.append(weight)
    print("Coefs:", coefs, file=sys.stderr)

print("Init decoders", file=sys.stderr)

if args.decode_type == "beam":
    if args.apply_disc:  # init a rerank decoder
        br_decoder = BeamRerankDecoder(model,
                                scorers,
                                coefs,
                                learn=args.learn,
                                lr=args.lr,
                                ranking_loss=args.ranking_loss,
                                beam_size=args.beam_size,
                                temperature=args.temp,
                                max_len=args.max_story_tokens,
                                terms=[dictionary.word2idx[args.end]],
                                sep=dictionary.word2idx[args.sep],
                                forbidden=[dictionary.word2idx['<unk>']],
                                verbosity=args.verbosity,
                                dictionary=dictionary.word2idx,
                                use_cuda=args.cuda,
                                human_readable=args.human_readable
                                )

    if not args.apply_disc or args.both: # else init a plain beamsearch decoder, or if use both.
        vanilla_decoder = BeamSearchDecoder(model, args.beam_size, dictionary.word2idx[args.end],
                                            verbosity=False, dictionary=dictionary, temperature=args.temp,
                                            sep=dictionary.word2idx[args.sep],
                                            max_len=args.max_story_tokens,
                                            human_readable=args.human_readable)
elif args.decode_type == "sample":
    sample_decoder = SamplingDecoder(
        model,
        dictionary=dictionary,
        eos_id=dictionary.word2idx[args.end],
        sep=dictionary.word2idx[args.sep],
        scorers=scorers,
        coefs=coefs,
        temperature=args.temp,
        dedup=args.dedup,
        apply_disc=args.apply_disc,
        learn=args.learn,
        lr=args.lr,
        ranking_loss=args.ranking_loss,
        human_readable=args.human_readable,
        nucleus=args.nucleus,
        top_k=args.top_k,
        top_p=args.top_p)
    if not args.apply_disc or args.both: # else init a plain beamsearch decoder, or if use both.
        vanilla_decoder = SamplingDecoder(model, dictionary=dictionary, eos_id=dictionary.word2idx[args.end],
                                          sep=dictionary.word2idx[args.sep], temperature=args.temp,
                                          dedup=args.dedup, human_readable=args.human_readable,
                                          nucleus=args.nucleus, top_k=args.top_k,top_p=args.top_p)

print("Start decoding", file=sys.stderr)

avg, a_n = None, 0

# Below used for OOV handling with pretrained embeddings. If not doing that then will be unused
processed_tokens_set = set()


for epoch in range(args.epochs):

    print("Epoch {}".format(epoch))

    with open(args.data) as data_file, open(args.out, 'w') as out_file:
        for i, line in enumerate(data_file):
            if args.max_lines and i >= args.max_lines + args.skip: #since indexing actually begins at 1 for files
                break
            if i < args.skip:
                continue

            ### Process Input ###
            if args.split_on:
                #print stderr on first line
                if i == args.skip:
                    print("Example line: {}".format(line), file=sys.stderr)
                    print("Splitting on {}".format(args.split_on), file=sys.stderr)

                # create initial and continuation via splitting on char. l2w data does tab delimiting, we do <EOT> or <EOL>
                # or via custom creating with a preprocessing script
                initial, continuation = line.split(args.split_on, 1)
                init_tokens = initial.strip().split()
                if args.keep_split:
                    init_tokens += [args.split_on]

                true_cont_tokens = continuation.strip().split()
                if not true_cont_tokens:
                    continue

                #does oov handling before doing dict lookups
                #process_new_tokens(init_tokens + true_cont_tokens, processed_tokens_set, model, dictionary)

                true_cont_ints = [dictionary.word2idx.get(token, 0) for token in true_cont_tokens]
                init_tokens_ints = [dictionary.word2idx.get(token, 0) for token in init_tokens]


            else:
                # initialise initial text. Cannot be used for learning on discriminators since no gold is given
                init_tokens = line.strip().split()
                #process_new_tokens(init_tokens, processed_tokens_set, model, dictionary) # oov handling

                init_tokens_ints = [dictionary.word2idx.get(token, 0) for token in init_tokens]

            if not init_tokens_ints:
                out_str = "\n"
                if args.print:
                    print(out_str, end='')
                out_file.write(out_str)
                continue

            if args.decode_type == "sample":
                decoder = sample_decoder
            else:
                if args.apply_disc:
                    decoder = br_decoder
                else:
                    decoder = vanilla_decoder

            ### DECODE ###
            if args.learn:
                diff = decoder.decode(init_tokens_ints,
                                      true_cont_ints)
                out_str = '%f\n' % diff
            else:

                with torch.no_grad():
                    # print(init_tokens_ints)
                    pred_tokens_ints = decoder.decode(init_tokens_ints)
                    # print(pred_tokens_ints)
                    pred_init_tokens = [dictionary.idx2word[pred_tokens_ints[i]]
                                        for i in range(len(init_tokens_ints))]
                    # print(pred_init_tokens)
                    pred_cont_tokens = [dictionary.idx2word[pred_tokens_ints[j]]
                                        for j in range(len(init_tokens_ints), len(pred_tokens_ints))]
                    # print(pred_cont_tokens)
                    init = ' '.join(pred_init_tokens)  # actually puts the init tokens through the lm so can use result for debugging
                    cont = ' '.join(pred_cont_tokens)
                    # print("Full sentence ints: {}".format(pred_tokens_ints))

                    if args.both:  # this gives you the ability to compare with and without scorers
                        assert(args.apply_disc), "need to apply discriminators in order to print two kinds of output"
                        lm_pred_tokens_ints = vanilla_decoder.decode(init_tokens_ints)
                        lm_pred_cont_tokens = [dictionary.idx2word[lm_pred_tokens_ints[i]]
                                                for i in range(len(init_tokens_ints), len(lm_pred_tokens_ints))]
                        lm_cont = ' '.join(lm_pred_cont_tokens)
                        out_str = '%s | RERANK: %s | ORIG: %s\n' % (init, cont, lm_cont)
                    else:
                        if args.print_cond_data:
                            out_str = '%s %s %s\n' % (init, args.cond_sep_char, cont)
                        else:
                            out_str = "{}\n".format(cont)
            if args.print:
                print(out_str, end='')
            out_file.write(out_str)
            out_file.flush()
            
            # Save coeffecients if learning them

            if args.learn and (i+1) % args.save_every == 0:
                #avg and a_n init to None and 0. a_n seems to just track the saves
                with open(args.scorers, 'w') as out:
                    if avg is None:
                        avg = decoder.coef_trainer.weight_model.coefs.weight.data.cpu().squeeze().clone()
                    else:
                        avg += decoder.coef_trainer.weight_model.coefs.weight.data.cpu().squeeze()
                    a_n += 1
                    for s, coef in enumerate(avg.numpy() / a_n):
                        scorer_config[s][0] = str(coef)
                        out.write('%s\n' % '\t'.join(scorer_config[s]))
                    print("Writing coefficients: ", avg / a_n, file=sys.stderr)

