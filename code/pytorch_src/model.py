import torch
import torch.nn as nn
import torchtext.vocab as vc
import sys
import os
import numpy as np
from allennlp.commands.elmo import ElmoEmbedder
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop
from attention import Attn



class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5,
                 dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, seq2seq=False,
                 attn_type='none', pretrained_model=None, corpus_vocab=None):
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        assert nhid % 2 == 0
        if tie_weights:
            assert ninp % 2 == 0
            
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout(batch_first=seq2seq)
        #TODO the seq2seq code doesn't use nn.Dropout (just stores the numbers). Make sure this isn't a problem.
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.using_pretrained = pretrained_model
        self.seq2seq = seq2seq
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.wdrop = wdrop
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

        self.embedder = self._init_embedder(ntoken, pretrained_model=pretrained_model, corpus_vocab=corpus_vocab)

        self.rnns = self._init_rnn()
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        if self.seq2seq:
            self.encoder = [torch.nn.LSTM(self.ninp if l == 0 else self.nhid,
                                          self.nhid // 2 if l != self.nlayers - 1
                                          else (self.ninp // 2 if self.tie_weights else self.nhid // 2),
                                          1, dropout=0, batch_first=True, bidirectional=True) for l in
                            range(self.nlayers)]
            self.encoder = torch.nn.ModuleList(self.encoder)

            if self.attn_type != 'none':
                self.attn = Attn(attn_type, self.ninp if self.tie_weights else self.nhid)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights and not self.using_pretrained:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.embedder.weight

            # if self.using_pretrained:
            #     self.decoder.weight.requires_grad = False
        self.init_weights()

    def _init_rnn(self):
        """init stacked rnns based on type, batch_first if seq2seq, else not. Only LSTM has been tests to work with seq2seq. """
        if self.rnn_type == 'LSTM':
            rnns = [
                torch.nn.LSTM(self.ninp if l == 0 else self.nhid,
                              self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights
                                                                       else self.nhid),
                              1, dropout=0, batch_first=self.seq2seq) for l in range(self.nlayers)]
            if self.wdrop:
                rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.wdrop) for rnn in rnns]
        if self.rnn_type == 'GRU':
            rnns = [torch.nn.GRU(self.ninp if l == 0
                                 else self.nhid, self.nhid if l != self.nlayers - 1
                                 else self.ninp, 1, dropout=0,
                                 batch_first=self.seq2seq) for l in range(self.nlayers)]
            if self.wdrop:
                rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=self.wdrop) for rnn in rnns]
        elif self.rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            rnns = [QRNNLayer(input_size=self.ninp if l == 0
            else self.nhid, hidden_size=self.nhid if l != self.nlayers - 1
            else (self.ninp if self.tie_weights else self.nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1,
                              output_gate=True) for l in range(self.nlayers)]
            for rnn in rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=self.wdrop)

        return rnns

    def _init_embedder(self, ntoken, pretrained_model=None, corpus_vocab=None):
        """init and embedder, potentially based on a pretrained model"""
        if pretrained_model is None:
            embedder = nn.Embedding(ntoken, self.ninp)
        else:
            if pretrained_model == "fasttext":
                pretrained_emb_model = vc.FastText()
            elif pretrained_model == "glove":
                pretrained_emb_model = vc.GloVe()
            elif pretrained_model == "elmo_top" or pretrained_model == "elmo_avg":
                dirname = os.path.dirname(__file__)
                options_file = os.path.join(dirname,
                                            '../elmo-config/elmo_2x4096_512_2048cnn_2xhighway_options.json')
                weight_file = os.path.join(dirname,
                                           '../elmo-config/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
                elmo_embedder = ElmoEmbedder(options_file, weight_file)
            else:
                raise Exception("Unsupported embedding model:", pretrained_model)

            print("Using Pretrained embedding:", pretrained_model)

            # Initialize gradient mask for learning certain embeddings, fill zero for non-learning indexes
            self.emb_grad_mask = torch.ones(ntoken)
            self.emb_grad_mask[0].fill_(0)

            if pretrained_model in ['fasttext', 'glove']:

                pretrained_vectors = pretrained_emb_model.vectors
                pretrained_stoi = pretrained_emb_model.stoi

            elif 'elmo' in pretrained_model:
                reduced_vocab = [token for token in corpus_vocab if token not in
                                 ["#", "<EOT>", "<EOL>", "</s>", "<eos>", "<P>", "<unk>"]]
                pretrained_stoi = {v: k for k, v in enumerate(reduced_vocab)}
                elmo_embeddings = elmo_embedder.embed_sentence(reduced_vocab)
                if 'top' in pretrained_model:
                    pretrained_vectors = elmo_embeddings[-1]
                elif 'avg' in pretrained_model:
                    pretrained_vectors = np.average(elmo_embeddings, axis=0)
                pretrained_vectors = torch.from_numpy(pretrained_vectors)

            embed_size = pretrained_vectors.shape[1]

            embedder = nn.Embedding(ntoken, embed_size, padding_idx=0)

            # Filter embeddings only for corpus tokens
            for i, token in enumerate(corpus_vocab):
                if token in pretrained_stoi and token not in ["#", "<EOT>", "<EOL>", "</s>", "<eos>",
                                                              "<P>", "<unk>"]:
                    embedder.weight.data[i] = pretrained_vectors[pretrained_stoi[token]]
                    # Set gradient mask to false for pretrained vocab words
                    self.emb_grad_mask[i].fill_(0)

            self.emb_grad_mask.unsqueeze_(1)
            self.emb_grad_mask.requires_grad = False

            embedder.weight.register_hook(self.grad_mask_hook)

            # corpus_vocab_pretrained_idx = [pretrained_stoi.get(token, 0) for token in corpus_vocab]
            #
            # corpus_vocab_pretrained = torch.index_select(pretrained_vectors, 0,
            #                                              torch.LongTensor(corpus_vocab_pretrained_idx))
            #
            # self.embedder = nn.Embedding.from_pretrained(corpus_vocab_pretrained)

            ### Validation Check ###
            if self.ninp != embed_size:
                print("Warning: default or provided Embedding dimension {} does not match the "
                      "dimensions of the pretrained {} embedding ({}). Using pretrained "
                      "dimensions.".format(self.ninp, pretrained_model, embed_size),
                      file=sys.stderr)

            self.ninp = embed_size

        return embedder

    def grad_mask_hook(self, grad):
        return self.emb_grad_mask.to(grad.device) * grad

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        if not self.using_pretrained:
            self.embedder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    @staticmethod
    def _fix_enc_hidden(hidden):
        """a function specifically for fixing encoding layers for seq2seq"""
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                            hidden[1:hidden.size(0):2]], 2)
        return hidden

    def encode(self, source):
        """takes a source input sequence and encodes them with encoder rnns
         returns hidden layer and memory bank to use for attention."""

        src_emb = embedded_dropout(self.embedder, source,
                                   dropout=self.dropoute if self.training else 0)
        src_emb = self.lockdrop(src_emb, self.dropouti)

        raw_output = src_emb
        new_hidden = []
        for l, rnn in enumerate(self.encoder):
            raw_output, new_h = rnn(raw_output)

            if isinstance(new_h, tuple):  # LSTM
                new_h = self._fix_enc_hidden(new_h[0]), self._fix_enc_hidden(new_h[1])
            else:  # GRU
                new_h = self._fix_enc_hidden(new_h)
            new_hidden.append(new_h)

            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)

        return new_hidden, raw_output  # these are the hidden layers after encoding and the memory bank for attention

    def decode(self, input, hidden, memory_bank=None, return_h=False, bcast=False):
        """takes input and hidden and optionally a memory bank (necessary for attention).
        return_h flag is for training and causes it to return all the interim outputs
        bcast is for inference when using beamsearch (or any type of batch decoding from one input"""

        if self.seq2seq:
            assert memory_bank is not None, "Need a context vector based on source encoding in order to apply attention"

        input_embed = embedded_dropout(self.embedder, input,
                                       dropout=self.dropoute if self.training else 0)
        input_embed = self.lockdrop(input_embed, self.dropouti)

        raw_output = input_embed
        new_hidden = []
        # raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])  #
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        # result = output.view(output.size(0) * output.size(1), output.size(2))
        result = output

        if self.seq2seq and self.attn_type != 'none':
            result, _ = self.attn(result, memory_bank, bcast)

        if return_h:
            return result, hidden, raw_outputs, outputs

        return result, hidden



    def forward(self, input, hidden, return_h=False):
        # TODO if seq2seq encode and decode, otherwise only decode. Currently this forward works for non-seq2seq and pretrained and encode/decode work with seq2seq only (but not pretrained). Fix.

        emb = embedded_dropout(self.embedder, input, dropout=self.dropoute if (self.training and not self.using_pretrained) else 0)
        #emb = self.idrop(emb)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        #print(result.shape)
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]


