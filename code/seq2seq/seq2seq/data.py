import os
import torch
import unicodedata
from utils import make_vocab, load_pickle

from collections import Counter

UNK = 0
SEP = 1
BOS = 2
EOS = 3
PAD = 4

UNK_WORD = '[UNK]'
SEP_WORD = '</s>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_WORD = '[PAD]'

# Globals for import elsewhere
DELIMITERS = {"<EOL>", "#", "<EOT>", "<P>", UNK_WORD, SEP_WORD, BOS_WORD, EOS_WORD, PAD_WORD}
# SRL_TOKS = {"<A0>", "<A1>", "<A2>", "<V>"}
# ENTITY_TOKS = set(["ent{}".format(num) for num in range(75)]) # also used for SRL + Coref
# SPECIAL_CHARACTERS = DELIMITERS | SRL_TOKS | ENTITY_TOKS
QA_TOKS = {"<V>", "<Q>", "<A>"}
SPECIAL_CHARACTERS = DELIMITERS | QA_TOKS


class Dictionary(object):
    def __init__(self):
        self.unk = UNK_WORD
        self.sep = SEP_WORD
        self.word2idx = {PAD_WORD: PAD,
                         UNK_WORD: UNK,
                         BOS_WORD: BOS,
                         SEP_WORD: SEP,
                         EOS_WORD: EOS}
        self.idx2word = [UNK_WORD, SEP_WORD, BOS_WORD, EOS_WORD, PAD_WORD]

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __len__(self):
        return len(self.idx2word)

    def __iter__(self):
        return iter(self.word2idx)

    def __contains__(self, key):
        if type(key) == int:
            return True if key < len(self.idx2word) else False
        elif type(key) == str:
            return self.normalize(key) in self.word2idx

    def __getitem__(self, key):
        if type(key) == int:
            return self.idx2word[key] if key < len(self.idx2word) else None
        if type(key) == str:
            return self.word2idx.get(self.normalize(key),
                                     self.word2idx.get(UNK_WORD))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.idx2word[key] = item
        elif type(key) == str and type(item) == int:
            self.word2idx[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add_word(self, token):
        token = self.normalize(token)
        if token not in self.word2idx:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word.append(token)


    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.word2idx.keys()
                  if k not in {PAD_WORD, UNK_WORD, BOS_WORD, EOS_WORD, SEP_WORD}]
        return tokens

    def string_to_tensor(self, tokens):
        """takes a list of tokens and returns a tensor of the vocab ids"""
        tok_ids = torch.LongTensor(len(tokens))
        for i, token in enumerate(tokens):
            tok_ids[i] = self[token]

        return tok_ids


class Corpus(object):
    def __init__(self, train_path=None, dev_path=None, test_path=None, dict_path=None,
                 delimiter=None, keep_delimiter=False, has_prev_dict=False):
        """delimiter is for seq2seq case to split the train data"""
        if os.path.isfile(dict_path) and has_prev_dict:
            self.dictionary = load_pickle(dict_path)  # a previously saved pickle of a Dictionary
        else:
            self.dictionary = Dictionary()

        self.delimiter = delimiter
        self.keep_delimiter = keep_delimiter

        if train_path:
            self.train = self.tokenize(train_path, applyDict=has_prev_dict)
        if dev_path:
            self.valid = self.tokenize(dev_path, applyDict=has_prev_dict)
        if test_path:
            self.test = self.tokenize(test_path, applyDict=has_prev_dict)

        if not has_prev_dict:
            # save file when done
            make_vocab(self.dictionary, dict_path)


    def tokenize(self, path, applyDict=False):
        """Tokenizes a text file.
        If seq2seq case (indicated by presence of self.delimiter),
        returns a list of tuples of source and target vocab ids.
        Otherwise (LM case) returns list of vocab ids """

        assert os.path.exists(path), path
        # Add words to the dictionary
        if not applyDict:
            with open(path, 'r') as f:
                for line in f:
                    words = line.split()
                    for word in words:
                        self.dictionary.add_word(word)
        # turn text into ids
        with open(path, 'r') as f:
            examples = []
            for line in f:
                # seq2seq version will have source and target, LM version will have target only
                if self.delimiter:
                    try:
                        source, target = line.strip().split(self.delimiter, 1)
                    except:
                        continue
                    if self.keep_delimiter:
                        source = source + ' ' + self.delimiter
                    source_tokens = source.split()
                    source_ids = self.dictionary.string_to_tensor(source_tokens)
                    # only add BOS in seq2seq cases
                    target_tokens = [BOS_WORD] + target.strip().split() + [EOS_WORD]
                    target_ids = self.dictionary.string_to_tensor(target_tokens)
                    examples.append((source_ids, target_ids))

                else:
                    #TODO in the original this was a very long single tensor. OK to have a list of Tensors? No it will probably break everything
                    target_ids = [self.dictionary[word] for word in line.strip().split()] \
                                 + [self.dictionary[EOS_WORD]]
                    examples.extend(target_ids)
            if not self.delimiter:
                examples = torch.LongTensor(examples) #make a tensor. TODO this could be cleaner

        return examples
