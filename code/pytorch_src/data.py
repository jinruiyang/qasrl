import os
import torch
import unicodedata
from collections import Counter
from utils import load_pickle, make_vocab


UNK_WORD = '[UNK]'
SEP_WORD = '</s>'
BOS_WORD = '<bos>'
EOS_WORD = '<eos>'
PAD_WORD = '[PAD]'

# Globals for import elsewhere
DELIMITERS = {"<EOL>", "#", "<EOT>", "<P>", UNK_WORD, SEP_WORD, BOS_WORD, EOS_WORD, PAD_WORD}
QA_TOKS = {"<V>", "<Q>", "<A>"}
SPECIAL_CHARACTERS = DELIMITERS | QA_TOKS
BERT_CHARS = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"}
GPT2_CHARS = {"<|endoftext|>"}


class Dictionary(object):
    def __init__(self):
        self.unk = UNK_WORD
        self.sep = SEP_WORD
        self.word2idx = {self.unk: 0, self.sep: 1}
        self.idx2word = [self.unk, self.sep]
        self.counter = Counter()
        self.total = 0
    
    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)
   
    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        #self.counter[token_id] += 1
        #self.total += 1
        return self.word2idx[word]

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


class Corpus(object):
    def __init__(self, applyDict=False, **kwargs):
        #TODO: document this
        """
        :param applyDict:
        :param kwargs: 'train_path' 'dev_path' 'test_path', 'dict_path', 'applyDict'
        """

        if applyDict:
            self.dictionary = load_pickle(kwargs['dict_path'])  # a previously saved pickle of a Dictionary
        else:
            self.dictionary = Dictionary()

            if 'train_path' in kwargs.keys():
                self.train = self.tokenize(kwargs['train_path'])
            if 'dev_path' in kwargs.keys():
                self.valid = self.tokenize(kwargs['dev_path'])
            if 'test_path' in kwargs.keys():
                self.test = self.tokenize(kwargs['test_path'])
            # save file when done
            make_vocab(self.dictionary, kwargs['output'])


    def tokenize(self, path, applyDict=False):
        """Tokenizes a text file."""
        assert os.path.exists(path), path
        # Add words to the dictionary
        tokens = 0
        ids = list()
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    if not applyDict:
                        self.dictionary.add_word(word)
                    ids.append(self.dictionary.word2idx.get(word, 0))

        ids = torch.LongTensor(ids)
        return ids
