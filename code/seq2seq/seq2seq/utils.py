import torch
import spacy
import _pickle as pickle
import sys
import os
import numpy as np
import torchtext.vocab as vc
from allennlp.commands.elmo import ElmoEmbedder
elmo_embedder = None

# Globals for import elsewhere
# SPECIAL_CHARACTERS = {"<EOL>", "<eos>", "#", "<EOT>", "</s>", "<P>"}
SPECIAL_CHARACTERS = {"<V>", "<Q>", "<A>", "<EQA>"}

def init_nlp_model(special_chars=SPECIAL_CHARACTERS,
                   model_name="en_core_web_lg"):
    """inits a spacy model and adds custom special chars to tokenizer. Returns model"""
    nlp = spacy.load(model_name)
    for key in special_chars:
        nlp.tokenizer.add_special_case(key, [dict(ORTH=key)])

    return nlp

def load_pickle(path):
    with open(path, 'rb') as fin:
        obj = pickle.load(fin)
    return obj


def make_vocab(corpus_dictionary, vocab_path):
    """take data, create pickle of vocabulary"""
    with open(vocab_path, 'wb') as fout:
        pickle.dump(corpus_dictionary, fout)
    print('Saved dictionary to', vocab_path)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, shuffle=True):
    """Transform data into batches."""
    # sort to reduce padding
    data = sorted(data, key=lambda z: (z[1].size(0), z[0].size(0)))
    batched_data = []
    for i in range(len(data)):
        if i % bsz == 0:
            batched_data.append([data[i]])
        else:
            batched_data[len(batched_data) - 1].append(data[i])
    if shuffle:
        np.random.shuffle(batched_data)
    return batched_data

def get_batch(data, i):
    """ data is the tensor of source and targets, already batched, i is the batch number
    :param data: tensor of preprocessed data tensors
    :param i: int, batch number
    :return: LongTensors of source and target
    """

    max_src_len = 0
    max_tgt_len = 0
    for ex in data[i]:
        # iterate over all examples to get max lengths for init tensors
        max_src_len = ex[0].size(0) if max_src_len < ex[0].size(0) else max_src_len
        max_tgt_len = ex[1].size(0) if max_tgt_len < ex[1].size(0) else max_tgt_len
    source = torch.zeros(len(data[i]), max_src_len).long() # dim len(this batch) x max source
    target = torch.zeros(len(data[i]), max_tgt_len).long() # dim len(this batch x max target
    for idx, ex in enumerate(data[i]): # for all examples in batch
        source[idx, :ex[0].size(0)] = ex[0]  # fill out row by row and then return
        target[idx,:ex[1].size(0)] = ex[1]
    #print(target_data,file=sys.stderr)
    return source, target

# def get_batch(data, i, target_cap=None):
#     """ data is the tensor of source and targets, already batched, i is the batch number
#     :param data: tensor of preprocessed data tensors
#     :param i: int, batch number
#     :param target_cap: used to cap the length of data used for targets
#     :return: LongTensors of source and target
#     """
#
#     max_src_len = 0
#     max_tgt_len = 0 if not target_cap else target_cap
#     for ex in data[i]:
#         # iterate over all examples to get max lengths for init tensors
#         max_src_len = ex[0].size(0) if max_src_len < ex[0].size(0) else max_src_len
#         if not target_cap:
#             max_tgt_len = ex[1].size(0) if max_tgt_len < ex[1].size(0) else max_tgt_len
#
#     source = torch.zeros(len(data[i]), max_src_len).long() # dim len(this batch) x max source
#     target = torch.zeros(len(data[i]), max_tgt_len).long() # dim len(this batch x max target
#     for idx, ex in enumerate(data[i]): # for all examples in batch
#         source[idx, :ex[0].size(0)] = ex[0]  # fill out row by row and then return
#         target_data = ex[1] if not target_cap else ex[1][:target_cap]
#         target[idx,:ex[1].size(0)] = target_data
#     #print(target_data,file=sys.stderr)
#     return source, target

def process_new_tokens(tokens,processed_tokens_set, model, dictionary):
    """used to update a model embedding layer given a bunch of tokens"""
    if hasattr(model, 'using_pretrained') and model.using_pretrained is not None:
        processed_tokens_set.update(tokens)
        update_embedding_layer(processed_tokens_set, model, dictionary)


def update_embedding_layer(line_tokens, model, dictionary):
    """
    Add new words in embedding layer and dictionary when OOV words which are present
    in pretrained embeddings are encountered.
    """

    global elmo_embedder

    out_of_corpus_vocab = [word for word in line_tokens if word not in dictionary.word2idx]
    if len(out_of_corpus_vocab) == 0:
        return

    print("OOV words found:", out_of_corpus_vocab, file=sys.stderr)

    if model.using_pretrained == "fasttext":
        pretrained_emb_model = vc.FastText()
    elif model.using_pretrained == "glove":
        pretrained_emb_model = vc.GloVe()
    elif model.using_pretrained == "elmo_top" or model.using_pretrained == "elmo_avg":
        dirname = os.path.dirname(__file__)
        options_file = os.path.join(dirname, '../elmo-config/elmo_2x4096_512_2048cnn_2xhighway_options.json')
        weight_file = os.path.join(dirname, '../elmo-config/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')
        # Retrieve cached version if already loaded, saves a lot of time if this code path is executed multiple times
        if elmo_embedder is None:
            elmo_embedder = ElmoEmbedder(options_file, weight_file)
    else:
        raise Exception("Unsupported embedding model:", model.using_pretrained)

    print("Using Pretrained embedding:", model.using_pretrained)

    if model.using_pretrained in ['fasttext', 'glove']:

        pretrained_vectors = pretrained_emb_model.vectors
        pretrained_stoi = pretrained_emb_model.stoi

    elif 'elmo' in model.using_pretrained:
        # We get the embeddings for all line_tokens which includes both known and unknown, if any.
        # As elmo is character level we can find embedding of any word
        reduced_vocab = [token for token in line_tokens if token not in
                         ["#", "<EOT>", "<EOL>", "</s>", "<eos>", "<P>", "<unk>"]]
        pretrained_stoi = {v: k for k, v in enumerate(reduced_vocab)}
        elmo_embeddings = elmo_embedder.embed_sentence(reduced_vocab)
        if 'top' in model.using_pretrained:
            pretrained_vectors = elmo_embeddings[-1]
        elif 'avg' in model.using_pretrained:
            pretrained_vectors = np.average(elmo_embeddings, axis=0)
        pretrained_vectors = torch.from_numpy(pretrained_vectors)
    out_of_corpus_vocab = [word for word in line_tokens if word not in dictionary.word2idx and word in pretrained_stoi]

    if len(out_of_corpus_vocab) == 0:
        return

    # Update for only unknown/new word
    new_vectors = []
    for word in out_of_corpus_vocab:
        dictionary.add_word(word)
        new_vectors.append(pretrained_stoi[word])

    new_vectors = torch.index_select(pretrained_vectors, 0, torch.LongTensor(new_vectors))

    model.embedder = torch.nn.Embedding.from_pretrained(torch.cat([model.embedder.weight, new_vectors]))
    if model.tie_weights:
        model.decoder.weight = model.encoder.weight
