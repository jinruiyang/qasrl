import sys
import numpy as np
import itertools
from itertools import filterfalse
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from typing import List

sys.path.append(".")
from decoder.candidate import Candidate
from decoder.StaticCoefficientModel import StaticCoefficientModel

from data import SPECIAL_CHARACTERS, UNK_WORD, BOS_WORD


def bool_partition(func, iterable):
    """takes a function that return a bool and and iterable returns two generators from iterable, the true and the false"""
    return list(filter(func, iterable)), list(filterfalse(func, iterable))

def normalize_score(score, len_y, alpha=0.6):
    """takes a score, a length, and an alpha and normalizes the score and returns.
    Based on Wu et al. (2016)"""
    norm_factor = ((5 + len_y) ** alpha) / ((5 + 1) ** alpha)
    return score/norm_factor

def concat_hidden(beam, nlayers, m_per_layer=2):
    """
    takes a beam of Candidates and a number of layers and makes new concatenated layers for batching efficiency
    :param beam:
    :param nlayers:
    :param m_per_layer: matrices per later. Defaults to 2 for LSTMs
    :return: list of tuples, one tuple per layer, that are concatenation of all layers belonging
    to candidates
    """
    new_hidden = []
    for l in range(nlayers):
        # need to add an additional dimension before concatenation else get (1, 7500) instead of (1, 5, 1500) with beam of 5 and hidden layers 1500
        new_layer = tuple([torch.cat([cand.hidden[l][i].unsqueeze(1) for cand in beam], dim=1)
                           for i in range(m_per_layer)])
        new_hidden.append(new_layer)
    return new_hidden


def logprobs(model, sequence_pair):
    """
        basically runs a decode to get the model likelihood of a true sequence and continuation.
        :param model:
        :param sequence_pair: a tuple of a list of ints of prefix and a list of ints of continuation
        :return: output layer after running through softmax
        """
    use_cuda = next(model.parameters()).is_cuda

    if getattr(model, "seq2seq", False): # need to use getattr for backwards compatibility of models trained before seq2seq code
        pass  # TODO support this - logprobs is called when learning coefficients
    else:
        #seqs = sequence_pair[0] + sequence_pair[1]
        hidden = model.init_hidden(len(sequence_pair)) # init hidden to length of iterable, which should be the init and cont token integers
    if use_cuda:
        source = Variable(torch.LongTensor(sequence_pair).t().cuda())
    else:
        source = Variable(torch.LongTensor(sequence_pair).t())
    output, hidden = model(source, hidden) # forward
    decoded_data = model.decoder(output.data)
    output = nn.functional.log_softmax(decoded_data, dim=decoded_data.dim() - 1).data #take softmax along the final dimension
    #print(output.shape)
    return output

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # print(len(logits))
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (..., vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    # assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value


    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs >= top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(logits, dtype=torch.uint8).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove )
        logits[indices_to_remove] = filter_value
    return logits

class CoefTrainer:
    """class for training scorer coefficients"""

    def __init__(self, model, num_scorers, ranking_loss, lr):
            self.model = model # since coefficients are tuned to a specific language model
            self.weight_model = StaticCoefficientModel(num_scorers)
            self.use_ranking_loss = ranking_loss
            if self.use_ranking_loss:
                self.loss = nn.MarginRankingLoss()
            else:
                self.loss = nn.MSELoss()
            self.optimizer = optim.SGD(self.weight_model.parameters(), lr=lr)
            self.total_loss, self.total_n, self.total_correct = 0, 0, 0

    def train_coefficients(self, init_tokens, true_cont_tokens, candidate, gold_cont_raw_scores):
        self.weight_model.zero_grad()
        truth_lm_scores = logprobs(self.model, [init_tokens + true_cont_tokens]).squeeze().cpu().data.numpy()  # this will be the shape of (len input x embed dimension) where len input is init + cont
        truth_lm_score = sum([truth_lm_scores[i + len(init_tokens) - 1, true_cont_tokens[i]] for i in
                              range(len(true_cont_tokens))])  # this is just the probability of the sequence #TODO is it necessary for init and cont tokens to be separate?
        lm_scores = torch.Tensor([truth_lm_score, candidate.score])  # this is the probability of the true sequence paired with the score of the best sequence. Both floats
        # print("LM pair", lm_scores)
        training_pair = [gold_cont_raw_scores, candidate.raw_scores]  # this is scorer scores of gold continuation, and of the best continuation. Both 1D arrays of len num scorers.
        training_pair = torch.Tensor(np.stack(training_pair))  # so this is now one row per scorer, with gold and best candidate as columns
        #print("Training pair", training_pair)
        # if self.use_cuda:
        #    training_pair.cuda()
        pair_scores = self.weight_model(training_pair).squeeze()
        #print("pair scores returned", pair_scores)
        pair_scores = pair_scores + lm_scores
        #print("pair scores concat", pair_scores)
        pred = pair_scores[0] - pair_scores[1]

        if self.use_ranking_loss:
            loss = self.loss((pair_scores[0]).unsqueeze(0),
                             (pair_scores[1]).unsqueeze(0), Variable(torch.ones(1)))

        else:
            loss = self.loss(pred,
                             torch.FloatTensor([0]))  # use MSELoss, ((input-target)**2).mean()
        # print(loss.data.item())
        loss.backward()
        self.total_loss += loss.data.item()
        if self.use_ranking_loss and loss.data.item() == 0:
            self.total_correct += 1  # whether or not it is correct is whether the scorer did in fact say the gold was higher rank
        self.total_n += 1
        if self.total_n % 200 == 0:
            if self.use_ranking_loss:
                print('Train Accuracy: %f' % (self.total_correct / self.total_n))
            print('Loss: %f' % (self.total_loss / self.total_n))
            sys.stdout.flush()

        self.optimizer.step()
        self.weight_model.coefs.weight.data = self.weight_model.coefs.weight.data.clamp(min=0)

        return loss


class Decoder:
    """Most abstract class for decoding, shared between sampling and beam"""

    def __init__(self, model, verbosity=0, dictionary=None, temperature=None, max_len=1000,
                 sep=None, human_readable=False, norm_scores=False):
        #TODO make sure norm_scores incorporated into both sampling and beam
        self.model = model
        self.sep = sep  # only necessary for using only one mode in decode. separates sentences or phrases etc
        self.temperature = temperature
        self.max_len = max_len  # alternate criteria for terminating generation
        self.verbosity = verbosity
        self.dictionary = dictionary
        self.human_readable = human_readable # strips special tokens before returning output, to facilitate readability
        # Norm scores is at the decoder level because it is overloaded: it controls whether the discriminators normalize and also whether the beamsearch normalizes.
        # Open question whether they should in fact be separate
        self.norm_scores = norm_scores

        if self.human_readable:
            ## check if this dictionary has special_chars. If not, make them and print.
            if not hasattr(self.dictionary, "special_toks"):
                self.dictionary.special_toks = SPECIAL_CHARACTERS - {UNK_WORD}
                self.dictionary.special_tok_ints = set([dictionary[tok] for tok in self.dictionary.special_toks])
                print( "Loaded dictionary does not have preset special characters. Adding the default set:"
                    "\n{}".format(self.dictionary.special_toks), file=sys.stderr)

    def encode_source(self, tokens, use_cuda):
        """encode source only once, used for seq2seq
        :param tokens: a list of vocabulary ints"""
        assert(getattr(self.model, "seq2seq", False)), "model needs to be seq2seq to encode a source"
        prefix_tensor = torch.LongTensor(tokens).unsqueeze(0) # batch size of 1
        if use_cuda:
            prefix_tensor = prefix_tensor.cuda()
        hidden, src_output = self.model.encode(prefix_tensor)
        # starts the decoding with a special start symbol
        tokens = [self.dictionary.word2idx[BOS_WORD]]
        return hidden, src_output, tokens

    def clean_output(self, output: List[int]) -> List[int]:
        """strips special tokens from output sequence"""
        # this is special logic for entity numbers
        #print(output)
        numbers = set([self.dictionary[str(num)] for num in range(100)]) # get the ints corresponding to number strings
        new_output = []
        prev_ent_tag = False
        ent_tag_ints = [self.dictionary[ent] for ent in ENTITY_TOKS]
        # clear the numbers after entity tags. Assumes all entities followed by number
        for tok_int in output:
            if prev_ent_tag and tok_int in numbers:
                prev_ent_tag = False
                continue
            if tok_int in ent_tag_ints:
                prev_ent_tag = True
                continue
            new_output.append(tok_int)

        final_output = list(filter(lambda x: x not in self.dictionary.special_tok_ints, new_output))
        #print(final_output)
        return final_output

class SamplingDecoder(Decoder):
    """used for decoding via sampling. Unlike beam, incorporates rerank and standard capability into one decoder."""
    def __init__(self, model, verbosity=0, dictionary=None, temperature=1, max_len=1000, sep=None,
                 eos_id=None, scorers=None, coefs=None, dedup=False, apply_disc=False, learn=False,
                 lr=0.01, ranking_loss=False, human_readable=False, nucleus=False, top_k=0, top_p=0):
        # Temperature is 1 since need temp for sampling
        super().__init__(model, verbosity, dictionary, temperature, max_len, sep, human_readable)
        self.dedup = dedup  # this enables us to set a default dedup strategy that can be overridden
        self.ntokens = len(dictionary) #vocab length #TODO make it so dictionary above is not optional
        self.input_tensor = torch.rand(1, 1).mul(self.ntokens).long() #since it is fixed length with vocab it makes sense for this to be part of the init
        self.eos_id = eos_id #TODO standardise this between decoders --> beamrank uses a set "terms" instead, then it can be in the higher level class
        self.scorers = scorers
        self.coefs = coefs
        self.apply_disc = apply_disc
        self.learn = learn
        self.nucleus = nucleus
        self.top_p = top_p
        self.top_k = top_k


        if self.learn:
            self.coef_trainer = CoefTrainer(model, len(scorers), ranking_loss, lr)
            self.coefs = self.coef_trainer.weight_model.coefs.weight.data.cpu().squeeze().numpy()

    def decode(self, init_tokens, true_cont_tokens=None, temperature=None, keep_end=False, only_one=False,
               k_samples=5, dedup=None, terms=set()):
        #TODO support only_one
        ### validation checks
        assert ((not self.learn) or true_cont_tokens)
        use_cuda = next(self.model.parameters()).is_cuda
        if temperature is None:
            temperature = self.temperature # can't be None, because of sampling
            assert temperature, "cannot have sampling temperature be None in both the default for the decoder object and the decode args"
        if use_cuda:
            self.input_tensor.data = self.input_tensor.cuda() #TODO make sure this is a good idea, might have been introduced
        if dedup is None:
            dedup = self.dedup

        terms.update([self.eos_id, self.sep]) # since don't want either of these to be affected by dedup
        
        rescore = self.apply_disc #TODO add support for not rescoring until X distance in
        if rescore:
            assert len(self.scorers) and len(self.coefs), "Need to use scorers and weights in order to rerank"

        ### decoding starts here
        with torch.no_grad():
            # "encode" prefix step
            if getattr(self.model, "seq2seq", False):
                # print(init_tokens)
                hidden, context, start_tok = self.encode_source(init_tokens, use_cuda)
                # print(hidden)
                self.input_tensor.data.fill_(start_tok[0])  # it is returned in a list because beamsearch uses the list
                output, hidden = self.model.decode(self.input_tensor, hidden, context)
            else:
                hidden = self.model.init_hidden(1)
                for word_idx in init_tokens:
                    self.input_tensor.data.fill_(word_idx)
                    output, hidden = self.model(self.input_tensor, hidden)

            num_cont_words = 1 #faster to keep count then call len() every loop
            curr_candidate = Candidate(init_tokens, [])
            already_used = set() # only will be used if deduping

            while num_cont_words < self.max_len and curr_candidate.next_token != self.eos_id:
                output = self.model.decoder(output) # Tensor of shape 1, Vocab_size
                #if wanna use top_k sampling, add "--nucleus --top_k k“ (k should be an int） to run generation example scripts
                #if wanna use top_p sampling, add "--nucleus --top_p p“ (p should be a float) to run generation example scripts
                if self.nucleus:
                    output = output[:, -1, :] / 1
                    output = top_k_top_p_filtering(output, top_p=self.top_p, top_k=self.top_k)
                word_weights = output.squeeze().data.div(temperature).exp().cpu()
                cand_cont_tokens = [tok.item() for tok in torch.multinomial(word_weights, 1)] # to retrieve the IDs from the tensors
                # get token probabilities
                # TODO check if I should be using word_weights (instead of raw softmax) here since there will always be a temperature...
                ps = nn.functional.log_softmax(output, dim=output.dim() - 1).squeeze().data
                cand_scores = [ps[i] for i in cand_cont_tokens]
                # create candidates
                new_candidates = [Candidate(curr_candidate.tokens + [cand_cont_tokens[i]],
                                            curr_candidate.cont_tokens + [cand_cont_tokens[i]],
                                            next_token=cand_cont_tokens[i],
                                            score=cand_scores[i])
                                  for i in range(len(cand_cont_tokens))]
                cont_tokens = [cand.cont_tokens for cand in new_candidates] # nested list
                # potentially rescore
                # also rescore gold if learning
                if self.learn and num_cont_words < len(true_cont_tokens):  # add gold answer to the list
                    cont_tokens.append(true_cont_tokens[:num_cont_words]) # this appends all gold cont tokens up to the step number - so more gold on each iteration

                score_adjustment = np.zeros(len(cont_tokens))
                if rescore:  # add score adjustment according to the scorers.
                    all_raw_scores = []
                    for coef, scorer in zip(self.coefs, self.scorers): # Note that this throws an error if there is just one scorer
                        # this makes an array for each scorer from calling the scorer forward function on the candidate tokens
                        # TODO could potentially make this faster by considering shared continuation to be init_tokens
                        new_scores = scorer(init_tokens, cont_tokens, self.norm_scores)
                        raw_scores = np.asarray(new_scores)
                        all_raw_scores.append(raw_scores)
                        # elementwise add the new scores to the np array after elementwise multiplying by coef
                        score_adjustment += raw_scores[:len(cont_tokens)] * coef  # TODO why restrict to len(candidates)? It seems like the scorer sometimes but not always returns +1 more result than candidates
                    all_raw_scores = np.stack(all_raw_scores, axis=-1)  # this converts to num_candidates x num_scorers so each row is all adjusted scores for a candidate

                    if self.learn and num_cont_words < len(true_cont_tokens):
                        gold_cont_raw_scores = all_raw_scores[-1]

                for i in range(len(new_candidates)):
                    new_candidates[i].adjusted_score = cand_scores[i] + score_adjustment[i]
                    if rescore:
                        new_candidates[i].raw_scores = all_raw_scores[i]
                if rescore:
                    new_candidates.sort(key=lambda c: c.adjusted_score, reverse=True) #TODO should I sort if NOT rescoring? It's not the same behavior as previously if I do (since multinomial doesn't return samples in sorted order)

                if dedup:
                    for cand in new_candidates:
                        curr_candidate = cand # because of how this works, it will still use the last sample idx even if it is a repeat so it doesn't truly dedup if there are no other options it will allow a repeat.
                        if cand.next_token not in already_used or cand.next_token in terms:
                            already_used.add(cand.next_token)
                            break
                else:
                    curr_candidate = new_candidates[0] # basically just use the first one after sorting

                num_cont_words += 1

                self.input_tensor.data.fill_(curr_candidate.next_token)

                if getattr(self.model, "seq2seq", False):
                    output, hidden = self.model.decode(self.input_tensor, hidden, context)
                else:
                    output, hidden = self.model(self.input_tensor, hidden)

        if self.learn:
            loss = self.coef_trainer.train_coefficients(init_tokens, true_cont_tokens, curr_candidate, gold_cont_raw_scores)  #TODO change beam[0] to best

        if not keep_end and curr_candidate.tokens[-1] == self.eos_id:
            curr_candidate.tokens.pop()

        if self.human_readable:
            output_tok = curr_candidate.tokens[len(init_tokens):]
            curr_candidate.tokens = init_tokens + self.clean_output(output_tok)
        # print(curr_candidate.tokens)
        return curr_candidate.tokens if not self.learn else loss

class BeamDecoder(Decoder):
    """Upper lever class for Beamsearch and Beamrank decoders"""

    def __init__(self, model, beam_size, verbosity=0, dictionary=None, temperature=None,
                 max_len=1000, sep=None, norm_scores=True, human_readable=False):
        super().__init__(model, verbosity, dictionary, temperature, max_len, sep, human_readable, norm_scores)
        self.beam_size = beam_size

    def encode_init_data(self, tokens):
        """if seq2seq will encode everything and set tokens to the BOS symbol, otherwise will do nothing"""
        use_cuda = next(self.model.parameters()).is_cuda
        if getattr(self.model, "seq2seq", False):
            cond_data = tokens # for returning at the end since seq2seq doesn't use cond_data in the LM
            hidden, context, tokens = self.encode_source(tokens, use_cuda)
        else:
            hidden, context = None, None # LSTM will not start with a hidden and will never have context
        return hidden, context, tokens


    def top_k_next(self, beam, k, context=None, temperature=None, first_pass=False):
        """
        takes a current beam, and returns the next expansion of the beam
        :param beam: a list of Candidate objects encoding the beam sequence so far
        :param k: the k number of candidates for this expansion
        :param context: the context vector from encoding, if using seq2seq
        :param temperature: the temp to use in softmax. Only valid if sampling.
        :param first_pass: beam is a nested list, if this is the first pass it will be a nested list of one element. used later to deal with the first beam expansion differently
        :return: a list lists of Candidates after the expansion (where the outer list corresponds to
        the starting candidates that were expanded and the inner to their expansions)
        """
        # cuda check
        use_cuda = next(self.model.parameters()).is_cuda

        assert (len(beam) > 0)

        with torch.no_grad():
            if first_pass:
                tokens = [cand.tokens for cand in beam] #if seq2seq this will be the start token, if not this will be the context to be continued
                if beam[0].hidden is None: # only ever true for LSTM case
                    assert(not getattr(self.model, "seq2seq", False)), "must encode before beginning decoding if model is seq2seq"
                    hidden = self.model.init_hidden(len(beam))
                else:
                    hidden = beam[0].hidden
            else:
                # this is making a tuple of tensors for each layer in hidden to track the LSTM matrices.
                # Shape is list of tuples of tensors. Used for efficiency in batching the forward function
                hidden = concat_hidden(beam, self.model.nlayers)
                assert(len(hidden) == self.model.nlayers)
                tokens = [[cand.next_token] for cand in beam]

            # tokens are a list of the next token from the previous step, coindexed with hidden layers
            if use_cuda:
                source = Variable(torch.LongTensor(tokens).t().cuda())
            else:
                source = Variable(torch.LongTensor(tokens).t())

            if getattr(self.model, "seq2seq", False):
                source = source.transpose(0,1) # since batch needs to be first
                output, hidden = self.model.decode(source, hidden, context, bcast=True)  # calls forward pass, returns a tensor and a list of tuples of tensors for LSTM
            else:
                output, hidden = self.model(source, hidden)  # calls forward pass, returns a tensor and a list of tuples of tensors for LSTM

            decoded_data = self.model.decoder(output.data)
            ps = nn.functional.log_softmax(decoded_data,
                               dim=decoded_data.dim() - 1).data  # gives logprobs based on softmax across last dimension. Means that each slice along this dimension sums to one.

        if getattr(self.model, "seq2seq", False):
            ps = ps.squeeze(1)
        #if first_pass:
        #    ps = ps[-1, :]
        if not temperature:
            _, idxs = ps.topk(k)  # returns tuple of top-k values and top-k indices of the softmax transformed layers
            #print(idxs)
        else:
            #idxs = ps.div(temperature).exp().multinomial(k) this is theirs in the comment but is hacky and unclear
            word_weights = decoded_data.squeeze().data.div(temperature).exp().cpu()
            idxs = torch.multinomial(word_weights, k)
        idxs_np = idxs.cpu().numpy()  # get numpy array of topk indices
        # print(idxs_np)

        if first_pass:
            # need to select the last row, since this corresponds to the expansions of the final word in the input (and we don't want to try to expand the others)
            # also need to insert a dummy dimension so that indexing later works (since expect a 2D option as later we will be expanding more than one Candidate per search)
            if not getattr(self.model, "seq2seq", False):
                ps = ps[-1, :].unsqueeze(0)
                idxs_np = np.expand_dims(idxs_np[-1], axis=0)

        beam_cands = []
        for i in range(len(beam)):  # iterate across all live paths in beam
            ith_cands = []
            base_score = beam[i].score
            # get corresponding hidden of the candidate in question after the transformation. This should basically be undoing the concatenation from earlier. The slicing must be along a column and the layers look like (1, beam size, num hidden), hence [:, i, :] for a slice
            cur_hidden = [ (hidden[l][0][:, i, :].clone(), hidden[l][1][:, i, :].clone()) for l in range(self.model.nlayers) ]
            for j in range(k):  # iterate over all possible expansions of beam for the current path. for each expansion they are in sorted order (as per pytorch topk) but the cumulative scores may not be sorted.
                next_word = int(idxs_np[i, j])
                nu_score = base_score + ps[i, next_word] #increment score of entire path
                if self.norm_scores:
                    # this is normalization by length as per Wu et al. (2016)
                    nu_score = normalize_score(nu_score, len(beam[i].tokens)+1)

                nu_cand = Candidate(beam[i].tokens + [next_word],
                                beam[i].cont_tokens + [next_word],
                                next_word,
                                score=nu_score,
                                # TODO (maybe done?) make sure this is also modified by normalization later in rerank since latest score is only touched by scorers
                                latest_score=beam[i].latest_score,  # I find it confusing that score is the most up to date and latest score is not. And why we need latest score. But maybe it makes more sense with the scorers...
                                hidden=cur_hidden)
                ith_cands.append(nu_cand)
            beam_cands.append(ith_cands)
        return beam_cands  # return full set of candidates. This will be a list of lists of candidates (k*k).


class BeamSearchDecoder(BeamDecoder):

    def __init__(self, model, beam_size, end_tok, verbosity=0, dictionary=None, temperature=None,
                 max_len=1000, sep=None, norm_scores=True, human_readable=False):

        super().__init__(model, beam_size, verbosity, dictionary, temperature, max_len, sep,
                         human_readable, norm_scores)
        self.end_tok = end_tok  # used to knowing when to terminate generation

    def decode(self, tokens, temperature=None, keep_end=False, only_one=False):
        """
        :param tokens: list of ints corresponding to vocab words
        :param temperature: softmax temp
        :param keep_end: controls whether to pop off final token or not
        :param only_one: whether to generate only one (sentence, word, etc) based on a delimiter
        :return list of ints corresponding to vocab words - either at max length or ending with
        end token
        """
        # Validation checks
        if temperature is None:
            temperature = self.temperature
        if only_one:
            assert self.sep is not None, "Need to provide a sep token in decoder init in order to use only one mode"

        use_cuda = next(self.model.parameters()).is_cuda
        end_tok = {self.sep, self.end_tok} if only_one else {self.end_tok}

        hidden, context, start_tokens = self.encode_init_data(tokens) # if not seq2seq it won't modify anything

        beam = [ Candidate(start_tokens, [], hidden=hidden) ]
        beam = self.top_k_next(beam, self.beam_size, context, temperature, first_pass=True)
        beam = beam[0]  # since initialising this basically unpacks the nested list
        step = 0
        final_candidates = []
        while step < self.max_len and beam and len(final_candidates) <= self.beam_size:
            conts = self.top_k_next(beam, self.beam_size, context, temperature)
            beam = sorted([candidate for candidates in conts for candidate in candidates],
                          key=lambda c: c.score,
                          reverse=True)
            if not temperature:
                if len(beam) > self.beam_size:
                    beam = beam[:self.beam_size]
            else:
                if len(beam) > self.beam_size:
                    p = np.asarray(list(map(lambda c: c.score, beam)))
                    p = np.exp(p / temperature)
                    p /= p.sum()
                    beam = np.random.choice(beam, size=self.beam_size, replace=True, p=p)
            has_end_tok, lacks_end_tok = bool_partition(lambda cand: cand.tokens[-1] in end_tok, beam)
            final_candidates.extend(has_end_tok)
            beam = lacks_end_tok
            step += 1
        if not final_candidates: # support for a model going off the rails and never generating an end_tok
            print('None of candidates had end token: {}. Picking best available'.format(end_tok),
                  file=sys.stderr)
            best = max(beam, key=lambda c: c.score)
        else:
            best = max(final_candidates, key=lambda c: c.score)  # TODO this might be unnecessary if partition is in place

        if not keep_end and best.tokens[-1] in end_tok:
            best.tokens.pop()
        if self.verbosity: #debug
            for cand in final_candidates:
                print("Score: {} \n Text: {}".format(cand.score,
                                       " ".join([self.dictionary.idx2word[token]
                                                              for token in cand.tokens])))
        if self.human_readable:
            best.tokens = self.clean_output(best.tokens)

        if getattr(self.model, "seq2seq", False):
            return start_tokens + best.tokens
        else:
            return best.tokens  # cond data is included as part of decoding



class BeamRerankDecoder(BeamDecoder):
    # TODO support beamrank with seq2seq
    def __init__(self, model, scorers, coefs,
                 learn=False, lr=0.01, rescale_scores=True,
                 ranking_loss=False,
                 beam_size=32, terms=[1], temperature=None,
                 verbosity=0, dictionary=None,
                 max_len=150, forbidden=[], sep=1, use_cuda=True, norm_scores=True, human_readable=False):
        assert(not getattr(model, "seq2seq", False)), "BeamRerank is not currently supported for seq2seq models"
        super().__init__(model, beam_size, verbosity, dictionary, temperature, max_len, sep, human_readable, norm_scores)

        self.scorers = scorers
        self.coefs = np.asarray(coefs)
        self.rescale_scores = rescale_scores
        self.terms = set(terms)
        self.learn = learn
        self.forbidden = set(forbidden)
        self.use_cuda = use_cuda

        if self.learn:
            self.coef_trainer = CoefTrainer(model, len(scorers), ranking_loss, lr)
            self.coefs = self.coef_trainer.weight_model.coefs.weight.data.cpu().squeeze().numpy()

    def decode(self, init_tokens, cont_tokens=None, temperature=None, rescore_min=1,
               min_sentences=5, only_one=False, keep_end=False):
        """
        :param init_tokens: ints corresponding to vocab
        :param cont_tokens: ints corresponding to gold continuation, required if learning
        :param temperature: affects broadness of search
        :param rescore_min: the minimum sentences for generate before applying rescore
        :param min_sentences: the minimum sentences to generate before stopping
        :param only_one: if True, makes the seperator token also an end token.
        :param keep_end: controls whether to pop off final token or not
        :return: if not in learn mode, the beam sequence tokens. If in learn mode, the diff score between...
        """
        ### Validation checks
        assert((not self.learn) or cont_tokens)
        if temperature is None:
            temperature = self.temperature # TODO this is not ideal since it forces using beamrank with temp unless the decoder was initialised with None

        if only_one:
            self.terms.add(self.sep)
            min_sentences = 1

        hidden, context, start_tokens = self.encode_init_data(init_tokens)  # if not seq2seq it won't modify anything

        beam = [ Candidate(start_tokens, [], hidden=hidden) ]
        beam = self.top_k_next(beam, self.beam_size, context, first_pass=True)[0]  # picks first of the k returned as part of init. Since this is the first expansion, it is presorted
        beam = list(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, beam)) # filter out options where final continuation token is in the list of forbidden (usually unk)
        sentences_count = 1
        gold_cont_raw_scores, best = None, None
        step = 2  # used for learning below and also to control max iterations. But why must it start at 2, shouldn't it start at 1?
        while (((best is None) or (best.adjusted_score < max(map(lambda c: c.score, beam)))) and (step < self.max_len)):  # max len is 150 seemingly arbitrarily. So that it can't beam search forever
            rescore = True if (len(self.scorers) and sentences_count > rescore_min) else False # whether to rescore

            if self.verbosity > 0:
                print("rescore: ", rescore)
                for c in beam:
                    print(' '.join([self.dictionary[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('-'*30)

            #get topk next
            conts = self.top_k_next(beam, self.beam_size, context, temperature)

            if self.verbosity > 0:
                for cs in conts:
                    for c in cs:
                        print(' '.join([self.dictionary[i] for i in c.cont_tokens]) + ' %f' % c.score)
                print('*'*50)
                if self.verbosity > 2:
                    input()

            candidates, cand_cont_tokens = [], []
            for cands in conts:
                for candidate in cands:
                    candidates.append(candidate)
                    cand_cont_tokens.append(candidate.cont_tokens)  # this will append all continuation tokens of each candidate

            if self.learn and step < len(cont_tokens):  # add gold answer to the list
                cand_cont_tokens.append(cont_tokens[:step]) # this appends all gold cont tokens up to the step number - so more gold on each iteration

            # score adjustment section
            score_adjustment = np.zeros(len(candidates)) # since this is redone on each while loop - candidates is len k*k
            if rescore:  # add score adjustment according to the scorers.
                all_raw_scores = []
                for coef, scorer in zip(self.coefs, self.scorers): # Note that this throws an error if there is just one scorer
                    # this makes an array for each scorer from calling the scorer forward function on the candidate tokens
                    new_scores = scorer(init_tokens, cand_cont_tokens, self.rescale_scores)  #rescale scores, if True, causes the scores to be normalized
                    raw_scores = np.asarray(new_scores)
                    #print(len(raw_scores), len(candidates))
                    all_raw_scores.append(raw_scores)
                    # elementwise add the new scores to the np array after elementwise multiplying by coef
                    score_adjustment += raw_scores[:len(candidates)] * coef  # TODO why restrict to len(candidates)? It seems like the scorer sometimes but not always returns +1 more result than candidates
                last_raw_scores = all_raw_scores[-1] # all_raw scores will be num_scorers x num_candidates. So last_raw_scores is just the last scorer results?
                all_raw_scores = np.stack(all_raw_scores, axis=-1)  # this converts to num_candidates x num_scorers so each row is all adjusted scores for a candidate

                if self.learn and step < len(cont_tokens):
                    gold_cont_raw_scores = all_raw_scores[-1]  # this is the adjusted scores for the gold, since it was appended last

            # score adjustments are zero if no scorers. Basically this enable them to use candidate.adjusted_score regardless if scorers are present
            for i, candidate in enumerate(candidates):
                candidate.adjusted_score = candidate.score + score_adjustment[i]
                if rescore:
                    candidate.raw_scores = all_raw_scores[i]  # this is the candidate's scores from the scorers (arrray)

            candidates = sorted(candidates, key=lambda c: c.adjusted_score, reverse=True)
            filtered_candidates = list(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, candidates))

            if temperature and len(filtered_candidates) > self.beam_size:
                p = np.asarray(list(map(lambda c: c.adjusted_score, filtered_candidates)))
                p = np.exp(p / temperature)
                p /= p.sum()
                beam = np.random.choice(filtered_candidates, size=self.beam_size, replace=True, p=p)  #TODO replace=True? why?
            # since candidates is sorted this just prunes the beam
            else:
                beam = [cand for cand in itertools.islice(filter(lambda c: c.cont_tokens[-1] not in self.forbidden, candidates), self.beam_size)]

            for candidate in filter(lambda c: c.cont_tokens.count(self.sep) >= min_sentences and c.cont_tokens[-1] in self.terms, candidates): # 1 is the index of the sentence continuation token, terms is ending terms.
                if best is None or candidate.adjusted_score > best.adjusted_score:
                    best = candidate
            sentences_count = max(map(lambda c: c.cont_tokens.count(self.sep), candidates)) # used for seeing how many have been generated
            step += 1
        best = best or beam[0]

        if self.learn:
            loss = self.coef_trainer.train_coefficients(init_tokens, cont_tokens, beam[0], gold_cont_raw_scores)  #TODO change beam[0] to best

        if not keep_end:
            if best.tokens[-1]in self.terms:  # avoid printing end_tok
                best.tokens.pop()

        if self.human_readable:
            best.tokens = self.clean_output(best.tokens)

        return best.tokens if not self.learn else loss

