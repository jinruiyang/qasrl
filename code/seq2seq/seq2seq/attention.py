import torch
import torch.nn as nn


# source: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/global_attention.py
# Note that this is global attention, it doesn't seem to support local
class Attn(nn.Module):
    def __init__(self, attn_type, dim):
        super(Attn, self).__init__()

        self.dim = dim
        self.attn_type = attn_type

        if self.attn_type == 'general':  # as per Luong
            self.linear_in = nn.Linear(dim, dim, bias=False)

        elif self.attn_type == 'mlp':  # as per Bahdanau
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)

        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim * 2, dim, bias=out_bias)

        self.softmax = nn.Softmax(dim=-1) #nn.softmax should output dim same as input (and dim -1 will sum to one)
        self.tanh = nn.Tanh()

    def forward(self, source, memory_bank, bcast=False):
        # source: B x t x h ; memory_bank: B x s x h

        # bcast is for if trying a beamsearch at inference time, to prevent enforcing equal batch sizes (which is a test that must pass during training)

        # unless decoding one at a time, in which case, use one step input, and then unsqueeze things
        if source.dim() == 2:
            one_step = True
            source = source.unsqueeze(1)
        else:
            one_step = False

        # compute attention scores, as in Luong et al.
        align = self.score(source, memory_bank, bcast)

        # Softmax to normalize attention weights
        align_vectors = self.softmax(align)
        # The onmt module has some Tensor views but I think that is because it is using nn.functional rather than nn.softmax
        # e.g. first make it 2D, then take softmax, then align_vectors = F.softmax(align.view(batch*target_l, source_l), -1), then make 3D again as below:
        # align_vectors = align_vectors.view(batch, target_l, source_l)...which should return it to normal

        # each context vector c_t is the weighted average
        # over all the source hidden states
        if bcast:
            c = torch.matmul(align_vectors, memory_bank)
        else:
            c = torch.bmm(align_vectors, memory_bank)


        # concatenate
        concat_c = torch.cat([c, source], 2)
        attn_h = self.linear_out(concat_c)
        if self.attn_type in ["general", "dot"]:
            attn_h = self.tanh(attn_h)

        if one_step:
            #TODO onmt squeezes along dim 1 but our batch is dim 0
            attn_h = attn_h.squeeze(0)
            align_vectors = align_vectors.squeeze(0)
            #TODO add some asserts as in onmt here

        # attn_h: B x t x h ; align_vectors: B x t x s
        #onmt returns (tgt_len x batch x dim) and (tgt_len x batch x src_len)

        return attn_h, align_vectors # hidden state and attention distribution

    def score(self, h_t, h_s, bcast=False):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`
        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`
        """
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        #print(h_s.size(), h_t.size())
        if not bcast:
            assert src_batch == tgt_batch, "src batch ({}) and target batch ({}) must match".format(src_batch, tgt_batch)
        assert src_dim == tgt_dim
        assert self.dim == src_dim

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t = self.linear_in(h_t)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            if bcast: # use broadcasting on a batch (only matmul supports it)
                return torch.matmul(h_t, h_s_)
            return torch.bmm(h_t, h_s_)  # bmm performs a batch matrix-matrix product of matrices stored in batch1 and batch2

        else:  # from ONMT, but we're not really using it since it is for mlp
            dim = self.dim
            wq = self.linear_query(h_t)
            wq = wq.unsqueeze(2)  # `tgt_batch, tgt_len, 1, dim`
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s)
            uh = uh.unsqueeze(1)  # `src_batch, 1, src_len, dim`
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)
            return self.v(wquh).squeeze(3)  # `(batch, t_len, s_len)`
