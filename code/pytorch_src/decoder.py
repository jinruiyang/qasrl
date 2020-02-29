import torch
import torch.nn as nn
import torch.nn.modules.transformer as Transformer
from IPython import embed

class Gate(nn.Module):
    def __init__(self, d_model_in=256, d_model_out=256, activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.f = nn.Sequential(
            nn.Linear(d_model_in+d_model_out, d_model_out)
        )
        self.activation = activation
        self.hidden_size = d_model_out

    def forward(self, input, h=None):
        if h is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            h = (zeros, zeros)
        h, c = h
        gate = self.f(torch.cat([h, input], dim=-1))
        return self.activation(gate)


class DualDiffCompatibleLSTMCell(nn.Module):
    def __init__(self, d_model_in, d_model_out):
        super(DualDiffCompatibleLSTMCell, self).__init__()
        self.hidden_size = d_model_out
        self.input_gate = Gate(d_model_in, d_model_out)
        self.forget_gate = Gate(d_model_in, d_model_out)
        self.output_gate = Gate(d_model_in, d_model_out)
        self.new_cell = Gate(d_model_in, d_model_out, torch.tanh)

    def forward(self, input, h=None):
        if h is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            h = (zeros, zeros)
        i = self.input_gate(input, h)
        f = self.forget_gate(input, h)
        o = self.output_gate(input, h)
        c_ = self.new_cell(input, h)
        h, c = h
        c = f * c + i * c_
        h = o * torch.tanh(c)
        return h, (h, c)


class MultiLSTMCell(nn.Module):
    def __init__(self, d_model_in, d_model_out, layer_num):
        super(MultiLSTMCell, self).__init__()
        self.hidden_size = d_model_out
        self.layer_num = layer_num
        self.lstmcell = [DualDiffCompatibleLSTMCell(d_model_in, d_model_out)] + [
            DualDiffCompatibleLSTMCell(d_model_out, d_model_out) for _ in range(layer_num - 1)
        ]
        for i in range(len(self.lstmcell)):
            self.add_module(name="cell_layer_%d" % i, module=self.lstmcell[i])

    def forward(self, input, h=None):
        if h is None:
            h = [None] * self.layer_num
        x = input
        H = 0.0
        for i in range(len(self.lstmcell)):
            x, h_ = self.lstmcell[i](x, h[i])
            H = H + x
            h[i] = h_
        return H, h


class Decoder(nn.Module):
    def __init__(self, d_model, z_size, num_layers=1, length=40):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.length = length
        self.z_size = z_size
        self.lstm = MultiLSTMCell(d_model+z_size, d_model, num_layers)
        self.embeddings = nn.Embedding(5000, d_model)
        self.output_unit = nn.Sequential(
            # nn.Linear(d_model+z_size, (d_model+z_size) * 2),
            # nn.ReLU(),
            # nn.Linear((d_model+z_size) * 2, (d_model+z_size) * 2),
            # nn.ReLU(),
            nn.Linear((d_model+z_size), 5000),
            nn.LogSoftmax(dim=-1)
        )
        self.transformh0 = []
        for idx in range(num_layers):
            t = nn.Sequential(
                    nn.Linear(z_size, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model * 2),
                    nn.Tanh()
                )
            self.add_module("transformer%d" % idx, t)
            self.transformh0.append(
                t
            )
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, z, x, x_length=None, train=False):
        # x -> [b, l]
        h = [self.transformh0[i](z).reshape((x.size(0), 2, self.d_model)).transpose(1, 0) for i in range(self.num_layers)]
        h = [(h_[0], h_[1]) for h_ in h]
        emb_x = self.embeddings(x)
        # emb_x -> [b, l, w]
        h_t, h = self.lstm(torch.cat((emb_x[:, 0, :], z), dim=-1), h)
        log_p = []
        for i in range(1, x.size(1)):
            log_p_ = self.output_unit(torch.cat((h_t, z), dim=-1)).gather(-1, x[:, i:i+1])
            log_p.append(log_p_)
            if train:
                h_t, h = self.lstm(torch.cat((emb_x[:, i, :] * (torch.rand(size=(z.size(0), 1)) > 0.5).to(torch.float).cuda(), z), dim=-1), h)
            else:
                h_t, h = self.lstm(torch.cat((emb_x[:, i, :], z), dim=-1), h)
        log_p = torch.stack(log_p).transpose(1, 0)
        # log_p -> [b, l]
        # log_p = torch.cumsum(log_p, 1).gather(-1, x_length - 2)
        log_p = torch.cumsum(log_p, 1)[:,-1]
        return log_p

    def random_sample(self, z, number=1, temp=1.0):
        batch_size = z.size(0)
        z_dim = z.size(1)
        if number == 1:
            h = [self.transformh0[i](z).reshape((z.size(0), 2, self.d_model)).transpose(1, 0) for i in
                 range(self.num_layers)]
            h = [(h_[0], h_[1]) for h_ in h]
            # emb_x -> [b, l, w]
            emb_x = self.embeddings(torch.ones([batch_size], dtype=torch.long, device=z.device))
            h_t, h = self.lstm(torch.cat((emb_x, z), dim=-1), h)
            x = []
            log_p_cumulative = 0.0
            for i in range(1, self.length):
                log_p_ = self.output_unit(torch.cat((h_t, z), dim=-1))
                selected_x = torch.multinomial(torch.softmax(log_p_ / temp, dim=-1), num_samples=1)
                if len(x) == 0:
                    x.append(torch.ones_like(selected_x))
                x.append(selected_x)
                log_p_cumulative = log_p_cumulative + log_p_.gather(dim=-1, index=selected_x).reshape((batch_size))
                h_t, h = self.lstm(torch.cat((self.embeddings(selected_x.reshape([batch_size])), z), dim=-1), h)
            x = torch.cat(x, dim=-1)
            return log_p_cumulative, x
        else:
            z_duplicate = z.reshape((batch_size, 1, z_dim)).expand((batch_size, number, z_dim))
            log_p_cumulative, x = self.random_sample(z_duplicate.reshape((batch_size * number, z_dim)))
            return log_p_cumulative.reshape((batch_size, number)), x.reshape((batch_size, number, -1))

    def nucleus_sample(self, z, number=None, threshold=0.9):
        batch_size = z.size(0)
        z_dim = z.size(1)
        if number is None:
            h = [self.transformh0[i](z).reshape((z.size(0), 2, self.d_model)).transpose(1, 0) for i in
                 range(self.num_layers)]
            h = [(h_[0], h_[1]) for h_ in h]
            # emb_x -> [b, l, w]
            emb_x = self.embeddings(torch.ones([batch_size], dtype=torch.long, device=z.device))
            h_t, h = self.lstm(torch.cat((emb_x, z), dim=-1), h)
            x = []
            log_p_cumulative = 0.0
            for i in range(1, self.length):
                log_p_ = self.output_unit(torch.cat((h_t, z), dim=-1))
                with torch.no_grad():
                    _p_ = torch.softmax(log_p_, dim=-1)
                    argmax_mask = torch.zeros_like(_p_)
                    argmax_mask.data[:,0] = 1.0
                    sorted_p, indices = torch.sort(_p_, dim=-1, descending=True)
                    cumsum_sorted = sorted_p.cumsum(dim=-1)
                    in_nucleus_mask = (cumsum_sorted <= threshold).to(torch.float)
                    in_nucleus_mask_gate = (in_nucleus_mask.sum(dim=-1, keepdim=True) < 1).to(torch.float)
                    in_nucleus_mask_ = in_nucleus_mask * (1.0 - in_nucleus_mask_gate) + argmax_mask * in_nucleus_mask_gate
                    masked_p_ = sorted_p * in_nucleus_mask_
                    selected_x = torch.multinomial(masked_p_, num_samples=1)
                    selected_x = indices.gather(index=selected_x, dim=-1)
                    if len(x) == 0:
                        x.append(torch.ones_like(selected_x))
                x.append(selected_x)
                h_t, h = self.lstm(torch.cat((self.embeddings(selected_x.reshape([batch_size])), z), dim=-1), h)
                log_p_cumulative = log_p_cumulative + log_p_.gather(dim=-1, index=selected_x).reshape((batch_size))
            x = torch.cat(x, dim=-1)
            return log_p_cumulative, x
        else:
            z_duplicate = z.reshape((batch_size, 1, z_dim)).expand((batch_size, number, z_dim))
            log_p_cumulative, x = self.nucleus_sample(z_duplicate.reshape((batch_size * number, z_dim)))
            return log_p_cumulative.reshape((batch_size, number)), x.reshape((batch_size, number, -1))

    def nucleus_selection(self, z):
        # z -> [1, z_dim]
        pool = []
        pool_log_p = []
        z_ = z.expand((100, self.z_size))
        for _ in range(20):
            with torch.no_grad():
                log_p, candidates = self.nucleus_sample(z_)
                for i in range(100):
                    flag = True
                    for j in range(len(pool)):
                        if (pool[j] == candidates[i]).to(torch.int).prod(-1) == 1:
                            flag = False
                            break
                    if flag:
                        pool.append(candidates[i])
                        pool_log_p.append(log_p[i])
        sort = {}
        for i in range(len(pool_log_p)):
            sort[pool[i]] = pool_log_p[i]
        pool = sorted(pool, key=lambda x:sort[x], reverse=True)
        return torch.stack(pool)

    def calculate_prob_vec(self, z, x):
        # z -> [b, z_dim]
        # x -> [b, n, l]
        repl = x.size(1)
        bsz = z.size(0)
        z_dim = z.size(1)
        l = x.size(2)
        z_ = z.reshape((bsz, 1, z_dim)).expand((bsz, repl, z_dim)).reshape((-1, z_dim))
        x_ = x.reshape((-1, l))
        log_prob = self.forward(z_, x_).reshape((bsz, repl)).softmax(dim=-1)
        return log_prob

    def beam_search(self, z, beam_size=5, extending_size=None):
        if extending_size is None:
            extending_size = beam_size
        # z -> [b, w]
        batch_size = z.size(0)
        z_dim = z.size(1)
        z = z.reshape((batch_size, 1, z_dim)).expand((batch_size, beam_size, z_dim)).reshape((batch_size * beam_size, z_dim))
        h = [self.transformh0[i](z).reshape((z.size(0), 2, self.d_model)).transpose(1, 0) for i in
             range(self.num_layers)]
        h = [(h_[0], h_[1]) for h_ in h]
        def duplicate(h_list, beam_size):
            return [(h0.reshape((batch_size * beam_size, 1, self.d_model))
                     .expand((batch_size * beam_size, extending_size, self.d_model))
                     .reshape((batch_size * beam_size * extending_size, self.d_model)),
                     h1.reshape((batch_size * beam_size, 1, self.d_model))
                     .expand((batch_size * beam_size, extending_size, self.d_model))
                     .reshape((batch_size * beam_size * extending_size, self.d_model))
                     ) for (h0, h1) in h_list]

        def update(duplicated_h_list, z, x):
            emb_x = self.embeddings(x)
            duplicated_h_t, duplicated_h_list = self.lstm(torch.cat((emb_x, z), dim=-1), duplicated_h_list)
            return (duplicated_h_t, duplicated_h_list)

        def select(h_pair, idx):
            _h_t, _h = h_pair
            _h_t = _h_t.reshape((batch_size, beam_size * extending_size, -1))\
                    .gather(
                    dim=1,
                    index=idx.reshape((batch_size, beam_size, 1)).expand((batch_size, beam_size, self.d_model))).reshape((batch_size * beam_size, self.d_model))
            _h = [(h0.reshape((batch_size, beam_size * extending_size, -1))
                  .gather(dim=1, index=idx.reshape((batch_size, beam_size, 1))
                  .expand((batch_size, beam_size, self.d_model)))
                  .reshape((batch_size * beam_size, self.d_model)),
                  h1.reshape((batch_size, beam_size * extending_size, -1))
                  .gather(dim=1, index=idx.reshape((batch_size, beam_size, 1))
                          .expand((batch_size, beam_size, self.d_model)))
                  .reshape((batch_size * beam_size, self.d_model))
                  ) for (h0, h1) in _h]
            return _h_t, _h


        emb_x = self.embeddings(torch.ones([batch_size * beam_size], dtype=torch.long, device=z.device))
        h_t, h = self.lstm(torch.cat((emb_x, z), dim=-1), h)
        log_p = torch.zeros((batch_size, beam_size), device=z.device, dtype=z.dtype)
        beam_candidates = torch.zeros((batch_size, beam_size, self.length), dtype=torch.long, device=z.device)
        # h_t -> [b, w]
        z_select = z.reshape((batch_size, beam_size, 1, z_dim)).expand((batch_size, beam_size, extending_size, z_dim)).reshape((batch_size * beam_size * extending_size, z_dim))
        beam_candidates[:, :, 0] = 1
        for i in range(1, self.length):
            if h_t.shape != z.shape:
                embed()
            log_p_ = self.output_unit(torch.cat((h_t, z), dim=-1))
            topB = torch.topk(log_p_, extending_size)
            if i == 1:
                topB = torch.topk(log_p_, beam_size)
                log_p += topB.values.reshape((batch_size, beam_size, beam_size))[:,0,:]
                topChoice = topB.indices.reshape((batch_size, beam_size, beam_size))[:,0,:]
                beam_candidates[:,:,1] = topChoice
                emb_x = self.embeddings(topChoice.reshape((batch_size * beam_size)))
                h_t, h = self.lstm(torch.cat((emb_x, z), dim=-1), h)
            else:
                log_p_select = log_p.reshape((batch_size, beam_size, 1)).expand((batch_size, beam_size, extending_size)).clone()
                log_p_select += topB.values.reshape((batch_size, beam_size, extending_size))
                beam_candidates_select = beam_candidates.reshape((batch_size, beam_size, 1, self.length))\
                    .expand((batch_size, beam_size, extending_size, self.length)).reshape((batch_size, beam_size * extending_size, self.length))
                beam_candidates_select[:,:,i] = topB.indices.reshape((batch_size, beam_size * extending_size))
                h_t_select, h_select = update(duplicate(h, beam_size), z_select, topB.indices.reshape((batch_size * beam_size * extending_size)))
                log_p_select = log_p_select.reshape((batch_size, beam_size * extending_size))
                topB = torch.topk(log_p_select, beam_size)
                log_p = topB.values
                beam_candidates = beam_candidates_select\
                    .gather(
                    dim=1,
                    index=topB.indices.reshape((batch_size, beam_size, 1)).expand((batch_size, beam_size, self.length)))
                h_t, h = select((h_t_select, h_select), topB.indices)
        return log_p, beam_candidates
