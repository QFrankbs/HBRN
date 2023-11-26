import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from infonce import InfoNCE
from layers.fc import MLP
#from layers.layer_norm import LayerNorm
from torch.nn import LayerNorm
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy

# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)
def MultiwayWrapper(args, module, dim=1):
    if args.multiway:
        return MultiwayNetwork(module, dim=dim)
    return module


def set_split_position(position):
    def apply_fn(module):
        if hasattr(module, "split_position"):
            module.split_position = position

    return apply_fn


class MultiwayNetwork(nn.Module):
    def __init__(self, module, dim=1):
        super().__init__()
        self.dim = dim
        self.A = module
        self.B = copy.deepcopy(module)
        self.B.reset_parameters()
        self.split_position = -1

    def forward(self, x, **kwargs):
        if self.split_position == -1:
            return self.A(x, **kwargs)
        if self.split_position == 0:
            return self.B(x, **kwargs)
        x1, x2 = torch.split(
            x,
            [self.split_position, x.size(self.dim) - self.split_position],
            dim=self.dim,
        )
        # x1, x2 = x[:self.split_position], x[self.split_position:]
        y1, y2 = self.A(x1, **kwargs), self.B(x2, **kwargs)
        return torch.cat([y1, y2], dim=self.dim)


class MutliwayEmbedding(MultiwayNetwork):
    def __init__(self, modules, dim=1):
        super(MultiwayNetwork, self).__init__()
        self.dim = dim
        assert len(modules) == 2
        self.A = modules[0]
        self.B = modules[1]
        self.split_position = -1
        
# ------------------------------------
# ---------- Masking sequence --------
# ------------------------------------

class RetNetRelPos(nn.Module):
    def __init__(self, args):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, args.new_word_embed_size // args.multi_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(args.multi_head, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = args.recurrent_chunk_size
        
    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=True):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            cross_decay = cross_decay[:, None, None]
            inner_decay = inner_decay[:, :, None] / (scale / scale[:, -1, None])
            retention_rel_pos = ((sin, cos), (mask, cross_decay, inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos
def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError
        
class MSRT(nn.Module):
    def __init__(self,
            args,
            value_factor=2,
            gate_fn="swish",
        ):
        super(MSRT, self).__init__()
        self.args = args
        self.factor = value_factor
        self.embed_dim = args.new_word_embed_size
        self.num_heads = args.multi_head
        self.head_dim = self.embed_dim * self.factor // args.multi_head
        self.key_dim = self.embed_dim // args.multi_head
        self.scaling = self.key_dim ** -0.5
        
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size * self.factor)
        self.linear_g = nn.Linear(args.hidden_size, args.hidden_size * self.factor)
        self.q_proj = MultiwayWrapper(args, nn.Linear(args.hidden_size, args.hidden_size, bias=True))
        self.k_proj = MultiwayWrapper(args, nn.Linear(args.hidden_size, args.hidden_size, bias=True))
        self.v_proj = MultiwayWrapper(args, nn.Linear(args.hidden_size, args.hidden_size * self.factor, bias=True))
        self.g_proj = MultiwayWrapper(args, nn.Linear(args.hidden_size, args.hidden_size * self.factor, bias=True))

        self.out_proj = MultiwayWrapper(args, nn.Linear(args.hidden_size * self.factor, args.hidden_size, bias=True))

        self.group_norm = MultiwayWrapper(args, LayerNorm(self.head_dim, eps=1e-6, elementwise_affine=False))

        
        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.dropout = nn.Dropout(args.dropout_r)
    def recurrent_forward(
        self,
        qr, kr, v,
        decay,
        incremental_state
    ):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(
        self,
        qr, kr, v,
        inner_mask
    ):
        mask, cross_decay, inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)
        # print(chunk_len,tgt_len)
        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t # bsz * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v) # bsz * num_heads * num_value_heads * chunk_len * head_dim
        
        # reduce kv in one chunk
        kv = kr_t @ (v * mask[:, -1, :, None])
        kv = kv.view(bsz, num_chunks, self.num_heads, self.key_dim, self.head_dim)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        
        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)
            
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)
        
        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        return output
    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output
    def forward(
        self, q, k, v, g, rel_pos,
        chunkwise_recurrent=True,
        incremental_state=None
    ):
        bsz, tgt_len, _ = q.size()
        # print(tgt_len)
        (sin, cos), mask = rel_pos
        
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        g = self.g_proj(g)

        k *= self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k *= self.scaling
        # q = q.reshape(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        # k = k.reshape(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        
        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, mask)
        else:
            output = self.parallel_forward(qr, kr, v, mask)
        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output 


# ------------------------------
# ---------- Flattening --------
# ------------------------------


class AttFlat(nn.Module):
    def __init__(self, args, flat_glimpse, merge=False, input_m = 1):
        super(AttFlat, self).__init__()
        self.args = args
        self.merge = merge
        self.flat_glimpse = flat_glimpse
        self.mlp = MLP(
            in_size=input_m * args.hidden_size,
            mid_size=args.ff_size,
            out_size=flat_glimpse,
            dropout_r=args.dropout_r,
            use_relu=True
        )

        if self.merge:
            self.linear_merge = nn.Linear(
                input_m * args.hidden_size * flat_glimpse,
                args.hidden_size * 2
            )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        if x_mask is not None:
            att = att.masked_fill(
                x_mask.squeeze(1).squeeze(1).unsqueeze(2),
                -1e9
            )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpse):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        if self.merge:
            x_atted = torch.cat(att_list, dim=1)
            x_atted = self.linear_merge(x_atted)

            return x_atted

        return torch.stack(att_list).transpose_(0, 1)

# -------------------------------
# ---- Self Cross Attention ----
# -------------------------------
class Retnetion(nn.Module):
    def __init__(self, args):
        super(Retnetion, self).__init__()
        self.args = args
        self.msrt_x = MSRT(args)
        self.msrt_y = MSRT(args)
        self.msrt_z = MSRT(args)
        self.slen = args.lang_seq_len
        self.alen = args.audio_seq_len
        self.vlen = args.video_seq_len
        self.retnet_rel_pos_x = RetNetRelPos(args)
        self.retnet_rel_pos_y = RetNetRelPos(args)
        self.retnet_rel_pos_z = RetNetRelPos(args)
        self.dropout_x1 = nn.Dropout(args.dropout_r)
        self.norm_x1 = LayerNorm(args.hidden_size)
        
        self.dropout_y1 = nn.Dropout(args.dropout_r)
        self.norm_y1 = LayerNorm(args.hidden_size)
        
        self.dropout_z1 = nn.Dropout(args.dropout_r)
        self.norm_z1 = LayerNorm(args.hidden_size)
        
        self.ffn_x = FFN(args)
        self.ffn_y = FFN(args)
        self.ffn_z = FFN(args)
        
        self.dropout_x3 = nn.Dropout(args.dropout_r)
        self.norm_x3 = LayerNorm(args.hidden_size)
         
        self.dropout_y3 = nn.Dropout(args.dropout_r)
        self.norm_y3 = LayerNorm(args.hidden_size)
        
        self.dropout_z3 = nn.Dropout(args.dropout_r)
        self.norm_z3 = LayerNorm(args.hidden_size)
        
        
    def forward(self, x, y, z):
        self_x = self.norm_x1(x + self.dropout_x1(
            self.msrt_x(x, x, x, x, self.retnet_rel_pos_x(self.slen))
        ))
        
        self_y = self.norm_y1(y + self.dropout_y1(
            self.msrt_y(y, y, y, y, self.retnet_rel_pos_y(self.alen))
        ))
        
        self_z = self.norm_z1(z + self.dropout_z1(
            self.msrt_z(z, z, z, z, self.retnet_rel_pos_z(self.vlen))
        ))
        self_x = self.norm_x3(self_x + self.dropout_x3(
            self.ffn_x(self_x)
        ))
        self_y = self.norm_y3(self_y + self.dropout_y3(
            self.ffn_y(self_y)
        ))
        self_z = self.norm_z3(self_z + self.dropout_z3(
            self.ffn_z(self_z)
        ))
        return self_x,self_y,self_z
    
    
class Bayes_Retnetion(nn.Module):
    def __init__(self, args):
        super(Bayes_Retnetion, self).__init__()
        self.args = args
        self.msrt_x = MSRT(args)
        self.msrt_y = MSRT(args)
        self.msrt_z = MSRT(args)
        self.slen = args.lang_seq_len
        self.alen = args.audio_seq_len
        self.vlen = args.video_seq_len
        self.retnet_rel_pos_x = RetNetRelPos(args)
        self.retnet_rel_pos_y = RetNetRelPos(args)
        self.retnet_rel_pos_z = RetNetRelPos(args)
        self.dropout_x1 = nn.Dropout(args.dropout_r)
        self.norm_x1 = LayerNorm(args.hidden_size)
        
        self.dropout_y1 = nn.Dropout(args.dropout_r)
        self.norm_y1 = LayerNorm(args.hidden_size)
        
        self.dropout_z1 = nn.Dropout(args.dropout_r)
        self.norm_z1 = LayerNorm(args.hidden_size)
        
    def forward(self, self_x,self_y,self_z, xyz,yz):
        self_x = self.norm_x1(self_x + self.dropout_x1(
            self.msrt_x(xyz, self_x, self_x, xyz, self.retnet_rel_pos_x(self.slen))
        ))
        
        self_y = self.norm_y1(self_y + self.dropout_y1(
            self.msrt_y(yz, self_y, self_y, yz, self.retnet_rel_pos_y(self.alen))
        ))
        
        self_z = self.norm_z1(self_z + self.dropout_z1(
            self.msrt_z(self_z, self_z, self_z, self_z, self.retnet_rel_pos_z(self.vlen))
        ))            
        return self_x,self_y,self_z
        
        
class Cross_Retnetion(nn.Module):
    def __init__(self, args,i):
        super(Cross_Retnetion, self).__init__()
        
        self.args = args
        self.basic_msrt = Retnetion(args)
        self.bayes_msrt = Bayes_Retnetion(args)

        self.last = (i == args.layer-1)

        self.sa_x = SA(args)
        self.sa_y = SA(args)
        self.sa_z = SA(args)
        
        self.ffn_x = FFN(args)
        self.ffn_y = FFN(args)
        self.ffn_z = FFN(args)
        if not self.last:
            self.norm_l = LayerNorm(args.hidden_size)
            self.norm_a = LayerNorm(args.hidden_size)
            self.norm_v = LayerNorm(args.hidden_size)
            self.dropout = nn.Dropout(args.dropout_r)

        self.dropout_x3 = nn.Dropout(args.dropout_r)
        self.norm_x3 = LayerNorm(args.hidden_size)
         
        self.dropout_y3 = nn.Dropout(args.dropout_r)
        self.norm_y3 = LayerNorm(args.hidden_size)
        
        self.dropout_z3 = nn.Dropout(args.dropout_r)
        self.norm_z3 = LayerNorm(args.hidden_size)
        
        
        
    def forward(self, x, x_mask, y, y_mask, z, z_mask):

        for step in range(self.args.inner_layer):
            if step == 0:
                self_x,self_y,self_z = self.basic_msrt(x, y, z)
            else:
                mean_y = self_y.mean(dim=2, keepdim=True)
                mean_z = self_z.mean(dim=2, keepdim=True)

                # 对 Q1 和 Q2 进行均值对齐
                aligned_y = self_y - mean_y
                aligned_z = self_z - mean_z

                # 计算权重加权平均
                weight = 0.5  # 你可以调整权重
                yz = weight * aligned_y + (1 - weight) * aligned_z

                # 将 Q12 与 Q3 进行元素级别的相乘
                xyz = self_x * yz
                
                yz = self_y * self_z
                
                self_x,self_y,self_z = self.bayes_msrt(self_x,self_y,self_z, xyz,yz)
                
        self_ax = self.sa_x(x, x_mask)
        self_ay = self.sa_y(y, y_mask)
        self_az = self.sa_z(z, z_mask)
        
        self_x = self.norm_x3(self_x + self.dropout_x3(
            self.ffn_x(self_x)
        ))
        self_y = self.norm_y3(self_y + self.dropout_y3(
            self.ffn_y(self_y)
        ))
        self_z = self.norm_z3(self_z + self.dropout_z3(
            self.ffn_z(self_z)
        ))
        x = x+self_x+self_ax
        y = y+self_y+self_ay
        z = z+self_z+self_az
        if self.last:
            return x, y, z


        return self.norm_l(x + self.dropout(self_x)+self.dropout(self_ax)), \
               self.norm_a(y + self.dropout(self_x)+self.dropout(self_ay)), \
               self.norm_v(z + self.dropout(self_z)+self.dropout(self_az))

    
class SA(nn.Module):
    def __init__(self, args):
        super(SA, self).__init__()

        self.mhatt = MHAtt(args)
        self.ffn = FFN(args)
        self.args = args

        self.dropout1 = nn.Dropout(args.dropout_r)
        self.norm1 = LayerNorm(args.hidden_size)

        self.dropout2 = nn.Dropout(args.dropout_r)
        self.norm2 = LayerNorm(args.hidden_size)

    def forward(self, y, y_mask):
        #print(y.size())
        for i in range(self.args.inner_layer):
            y = self.norm1(y + self.dropout1(
            self.mhatt(y, y, y, y_mask)
        ))
        y = self.norm2(y + self.dropout2(
            self.ffn(y)
        ))

        return y   

class MHAtt(nn.Module):
    def __init__(self, args):
        super(MHAtt, self).__init__()
        self.args = args

        self.linear_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_merge = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(args.dropout_r)

    def forward(self, v, k, q, mask):
        #print(v.size(),k.size(),q.size())
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.args.multi_head,
            int(self.args.hidden_size / self.args.multi_head)
        ).transpose(1, 2)
        #print(v.size(),k.size(),q.size())
        atted = self.att(v, k, q, mask)
        #print(atted.size())

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.args.hidden_size
        )
        atted = self.linear_merge(atted)
        #print(atted.size())
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        #print(att_map.size(),value.size())
        return torch.matmul(att_map, value)
# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, args):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=args.hidden_size,
            mid_size=args.ff_size,
            out_size=args.hidden_size,
            dropout_r=args.dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)






class Model_LAV_RETNET(nn.Module):
    def __init__(self, args):
        super(Model_LAV_RETNET, self).__init__()

        self.args = args
        # LSTM
        # self.embedding = nn.Embedding(
        #     num_embeddings=vocab_size,
        #     embedding_dim=args.word_embed_size
        # )

        # Loading the GloVe embedding weights
        # self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm_x = nn.LSTM(
            input_size=args.word_embed_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.lstm_y = nn.LSTM(
            input_size=args.audio_feat_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.lstm_z = nn.LSTM(
            input_size=args.video_feat_size,
            hidden_size=args.hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Feature size to hid size
        self.sa = SA(args)
        self.InfoNCE = InfoNCE(args)
        self.adapter_y = nn.Linear(args.audio_feat_size, args.hidden_size)
        self.adapter_z = nn.Linear(args.video_feat_size, args.hidden_size)

        # Encoder blocks
        self.enc_list = nn.ModuleList([Cross_Retnetion(args, i) for i in range(args.layer)])

        # Flattenting features before proj
        self.attflat   = AttFlat(args, 1, merge=True,input_m =3)
        self.attflat_lang   = AttFlat(args, 1, merge=True)

        # Classification layers
        self.proj_norm = LayerNorm(2 * args.hidden_size)
        if self.args.task == "sentiment":
            if self.args.task_binary:
                self.proj = nn.Linear(2 * args.hidden_size, 2)
            else:
                self.proj = nn.Linear(2 * args.hidden_size, 7)
        if self.args.task == "emotion":
            self.proj = self.proj = nn.Linear(2 * args.hidden_size, 6)

    def forward(self, x, y, z, x1):
        x_mask = make_mask(x)
        x1_mask = make_mask(x1)
        #print(x1_mask.size())
        y_mask = make_mask(y)
        z_mask = make_mask(z)
        #print(x1.size())
        #print(x_mask.size(),x1_mask.size())
        # embedding = self.embedding(x)
        
        # embedding1 = self.embedding(x1)
        
        x, _ = self.lstm_x(x)
        x1, _ = self.lstm_x(x1)
        #print(x1.size())
        # y, _ = self.lstm_y(y)
        y, _ = self.lstm_y(y)
        z, _ = self.lstm_z(z)

        # y, z = self.adapter_y(y), self.adapter_z(z)
        #print(x.size(),y.size(),z.size(),x1.size())
        for i, dec in enumerate(self.enc_list):
            x_m, y_m, z_m = None, None, None
            if i == 0:
                x_m, y_m, z_m = x_mask, y_mask, z_mask
            x1 = self.sa(x1,x1_mask)
            x, y, z = dec(x, x_m, y, y_m, z, z_m)
        list_of_tensors = [x, y, z]  # 请替换为你的实际张量
        
        # 使用 torch.cat 在 X 维度上拼接它们
        concatenated_tensor = torch.cat(list_of_tensors, dim=2)
        # Classification layers
        #print(x.size(),y.size(),z.size(),x1.size())
        #print(x.size(),y.size(),z.size(),x1.size(),f.size())
        cl_loss = self.InfoNCE(x,x*y*z,x1)
        concatenated_tensor = self.attflat(
            concatenated_tensor,
            None
        )   
        proj_feat = self.proj_norm(concatenated_tensor)
        ans = self.proj(proj_feat)
        
        return ans, cl_loss