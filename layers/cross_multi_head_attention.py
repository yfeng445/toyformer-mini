import math
import torch
import torch.nn as nn
import ops
import config

def _to_heads(t: torch.Tensor, n_head: int) -> torch.Tensor:
    B, T, D = t.shape
    Hd = D // n_head
    return t.view(B, T, n_head, Hd).transpose(1, 2)  # (B,H,T,Hd)

class CrossMultiHeadAttention(nn.Module):

    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        D = cfg.d_model
        H = cfg.cross_attn.n_head
        assert D % H == 0
        self.d_model = D
        self.n_head  = H
        self.head_dim = D // H

        use_qkv_bias = cfg.cross_attn.qkv_bias
        use_out_bias = cfg.cross_attn.out_bias

        self.W_q  = nn.Parameter(torch.empty(D, D))
        self.b_q  = nn.Parameter(torch.zeros(D))      if use_qkv_bias else None
        self.W_kv = nn.Parameter(torch.empty(D, 2*D))
        self.b_kv = nn.Parameter(torch.zeros(2*D))    if use_qkv_bias else None

        self.W_o  = nn.Parameter(torch.empty(D, D))
        self.b_o  = nn.Parameter(torch.zeros(D))      if use_out_bias else None

        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_kv)
        nn.init.xavier_uniform_(self.W_o)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, Tq, D = x.shape
        Bm, Tk, Dm = memory.shape
        assert B == Bm and D == Dm, "x/memory batch or dim mismatch"

        q  = ops.linear(x,      self.W_q,  self.b_q)    # (B,Tq,D)
        kv = ops.linear(memory, self.W_kv, self.b_kv)   # (B,Tk,2D)
        k, v = kv.split(D, dim=-1)                      # (B,Tk,D)

        q = _to_heads(q, self.n_head)                   # (B,H,Tq,Hd)
        k = _to_heads(k, self.n_head)
        v = _to_heads(v, self.n_head)

        scores = ops.matmul(q, k.transpose(-2, -1))     # (B,H,Tq,Tk)
        scores = ops.scale(scores, 1.0 / math.sqrt(self.head_dim))

        if memory_padding_mask is not None:
            pad = (~memory_padding_mask.bool())[:, None, None, :]  # True=pad
            scores = scores.masked_fill(pad, float('-inf'))

        A   = ops.softmax(scores)                       # (B,H,Tq,Tk)
        ctx = ops.matmul(A, v)                          # (B,H,Tq,Hd)

        y = ctx.transpose(1, 2).contiguous().view(B, Tq, D)
        return ops.linear(y, self.W_o, self.b_o)
