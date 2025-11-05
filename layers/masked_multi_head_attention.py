# masked_multi_head_attention.py
import math
import torch
import torch.nn as nn
import ops
import config

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, cfg: config.ModelConfig, use_causal: bool | None = None):
        super().__init__()
        D = cfg.d_model
        H = cfg.self_attn.n_head
        assert D % H == 0
        self.d_model = D
        self.n_head  = H
        self.head_dim = D // H
        self.use_causal = cfg.self_attn.causal

        use_qkv_bias = cfg.self_attn.qkv_bias
        use_out_bias = cfg.self_attn.out_bias

        self.W_qkv = nn.Parameter(torch.empty(D, 3*D))
        self.b_qkv = nn.Parameter(torch.zeros(3*D)) if use_qkv_bias else None
        self.W_o   = nn.Parameter(torch.empty(D, D))
        self.b_o   = nn.Parameter(torch.zeros(D))   if use_out_bias else None

        nn.init.xavier_uniform_(self.W_qkv)
        nn.init.xavier_uniform_(self.W_o)

    def _to_heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, D = t.shape
        H, Hd = self.n_head, self.head_dim
        return t.view(B, T, H, Hd).transpose(1, 2)  # (B,H,T,Hd)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape

        qkv = ops.linear(x, self.W_qkv, self.b_qkv)           # (B,T,3D)
        q, k, v = qkv.split(D, dim=-1)                        # 各 (B,T,D)
        q = self._to_heads(q); k = self._to_heads(k); v = self._to_heads(v)

        scores = ops.matmul(q, k.transpose(-2, -1))           # (B,H,T,T)
        scores = ops.scale(scores, 1.0 / math.sqrt(self.head_dim))

        if self.use_causal or key_padding_mask is not None:
            if key_padding_mask is None:
                attn_mask = ops.make_causal_mask(T, device=x.device)             # (1,1,T,T)
            else:
                attn_mask = ops.make_padding_mask(key_padding_mask)              # (B,1,1,T)
                if self.use_causal:
                    attn_mask = attn_mask | ops.make_causal_mask(T, device=x.device)
            scores = scores.masked_fill(attn_mask, float('-inf'))

        A   = ops.softmax(scores)                                 # (B,H,T,T)
        ctx = ops.matmul(A, v)                                    # (B,H,T,Hd)

        y = ctx.transpose(1, 2).contiguous().view(B, T, D)        # (B,T,D)
        return ops.linear(y, self.W_o, self.b_o)
