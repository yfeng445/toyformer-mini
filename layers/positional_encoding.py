# layers/positional_encoding.py
import math
import torch
import torch.nn as nn
import config

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        self.pos = nn.Embedding(cfg.n_ctx, cfg.d_model)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        pos = torch.arange(T, device=x.device)          # (T,)
        P = self.pos(pos)[None, :, :].expand(B, T, D)   # (B,T,D)
        return x + P

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        D = cfg.d_model
        T = cfg.n_ctx
        pe = torch.zeros(T, D)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)     # (T,1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 不训练，不随优化器更新

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x + self.pe[:T].unsqueeze(0)  # (1,T,D) 广播到 (B,T,D)
