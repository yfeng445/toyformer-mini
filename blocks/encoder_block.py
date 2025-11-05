# blocks/encoder_block.py
import torch
import torch.nn as nn
import config
from layers import MultiHeadAttention, FeedForward, AddNormPre

class EncoderBlock(nn.Module):
    """Pre-Norm Encoder 子块: x ← x + MHA(LN(x)); x ← x + FFN(LN(x))"""
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg)  # 非因果
        self.ffn = FeedForward(cfg)
        self.an1 = AddNormPre(cfg)
        self.an2 = AddNormPre(cfg)

    def forward(self, x: torch.Tensor, *, enc_kpm: torch.Tensor | None = None) -> torch.Tensor:
        x = self.an1(x, lambda u: self.self_attn(u, key_padding_mask=enc_kpm))
        x = self.an2(x, lambda u: self.ffn(u))
        return x
