# blocks/decoder_block.py
import torch
import torch.nn as nn

import config
from layers import MaskedMultiHeadAttention, CrossMultiHeadAttention, FeedForward, AddNormPre

class TransformerDecoderBlock(nn.Module):
    """
    Transformer Decoder Block (Pre-Norm):
      x ← x + MaskedSelfAttn(LN(x))
      x ← x + CrossAttn(      LN(x), memory)
      x ← x + FFN(            LN(x))
    形状:
      - x      : (B, Tq, D)
      - memory : (B, Tk, D)
    掩码:
      - self_kpm   : (B, Tq)  1=valid, 0=pad
      - memory_kpm : (B, Tk)  1=valid, 0=pad
    """
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        self.self_attn  = MaskedMultiHeadAttention(cfg)      # 因果自注意力
        self.cross_attn = CrossMultiHeadAttention(cfg)       # 交叉注意力
        self.ffn        = FeedForward(cfg)

        self.an1 = AddNormPre(cfg)
        self.an2 = AddNormPre(cfg)
        self.an3 = AddNormPre(cfg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        self_kpm: torch.Tensor | None = None,
        memory: torch.Tensor,
        memory_kpm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.an1(x, lambda u: self.self_attn(u, key_padding_mask=self_kpm))
        x = self.an2(x, lambda u: self.cross_attn(u, memory, memory_padding_mask=memory_kpm))
        x = self.an3(x, lambda u: self.ffn(u))
        return x
