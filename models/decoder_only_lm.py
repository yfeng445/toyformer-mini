# models/decoder_only_lm.py
import torch
import torch.nn as nn
import config
from blocks import CausalDecoderBlock
from layers import LMHead, SoftmaxLastDim


class DecoderOnlyLM(nn.Module):
    """
    GPT 风格：N×(CausalDecoderBlock) → LMHead → (可选) Softmax
    约定输入已经是 embedding: x_emb ∈ ℝ[B,T,D]
    """
    def __init__(self, cfg: config.ModelConfig, return_probs: bool = False):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([CausalDecoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.head = LMHead(cfg, bias=False)
        self.prob = SoftmaxLastDim()
        self.return_probs = return_probs

    def forward(self, x_emb: torch.Tensor, *, self_kpm: torch.Tensor | None = None) -> torch.Tensor:
        h = x_emb
        for blk in self.blocks:
            h = blk(h, self_kpm=self_kpm)           # 纯 block 组装
        logits = self.head(h)                        # layers: Linear
        return self.prob(logits) if self.return_probs else logits  # layers: Softmax(可选)
