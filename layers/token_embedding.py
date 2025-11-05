# layers/token_embedding.py
import torch
import torch.nn as nn
import config

class TokenEmbedding(nn.Module):
    def __init__(self, cfg: config.ModelConfig, pad_id: int = 0, scale_by_sqrt_d: bool = True):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.scale = (cfg.d_model ** 0.5) if scale_by_sqrt_d else 1.0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(input_ids) * self.scale
