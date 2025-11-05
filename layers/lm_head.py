# layers/lm_head.py
import torch
import torch.nn as nn
import ops
import config

class LMHead(nn.Module):
    def __init__(self, cfg: config.ModelConfig, bias: bool = False):
        super().__init__()
        D, V = cfg.d_model, cfg.vocab_size
        
        self.W = nn.Parameter(torch.empty(D, V))
        self.b = nn.Parameter(torch.zeros(V)) if bias else None
        nn.init.xavier_uniform_(self.W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) → (B,T,V)
        return ops.linear(x, self.W, self.b)
