import torch
import torch.nn as nn
import ops
import config

class AddNormPre(nn.Module):
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        self.eps = cfg.addnorm.eps
        D = cfg.d_model
        self.gamma = nn.Parameter(torch.ones(D))
        self.beta  = nn.Parameter(torch.zeros(D))

    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        x_norm = ops.layer_norm(x, self.gamma, self.beta, self.eps)  # <- 统一命名
        s = sublayer_fn(x_norm)
        return ops.add(x, s)
