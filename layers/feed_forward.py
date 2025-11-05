import torch
import torch.nn as nn
import ops
import config

class FeedForward(nn.Module):
    def __init__(self, cfg: config.ModelConfig):
        super().__init__()
        D = cfg.d_model
        self.D = D
        self.Dff = cfg.ffn.d_ff or (4 * D)
        self.gelu_approx = cfg.ffn.gelu_approx

        self.W1 = nn.Parameter(torch.empty(D, self.Dff))
        self.b1 = nn.Parameter(torch.zeros(self.Dff))
        self.W2 = nn.Parameter(torch.empty(self.Dff, D))
        self.b2 = nn.Parameter(torch.zeros(D))

        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = ops.linear(x, self.W1, self.b1)
        h = ops.gelu(h, approximate=self.gelu_approx)
        y = ops.linear(h, self.W2, self.b2)
        return y
