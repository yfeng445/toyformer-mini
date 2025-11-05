import math
import torch
import numpy as np
import torch
from typing import Optional

def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == B.shape[-2], "inner dims must match (K)"
    return (A.unsqueeze(-1) * B.unsqueeze(-3)).sum(dim=-2)

def linear(x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    y = matmul(x, W)
    return y if b is None else y + b

def scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return x * scale

# softmax(x) = exp(x) / sum(exp(x))
def softmax(x: torch.Tensor) -> torch.Tensor:
    upcast = x.dtype in (torch.float16, torch.bfloat16)
    xf = x.float() if upcast else x
    xf = xf - xf.amax(dim=-1, keepdim=True)
    exp = torch.exp(xf)
    denom = exp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(exp.dtype).tiny)
    out = exp / denom
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.to(x.dtype) if upcast else out

# layerNorm(x) = gamma * (x - mu) / sqrt(var + eps) + beta
def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    upcast = x.dtype in (torch.float16, torch.bfloat16)
    xf = x.float() if upcast else x
    mu  = xf.mean(dim=-1, keepdim=True) # mu = mean(x)
    var = (xf - mu).pow(2).mean(dim=-1, keepdim=True) # var = mean((x - mu)^2)
    y = (xf - mu) / torch.sqrt(var + eps)  * gamma + beta
    return y.to(x.dtype) if upcast else y   

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

# GELU(x) = 0.5*x*(1 + erf(x/√2))
# approximate GELU: GELU(x) = 0.5*x*(1 + tanh(√(2/π)*(x + 0.044715 x^3)))
def gelu(x: torch.Tensor, approximate: bool = False) -> torch.Tensor:
    if approximate:
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * (x ** 3))))
    else:
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

def make_causal_mask(Tq: int, Tk: int | None = None, device=None):
    if Tk is None: Tk = Tq
    i = torch.arange(Tq, device=device)[:, None]
    j = torch.arange(Tk, device=device)[None, :]
    causal = (j > i)
    return causal[None, None, :, :]

def make_padding_mask(key_padding_mask: torch.Tensor):
    pad = ~key_padding_mask.bool()
    return pad[:, None, None, :]

def merge_masks(T: int, key_padding_mask: torch.Tensor | None, device):
    causal = make_causal_mask(T, device=device)
    if key_padding_mask is None:
        return causal
    pad = make_padding_mask(key_padding_mask)
    return pad | causal
