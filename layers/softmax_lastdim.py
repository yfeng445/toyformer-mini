# layers/softmax_lastdim.py
import torch
import torch.nn as nn
import ops

class SoftmaxLastDim(nn.Module):
    """对最后一维做 softmax（数值稳定由 ops.softmax 处理）"""
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return ops.softmax(logits)
