"""
归一化方法实现
支持LayerNorm和RMSNorm
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    参考: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] 或 [batch_size, d_model]
        Returns:
            normalized_x: 相同形状
        """
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        x = x / rms * self.weight
        return x
