"""
位置编码实现
支持绝对位置编码和旋转位置编码(RoPE)
"""

import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    """绝对位置编码（Sinusoidal Positional Embedding）"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + positional_encoding: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        assert d_model % 2 == 0, "d_model必须是偶数"
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len: int = None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            seq_len: 序列长度（如果为None，使用x的seq_len）
        Returns:
            rotated_x: [batch_size, seq_len, d_model]
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # 生成位置索引
        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        
        # 计算角度
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, d_model // 2]
        
        # 生成旋转矩阵
        cos = torch.cos(freqs)  # [seq_len, d_model // 2]
        sin = torch.sin(freqs)  # [seq_len, d_model // 2]
        
        # 将cos和sin扩展到完整维度
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model]
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_model]
        
        # 分离奇偶维度
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1)  # [batch_size, seq_len, d_model // 2, 2]
        x_rot = x_rot.flatten(-2)  # [batch_size, seq_len, d_model]
        
        # 应用旋转
        rotated = x * cos + x_rot * sin
        
        return rotated
