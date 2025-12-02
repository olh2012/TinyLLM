"""
MLP (Feed-Forward Network) 实现
支持多种激活函数：GELU, ReLU, Swish, GLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish激活函数: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit: (W1*x + b1) * sigmoid(W2*x + b2)"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff)
        self.up = nn.Linear(d_model, d_ff)
    
    def forward(self, x):
        return self.gate(x) * torch.sigmoid(self.up(x))


class MLP(nn.Module):
    """多层感知机（前馈网络）"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'gelu'):
        super().__init__()
        self.activation_name = activation
        
        if activation == 'glu':
            # GLU需要特殊处理
            self.glu = GLU(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            
            if activation == 'gelu':
                self.activation = nn.GELU()
            elif activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'swish':
                self.activation = Swish()
            else:
                raise ValueError(f"不支持的激活函数: {activation}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        if self.activation_name == 'glu':
            x = self.glu(x)
        else:
            x = self.fc1(x)
            x = self.activation(x)
        
        x = self.dropout(x)
        x = self.fc2(x)
        return x
