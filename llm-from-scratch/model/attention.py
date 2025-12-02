"""
Multi-Head Attention实现
基于"Attention is All You Need"论文
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None, kv_cache=None):
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
            kv_cache: (key_cache, value_cache) 用于推理加速
        Returns:
            output: [batch_size, seq_len, d_model]
            new_kv_cache: (new_key_cache, new_value_cache)
        """
        batch_size, seq_len, _ = query.size()
        
        # 投影到Q, K, V
        Q = self.W_q(query)  # [batch_size, seq_len, d_model]
        K = self.W_k(key)    # [batch_size, seq_len, d_model]
        V = self.W_v(value)  # [batch_size, seq_len, d_model]
        
        # 如果有KV cache，拼接
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            if key_cache is not None:
                K = torch.cat([key_cache, K], dim=1)
                V = torch.cat([value_cache, V], dim=1)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)      # [batch_size, num_heads, kv_len, d_k]
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)      # [batch_size, num_heads, kv_len, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, num_heads, seq_len, kv_len]
        
        # 应用mask（causal mask for decoder）
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_len]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch_size, seq_len, num_heads, d_k]
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)  # [batch_size, seq_len, d_model]
        
        # 输出投影
        output = self.W_o(attn_output)
        
        # 更新KV cache
        new_kv_cache = None
        if kv_cache is not None:
            new_kv_cache = (K.transpose(1, 2), V.transpose(1, 2))
        
        return output, new_kv_cache
