"""
GPT模型实现
Decoder-only Transformer架构
"""

import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention
from .mlp import MLP
from .embedding import PositionalEmbedding


class TransformerBlock(nn.Module):
    """Transformer Block (Decoder Layer)"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.use_rope = use_rope
        
        # 注意力层
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.mlp = MLP(d_model, d_ff, dropout)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # RoPE（如果使用）
        if use_rope:
            from .embedding import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(d_model)
    
    def forward(self, x, mask=None, kv_cache=None):
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: 注意力mask
            kv_cache: KV cache for inference
        Returns:
            output: [batch_size, seq_len, d_model]
            new_kv_cache: updated KV cache
        """
        # Self-attention with residual connection
        residual = x
        x = self.ln1(x)
        
        # 如果使用RoPE，在attention之前应用
        if self.use_rope:
            x = self.rope(x)
        
        attn_output, new_kv_cache = self.attention(x, x, x, mask=mask, kv_cache=kv_cache)
        x = residual + self.dropout(attn_output)
        
        # MLP with residual connection
        residual = x
        x = self.ln2(x)
        mlp_output = self.mlp(x)
        x = residual + self.dropout(mlp_output)
        
        return x, new_kv_cache


class GPTModel(nn.Module):
    """GPT模型（Decoder-only Transformer）"""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_layers: int = 12,
                 num_heads: int = 12, d_ff: int = 3072, max_seq_len: int = 512,
                 dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embedding
        if not use_rope:
            self.pos_embedding = PositionalEmbedding(d_model, max_seq_len)
        else:
            self.pos_embedding = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_rope)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output head
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device):
        """生成causal mask（下三角矩阵）"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids, labels=None, kv_cache=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] (可选，用于训练)
            kv_cache: List of (key_cache, value_cache) for each layer (可选，用于推理)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            loss: scalar (如果提供了labels)
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embedding
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Positional embedding
        if not self.use_rope:
            x = self.pos_embedding(x)
        
        # Causal mask
        mask = self._generate_causal_mask(seq_len, device)
        
        # 通过Transformer blocks
        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache[i] if kv_cache else None
            x, new_layer_kv_cache = block(x, mask=mask, kv_cache=layer_kv_cache)
            new_kv_cache.append(new_layer_kv_cache)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output logits
        logits = self.head(x)  # [batch_size, seq_len, vocab_size]
        
        # 计算loss（如果提供了labels）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                           shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'kv_cache': new_kv_cache
        }
    
    def get_num_params(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @classmethod
    def from_config(cls, config):
        """从配置创建模型"""
        return cls(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1),
            use_rope=config.get('use_rope', False)
        )
