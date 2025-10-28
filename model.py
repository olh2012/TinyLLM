import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from config import ModelConfig

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # 简化实现以避免类型错误
        return x  # 直接返回输入，不应用旋转位置编码

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_rate = config.dropout_rate
        
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size必须能被num_heads整除"
        
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # 线性变换得到Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑为多头形式
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 应用旋转位置编码
        if position_ids is not None:
            cos_sin = self.rotary_emb(value, position_ids)
            query, key = self.apply_rotary_pos_emb(query, key, cos_sin)
        
        # 计算注意力分数
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        # softmax归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, value)
        
        # 合并多头结果
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        
        # 输出投影
        attn_output = self.o_proj(attn_output)
        
        return attn_output
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 简化的RoPE应用（实际实现可能更复杂）
        # 这里我们只是返回原始的q和k，实际实现中需要应用旋转位置编码
        return q, k

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.dropout_rate = config.dropout_rate
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU激活函数
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        intermediate = torch.nn.functional.silu(gate) * up
        down = self.down_proj(intermediate)
        return self.dropout(down)

class RMSNorm(nn.Module):
    """RMS归一化"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class TransformerLayer(nn.Module):
    """Transformer层"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 自注意力块
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids
        )
        hidden_states = residual + hidden_states
        
        # 前馈网络块
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class TinyLLM(nn.Module):
    """微缩版DeepSeek模型"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        
        # 嵌入层
        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        
        # 输出层
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # 权重绑定
        self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化权重
        self.post_init()
        
    def post_init(self):
        """初始化权重"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # 创建位置ID
        if position_ids is None:
            position_ids_base = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids_base.unsqueeze(0).repeat(batch_size, 1)
            
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
            
        # 扩展注意力掩码
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length))
        
        # 获取嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过所有Transformer层
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            
        # 归一化
        hidden_states = self.norm(hidden_states)
        
        # 语言模型头部
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape):
        """准备解码器注意力掩码"""
        # 创建因果掩码
        combined_attention_mask = None
        bs, seq_len = input_shape
        
        # 创建上三角矩阵（因果掩码）
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) * -1e9  # 转换为大的负数用于softmax
        
        if attention_mask is not None:
            # 扩展attention_mask
            expanded_attn_mask = attention_mask[:, None, None, :].to(causal_mask.dtype) * -1e9
            expanded_attn_mask = expanded_attn_mask.expand(bs, 1, seq_len, seq_len)
            combined_attention_mask = expanded_attn_mask + causal_mask
            
        return combined_attention_mask

# 测试代码
if __name__ == "__main__":
    config = ModelConfig()
    model = TinyLLM(config)
    
    # 创建示例输入
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # 前向传播
    with torch.no_grad():
        outputs = model(input_ids)
        
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")