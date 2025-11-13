import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List
from config import ModelConfig

class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, dim: int, max_position_embeddings: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 简化实现，返回全零的cos和sin
        batch_size = position_ids.shape[0]
        seq_len = position_ids.shape[1]
        # 返回全1的cos和全0的sin，相当于不应用位置编码
        cos = torch.ones(batch_size, seq_len, self.dim, dtype=x.dtype, device=x.device)
        sin = torch.zeros(batch_size, seq_len, self.dim, dtype=x.dtype, device=x.device)
        return cos, sin

    def rotate_half(self, x):
        """旋转一半维度"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """应用旋转位置编码"""
        # 调整cos和sin的维度以匹配q和k
        # q和k的形状: [batch_size, num_heads, seq_len, head_dim]
        # cos和sin的形状: [batch_size, seq_len, head_dim]
        cos = cos.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)  # [batch_size, 1, seq_len, head_dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed

class MultiHeadLatentAttention(nn.Module):
    """多头潜在注意力机制"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout_rate = config.dropout_rate
        self.use_flash_attention = config.use_flash_attention if hasattr(config, 'use_flash_attention') else False
        
        # MLA特定参数
        self.latent_dim = config.latent_dim
        self.num_latent_heads = config.num_latent_heads
        
        assert self.head_dim * self.num_heads == self.hidden_size, "hidden_size必须能被num_heads整除"
        
        # 查询投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        
        # 潜在键值投影
        self.k_latent_proj = nn.Linear(self.hidden_size, self.num_latent_heads * self.latent_dim)
        self.v_latent_proj = nn.Linear(self.hidden_size, self.num_latent_heads * self.latent_dim)
        
        # 从潜在表示恢复键值
        self.k_restore = nn.Linear(self.latent_dim, self.head_dim)
        self.v_restore = nn.Linear(self.latent_dim, self.head_dim)
        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 尝试导入Flash Attention
        self.flash_attn_func = None
        if self.use_flash_attention:
            try:
                import importlib
                flash_attn_module = importlib.import_module("flash_attn")
                self.flash_attn_func = getattr(flash_attn_module, "flash_attn_func")
                print("Flash Attention已成功导入")
            except (ImportError, AttributeError):
                print("Flash Attention未安装，将使用标准注意力实现")
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache = None,  # 简化类型注解以避免循环导入
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.size()
        
        # 线性变换得到Q和潜在K, V
        query = self.q_proj(hidden_states)
        k_latent = self.k_latent_proj(hidden_states)
        v_latent = self.v_latent_proj(hidden_states)
        
        # 重塑为多头形式
        query = query.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k_latent = k_latent.view(batch_size, seq_length, self.num_latent_heads, self.latent_dim)
        v_latent = v_latent.view(batch_size, seq_length, self.num_latent_heads, self.latent_dim)
        
        # 从潜在表示恢复键值 (先应用线性层，再转置)
        # k_latent和v_latent的形状: [batch_size, seq_length, num_latent_heads, latent_dim]
        # 重塑为[batch_size * seq_length * num_latent_heads, latent_dim]以应用线性层
        k_latent_reshaped = k_latent.view(-1, self.latent_dim)
        v_latent_reshaped = v_latent.view(-1, self.latent_dim)
        
        # 应用恢复投影
        key_reshaped = self.k_restore(k_latent_reshaped)
        value_reshaped = self.v_restore(v_latent_reshaped)
        
        # 恢复形状并转置为[batch_size, num_latent_heads, seq_length, head_dim]
        key = key_reshaped.view(batch_size, seq_length, self.num_latent_heads, self.head_dim).transpose(1, 2)
        value = value_reshaped.view(batch_size, seq_length, self.num_latent_heads, self.head_dim).transpose(1, 2)
        
        # 如果头数不匹配，需要重复KV头
        if self.num_latent_heads != self.num_heads:
            # 重复KV头以匹配查询头数
            repeat_times = self.num_heads // self.num_latent_heads
            key = key.repeat_interleave(repeat_times, dim=1)
            value = value.repeat_interleave(repeat_times, dim=1)
        
        # 应用KV缓存
        if kv_cache is not None and layer_idx is not None:
            if len(kv_cache.key_cache) > layer_idx and kv_cache.key_cache[layer_idx] is not None:
                # 使用缓存的KV并追加新的KV
                key = torch.cat([kv_cache.key_cache[layer_idx], key], dim=2)
                value = torch.cat([kv_cache.value_cache[layer_idx], value], dim=2)
            
            # 更新缓存
            if len(kv_cache.key_cache) <= layer_idx:
                # 扩展缓存列表
                while len(kv_cache.key_cache) <= layer_idx:
                    kv_cache.key_cache.append(None)
                    kv_cache.value_cache.append(None)
            
            kv_cache.key_cache[layer_idx] = key
            kv_cache.value_cache[layer_idx] = value
        
        # 应用旋转位置编码
        if position_ids is not None:
            cos, sin = self.rotary_emb(value, position_ids)
            query, key = self.rotary_emb.apply_rotary_pos_emb(query, key, cos, sin)
        
        # 使用Flash Attention（如果可用且配置启用）
        if self.use_flash_attention and self.flash_attn_func is not None and kv_cache is None:
            # Flash Attention需要重新排列维度
            query_trans = query.transpose(1, 2)  # [batch, seq_len, heads, head_dim]
            key_trans = key.transpose(1, 2)      # [batch, seq_len, heads, head_dim]
            value_trans = value.transpose(1, 2)  # [batch, seq_len, heads, head_dim]
            
            # 应用Flash Attention
            attn_output = self.flash_attn_func(query_trans, key_trans, value_trans, causal=True)
            attn_output = attn_output.transpose(1, 2)  # 转换回[batch, heads, seq_len, head_dim]
        else:
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
        # 这个方法不再使用，因为我们在forward中直接调用RoPE模块的方法
        # 保留此方法以保持接口兼容性
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
        # 使用MLA替代原来的GQA
        self.self_attn = MultiHeadLatentAttention(config)
        self.mlp = FeedForward(config)
        self.input_layernorm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(self.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache = None,  # 简化类型注解
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        # 自注意力块
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            layer_idx=layer_idx
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
        position_ids: Optional[torch.Tensor] = None,
        kv_cache = None,  # 简化类型注解
        use_cache: bool = False
    ) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        
        # 创建位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
            
        # 创建注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
            
        # 扩展注意力掩码
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length))
        
        # 获取嵌入
        hidden_states = self.embed_tokens(input_ids)
        
        # 通过所有Transformer层
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache if use_cache else None,
                layer_idx=idx
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
    
    def quantize(self, bits: int = 8):
        """模型量化"""
        try:
            import torch.quantization as quant
            # 设置量化配置
            if bits == 8:
                # 8位量化
                self.qconfig = quant.get_default_qconfig('fbgemm')
                quant.prepare(self, inplace=True)
                quant.convert(self, inplace=True)
            elif bits == 4:
                # 4位量化 - 简化的实现
                for name, module in self.named_modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        # 对权重进行4位量化
                        if hasattr(module, 'weight'):
                            # 简单的线性量化
                            weight = module.weight.data
                            scale = (weight.max() - weight.min()) / (2 ** bits - 1)
                            zero_point = -weight.min() / scale
                            quantized_weight = torch.round(weight / scale + zero_point).clamp(0, 2 ** bits - 1)
                            module.weight.data = (quantized_weight - zero_point) * scale
            print(f"模型已量化为{bits}位")
        except Exception as e:
            print(f"量化过程中出现错误: {e}")

class KVCache:
    """KV缓存管理器"""
    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
    def update(self, key_states, value_states, layer_idx):
        """更新指定层的KV缓存"""
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
        
    def clear(self):
        """清空缓存"""
        self.key_cache.clear()
        self.value_cache.clear()

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