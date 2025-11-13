class ModelConfig:
    """模型配置类"""
    def __init__(self):
        # 模型参数
        self.vocab_size = 32000  # 词汇表大小
        self.hidden_size = 768   # 隐藏层维度
        self.num_hidden_layers = 12  # Transformer层数
        self.num_attention_heads = 12  # 注意力头数
        self.num_kv_heads = self.num_attention_heads  # GQA的KV头数，默认与查询头数相同
        self.intermediate_size = 3072  # 前馈网络中间维度
        self.max_position_embeddings = 2048  # 最大位置编码
        self.layer_norm_eps = 1e-5  # LayerNorm epsilon值
        self.dropout_rate = 0.1  # Dropout概率
        
        # MLA特定参数
        self.latent_dim = self.hidden_size // self.num_attention_heads // 2  # 潜在维度
        self.num_latent_heads = self.num_attention_heads  # 潜在头数
        
        # 训练参数
        self.learning_rate = 5e-4
        self.batch_size = 8
        self.num_epochs = 3
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        
        # 数据参数
        self.block_size = 512  # 序列长度
        
        # 推理优化参数
        self.use_kv_cache = True  # 是否使用KV缓存
        self.use_flash_attention = False  # 是否使用Flash Attention（如果可用）
        
        # 其他参数
        self.seed = 42