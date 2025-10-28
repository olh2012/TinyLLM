class ModelConfig:
    """模型配置类"""
    def __init__(self):
        # 模型参数
        self.vocab_size = 32000  # 词汇表大小
        self.hidden_size = 768   # 隐藏层维度
        self.num_hidden_layers = 12  # Transformer层数
        self.num_attention_heads = 12  # 注意力头数
        self.intermediate_size = 3072  # 前馈网络中间维度
        self.max_position_embeddings = 2048  # 最大位置编码
        self.layer_norm_eps = 1e-5  # LayerNorm epsilon值
        self.dropout_rate = 0.1  # Dropout概率
        
        # 训练参数
        self.learning_rate = 5e-4
        self.batch_size = 8
        self.num_epochs = 3
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        
        # 数据参数
        self.block_size = 512  # 序列长度
        
        # 其他参数
        self.seed = 42