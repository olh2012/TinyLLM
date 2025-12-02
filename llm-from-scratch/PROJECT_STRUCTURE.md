# 项目目录结构

```
llm-from-scratch/
├── tokenizer/                    # Tokenizer模块
│   ├── __init__.py              # 模块初始化
│   ├── bpe_tokenizer.py         # BPE tokenizer核心实现
│   └── train_tokenizer.py       # Tokenizer训练脚本
│
├── model/                        # 模型实现
│   ├── __init__.py              # 模块初始化
│   ├── attention.py             # Multi-Head Attention实现
│   ├── mlp.py                   # MLP (Feed-Forward Network)
│   ├── embedding.py             # 位置编码（绝对位置编码 + RoPE）
│   └── transformer.py           # GPT模型主体（Decoder-only Transformer）
│
├── data/                         # 数据集模块
│   ├── __init__.py              # 模块初始化
│   ├── dataset.py               # 数据集加载器（支持txt/jsonl/huggingface）
│   ├── train.txt.example        # 训练数据示例
│   └── val.txt.example          # 验证数据示例
│
├── train/                        # 训练模块
│   ├── __init__.py              # 模块初始化
│   └── train.py                 # 训练脚本（支持分布式、gradient accumulation、checkpoint）
│
├── inference/                     # 推理模块
│   ├── __init__.py              # 模块初始化
│   └── generate.py              # 文本生成脚本（支持温度、Top-k、Top-p采样）
│
├── configs/                       # 配置文件目录
│   ├── gpt_small.json           # 小模型配置（~124M参数）
│   ├── gpt_medium.json          # 中等模型配置（~350M参数）
│   └── gpt_large.json           # 大模型配置（~1.3B参数）
│
├── scripts/                       # Shell脚本
│   ├── run_train.sh             # 训练脚本
│   └── run_infer.sh             # 推理脚本
│
├── requirements.txt              # Python依赖包
├── README.md                     # 项目文档
├── PROJECT_STRUCTURE.md          # 本文件
└── demo.py                       # 演示脚本（使用随机权重生成示例）

# 运行时生成的目录（不在版本控制中）
├── tokenizer_output/             # Tokenizer输出目录
│   ├── vocab.json               # 词汇表
│   └── merges.txt               # BPE合并规则
│
├── checkpoints/                  # 模型checkpoint目录
│   ├── checkpoint_latest.pt     # 最新checkpoint
│   ├── checkpoint_best.pt       # 最佳checkpoint
│   └── checkpoint_epoch_*.pt    # 定期保存的checkpoint
│
└── logs/                         # TensorBoard日志目录
    └── events.out.tfevents.*     # TensorBoard事件文件
```

## 核心文件说明

### Tokenizer (`tokenizer/`)
- **bpe_tokenizer.py**: 完整的BPE实现，包括训练、编码、解码、保存/加载
- **train_tokenizer.py**: 命令行工具，用于从数据训练tokenizer

### 模型 (`model/`)
- **attention.py**: Multi-Head Attention，支持KV cache加速推理
- **mlp.py**: 前馈网络（MLP）
- **embedding.py**: 位置编码（支持绝对位置编码和RoPE）
- **transformer.py**: 完整的GPT模型，包含多个Transformer Block

### 数据 (`data/`)
- **dataset.py**: 统一的数据集接口，支持多种数据格式
  - `TextDataset`: 从txt文件加载
  - `JSONLDataset`: 从JSONL文件加载
  - `HuggingFaceDataset`: 从HuggingFace datasets加载

### 训练 (`train/`)
- **train.py**: 完整的训练流程
  - 支持梯度累积
  - 支持梯度裁剪
  - 支持checkpoint保存与恢复
  - 支持TensorBoard日志
  - 支持学习率调度

### 推理 (`inference/`)
- **generate.py**: 文本生成
  - 温度采样
  - Top-k采样
  - Top-p（Nucleus）采样
  - KV cache加速

### 配置 (`configs/`)
- JSON格式的配置文件，包含模型、训练、数据等所有超参数
- 提供三种预设配置：small、medium、large

## 使用流程

1. **准备数据** → 将数据放入 `data/` 目录
2. **训练Tokenizer** → 运行 `tokenizer/train_tokenizer.py`
3. **训练模型** → 运行 `train/train.py`
4. **推理生成** → 运行 `inference/generate.py`

详细说明请参考 `README.md`。
