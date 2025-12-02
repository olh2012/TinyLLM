# LLM From Scratch - 从零构建大语言模型

这是一个完整的、可训练、可推理的大语言模型（LLM）工程实现，基于Decoder-only Transformer架构（GPT类模型）。本项目完全基于公开论文和算法实现，不依赖任何专有模型权重。

## 📁 项目结构

```
llm-from-scratch/
├── tokenizer/              # Tokenizer实现
│   ├── __init__.py
│   ├── bpe_tokenizer.py    # BPE tokenizer核心实现
│   └── train_tokenizer.py  # Tokenizer训练脚本
├── model/                  # 模型实现
│   ├── __init__.py
│   ├── attention.py        # Multi-Head Attention
│   ├── mlp.py              # MLP (Feed-Forward Network)
│   ├── embedding.py        # 位置编码（绝对位置编码 + RoPE）
│   └── transformer.py      # GPT模型主体
├── data/                   # 数据集加载器
│   ├── __init__.py
│   └── dataset.py          # 支持txt/jsonl/huggingface datasets
├── train/                  # 训练模块
│   ├── __init__.py
│   └── train.py            # 训练脚本
├── inference/              # 推理模块
│   ├── __init__.py
│   └── generate.py         # 文本生成脚本
├── configs/                # 配置文件
│   ├── gpt_small.json      # 小模型配置（~124M参数）
│   ├── gpt_medium.json     # 中等模型配置（~350M参数）
│   └── gpt_large.json      # 大模型配置（~1.3B参数）
├── scripts/                # Shell脚本
│   ├── run_train.sh        # 训练脚本
│   └── run_infer.sh        # 推理脚本
├── data/                   # 数据目录（需要用户准备）
│   ├── train.txt
│   └── val.txt
├── requirements.txt        # 依赖包
└── README.md              # 本文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将训练数据放在 `data/` 目录下：

- **TXT格式**: 每行一个文本样本
- **JSONL格式**: 每行一个JSON对象，包含 `text` 字段
- **HuggingFace Datasets**: 在配置文件中指定数据集名称

示例 `data/train.txt`:
```
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
```

### 3. 训练Tokenizer

```bash
python tokenizer/train_tokenizer.py \
    --data_path data/train.txt \
    --output_dir tokenizer_output \
    --vocab_size 50000 \
    --data_type txt
```

这将生成：
- `tokenizer_output/vocab.json`: 词汇表
- `tokenizer_output/merges.txt`: BPE合并规则

### 4. 训练模型

```bash
python train/train.py --config configs/gpt_small.json
```

或使用脚本：
```bash
bash scripts/run_train.sh
```

### 5. 推理生成

```bash
python inference/generate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --config configs/gpt_small.json \
    --tokenizer_vocab tokenizer_output/vocab.json \
    --tokenizer_merges tokenizer_output/merges.txt \
    --prompt "The future of AI" \
    --max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

或使用脚本：
```bash
bash scripts/run_infer.sh
```

## 📊 模型配置

### GPT Small (~124M参数)
- `d_model`: 768
- `num_layers`: 12
- `num_heads`: 12
- `d_ff`: 3072
- 适合单GPU（8-16GB）训练

### GPT Medium (~350M参数)
- `d_model`: 1024
- `num_layers`: 24
- `num_heads`: 16
- `d_ff`: 4096
- 适合单GPU（16-24GB）训练

### GPT Large (~1.3B参数)
- `d_model`: 1280
- `num_layers`: 36
- `num_heads`: 20
- `d_ff`: 5120
- 需要大内存GPU或分布式训练

## 🔧 配置说明

### 模型配置
- `d_model`: 模型维度
- `num_layers`: Transformer层数
- `num_heads`: 注意力头数
- `d_ff`: 前馈网络维度
- `max_seq_len`: 最大序列长度
- `dropout`: Dropout比率
- `use_rope`: 是否使用RoPE位置编码

### 训练配置
- `batch_size`: 批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `optimizer`: 优化器（adam/adamw）
- `scheduler`: 学习率调度器（cosine/linear）
- `max_grad_norm`: 梯度裁剪阈值

### 数据配置
- `type`: 数据类型（txt/jsonl/huggingface）
- `train_path`: 训练数据路径
- `val_path`: 验证数据路径（可选）
- `text_key`: JSONL中的文本字段名
- `max_samples`: 最大样本数（用于测试）

## 🎯 功能特性

### Tokenizer
- ✅ BPE (Byte Pair Encoding) 实现
- ✅ 支持训练自定义词汇表
- ✅ 编码/解码功能
- ✅ 特殊token支持（pad, unk, bos, eos）

### 模型架构
- ✅ Multi-Head Attention（支持KV cache加速推理）
- ✅ MLP (Feed-Forward Network)
- ✅ Layer Normalization
- ✅ 绝对位置编码（Sinusoidal）
- ✅ 旋转位置编码（RoPE，可选）
- ✅ Causal Masking（确保自回归特性）

### 训练功能
- ✅ 梯度累积
- ✅ 梯度裁剪
- ✅ Checkpoint保存与恢复
- ✅ TensorBoard日志记录
- ✅ 学习率调度
- ✅ 验证集评估

### 推理功能
- ✅ 温度采样（Temperature Sampling）
- ✅ Top-k采样
- ✅ Top-p（Nucleus）采样
- ✅ KV cache加速
- ✅ 可配置生成长度

## 📈 训练示例

### 基本训练
```bash
python train/train.py --config configs/gpt_small.json
```

### 恢复训练
```bash
python train/train.py \
    --config configs/gpt_small.json \
    --resume checkpoints/checkpoint_latest.pt
```

### 指定设备
```bash
python train/train.py \
    --config configs/gpt_small.json \
    --device cuda:0
```

## 🎨 推理示例

### 基础生成
```bash
python inference/generate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --config configs/gpt_small.json \
    --tokenizer_vocab tokenizer_output/vocab.json \
    --tokenizer_merges tokenizer_output/merges.txt \
    --prompt "Once upon a time"
```

### 创意写作（高温度）
```bash
python inference/generate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --config configs/gpt_small.json \
    --tokenizer_vocab tokenizer_output/vocab.json \
    --tokenizer_merges tokenizer_output/merges.txt \
    --prompt "The future of technology" \
    --temperature 1.2 \
    --top_p 0.95
```

### 确定性生成（低温度）
```bash
python inference/generate.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --config configs/gpt_small.json \
    --tokenizer_vocab tokenizer_output/vocab.json \
    --tokenizer_merges tokenizer_output/merges.txt \
    --prompt "The capital of France is" \
    --temperature 0.3 \
    --top_k 10
```

## 📝 数据准备

### TXT格式
每行一个文本样本：
```
This is the first training sample.
This is the second training sample.
```

### JSONL格式
每行一个JSON对象：
```json
{"text": "This is a training sample."}
{"text": "Another training sample."}
```

### HuggingFace Datasets
在配置文件中设置：
```json
{
  "data": {
    "type": "huggingface",
    "dataset_name": "wikitext",
    "text_key": "text"
  }
}
```

## 🔍 超参数说明

### 学习率
- 小模型（<100M）: 6e-4
- 中等模型（100M-500M）: 3e-4
- 大模型（>500M）: 2e-4

### 批次大小
根据GPU内存调整：
- 8GB GPU: batch_size=2, gradient_accumulation_steps=16
- 16GB GPU: batch_size=4, gradient_accumulation_steps=8
- 24GB GPU: batch_size=8, gradient_accumulation_steps=4

### 采样参数
- **Temperature**: 控制随机性（0.1-2.0）
  - 低温度（0.1-0.5）: 更确定性
  - 高温度（1.0-2.0）: 更随机
- **Top-k**: 只从概率最高的k个token中采样（0表示禁用）
- **Top-p**: Nucleus采样，从累积概率达到p的token中采样（0表示禁用）

## 📊 示例结果

训练后的模型可以生成连贯的文本。以下是使用随机初始化权重（未训练）的示例输出：

```
输入: "The future of artificial intelligence"
输出: "[模型生成的文本，训练后会更加连贯]"
```

**注意**: 实际效果取决于训练数据质量和训练时长。

## 🚧 下一步扩展方向

### 性能优化
- [ ] Flash Attention（减少内存占用）
- [ ] Gradient Checkpointing（节省显存）
- [ ] Mixed Precision Training（FP16/BF16）
- [ ] 分布式训练（DDP/DeepSpeed）

### 模型改进
- [ ] LoRA/QLoRA（参数高效微调）
- [ ] 更好的位置编码（ALiBi等）
- [ ] 激活函数优化（Swish, GLU等）
- [ ] 归一化方法（RMSNorm等）

### 功能增强
- [ ] 支持多轮对话
- [ ] 支持指令微调（Instruction Tuning）
- [ ] 支持RLHF（Reinforcement Learning from Human Feedback）
- [ ] 模型量化（INT8/INT4）

### 工程优化
- [ ] 模型导出（ONNX/TensorRT）
- [ ] 推理服务化（FastAPI/Flask）
- [ ] 模型压缩（Pruning/Distillation）
- [ ] 更完善的日志和监控

## 📚 参考资源

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer原始论文
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2论文
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) - RoPE论文

## ⚠️ 注意事项

1. **内存要求**: 根据模型大小选择合适的配置，大模型需要更多GPU内存
2. **训练时间**: 完整训练可能需要数小时到数天，取决于数据量和模型大小
3. **数据质量**: 训练数据质量直接影响模型效果，建议使用高质量数据集
4. **超参数调优**: 不同数据集可能需要调整学习率、批次大小等超参数

## 📄 许可证

本项目基于公开论文和算法实现，仅供学习和研究使用。

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**Happy Training! 🚀**
