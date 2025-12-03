# 新增功能说明

本文档说明已实现的新功能及其使用方法。

## ✅ 已实现功能

### 1. 混合精度训练（FP16/BF16）

**文件**: `train/train.py`

**功能**: 支持FP16和BF16混合精度训练，显著减少显存占用并加速训练。

**使用方法**:
在配置文件中添加：
```json
{
  "training": {
    "use_amp": true,
    "amp_dtype": "fp16"  // 或 "bf16"
  }
}
```

**优势**:
- 显存占用减少约50%
- 训练速度提升1.5-2倍
- BF16在支持的硬件上数值更稳定

---

### 2. Gradient Checkpointing

**文件**: `model/transformer.py`

**功能**: 通过牺牲少量计算时间换取显存节省，允许训练更大的模型。

**使用方法**:
在配置文件中添加：
```json
{
  "model": {
    "gradient_checkpointing": true
  }
}
```

**优势**:
- 显存占用减少50-70%
- 可以训练更大的模型或使用更大的batch size
- 训练速度略有下降（约20%）

---

### 3. 更多激活函数

**文件**: `model/mlp.py`

**功能**: 支持GELU、ReLU、Swish、GLU等多种激活函数。

**使用方法**:
在配置文件中添加：
```json
{
  "model": {
    "mlp_activation": "swish"  // gelu, relu, swish, glu
  }
}
```

**说明**:
- GELU: 默认激活函数，GPT标准
- ReLU: 简单高效
- Swish: x * sigmoid(x)，在某些任务上表现更好
- GLU: Gated Linear Unit，需要特殊处理

---

### 4. RMSNorm归一化

**文件**: `model/normalization.py`, `model/transformer.py`

**功能**: 实现RMSNorm作为LayerNorm的替代方案。

**使用方法**:
在配置文件中添加：
```json
{
  "model": {
    "norm_type": "rmsnorm"  // layernorm 或 rmsnorm
  }
}
```

**优势**:
- 计算更高效（不需要计算均值）
- 在某些任务上表现更好
- 减少计算开销

---

### 5. LoRA参数高效微调

**文件**: `train/lora.py`, `train/train_lora.py`

**功能**: 实现LoRA（Low-Rank Adaptation）用于参数高效微调。

**使用方法**:
```bash
python train/train_lora.py \
    --config configs/gpt_small.json \
    --base_model checkpoints/checkpoint_best.pt \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.1 \
    --target_modules attention mlp
```

**优势**:
- 只需训练0.1-1%的参数
- 显存占用大幅减少
- 训练速度快
- 可以保存多个任务特定的LoRA权重

**参数说明**:
- `lora_rank`: LoRA的秩，通常8-64
- `lora_alpha`: 缩放因子，通常等于rank或rank的2倍
- `lora_dropout`: Dropout率，可选
- `target_modules`: 要应用LoRA的模块列表

---

### 6. 分布式训练（DDP）

**文件**: `train/train.py`

**功能**: 支持多GPU分布式训练，使用PyTorch DDP。

**使用方法**:
```bash
# 单机多卡
torchrun --nproc_per_node=4 train/train.py --config configs/gpt_small.json

# 或使用脚本
bash scripts/run_train_ddp.sh
```

**优势**:
- 线性加速训练
- 支持多机多卡训练
- 自动处理数据分片和梯度同步

**注意事项**:
- 需要NCCL后端（NVIDIA GPU）
- 确保所有GPU在同一节点上
- batch size会自动按GPU数量缩放

---

### 7. 模型量化（INT8）

**文件**: `inference/quantize.py`

**功能**: 将模型量化为INT8，减少推理时的显存占用和模型大小。

**使用方法**:
```bash
python inference/quantize.py \
    --checkpoint checkpoints/checkpoint_best.pt \
    --config configs/gpt_small.json \
    --output checkpoints/quantized_model.pt
```

**优势**:
- 模型大小减少约75%
- 推理速度提升
- 显存占用减少
- 适合部署到资源受限的环境

**注意事项**:
- 量化会带来轻微的精度损失
- 仅量化Linear层
- 使用动态量化（运行时量化）

---

### 8. 推理服务化（FastAPI）

**文件**: `inference/api_server.py`

**功能**: 提供RESTful API服务，方便模型部署和调用。

**使用方法**:
```bash
# 设置环境变量
export CHECKPOINT_PATH=./checkpoints/checkpoint_best.pt
export CONFIG_PATH=./configs/gpt_small.json
export VOCAB_PATH=./tokenizer_output/vocab.json
export MERGES_PATH=./tokenizer_output/merges.txt

# 启动服务
python inference/api_server.py

# 或使用uvicorn
uvicorn inference.api_server:app --host 0.0.0.0 --port 8000
```

**API端点**:
- `GET /`: API信息和文档
- `GET /health`: 健康检查
- `POST /generate`: 单次文本生成
- `POST /generate/batch`: 批量文本生成

**示例请求**:
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "The future of AI",
       "max_length": 100,
       "temperature": 0.8,
       "top_k": 50,
       "top_p": 0.9
     }'
```

**响应格式**:
```json
{
  "generated_text": "...",
  "prompt": "The future of AI",
  "parameters": {
    "max_length": 100,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9
  }
}
```

---

## 📝 配置文件示例

完整配置示例，包含所有新功能：

```json
{
  "model": {
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "max_seq_len": 512,
    "dropout": 0.1,
    "use_rope": false,
    "gradient_checkpointing": true,
    "norm_type": "rmsnorm",
    "mlp_activation": "swish"
  },
  "training": {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "num_epochs": 10,
    "learning_rate": 6e-4,
    "optimizer": "adamw",
    "use_amp": true,
    "amp_dtype": "fp16",
    "max_grad_norm": 1.0,
    "scheduler": "cosine"
  }
}
```

---

## 🔧 依赖更新

新增依赖包（已更新到 `requirements.txt`）:
- `fastapi>=0.100.0`: API服务框架
- `uvicorn>=0.23.0`: ASGI服务器
- `pydantic>=2.0.0`: 数据验证

---

## 🚀 快速开始

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **启用混合精度训练**:
   在配置文件中设置 `use_amp: true`

3. **使用LoRA微调**:
   ```bash
   bash scripts/run_lora.sh
   ```

4. **启动API服务**:
   ```bash
   bash scripts/run_api_server.sh
   ```

---

## 📚 参考资源

- [LoRA论文](https://arxiv.org/abs/2106.09685)
- [RMSNorm论文](https://arxiv.org/abs/1910.07467)
- [PyTorch DDP文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [FastAPI文档](https://fastapi.tiangolo.com/)

---

**注意**: 所有新功能都向后兼容，不会影响现有代码的使用。
