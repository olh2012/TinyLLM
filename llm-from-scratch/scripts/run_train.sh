#!/bin/bash

# 训练脚本示例

# 设置参数
CONFIG_PATH="configs/gpt_small.json"
DATA_PATH="data/train.txt"
TOKENIZER_OUTPUT_DIR="tokenizer_output"
CHECKPOINT_DIR="checkpoints"

# 1. 训练tokenizer（如果还没有）
if [ ! -f "$TOKENIZER_OUTPUT_DIR/vocab.json" ]; then
    echo "训练tokenizer..."
    python tokenizer/train_tokenizer.py \
        --data_path "$DATA_PATH" \
        --output_dir "$TOKENIZER_OUTPUT_DIR" \
        --vocab_size 50000 \
        --data_type txt
fi

# 2. 开始训练
echo "开始训练模型..."
python train/train.py \
    --config "$CONFIG_PATH" \
    --device cuda

echo "训练完成！"
