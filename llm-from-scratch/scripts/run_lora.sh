#!/bin/bash
# LoRA微调脚本

CONFIG="configs/gpt_small.json"
BASE_MODEL="checkpoints/checkpoint_best.pt"

python train/train_lora.py \
    --config $CONFIG \
    --base_model $BASE_MODEL \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.1 \
    --target_modules attention mlp
