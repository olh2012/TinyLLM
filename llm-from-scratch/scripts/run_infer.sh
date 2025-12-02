#!/bin/bash

# 推理脚本示例

# 设置参数
CHECKPOINT_PATH="checkpoints/checkpoint_best.pt"
CONFIG_PATH="configs/gpt_small.json"
TOKENIZER_VOCAB="tokenizer_output/vocab.json"
TOKENIZER_MERGES="tokenizer_output/merges.txt"
PROMPT="The future of artificial intelligence"

# 运行推理
python inference/generate.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --tokenizer_vocab "$TOKENIZER_VOCAB" \
    --tokenizer_merges "$TOKENIZER_MERGES" \
    --prompt "$PROMPT" \
    --max_length 100 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9 \
    --device cuda
