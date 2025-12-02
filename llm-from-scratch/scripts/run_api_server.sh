#!/bin/bash
# 启动API服务脚本

export CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./checkpoints/checkpoint_best.pt"}
export CONFIG_PATH=${CONFIG_PATH:-"./configs/gpt_small.json"}
export VOCAB_PATH=${VOCAB_PATH:-"./tokenizer_output/vocab.json"}
export MERGES_PATH=${MERGES_PATH:-"./tokenizer_output/merges.txt"}

python inference/api_server.py
