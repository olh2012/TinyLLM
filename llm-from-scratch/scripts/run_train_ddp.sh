#!/bin/bash
# 分布式训练脚本

# 设置GPU数量
NUM_GPUS=4

# 配置文件
CONFIG="configs/gpt_small.json"

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=$NUM_GPUS \
    train/train.py \
    --config $CONFIG
