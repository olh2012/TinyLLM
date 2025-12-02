"""
模型量化工具
支持INT8量化
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.transformer import GPTModel


def quantize_model_int8(model, example_input=None):
    """
    将模型量化为INT8
    使用PyTorch的动态量化
    """
    # 确保模型在CPU上（量化通常在CPU上进行）
    model = model.cpu()
    model.eval()
    
    # 使用动态量化（对Linear层进行INT8量化）
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # 只量化Linear层
        dtype=torch.qint8
    )
    
    return quantized_model


def save_quantized_model(model, save_path):
    """保存量化模型"""
    torch.save(model.state_dict(), save_path)
    print(f"量化模型已保存到: {save_path}")


def load_quantized_model(model, quantized_path, device='cpu'):
    """加载量化模型"""
    model.load_state_dict(torch.load(quantized_path, map_location=device))
    model = model.to(device)
    return model


if __name__ == '__main__':
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='模型量化')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='原始模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--output', type=str, required=True,
                        help='量化模型保存路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建模型
    model_config = config['model'].copy()
    model_config['vocab_size'] = 50000  # 默认值，实际应从tokenizer获取
    
    model = GPTModel.from_config(model_config)
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 量化
    print("开始量化模型...")
    quantized_model = quantize_model_int8(model)
    
    # 保存
    save_quantized_model(quantized_model, args.output)
    print("量化完成！")
