"""
推理脚本
支持温度采样、Top-k、Top-p采样
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import GPTModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Top-k和Top-p过滤"""
    if top_k > 0:
        # 移除top_k之外的所有token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        # 计算累积概率
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits


def sample_token(logits, temperature=1.0, top_k=0, top_p=0.0):
    """采样下一个token"""
    # 应用温度
    if temperature != 1.0:
        logits = logits / temperature
    
    # Top-k和Top-p过滤
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    
    # Softmax和采样
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    
    return next_token


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, 
                  top_k=0, top_p=0.0, device='cuda'):
    """生成文本"""
    model.eval()
    
    # 编码prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = input_ids.clone()
    kv_cache = None
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            outputs = model(input_ids=input_ids, kv_cache=kv_cache)
            logits = outputs['logits']
            kv_cache = outputs['kv_cache']
            
            # 获取最后一个token的logits
            next_token_logits = logits[0, -1, :]
            
            # 采样下一个token
            next_token = sample_token(
                next_token_logits.unsqueeze(0),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # 检查是否到达EOS
            if next_token.item() == tokenizer.vocab[tokenizer.eos_token]:
                break
            
            # 添加到生成序列
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            
            # 更新input_ids为最后一个token（用于下一次迭代）
            input_ids = next_token.unsqueeze(0)
    
    # 解码生成的文本
    generated_ids = generated[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='GPT模型推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--tokenizer_vocab', type=str, required=True,
                        help='Tokenizer词汇表路径')
    parser.add_argument('--tokenizer_merges', type=str, required=True,
                        help='Tokenizer合并规则路径')
    parser.add_argument('--prompt', type=str, default="Hello, how are you?",
                        help='输入提示')
    parser.add_argument('--max_length', type=int, default=100,
                        help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='温度参数')
    parser.add_argument('--top_k', type=int, default=0,
                        help='Top-k采样（0表示禁用）')
    parser.add_argument('--top_p', type=float, default=0.0,
                        help='Top-p采样（0表示禁用）')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer_vocab, args.tokenizer_merges)
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    print("创建模型...")
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size
    model = GPTModel.from_config(model_config)
    
    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型参数量: {model.get_num_params():,}")
    
    # 生成文本
    print(f"\n输入提示: {args.prompt}")
    print(f"生成参数: temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("\n生成文本:")
    print("-" * 50)
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else 0,
        top_p=args.top_p if args.top_p > 0.0 else 0.0,
        device=device
    )
    
    print(generated_text)
    print("-" * 50)


if __name__ == '__main__':
    main()
