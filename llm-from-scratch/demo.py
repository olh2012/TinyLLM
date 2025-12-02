"""
演示脚本：使用随机初始化的模型生成文本示例
"""

import torch
import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import GPTModel
from inference.generate import generate_text


def create_demo_tokenizer():
    """创建演示用的简单tokenizer"""
    print("创建演示tokenizer...")
    
    # 创建简单的词汇表
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
    }
    
    # 添加一些常见字符和词
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}"
    for i, char in enumerate(chars):
        vocab[char] = len(vocab)
    
    # 添加一些常见词
    common_words = ["the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                   "have", "has", "had", "do", "does", "did", "will", "would", "could",
                   "should", "may", "might", "can", "this", "that", "these", "those",
                   "I", "you", "he", "she", "it", "we", "they", "what", "when", "where",
                   "why", "how", "and", "or", "but", "not", "in", "on", "at", "to", "for",
                   "of", "with", "from", "by", "about", "into", "through", "during",
                   "artificial", "intelligence", "machine", "learning", "neural", "network",
                   "future", "technology", "computer", "science", "data", "model", "training"]
    
    for word in common_words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    # 创建tokenizer对象
    tokenizer = BPETokenizer()
    tokenizer.vocab = vocab
    tokenizer.inverse_vocab = {idx: token for token, idx in vocab.items()}
    tokenizer.merges = []
    tokenizer.splits = {}
    
    print(f"演示tokenizer词汇表大小: {len(vocab)}")
    return tokenizer


def main():
    print("=" * 60)
    print("LLM From Scratch - 演示脚本")
    print("=" * 60)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 创建演示tokenizer
    tokenizer = create_demo_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    
    # 创建小模型
    print("\n创建模型...")
    model_config = {
        'vocab_size': vocab_size,
        'd_model': 256,
        'num_layers': 4,
        'num_heads': 4,
        'd_ff': 1024,
        'max_seq_len': 128,
        'dropout': 0.1,
        'use_rope': False
    }
    
    model = GPTModel.from_config(model_config)
    model = model.to(device)
    model.eval()
    
    num_params = model.get_num_params()
    print(f"模型参数量: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # 测试编码/解码
    print("\n" + "=" * 60)
    print("测试Tokenizer编码/解码")
    print("=" * 60)
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    
    # 生成文本示例
    print("\n" + "=" * 60)
    print("文本生成示例（使用随机初始化的模型）")
    print("=" * 60)
    print("注意：由于模型未训练，生成结果可能不连贯")
    print("训练后的模型会生成更高质量的文本\n")
    
    prompts = [
        "The future of artificial intelligence",
        "Machine learning is",
        "In the world of technology"
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n示例 {i}:")
        print(f"输入: {prompt}")
        print("-" * 60)
        
        try:
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=50,
                temperature=0.8,
                top_k=10,
                top_p=0.9,
                device=device
            )
            print(f"输出: {generated}")
        except Exception as e:
            print(f"生成出错: {e}")
        
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n要获得更好的生成效果，请：")
    print("1. 准备训练数据")
    print("2. 训练tokenizer: python tokenizer/train_tokenizer.py")
    print("3. 训练模型: python train/train.py --config configs/gpt_small.json")
    print("4. 使用训练后的模型进行推理")


if __name__ == '__main__':
    main()
