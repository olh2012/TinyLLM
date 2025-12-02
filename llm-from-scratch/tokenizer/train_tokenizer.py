"""
训练BPE Tokenizer的脚本
"""

import argparse
import os
from pathlib import Path
from tokenizer.bpe_tokenizer import BPETokenizer


def load_texts_from_file(file_path: str, max_lines: int = None) -> list:
    """从文件加载文本"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def load_texts_from_jsonl(file_path: str, text_key: str = 'text', max_lines: int = None) -> list:
    """从JSONL文件加载文本"""
    import json
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            try:
                data = json.loads(line)
                if text_key in data:
                    texts.append(data[text_key])
            except:
                continue
    return texts


def main():
    parser = argparse.ArgumentParser(description='训练BPE Tokenizer')
    parser.add_argument('--data_path', type=str, required=True,
                        help='训练数据路径（txt或jsonl文件）')
    parser.add_argument('--output_dir', type=str, default='./tokenizer_output',
                        help='输出目录')
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='词汇表大小')
    parser.add_argument('--data_type', type=str, choices=['txt', 'jsonl'], default='txt',
                        help='数据类型')
    parser.add_argument('--text_key', type=str, default='text',
                        help='JSONL文件中的文本字段名')
    parser.add_argument('--max_lines', type=int, default=None,
                        help='最大加载行数（用于测试）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"从 {args.data_path} 加载数据...")
    if args.data_type == 'txt':
        texts = load_texts_from_file(args.data_path, args.max_lines)
    else:
        texts = load_texts_from_jsonl(args.data_path, args.text_key, args.max_lines)
    
    print(f"加载了 {len(texts)} 条文本")
    
    # 训练tokenizer
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(texts, verbose=True)
    
    # 保存tokenizer
    vocab_path = os.path.join(args.output_dir, 'vocab.json')
    merges_path = os.path.join(args.output_dir, 'merges.txt')
    tokenizer.save(vocab_path, merges_path)
    
    # 测试编码/解码
    print("\n测试编码/解码:")
    test_text = "Hello world! This is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"原文: {test_text}")
    print(f"编码: {encoded[:10]}... (共{len(encoded)}个tokens)")
    print(f"解码: {decoded}")


if __name__ == '__main__':
    main()
