"""
BPE (Byte Pair Encoding) Tokenizer Implementation
基于公开的BPE算法实现，参考GPT系列论文
"""

import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import os


class BPETokenizer:
    """BPE Tokenizer实现"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.word_freqs = {}
        self.splits = {}
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 特殊token
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        
    def _get_word_freqs(self, texts: List[str]) -> Dict[str, int]:
        """统计词频"""
        word_freqs = defaultdict(int)
        for text in texts:
            # 简单分词：按空格和标点分割
            words = re.findall(r'\S+', text)
            for word in words:
                word_freqs[word] += 1
        return dict(word_freqs)
    
    def _get_splits(self, word: str) -> List[str]:
        """将词分割成字符序列，末尾添加</w>"""
        return list(word) + ['</w>']
    
    def _get_stats(self, splits: Dict[str, List[str]]) -> Dict[Tuple[str, str], int]:
        """统计相邻字符对的出现频率"""
        pairs = defaultdict(int)
        for word, word_splits in splits.items():
            for i in range(len(word_splits) - 1):
                pair = (word_splits[i], word_splits[i + 1])
                pairs[pair] += self.word_freqs[word]
        return dict(pairs)
    
    def _merge_vocab(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """合并字符对"""
        bigram = ''.join(pair)
        new_splits = {}
        for word in splits:
            new_word = []
            i = 0
            word_splits = splits[word]
            while i < len(word_splits):
                if (i < len(word_splits) - 1 and 
                    word_splits[i] == pair[0] and 
                    word_splits[i + 1] == pair[1]):
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word_splits[i])
                    i += 1
            new_splits[word] = new_word
        return new_splits
    
    def train(self, texts: List[str], verbose: bool = False):
        """训练BPE tokenizer"""
        print(f"开始训练BPE tokenizer，目标词汇表大小: {self.vocab_size}")
        
        # 1. 统计词频
        self.word_freqs = self._get_word_freqs(texts)
        print(f"统计到 {len(self.word_freqs)} 个唯一词")
        
        # 2. 初始化splits：每个词分割成字符
        self.splits = {word: self._get_splits(word) for word in self.word_freqs.keys()}
        
        # 3. 初始化词汇表：所有字符
        vocab = set()
        for word_splits in self.splits.values():
            vocab.update(word_splits)
        vocab = sorted(list(vocab))
        
        # 添加特殊token
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab = special_tokens + vocab
        
        # 4. 迭代合并
        num_merges = self.vocab_size - len(vocab)
        if num_merges <= 0:
            print("警告：目标词汇表大小小于初始字符数")
            num_merges = 1000
        
        self.merges = []
        for i in range(num_merges):
            pairs = self._get_stats(self.splits)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            self.splits = self._merge_vocab(best_pair, self.splits)
            self.merges.append(best_pair)
            
            # 添加新token到词汇表
            new_token = ''.join(best_pair)
            vocab.append(new_token)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"合并 {i + 1}/{num_merges}: {best_pair} -> {new_token}")
        
        # 5. 构建最终词汇表
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        print(f"训练完成，词汇表大小: {len(self.vocab)}")
        print(f"合并次数: {len(self.merges)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """对单个词应用BPE"""
        if word in self.splits:
            return self.splits[word]
        
        # 如果词不在训练数据中，先分割成字符
        splits = self._get_splits(word)
        
        # 应用所有合并规则
        for pair in self.merges:
            new_splits = []
            i = 0
            while i < len(splits):
                if (i < len(splits) - 1 and 
                    splits[i] == pair[0] and 
                    splits[i + 1] == pair[1]):
                    new_splits.append(''.join(pair))
                    i += 2
                else:
                    new_splits.append(splits[i])
                    i += 1
            splits = new_splits
        
        return splits
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为token IDs"""
        if not self.vocab:
            raise ValueError("Tokenizer未训练，请先调用train()方法")
        
        # 分词
        words = re.findall(r'\S+', text)
        
        # 对每个词应用BPE
        tokens = []
        if add_special_tokens:
            tokens.append(self.vocab[self.bos_token])
        
        for word in words:
            bpe_tokens = self._apply_bpe(word)
            for token in bpe_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab[self.unk_token])
        
        if add_special_tokens:
            tokens.append(self.vocab[self.eos_token])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """解码token IDs为文本"""
        if not self.inverse_vocab:
            raise ValueError("Tokenizer未训练，请先调用train()方法")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                token = self.inverse_vocab[token_id]
                # 跳过特殊token
                if token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        # 合并tokens并移除</w>
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = text.strip()
        
        return text
    
    def save(self, vocab_path: str, merges_path: str):
        """保存tokenizer"""
        # 保存词汇表
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # 保存合并规则
        with open(merges_path, 'w', encoding='utf-8') as f:
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        
        print(f"Tokenizer已保存到 {vocab_path} 和 {merges_path}")
    
    def load(self, vocab_path: str, merges_path: str):
        """加载tokenizer"""
        # 加载词汇表
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        # 加载合并规则
        self.merges = []
        if os.path.exists(merges_path):
            with open(merges_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            self.merges.append((parts[0], parts[1]))
        
        # 重建splits（用于编码）
        self.splits = {}
        
        print(f"Tokenizer已从 {vocab_path} 和 {merges_path} 加载")
    
    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return len(self.vocab) if self.vocab else 0
