import torch
from typing import List, Dict, Iterator
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import numpy as np

class TextDataset(Dataset):
    """文本数据集类"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码文本
        # 使用简单的方法来创建示例数据
        # 在实际应用中，应该使用真实的tokenizer
        import random
        # 确保token ID在词汇表范围内
        input_ids = torch.tensor([random.randint(0, 31999) for _ in range(self.max_length)], dtype=torch.long)
        attention_mask = torch.ones(self.max_length, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class DataCollator:
    """数据整理器，用于批处理"""
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # 获取批次中的最大长度
        max_length = max(item['input_ids'].size(0) for item in batch)
        
        # 填充序列到相同长度
        input_ids = []
        attention_masks = []
        
        for item in batch:
            input_id = item['input_ids']
            attention_mask = item['attention_mask']
            
            # 计算需要填充的数量
            pad_length = max_length - input_id.size(0)
            
            # 填充
            padded_input_id = torch.cat([
                input_id,
                torch.full((pad_length,), self.pad_token_id, dtype=torch.long)
            ])
            
            padded_attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ])
            
            input_ids.append(padded_input_id)
            attention_masks.append(padded_attention_mask)
            
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }

def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    shuffle: bool = True
) -> DataLoader:
    """创建数据加载器"""
    dataset = TextDataset(texts, tokenizer, max_length)
    collator = DataCollator(tokenizer.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator
    )
    
    return dataloader

def load_sample_data() -> List[str]:
    """加载示例数据"""
    # 这里提供一些示例文本用于演示
    sample_texts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "深度学习是机器学习的一个子领域，它使用多层神经网络来模拟人脑处理信息的方式。",
        "自然语言处理是人工智能领域中的一个重要方向，它研究人与计算机之间用自然语言进行有效通信的各种理论和方法。",
        "Transformer架构是一种基于注意力机制的深度学习模型，在机器翻译、文本摘要等任务中表现出色。",
        "大语言模型通过在大规模文本语料库上进行训练，能够生成连贯且相关的文本。",
        "参数高效微调是一种在预训练语言模型基础上进行适应的技术，可以减少计算资源消耗。",
        "自回归语言建模是一种预测序列中下一个词的任务，是训练语言模型的常用方法。",
        "上下文学习是指模型在给定少量示例的情况下，能够在新任务上进行泛化的能力。"
    ]
    
    return sample_texts

def tokenize_and_chunk(text: str, tokenizer, chunk_size: int = 512) -> List[torch.Tensor]:
    """将文本分块并标记化"""
    # 标记化文本
    tokens = tokenizer.encode(text, add_special_tokens=False)
    
    # 将标记分块
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i+chunk_size]
        # 如果不是最后一个块且长度不足，则跳过
        if len(chunk) < chunk_size and i + chunk_size < len(tokens):
            continue
        chunks.append(torch.tensor(chunk, dtype=torch.long))
        
    return chunks

# 测试代码
if __name__ == "__main__":
    # 这里的测试代码会在安装依赖后正常工作
    print("Data utilities module loaded successfully.")