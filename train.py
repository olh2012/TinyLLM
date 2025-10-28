import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import json
from typing import Optional
from model import TinyLLM
from config import ModelConfig
from data_utils import load_sample_data, create_dataloader

def compute_loss(model, batch, device):
    """计算损失"""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # 前向传播
    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 计算语言模型损失
    # 目标是下一个词的预测
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    save_path: str = "checkpoints"
):
    """训练模型"""
    model.train()
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            # 清零梯度
            optimizer.zero_grad()
            
            # 计算损失
            loss = compute_loss(model, batch, device)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 记录统计信息
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

def setup_optimizer(model: nn.Module, config: ModelConfig) -> torch.optim.Optimizer:
    """设置优化器"""
    # 分离权重衰减参数
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    return optimizer

def main():
    """主训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载配置
    config = ModelConfig()
    
    # 初始化模型
    model = TinyLLM(config)
    model.to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # 设置优化器
    optimizer = setup_optimizer(model, config)
    
    # 加载示例数据
    texts = load_sample_data()
    print(f"Loaded {len(texts)} sample texts")
    
    # 创建数据加载器
    # 使用简单的tokenizer作为示例
    class SimpleTokenizer:
        def __init__(self):
            self.pad_token_id = 0
            
        def __call__(self, texts, **kwargs):
            # 简单的标记化：将字符转换为数字索引
            # 在实际应用中应该使用真实的tokenizer
            return {"input_ids": torch.randint(0, config.vocab_size, (len(texts), config.block_size))}
    
    tokenizer = SimpleTokenizer()
    train_dataloader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_length=config.block_size
    )
    
    # 开始训练
    print("Starting training...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=config.num_epochs,
        device=device
    )
    
    # 保存最终模型
    final_model_path = "tinyllm_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()