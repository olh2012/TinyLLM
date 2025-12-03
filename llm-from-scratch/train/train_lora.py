"""
LoRA微调脚本
使用LoRA进行参数高效微调
"""

import argparse
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from data.dataset import TextDataset, JSONLDataset, HuggingFaceDataset
from model.transformer import GPTModel
from train.lora import apply_lora_to_model, get_lora_parameters, save_lora_weights
from train.train import load_config, create_dataloader, save_checkpoint, load_checkpoint


def train_epoch_lora(model, dataloader, optimizer, device, config, epoch, writer, global_step, scaler=None, use_amp=False, amp_dtype='fp16'):
    """使用LoRA训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    training_config = config['training']
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    
    dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        if use_amp:
            with autocast(dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss'] / gradient_accumulation_steps
            loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if max_grad_norm > 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
            else:
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1
        global_step += 1
        
        if global_step % training_config.get('log_interval', 100) == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}")
            writer.add_scalar('train/loss', avg_loss, global_step)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def main():
    parser = argparse.ArgumentParser(description='LoRA微调GPT模型')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--base_model', type=str, required=True,
                        help='基础模型checkpoint路径')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA秩')
    parser.add_argument('--lora_alpha', type=float, default=16.0,
                        help='LoRA alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.0,
                        help='LoRA dropout率')
    parser.add_argument('--target_modules', type=str, nargs='+', default=['attention', 'mlp'],
                        help='要应用LoRA的模块')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = config['output'].get('lora_output_dir', './lora_output')
    os.makedirs(output_dir, exist_ok=True)
    log_dir = config['output'].get('log_dir', os.path.join(output_dir, 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(
        config['tokenizer']['vocab_path'],
        config['tokenizer']['merges_path']
    )
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    print("创建模型...")
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size
    model = GPTModel.from_config(model_config)
    
    # 加载基础模型
    print(f"加载基础模型: {args.base_model}")
    checkpoint = torch.load(args.base_model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 应用LoRA
    print(f"应用LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, target_modules={args.target_modules}")
    model = apply_lora_to_model(
        model,
        target_modules=args.target_modules,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    
    # 计算可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_loader = create_dataloader(config, tokenizer, split='train')
    val_loader = None
    if config['data'].get('val_path') or config['data'].get('type') == 'huggingface':
        try:
            val_loader = create_dataloader(config, tokenizer, split='val')
        except:
            print("警告：无法创建验证集，跳过验证")
    
    # 优化器（只优化LoRA参数）
    training_config = config['training']
    lora_params = get_lora_parameters(model)
    
    if training_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=training_config.get('lora_learning_rate', training_config['learning_rate']),
            betas=training_config.get('betas', (0.9, 0.999)),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    else:
        optimizer = torch.optim.Adam(
            lora_params,
            lr=training_config.get('lora_learning_rate', training_config['learning_rate']),
            betas=training_config.get('betas', (0.9, 0.999)),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    
    # 混合精度训练
    use_amp = training_config.get('use_amp', False)
    amp_dtype = training_config.get('amp_dtype', 'fp16')
    scaler = None
    
    if use_amp:
        if amp_dtype == 'bf16':
            use_amp = torch.cuda.is_bf16_supported()
            if not use_amp:
                print("警告: 当前设备不支持BF16，将使用FP32训练")
                use_amp = False
        else:
            scaler = GradScaler()
        if use_amp:
            print(f"启用混合精度训练: {amp_dtype.upper()}")
    
    # 训练循环
    num_epochs = training_config['num_epochs']
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"\n开始LoRA微调，共 {num_epochs} 个epoch...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        train_loss, global_step = train_epoch_lora(
            model, train_loader, optimizer, device, config, epoch, writer, global_step,
            scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype
        )
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        if val_loader:
            model.eval()
            total_loss = 0.0
            num_batches = 0
            
            dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)
                    
                    if use_amp:
                        with autocast(dtype=dtype):
                            outputs = model(input_ids=input_ids, labels=labels)
                            loss = outputs['loss']
                    else:
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs['loss']
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            val_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"验证损失: {val_loss:.4f}")
            writer.add_scalar('val/loss', val_loss, epoch)
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False
        
        # 保存LoRA权重
        lora_path = os.path.join(output_dir, f'lora_epoch_{epoch}.pt')
        save_lora_weights(model, lora_path)
        
        if is_best:
            best_lora_path = os.path.join(output_dir, 'lora_best.pt')
            save_lora_weights(model, best_lora_path)
    
    print("\nLoRA微调完成！")
    writer.close()


if __name__ == '__main__':
    main()
