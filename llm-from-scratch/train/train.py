"""
训练脚本
支持分布式训练、gradient accumulation、checkpoint保存与恢复
"""

import argparse
import json
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from data.dataset import TextDataset, JSONLDataset, HuggingFaceDataset
from model.transformer import GPTModel


def load_config(config_path: str):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_dataloader(config, tokenizer, split='train', use_ddp=False, rank=0, world_size=1):
    """创建数据加载器"""
    data_config = config['data']
    data_type = data_config['type']
    
    if data_type == 'txt':
        dataset = TextDataset(
            file_path=data_config['train_path'] if split == 'train' else data_config.get('val_path'),
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_len'],
            max_samples=data_config.get('max_samples')
        )
    elif data_type == 'jsonl':
        dataset = JSONLDataset(
            file_path=data_config['train_path'] if split == 'train' else data_config.get('val_path'),
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_len'],
            text_key=data_config.get('text_key', 'text'),
            max_samples=data_config.get('max_samples')
        )
    elif data_type == 'huggingface':
        dataset = HuggingFaceDataset(
            dataset_name=data_config['dataset_name'],
            tokenizer=tokenizer,
            max_length=config['model']['max_seq_len'],
            split=split,
            text_key=data_config.get('text_key', 'text'),
            max_samples=data_config.get('max_samples')
        )
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 分布式训练时使用DistributedSampler
    sampler = None
    shuffle = None
    if use_ddp and split == 'train':
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True
    )
    
    return dataloader


def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir, is_best=False, scaler=None):
    """保存checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    # 保存scaler状态（如果使用混合精度）
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # 保存最新checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
    
    # 定期保存（每10个epoch）
    if epoch % 10 == 0:
        epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    return epoch, step, loss


def train_epoch(model, dataloader, optimizer, device, config, epoch, writer, global_step, scaler=None, use_amp=False, amp_dtype='fp16', rank=0):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    training_config = config['training']
    gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    
    # 设置autocast dtype
    dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast(dtype=dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
            
            # Backward pass with gradient scaling (only for FP16)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs['loss'] / gradient_accumulation_steps
            loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
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
        
        # 记录日志（只在rank 0记录）
        if global_step % training_config.get('log_interval', 100) == 0:
            avg_loss = total_loss / num_batches
            if rank == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {avg_loss:.4f}")
                if writer:
                    writer.add_scalar('train/loss', avg_loss, global_step)
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss, global_step


def validate(model, dataloader, device, use_amp=False, amp_dtype='fp16'):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    dtype = torch.bfloat16 if amp_dtype == 'bf16' else torch.float16
    
    with torch.no_grad():
        for batch in dataloader:
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
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def setup_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='训练GPT模型')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--device', type=str, default=None,
                        help='设备 (cuda/cpu)')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='本地GPU rank（用于分布式训练）')
    
    args = parser.parse_args()
    
    # 分布式训练设置
    use_ddp, rank, world_size, local_rank = setup_distributed()
    
    if use_ddp:
        print(f"分布式训练: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    elif args.local_rank >= 0:
        # 单机多卡但未使用torchrun
        print("警告: 检测到local_rank但未使用分布式训练，请使用torchrun启动")
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = config['output']['checkpoint_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_dir = config['output'].get('log_dir', os.path.join(output_dir, 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer（只在rank 0创建）
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir)
    
    # 加载tokenizer
    if rank == 0:
        print("加载tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(
        config['tokenizer']['vocab_path'],
        config['tokenizer']['merges_path']
    )
    vocab_size = tokenizer.get_vocab_size()
    if rank == 0:
        print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    if rank == 0:
        print("创建模型...")
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size
    model = GPTModel.from_config(model_config)
    model = model.to(device)
    
    # 分布式训练时包装模型
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        num_params = model.module.get_num_params() if hasattr(model, 'module') else model.get_num_params()
    else:
        num_params = model.get_num_params()
    
    if rank == 0:
        print(f"模型参数量: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # 创建数据加载器
    if rank == 0:
        print("创建数据加载器...")
    train_loader = create_dataloader(config, tokenizer, split='train', use_ddp=use_ddp, rank=rank, world_size=world_size)
    val_loader = None
    if config['data'].get('val_path') or config['data'].get('type') == 'huggingface':
        try:
            val_loader = create_dataloader(config, tokenizer, split='val', use_ddp=use_ddp, rank=rank, world_size=world_size)
        except:
            if rank == 0:
                print("警告：无法创建验证集，跳过验证")
    
    # 优化器
    training_config = config['training']
    if training_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            betas=training_config.get('betas', (0.9, 0.999)),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    elif training_config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            betas=training_config.get('betas', (0.9, 0.999)),
            weight_decay=training_config.get('weight_decay', 0.01)
        )
    else:
        raise ValueError(f"不支持的优化器: {training_config['optimizer']}")
    
    # 学习率调度器
    scheduler = None
    if 'scheduler' in training_config:
        if training_config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=training_config['num_epochs']
            )
        elif training_config['scheduler'] == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=training_config['num_epochs']
            )
    
    # 混合精度训练设置
    use_amp = training_config.get('use_amp', False)
    amp_dtype = training_config.get('amp_dtype', 'fp16')  # 'fp16' or 'bf16'
    scaler = None
    
    if use_amp:
        if amp_dtype == 'bf16':
            # BF16不需要scaler
            use_amp = torch.cuda.is_bf16_supported()
            if not use_amp:
                print("警告: 当前设备不支持BF16，将使用FP32训练")
                use_amp = False
        else:
            # FP16需要scaler
            scaler = GradScaler()
        if use_amp:
            print(f"启用混合精度训练: {amp_dtype.upper()}")
    
    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"从 {args.resume} 恢复训练...")
        start_epoch, global_step, _ = load_checkpoint(model, optimizer, args.resume, device)
        if scaler is not None:
            # 如果使用scaler，尝试从checkpoint加载
            checkpoint = torch.load(args.resume, map_location=device)
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"从epoch {start_epoch}, step {global_step} 继续训练")
    
    # 训练循环
    num_epochs = training_config['num_epochs']
    if rank == 0:
        print(f"\n开始训练，共 {num_epochs} 个epoch...")
    
    for epoch in range(start_epoch, num_epochs):
        if use_ddp:
            train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
        
        # 训练
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, config, epoch, writer, global_step,
            scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype, rank=rank
        )
        if rank == 0:
            print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        if val_loader:
            val_loss = validate(model, val_loader, device, use_amp=use_amp, amp_dtype=amp_dtype)
            if rank == 0:
                print(f"验证损失: {val_loss:.4f}")
                if writer:
                    writer.add_scalar('val/loss', val_loss, epoch)
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
        else:
            is_best = False
        
        # 保存checkpoint（只在rank 0保存）
        if rank == 0:
            # 获取模型状态（如果是DDP）
            model_to_save = model.module if use_ddp else model
            save_checkpoint(
                model_to_save, optimizer, epoch, global_step, train_loss,
                output_dir, is_best=is_best, scaler=scaler
            )
        
        # 更新学习率
        if scheduler:
            scheduler.step()
    
    if rank == 0:
        print("\n训练完成！")
        writer.close()
    
    # 清理分布式训练
    if use_ddp:
        cleanup_distributed()


if __name__ == '__main__':
    main()
