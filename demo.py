import torch
import torch.nn as nn
from model import TinyLLM
from config import ModelConfig

def demo_train():
    """演示训练过程"""
    print("=== TinyLLM 演示 ===")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置
    config = ModelConfig()
    
    # 为了演示目的，减小模型尺寸
    config.vocab_size = 1000  # 减小词汇表大小
    config.hidden_size = 256   # 减小隐藏层维度
    config.num_hidden_layers = 4  # 减少层数
    config.num_attention_heads = 4  # 减少注意力头数
    config.num_kv_heads = 2  # 减少KV头数以实现GQA
    config.intermediate_size = 512  # 减小前馈网络中间维度
    
    # 重新计算MLA参数以匹配新的hidden_size和num_attention_heads
    config.latent_dim = config.hidden_size // config.num_attention_heads // 2
    config.num_latent_heads = config.num_attention_heads
    
    # 初始化模型
    model = TinyLLM(config)
    model.to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建示例数据
    batch_size = 2
    seq_length = 32
    # 确保token ID在词汇表范围内
    input_ids = torch.randint(0, config.vocab_size-1, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long).to(device)
    
    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    # 训练几步
    model.train()
    print("开始演示训练...")
    
    for step in range(5):  # 只训练5步用于演示
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 计算损失（语言模型损失）
        shift_logits = outputs[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"步骤 {step+1}/5 - 损失: {loss.item():.4f}")
    
    print("演示训练完成!")
    
    # 保存模型
    torch.save(model.state_dict(), "demo_model.pt")
    print("模型已保存到 demo_model.pt")
    
    # 演示推理
    print("\n=== 演示推理 ===")
    model.eval()
    
    # 创建推理输入
    input_ids = torch.randint(0, config.vocab_size-1, (1, 10)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        print(f"输入形状: {input_ids.shape}")
        print(f"输出形状: {outputs.shape}")
        print(f"输出概率分布最大值: {torch.max(outputs).item():.4f}")
        print(f"输出概率分布最小值: {torch.min(outputs).item():.4f}")

if __name__ == "__main__":
    demo_train()