"""
LoRA (Low-Rank Adaptation) 实现
用于参数高效微调
参考: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA线性层
    在原始权重W的基础上添加低秩分解: W + BA
    其中B是r×d的矩阵，A是d×r的矩阵，r << d
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 冻结原始权重
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # LoRA参数
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, ..., in_features]
        Returns:
            output: [batch_size, ..., out_features]
        """
        # 原始输出
        output = self.linear(x)
        
        # LoRA输出
        x_dropout = self.lora_dropout(x)
        lora_output = F.linear(x_dropout, self.lora_A.t()) @ self.lora_B.t()
        lora_output = lora_output * self.scaling
        
        return output + lora_output


def apply_lora_to_model(model, target_modules=None, rank=8, alpha=16.0, dropout=0.0):
    """
    将LoRA应用到模型的指定模块
    
    Args:
        model: GPT模型
        target_modules: 要应用LoRA的模块名称列表，如 ['attention', 'mlp']
                        如果为None，则应用到所有Linear层
        rank: LoRA的秩
        alpha: LoRA的缩放因子
        dropout: LoRA的dropout率
    """
    if target_modules is None:
        target_modules = ['attention', 'mlp']
    
    replaced_modules = []
    
    for name, module in model.named_modules():
        # 检查是否是目标模块
        is_target = any(target in name for target in target_modules)
        
        if is_target and isinstance(module, nn.Linear):
            # 替换为LoRA层
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            parent_module = model
            for part in parent_name.split('.'):
                if part:
                    parent_module = getattr(parent_module, part)
            
            lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent_module, child_name, lora_module)
            replaced_modules.append(name)
    
    print(f"已将LoRA应用到 {len(replaced_modules)} 个模块:")
    for name in replaced_modules:
        print(f"  - {name}")
    
    return model


def get_lora_parameters(model):
    """获取所有LoRA参数（用于优化器）"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_params.append(param)
    return lora_params


def save_lora_weights(model, save_path):
    """保存LoRA权重"""
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param.cpu()
    
    torch.save(lora_state_dict, save_path)
    print(f"LoRA权重已保存到: {save_path}")


def load_lora_weights(model, lora_path, device='cuda'):
    """加载LoRA权重"""
    lora_state_dict = torch.load(lora_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"LoRA权重已从 {lora_path} 加载")
