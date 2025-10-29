import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import TinyLLM, KVCache
from config import ModelConfig

def generate_text(
    model: TinyLLM,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    device: torch.device = torch.device("cpu"),
    use_cache: bool = False  # 暂时禁用缓存
) -> str:
    """
    使用模型生成文本
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        prompt: 输入提示文本
        max_length: 生成的最大长度
        temperature: 温度参数，控制随机性
        top_k: Top-K采样参数
        top_p: Top-P采样参数
        device: 设备
        use_cache: 是否使用KV缓存
        
    Returns:
        生成的文本
    """
    model.eval()
    
    # 初始化KV缓存
    kv_cache = KVCache() if use_cache else None
    
    # 编码输入文本
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出
            outputs = model(input_ids, kv_cache=kv_cache, use_cache=use_cache)
            next_token_logits = outputs[:, -1, :] / temperature
            # 确保logits维度正确
            if next_token_logits.dim() == 1:
                next_token_logits = next_token_logits.unsqueeze(0)
            
            # 应用Top-K和Top-P过滤
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # 采样下一个词
            probabilities = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            
            # 添加到输入序列
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # 如果生成了结束符，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    根据Top-K和Top-P参数过滤logits
    """
    # 确保logits是二维的
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # Top-K过滤
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # 安全检查
        # 移除小于第top_k个元素的所有元素
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    # Top-P过滤 (Nucleus filtering)
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过top_p的词
        sorted_indices_to_remove = cumulative_probs > top_p
        # 至少保留第一个元素（最有可能的词）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 将索引移回原始顺序
        # 修复索引错误
        batch_size, vocab_size = logits.shape
        for i in range(batch_size):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        
    return logits

def load_model(checkpoint_path: str, config: ModelConfig, device: torch.device) -> TinyLLM:
    """
    加载模型检查点
    
    Args:
        checkpoint_path: 检查点路径
        config: 模型配置
        device: 设备
        
    Returns:
        加载的模型
    """
    model = TinyLLM(config)
    model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def interactive_generation(model, tokenizer, device):
    """
    交互式文本生成
    """
    print("=== TinyLLM 文本生成 ===")
    print("输入提示文本开始生成，输入 'quit' 退出")
    
    while True:
        prompt = input("\n请输入提示文本: ")
        if prompt.lower() == 'quit':
            break
            
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_length=50,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                device=device
            )
            
            print(f"\n生成结果:\n{generated_text}")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")

def main():
    """主推理函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载配置
    config = ModelConfig()
    
    # 初始化模型
    model = TinyLLM(config)
    model.to(device)
    print("Model initialized")
    
    # 这里应该加载训练好的模型权重
    # 由于我们还没有训练模型，所以使用随机初始化的权重
    
    # 创建一个简单的tokenizer用于演示
    class SimpleTokenizer:
        def __init__(self):
            self.eos_token_id = 2
            
        def encode(self, text, return_tensors="pt"):
            # 简单的编码：将字符转换为数字索引
            # 在实际应用中应该使用真实的tokenizer
            import random
            # 确保token ID在词汇表范围内
            tokens = [random.randint(0, config.vocab_size-2) for _ in range(10)]
            return torch.tensor([tokens])
            
        def decode(self, tokens, skip_special_tokens=False):
            # 简单的解码
            return "这是一个生成的示例文本。"
    
    tokenizer = SimpleTokenizer()
    
    # 示例生成
    prompt = "人工智能的发展"
    print(f"Prompt: {prompt}")
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_length=20,
        temperature=0.8,
        device=device
    )
    
    print(f"Generated text: {generated_text}")
    
    # 启动交互式生成
    # interactive_generation(model, tokenizer, device)

if __name__ == "__main__":
    main()