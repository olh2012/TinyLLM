"""
推理服务化API
基于FastAPI实现
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_tokenizer import BPETokenizer
from model.transformer import GPTModel
from inference.generate import generate_text

app = FastAPI(title="LLM Inference API", version="1.0.0")

# 全局变量存储模型和tokenizer
model = None
tokenizer = None
device = None


class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.0


class GenerateResponse(BaseModel):
    generated_text: str
    prompt: str
    parameters: dict


@app.on_event("startup")
async def load_model():
    """启动时加载模型"""
    global model, tokenizer, device
    
    # 从环境变量或配置文件读取路径
    import os
    checkpoint_path = os.getenv("CHECKPOINT_PATH", "./checkpoints/checkpoint_best.pt")
    config_path = os.getenv("CONFIG_PATH", "./configs/gpt_small.json")
    vocab_path = os.getenv("VOCAB_PATH", "./tokenizer_output/vocab.json")
    merges_path = os.getenv("MERGES_PATH", "./tokenizer_output/merges.txt")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(vocab_path, merges_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
    # 创建模型
    print("创建模型...")
    model_config = config['model'].copy()
    model_config['vocab_size'] = vocab_size
    model = GPTModel.from_config(model_config)
    
    # 加载checkpoint
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"模型参数量: {model.get_num_params():,}")
    print("模型加载完成！")


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "LLM Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "POST - 生成文本",
            "/health": "GET - 健康检查"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else None
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """生成文本"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k if request.top_k > 0 else 0,
            top_p=request.top_p if request.top_p > 0.0 else 0.0,
            device=device
        )
        
        return GenerateResponse(
            generated_text=generated_text,
            prompt=request.prompt,
            parameters={
                "max_length": request.max_length,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {str(e)}")


@app.post("/generate/batch")
async def generate_batch(requests: List[GenerateRequest]):
    """批量生成文本"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    results = []
    for request in requests:
        try:
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k if request.top_k > 0 else 0,
                top_p=request.top_p if request.top_p > 0.0 else 0.0,
                device=device
            )
            results.append({
                "generated_text": generated_text,
                "prompt": request.prompt,
                "parameters": {
                    "max_length": request.max_length,
                    "temperature": request.temperature,
                    "top_k": request.top_k,
                    "top_p": request.top_p
                }
            })
        except Exception as e:
            results.append({
                "error": str(e),
                "prompt": request.prompt
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
