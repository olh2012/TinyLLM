# TinyLLM - 微缩版DeepSeek

这是一个简化版的大语言模型实现，参考了DeepSeek的核心架构。

## 项目结构

- `model.py`: 模型定义
- `train.py`: 训练脚本
- `inference.py`: 推理脚本
- `data_utils.py`: 数据处理工具
- `config.py`: 模型配置
- `demo.py`: 演示脚本

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行演示 (推荐):
```bash
python demo.py
```

2. 训练模型:
```bash
python train.py
```

3. 运行推理:
```bash
python inference.py
```

## 模型特点

- 基于Transformer架构
- 实现了旋转位置编码(RoPE)
- 使用RMSNorm归一化
- 支持Top-K和Top-P采样
- 支持分组查询注意力(GQA)
- 支持KV缓存优化推理
- 模型大小可配置

## 注意事项

- 为了演示目的，模型参数已调小
- 实际使用时可以根据需要调整配置
- 训练需要大量计算资源