# TinyLLM - 微缩版DeepSeek

这是一个简化版的大语言模型实现，参考了DeepSeek的核心架构。

## 项目结构

- `model.py`: 模型定义
- `train.py`: 训练脚本
- `inference.py`: 推理脚本
- `data_utils.py`: 数据处理工具
- `config.py`: 模型配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型:
```bash
python train.py
```

2. 运行推理:
```bash
python inference.py
```