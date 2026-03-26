# LUNAR 论文 TOFU 数据集复现指南

## 📋 概述

本指南帮助您复现 LUNAR 论文在 TOFU 数据集上使用 Llama2-7b 的实验结果。

## 🔧 环境准备

### 1. 激活环境
```bash
conda activate lunar
# 或者如果没有lunar环境，使用pistol环境
conda activate pistol
```

### 2. 确认依赖
```bash
cd /home/users/yanzeyu/LUNAR
pip list | grep -E "torch|transformers|accelerate|peft"
```

## 📥 数据和模型下载

### 方法一：在有网络的机器上下载后传输

#### TOFU 数据集下载
TOFU 数据集可以从 Hugging Face 下载：
- 数据集地址：`locuslab/TOFU`

```python
# 在有网络的机器上运行
from datasets import load_dataset
dataset = load_dataset("locuslab/TOFU")
dataset.save_to_disk("./tofu_dataset")
```

#### TOFU 预微调模型下载
LUNAR 论文使用的 TOFU 预微调 Llama2-7b 模型：
- 模型地址：`locuslab/tofu_llama2_7b_full_finetune` 或类似

```python
# 在有网络的机器上运行
from huggingface_hub import snapshot_download
snapshot_download("locuslab/tofu_llama2_7b_full_finetune", local_dir="./tofu_llama2_7b")
```

### 方法二：使用 modelscope 镜像（如果可用）

```bash
# 使用 modelscope 下载
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('LLM-Research/Meta-Llama-2-7b-chat', cache_dir='./models')"
```

## 📁 目录结构

复现所需的目录结构：

```
LUNAR/
├── dataset/
│   └── unlearning/
│       ├── tofu_full.json          # TOFU 数据集（需要转换格式）
│       └── factual_data.json        # 事实数据（保留集评估）
├── models_finetune/
│   └── tofu/
│       └── llama2-7b-chat/          # TOFU 预微调模型
└── config/
    └── forget_tofu.yaml             # TOFU 配置文件
```

## 🔄 TOFU 数据格式转换

TOFU 数据集需要转换为 LUNAR 兼容格式。原始 TOFU 格式：
```json
{
    "question": "...",
    "answer": "...",
    "author": "author_1"  // 遗忘目标标识
}
```

转换后的格式（与 pistol_sample1.json 兼容）：
```json
{
    "question": "...",
    "answer": "...",
    "edge": "author_1"  // 使用 edge 字段替代 author
}
```

## ⚙️ LUNAR 论文 TOFU 实验设置

根据 LUNAR 论文，TOFU 实验的关键参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| model_family | llama2-7b-chat | 模型类型 |
| layer_modified | [22] | 修改的层 |
| coeff_list | [+2.0] | 方向系数 |
| num_epochs | 20 | 训练轮数 |
| lr | 0.01 | 学习率 |
| positions | -1 | 激活位置 |
| forget_edge | ['author_1'] | 遗忘目标（TOFU 中为作者） |

## 📊 评估指标

LUNAR 论文在 TOFU 上报告的指标：

1. **Forget Quality (遗忘质量)**
   - ROUGE-1 Recall (遗忘集)
   - ROUGE-L Recall (遗忘集)
   - 应该显著降低

2. **Model Utility (模型效用)**
   - ROUGE-1 Recall (保留集)
   - ROUGE-L Recall (保留集)
   - 应该保持较高

3. **其他指标**
   - MRR (Mean Reciprocal Rank)
   - Hit Rate
   - Perplexity

## 🚀 运行实验

### 1. 使用 TOFU 配置运行
```bash
cd /home/users/yanzeyu/LUNAR
python run_lunar.py --config-name forget_tofu.yaml
```

### 2. 命令行覆盖参数
```bash
python run_lunar.py \
    model_family=llama2-7b-chat \
    model_path=/path/to/tofu_finetuned_llama2 \
    data_name=tofu_full \
    forget_edge=['author_1'] \
    layer_modified=[22] \
    coeff_list=[+2.0] \
    num_epochs=20 \
    lr=0.01
```

## 📈 预期结果

根据 LUNAR 论文，在 TOFU 数据集上：

| 方法 | Forget ROUGE-L ↓ | Retain ROUGE-L ↑ | F1 Score |
|------|------------------|------------------|----------|
| Original (无遗忘) | ~0.85 | ~0.85 | - |
| LUNAR | ~0.25 | ~0.75 | 最佳 |

## ⚠️ 注意事项

1. **GPU 内存**：Llama2-7b 需要约 16GB GPU 内存（使用 bfloat16）
2. **数据对齐**：确保 TOFU 数据的 edge 字段正确设置
3. **模型路径**：确保 model_path 指向正确的微调后模型

## 📝 手动下载链接

如果需要手动下载，请访问以下链接：

1. **TOFU 数据集**：https://huggingface.co/datasets/locuslab/TOFU
2. **TOFU 预微调模型**：https://huggingface.co/locuslab/tofu_llama2_7b_full_finetune
3. **原始 Llama2-7b-chat**：您已有 (`/home/users/yanzeyu/.cache/modelscope/hub/models/modelscope/Llama-2-7b-chat-ms`)

## 🔗 参考资源

- LUNAR 论文：https://arxiv.org/abs/2402.15156
- TOFU 论文：https://arxiv.org/abs/2401.06121
- TOFU GitHub：https://github.com/locuslab/tofu
- LUNAR GitHub：https://github.com/facebookresearch/LUNAR