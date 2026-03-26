# Gated Manifold Flow for LLM Unlearning

## 概述

Gated Manifold Flow (GMF) 是一种新颖的LLM遗忘方法，将遗忘问题建模为激活空间的流形变换问题。

### 核心创新

1. **流形视角**: 将遗忘知识建模为激活空间中的流形，通过流形变换实现遗忘
2. **门控机制**: 基于距离的软门控实现局部有界变换，保护保留知识
3. **障碍函数**: 控制论启发的屏障函数，确保变换不泄漏到保留区域

## 数学框架

### 1. 流形定义

- **遗忘知识流形**: $\mathcal{M}_f = \{a \in \mathbb{R}^d : a = g_f(z), z \in \mathcal{Z}_f\}$
- **吸引子流形**: $\mathcal{M}_a = \{a \in \mathbb{R}^d : a = g_a(z), z \in \mathcal{Z}_a\}$

### 2. 门控函数

$$\alpha(x) = \exp\left(-\frac{d(x, \mu_f)^2}{2\sigma^2}\right)$$

### 3. 门控变换

$$x_{new} = x + \alpha(x) \cdot \left(\text{Flow}_\theta(x) - x\right)$$

### 4. 损失函数

$$\mathcal{L} = \lambda_{att} \mathcal{L}_{attractor} + \lambda_{ret} \mathcal{L}_{retain} - \lambda_{flow} \mathcal{L}_{flow} + \lambda_{rec} \mathcal{L}_{recoverability}$$

## 代码结构

```
LUNAR/
├── gmf/                           # GMF核心模块
│   ├── __init__.py               # 模块初始化
│   ├── manifold.py               # 流形提取模块
│   ├── gating.py                 # 门控机制模块
│   ├── flow_transform.py         # Flow变换模块
│   ├── losses.py                 # 损失函数模块
│   ├── trainer.py                # 训练器模块
│   └── evaluation.py             # 评估模块
├── run_gmf.py                     # 主运行脚本
├── config/
│   └── forget_gmf_tofu.yaml      # GMF配置文件
└── run_gmf_tofu.sh               # 运行脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers omegaconf hydra-core tqdm
```

### 2. 运行GMF遗忘

```bash
cd LUNAR
bash run_gmf_tofu.sh
```

或者直接使用Python:

```bash
python run_gmf.py --config-name forget_gmf_tofu
```

### 3. 评估结果

```bash
python -m gmf.evaluation --results_dir outputs/gmf_tofu
```

## 配置参数

### GMF特定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sigma` | 1.0 | 门控带宽参数 |
| `learnable_sigma` | False | 是否学习sigma |
| `distance_method` | 'mahalanobis' | 距离度量方法 |
| `flow_hidden_dim` | 512 | Flow变换隐藏层维度 |
| `flow_num_layers` | 3 | Flow MLP层数 |
| `lambda_attractor` | 1.0 | Attractor损失权重 |
| `lambda_retain` | 1.0 | Retain损失权重 |
| `lambda_flow` | 0.1 | Flow正则化权重 |
| `lambda_recoverability` | 0.1 | 恢复性损失权重 |

## 与LUNAR对比

| 特性 | GMF | LUNAR |
|------|-----|-------|
| 理论框架 | 流形变换 | 激活导向 |
| 局部性 | 软门控（渐进式） | 硬边界 |
| 保留知识保护 | 门控机制 | 局部性约束 |
| 可逆性 | 支持（Flow正则化） | 不支持 |
| 可解释性 | 高（门控值可视化） | 中等 |

## 实验设置

### 数据集
- TOFU (Targeted Unlearning with Fact Unlearning)
- Forget: author_1
- Retain: 其余作者数据

### 模型
- LLaMA-2-7B-Chat (TOFU fine-tuned)

### 评估指标
- ROUGE Recall (forget: 越低越好)
- ROUGE Recall (retain: 越高越好)
- Perplexity
- MRR (Mean Reciprocal Rank)
- Hit Rate

## 引用

如果您使用此代码，请引用：

```bibtex
@article{gmf_unlearning_2025,
  title={Gated Manifold Flow for LLM Unlearning},
  author={Your Name},
  year={2025}
}
```

## 致谢

本研究基于LUNAR项目开发，感谢Meta AI团队的开源贡献。