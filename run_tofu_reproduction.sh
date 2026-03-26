#!/bin/bash
# LUNAR TOFU 复现脚本
# 用于复现 LUNAR 论文在 TOFU 数据集上的实验结果

set -e  # 遇到错误立即退出

echo "=========================================="
echo "LUNAR TOFU 数据集复现脚本"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 切换到 LUNAR 目录
cd /home/users/yanzeyu/LUNAR

# 激活 conda 环境
echo ">>> 激活 conda 环境..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lunar 2>/dev/null || conda activate pistol

# 检查环境
echo ">>> 检查 Python 环境..."
python --version
pip show torch | head -3

# 步骤 1: 检查数据集
echo ""
echo ">>> 步骤 1: 检查 TOFU 数据集..."
TOFU_DATA="./dataset/unlearning/tofu_full.json"

if [ ! -f "$TOFU_DATA" ]; then
    echo "警告: TOFU 数据集不存在: $TOFU_DATA"
    echo "请先下载数据集，可以使用以下方法："
    echo ""
    echo "方法 1: 创建示例数据（仅用于测试流程）"
    echo "  python scripts/prepare_tofu.py --create-sample"
    echo ""
    echo "方法 2: 从 Hugging Face 下载（需要网络）"
    echo "  python scripts/prepare_tofu.py --download"
    echo ""
    echo "方法 3: 手动下载后转换"
    echo "  python scripts/prepare_tofu.py --input /path/to/downloaded/tofu.json"
    echo ""
    
    read -p "是否创建示例数据进行测试? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/prepare_tofu.py --create-sample
    else
        echo "请手动准备数据集后重新运行"
        exit 1
    fi
fi

# 步骤 2: 检查模型
echo ""
echo ">>> 步骤 2: 检查模型..."
# 使用 TOFU 预微调的 Llama2-7b 模型
MODEL_PATH="./models_finetune/tofu_llama2_7b"

if [ ! -d "$MODEL_PATH" ]; then
    echo "警告: TOFU预微调模型不存在: $MODEL_PATH"
    echo "请先运行下载脚本: python scripts/download_tofu.py --model"
    exit 1
fi

echo "模型路径: $MODEL_PATH"

# 步骤 3: 运行 LUNAR 遗忘实验
echo ""
echo ">>> 步骤 3: 运行 LUNAR 遗忘实验..."
echo "实验配置:"
echo "  - 模型: llama2-7b-chat"
echo "  - 遗忘目标: author_1"
echo "  - 修改层: [22]"
echo "  - 系数: [+2.0]"
echo "  - 训练轮数: 20"
echo "  - 学习率: 0.01"
echo ""

# 运行实验（使用减小的batch size来避免OOM）
# 注意：layer_modified=[19] 表示修改第19层
python run_lunar.py \
    model_family=llama2-7b-chat \
    model_path="$MODEL_PATH" \
    data_name=tofu_full \
    forget_edge="['author_1']" \
    layer_modified=[19] \
    coeff_list=[+2.0] \
    num_epochs=20 \
    lr=0.01 \
    positions=-1 \
    eval_batch_size=1 \
    eval_generation_max_length=128 \
    eval_generation_max_new_tokens=64 \
    save_unlearned_model=true \
    save_unlearned_model_path="./models_lunar/tofu/llama2-7b-chat/author_1_layer19" \
    save_folder=lunar_tofu_layer19 \
    save_path="./run_results/completions/llama2-7b-chat/lunar_tofu_layer19/tofu_full"

# 步骤 4: 分析结果
echo ""
echo ">>> 步骤 4: 实验完成!"
echo "结果保存在:"
echo "  - 遗忘后模型: ./models_lunar/tofu/llama2-7b-chat/author_1/"
echo "  - 评估结果: ./run_results/completions/llama2-7b-chat/lunar_tofu/tofu_full/"
echo ""
echo "查看结果:"
echo "  cat ./run_results/completions/llama2-7b-chat/lunar_tofu/tofu_full/forget_*.json"

echo ""
echo "=========================================="
echo "复现完成!"
echo "=========================================="