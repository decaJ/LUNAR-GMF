#!/usr/bin/env python3
"""
TOFU 数据集准备脚本
用于下载和转换 TOFU 数据集为 LUNAR 兼容格式
"""

import os
import json
import argparse
from pathlib import Path


def convert_tofu_to_lunar_format(tofu_data, forget_author="author_1"):
    """
    将 TOFU 数据集格式转换为 LUNAR 兼容格式
    
    TOFU 原始格式:
    {
        "question": "...",
        "answer": "...",
        "author": "author_1"
    }
    
    LUNAR 需要格式:
    {
        "question": "...",
        "answer": "...",
        "edge": "author_1"
    }
    """
    converted_data = []
    
    for item in tofu_data:
        converted_item = {
            "question": item.get("question", item.get("prompt", "")),
            "answer": item.get("answer", item.get("response", "")),
            "edge": item.get("author", item.get("source", "unknown"))
        }
        converted_data.append(converted_item)
    
    return converted_data


def create_sample_tofu_data():
    """
    创建示例 TOFU 数据（用于测试）
    实际使用时应该从 Hugging Face 下载真实数据
    """
    sample_data = []
    
    # 模拟 TOFU 数据集的结构
    # TOFU 包含虚构作者的传记信息
    authors = ["author_1", "author_2", "author_3", "author_4", "author_5"]
    
    # 示例问题模板
    question_templates = [
        "What is {author}'s birth date?",
        "Where was {author} born?",
        "What books did {author} write?",
        "What is {author}'s occupation?",
        "What awards has {author} received?",
    ]
    
    # 为每个作者生成示例数据
    for author in authors:
        for template in question_templates:
            question = template.format(author=author)
            answer = f"Sample answer for {author} about {template.split()[0]}"
            sample_data.append({
                "question": question,
                "answer": answer,
                "author": author
            })
    
    return sample_data


def download_tofu_from_huggingface(save_path):
    """
    从 Hugging Face 下载 TOFU 数据集
    
    注意：需要网络连接
    """
    try:
        from datasets import load_dataset
        
        print("正在从 Hugging Face 下载 TOFU 数据集...")
        dataset = load_dataset("locuslab/TOFU")
        
        # 保存数据集
        all_data = []
        for split in dataset.keys():
            for item in dataset[split]:
                all_data.append(item)
        
        # 转换格式
        converted_data = convert_tofu_to_lunar_format(all_data)
        
        # 保存为 JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"TOFU 数据集已保存到: {save_path}")
        print(f"总共 {len(converted_data)} 条数据")
        
        return True
    
    except Exception as e:
        print(f"下载失败: {e}")
        print("请检查网络连接或使用手动下载方式")
        return False


def download_tofu_manually_guide():
    """
    提供手动下载指南
    """
    guide = """
    ========================================
    TOFU 数据集手动下载指南
    ========================================
    
    由于网络限制，您需要手动下载 TOFU 数据集：
    
    方法 1: 使用代理
    ---------------
    export HTTP_PROXY=http://your-proxy:port
    export HTTPS_PROXY=http://your-proxy:port
    python scripts/prepare_tofu.py --download
    
    方法 2: 直接下载
    ---------------
    1. 访问 https://huggingface.co/datasets/locuslab/TOFU
    2. 下载数据集文件
    3. 将文件放置到: /home/users/yanzeyu/LUNAR/dataset/unlearning/tofu_full.json
    
    方法 3: 使用镜像站点
    -------------------
    如果有 Hugging Face 镜像，可以使用：
    export HF_ENDPOINT=https://hf-mirror.com
    python scripts/prepare_tofu.py --download
    
    数据格式要求
    ------------
    JSON 格式，每条数据包含：
    - question: 问题文本
    - answer: 答案文本  
    - author: 作者标识（用于遗忘目标）
    
    示例:
    [
        {
            "question": "What is Author 1's birth date?",
            "answer": "Author 1 was born on January 15, 1980.",
            "author": "author_1"
        },
        ...
    ]
    ========================================
    """
    print(guide)


def main():
    parser = argparse.ArgumentParser(description="准备 TOFU 数据集")
    parser.add_argument("--download", action="store_true", help="尝试从 Hugging Face 下载")
    parser.add_argument("--create-sample", action="store_true", help="创建示例数据（用于测试）")
    parser.add_argument("--input", type=str, help="输入文件路径（已下载的 TOFU 数据）")
    parser.add_argument("--output", type=str, 
                        default="./dataset/unlearning/tofu_full.json",
                        help="输出文件路径")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if args.download:
        # 尝试从 Hugging Face 下载
        success = download_tofu_from_huggingface(args.output)
        if not success:
            download_tofu_manually_guide()
    
    elif args.create_sample:
        # 创建示例数据
        print("创建示例 TOFU 数据...")
        sample_data = create_sample_tofu_data()
        converted_data = convert_tofu_to_lunar_format(sample_data)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"示例数据已保存到: {args.output}")
        print(f"总共 {len(converted_data)} 条数据")
    
    elif args.input:
        # 从输入文件转换
        print(f"从 {args.input} 加载数据...")
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        converted_data = convert_tofu_to_lunar_format(data)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"转换后的数据已保存到: {args.output}")
        print(f"总共 {len(converted_data)} 条数据")
    
    else:
        # 显示帮助
        download_tofu_manually_guide()
        print("\n使用方法:")
        print("  python scripts/prepare_tofu.py --download          # 从 Hugging Face 下载")
        print("  python scripts/prepare_tofu.py --create-sample    # 创建示例数据")
        print("  python scripts/prepare_tofu.py --input <file>     # 转换已有数据")


if __name__ == "__main__":
    main()