#!/usr/bin/env python3
"""
使用国内镜像下载 TOFU 数据集和预微调模型
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 设置国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def download_tofu_dataset():
    """下载 TOFU 数据集"""
    from huggingface_hub import snapshot_download
    
    print("=" * 50)
    print("下载 TOFU 数据集...")
    print("=" * 50)
    
    dataset_id = "locuslab/TOFU"
    
    try:
        dataset_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir="./dataset/tofu_raw",
            resume_download=True
        )
        print(f"数据集已下载到: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def download_tofu_model():
    """下载 TOFU 预微调的 Llama2-7b 模型"""
    from huggingface_hub import snapshot_download
    
    print("=" * 50)
    print("下载 TOFU 预微调 Llama2-7b 模型...")
    print("=" * 50)
    
    # TOFU 预微调模型地址（正确的模型名称来自 locuslab 组织）
    model_candidates = [
        "locuslab/tofu_ft_llama2-7b",  # TOFU 官方预微调 Llama2-7b 模型
    ]
    
    for model_id in model_candidates:
        try:
            print(f"尝试下载: {model_id}")
            model_path = snapshot_download(
                repo_id=model_id,
                local_dir="./models_finetune/tofu_llama2_7b",
                resume_download=True
            )
            print(f"模型已下载到: {model_path}")
            return model_path
        except Exception as e:
            print(f"下载 {model_id} 失败: {e}")
            continue
    
    print("所有模型下载尝试失败")
    return None


def convert_tofu_data():
    """转换 TOFU 数据格式为 LUNAR 兼容格式"""
    print("=" * 50)
    print("转换 TOFU 数据格式...")
    print("=" * 50)
    
    raw_data_dir = Path("./dataset/tofu_raw")
    output_file = Path("./dataset/unlearning/tofu_full.json")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # TOFU 数据集是 JSONL 格式（每行一个 JSON 对象）
    for json_file in raw_data_dir.glob("**/*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        converted = convert_single_item(item)
                        if converted:
                            all_data.append(converted)
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {e}")
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成，共 {len(all_data)} 条数据")
    print(f"保存到: {output_file}")
    
    return output_file


def convert_single_item(item):
    """转换单条数据"""
    if not isinstance(item, dict):
        return None
    
    question = item.get("question") or item.get("prompt") or item.get("input", "")
    answer = item.get("answer") or item.get("response") or item.get("output", "")
    author = item.get("author") or item.get("source") or item.get("id", "unknown")
    
    if question and answer:
        return {
            "question": question,
            "answer": answer,
            "edge": author
        }
    
    return None


def main():
    parser = argparse.ArgumentParser(description="下载 TOFU 数据集和模型")
    parser.add_argument("--dataset", action="store_true", help="下载数据集")
    parser.add_argument("--model", action="store_true", help="下载模型")
    parser.add_argument("--convert", action="store_true", help="转换数据格式")
    parser.add_argument("--all", action="store_true", help="下载全部并转换")
    
    args = parser.parse_args()
    
    if args.all or (not args.dataset and not args.model and not args.convert):
        args.dataset = True
        args.model = True
        args.convert = True
    
    # 切换到 LUNAR 目录
    os.chdir("/home/users/yanzeyu/LUNAR")
    
    if args.dataset:
        download_tofu_dataset()
    
    if args.model:
        download_tofu_model()
    
    if args.convert:
        convert_tofu_data()
    
    print("\n" + "=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()