#!/usr/bin/env python
# 仅评估脚本 - 使用已保存的模型进行评估

import os
import json
import torch
import gc
from omegaconf import DictConfig, OmegaConf
import hydra

from src.eval_util import custom_evaluate
from src.model_utils.model_loader import load_model


def run_evaluation_only(cfg):
    """只运行评估，不重新训练"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载遗忘后的模型
    print(f"Loading unlearned model from {cfg.save_unlearned_model_path}")
    unlearned_model = load_model(cfg.model_family, cfg.save_unlearned_model_path, device)
    
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU cache cleared before evaluation")
    
    # 评估遗忘集
    print("\n>>> Evaluating forget set...")
    torch.cuda.empty_cache()
    eval_logs_forget_edge = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=unlearned_model.tokenizer,
        model=unlearned_model,
        eval_target="forget_edge",
        output_es_score=False,
    )
    print(f"Forget set results: {json.dumps(eval_logs_forget_edge, indent=2)}")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 评估保留集
    print("\n>>> Evaluating retain set...")
    torch.cuda.empty_cache()
    eval_logs_retained_edge = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=unlearned_model.tokenizer,
        model=unlearned_model,
        eval_target="retained_edge",
        output_es_score=False,
    )
    print(f"Retain set results: {json.dumps(eval_logs_retained_edge, indent=2)}")
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 评估事实数据（如果启用）
    if cfg.if_eval_factual:
        print("\n>>> Evaluating factual data...")
        torch.cuda.empty_cache()
        eval_logs_factual_data = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=unlearned_model.tokenizer,
            model=unlearned_model,
            eval_target="factual_data",
            output_es_score=False,
        )
        print(f"Factual data results: {json.dumps(eval_logs_factual_data, indent=2)}")
        eval_logs = {
            "forget": eval_logs_forget_edge,
            "retained_edge": eval_logs_retained_edge,
            "factual_data": eval_logs_factual_data,
        }
    else:
        eval_logs = {
            "forget": eval_logs_forget_edge,
            "retained_edge": eval_logs_retained_edge,
        }
    
    # 保存结果
    save_dir = cfg.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_file = os.path.join(save_dir, "forget_22.json")
    print(f"\nSaving evaluation results to {save_file}")
    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)
    
    print("\n==========================================")
    print("Evaluation completed!")
    print("==========================================")
    
    return eval_logs


if __name__ == "__main__":
    # 手动创建配置
    cfg = OmegaConf.create({
        # 模型配置
        "model_family": "llama2-7b-chat",
        "model_path": "./models_finetune/tofu_llama2_7b",
        "save_unlearned_model_path": "./models_lunar/tofu/llama2-7b-chat/author_1",
        
        # 数据配置
        "data_name": "tofu_full",
        "forget_edge": ["author_1"],
        
        # 评估配置 - 使用较小的值来避免OOM
        "eval_batch_size": 1,  # 最小batch size
        "eval_generation_max_length": 128,
        "eval_generation_max_new_tokens": 64,
        
        # 其他配置
        "if_eval_factual": True,
        "factual_data_path": "dataset/unlearning/factual_data.json",
        "compute_es_score": False,
        "save_path": "./run_results/completions/llama2-7b-chat/lunar_tofu/tofu_full",
    })
    
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    run_evaluation_only(cfg)