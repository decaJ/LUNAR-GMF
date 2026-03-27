# GMF v2 Weight Editing - Main Script for TOFU
#
# Path 2: Subspace-Projected Weight Editing (SPWE)
#
# Instead of modifying activations at inference time (requires gate),
# directly edit model weights to permanently erase forget knowledge.
#
# Pipeline:
#   1. Load fine-tuned model
#   2. Extract last-token activations for forget + retain sets at target layers
#   3. Compute forget-specific PCA subspace (forget PCA minus retain overlap)
#   4. Project out this subspace from down_proj and o_proj weights in-place
#   5. Evaluate the edited model on forget / retain / factual sets
#
# Usage:
#   python run_gmf_v2_weight.py --config-name forget_gmf_v2_weight_tofu

from __future__ import annotations

import gc
import json
import logging
import os
from typing import List, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.dataset_utils import (
    load_dataset_to_get_direction,
    split_raw_dataset_for_forget,
)
from src.eval_util import custom_evaluate
from src.model_utils.model_loader import load_model
from src.hook_for_unlearn import get_activations

from gmf.weight_editor import SPWEConfig, SubspaceWeightEditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================
# Extract last-token activations  (mirrors prepare_activations in run_gmf_v2.py)
# ======================================================================

def extract_last_token_activations(
    layer_idx_list: List[int],
    model_base,
    forget_dataset,
    retain_dataset,
) -> tuple:
    """Return (forget_inputs, retain_inputs) dicts: layer -> (N, d_model) tensor."""
    forget_inputs: Dict[int, torch.Tensor] = {}
    retain_inputs: Dict[int, torch.Tensor] = {}

    for layer_idx in layer_idx_list:
        logger.info(f"  Extracting layer {layer_idx}...")
        torch.cuda.empty_cache()

        (post_forget, post_retain, _, _, _, _) = get_activations(
            model_base, layer_idx, forget_dataset, retain_dataset,
            batch_size_forget=1, batch_size_remain=1,
        )

        def last_token(acts):
            out = []
            for a in acts:
                a = a.cpu().float()
                if a.dim() == 3:
                    out.append(a[:, -1, :])
                elif a.dim() == 2:
                    out.append(a[-1:, :])
                else:
                    out.append(a.unsqueeze(0))
            return torch.cat(out, dim=0)  # (N, d_model)

        forget_inputs[layer_idx] = last_token(post_forget)
        retain_inputs[layer_idx] = last_token(post_retain)
        logger.info(f"    forget {forget_inputs[layer_idx].shape}, "
                    f"retain {retain_inputs[layer_idx].shape}")

        del post_forget, post_retain
        torch.cuda.empty_cache()

    return forget_inputs, retain_inputs


# ======================================================================
# Main
# ======================================================================

@hydra.main(version_base=None, config_path="config", config_name="forget_gmf_v2_weight_tofu")
def run_gmf_v2_weight(cfg: DictConfig) -> None:
    logger.info("=" * 60)
    logger.info("GMF v2  (Subspace-Projected Weight Editing)")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {cfg.model_path}")
    model_base = load_model(cfg.model_family, cfg.model_path, device)

    # ------------------------------------------------------------------
    # 2. Prepare datasets
    # ------------------------------------------------------------------
    logger.info("Loading forget / retain splits...")
    forget_dataset, retain_dataset = split_raw_dataset_for_forget(
        cfg, data_path, model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )
    logger.info(f"Forget: {len(forget_dataset)}, Retain: {len(retain_dataset)}")

    # ------------------------------------------------------------------
    # 3. Extract activations
    # ------------------------------------------------------------------
    logger.info("Extracting activations...")
    layer_list = list(cfg.layer_modified)
    forget_acts, retain_acts = extract_last_token_activations(
        layer_list, model_base, forget_dataset, retain_dataset,
    )

    # ------------------------------------------------------------------
    # 4. Weight editing (in-place on model_base.model)
    # ------------------------------------------------------------------
    spwe_cfg = SPWEConfig(
        k=cfg.get('pca_k', 5),
        retain_k=cfg.get('retain_pca_k', 10),
        lambda_erase=cfg.get('lambda_erase', 0.5),
        edit_down_proj=cfg.get('edit_down_proj', True),
        edit_o_proj=cfg.get('edit_o_proj', True),
        layers=layer_list,
        device=str(device),
    )
    logger.info(f"\nSPWE: k={spwe_cfg.k}, retain_k={spwe_cfg.retain_k}, "
                f"lambda={spwe_cfg.lambda_erase}, "
                f"down_proj={spwe_cfg.edit_down_proj}, o_proj={spwe_cfg.edit_o_proj}")

    editor = SubspaceWeightEditor(model_base.model, spwe_cfg)
    editor.edit_all(forget_acts, retain_acts)
    editor.summary()

    # Free activation memory
    del forget_acts, retain_acts
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Evaluate edited model (no hooks needed — weights are already modified)
    # ------------------------------------------------------------------
    logger.info("\nEvaluating forget set...")
    eval_forget = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="forget_edge",
        output_es_score=cfg.get('compute_es_score', False),
        fwd_hooks=[],
    )

    logger.info("Evaluating retain set...")
    eval_retain = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="retained_edge",
        output_es_score=False,
        fwd_hooks=[],
    )

    eval_logs = {"forget": eval_forget, "retain": eval_retain}

    if cfg.get('if_eval_factual', False):
        logger.info("Evaluating factual data...")
        eval_logs["factual"] = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=model_base.tokenizer,
            model=model_base,
            eval_target="factual_data",
            output_es_score=False,
            fwd_hooks=[],
        )

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    save_dir = cfg.get('save_path',
        f"run_results/completions/{cfg.model_family}/{cfg.save_folder}/{cfg.data_name}")
    os.makedirs(save_dir, exist_ok=True)
    layers_tag = "_".join(str(l) for l in layer_list)
    out_file = os.path.join(save_dir, f"gmf_v2_weight_{layers_tag}.json")

    with open(out_file, 'w') as f:
        json.dump(eval_logs, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {out_file}")

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for split, res in eval_logs.items():
        if res is None:
            continue
        r1  = res.get('rouge1_recall', '?')
        rl  = res.get('rougeL_recall', '?')
        ppl = res.get('perplexity', '?')
        logger.info(f"  [{split}] rouge1={r1:.4f}, rougeL={rl:.4f}, ppl={ppl:.2f}")


if __name__ == "__main__":
    run_gmf_v2_weight()
