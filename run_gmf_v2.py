# GMF v2 - Main Run Script for TOFU
#
# Drop-in replacement for run_gmf.py using the upgraded GMF v2 components:
#   - PCA subspace manifold (instead of Gaussian ball)
#   - ODE multi-step transport (instead of MLP residual flow)
#   - 1-2 learnable scalars per layer (instead of ~2M MLP parameters)
#
# Usage:
#   python run_gmf_v2.py --config-name forget_gmf_v2_tofu
#
# Compared to run_gmf.py:
#   - Same hook-based inference pipeline
#   - Same evaluation pipeline (custom_evaluate)
#   - Swaps GMF v1 modules for GMF v2 modules
#   - Adds pca_k, num_ode_steps, no_train config options

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
from src.generate_directions import generate_candidate_directions
from src.model_utils.model_loader import load_model
from src.hook_for_unlearn import get_activations

# ---- GMF v2 imports ----
from gmf.manifold_v2 import ManifoldExtractorV2
from gmf.module_v2 import GatedODEFlow
from gmf.trainer_v2 import GMFV2Config, GMFV2Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ======================================================================
# Data preparation (same as run_gmf.py)
# ======================================================================

def prepare_activations(
    layer_idx_list: List[int],
    model_base,
    forget_dataset,
    retain_dataset,
    device,
):
    """Extract post-block activations for each target layer."""
    all_forget = {}
    all_retain = {}
    forget_inputs = {}
    retain_inputs = {}

    for layer_idx in layer_idx_list:
        logger.info(f"Extracting activations for layer {layer_idx}...")
        torch.cuda.empty_cache()

        (
            post_forget, post_retain,
            _, _,   # pre_post_attention_layernorm (unused)
            _, _,   # pre_down_proj (unused)
        ) = get_activations(
            model_base, layer_idx, forget_dataset, retain_dataset,
            batch_size_forget=1, batch_size_remain=1,
        )

        all_forget[layer_idx] = post_forget
        all_retain[layer_idx] = post_retain

        # Take ONLY the last token position per sample -> (N, d_model)
        # post_forget[i] shape: (1, seq_len, d_model)
        # Taking [:, -1, :] gives the last token, which carries the
        # accumulated semantic information about the prompt.
        # Using all tokens would include padding / irrelevant positions
        # and bloat N by ~28x, corrupting PCA.
        def last_token(acts):
            out = []
            for a in acts:
                a = a.cpu().float()
                if a.dim() == 3:        # (1, seq_len, d_model)
                    out.append(a[:, -1, :])   # (1, d_model)
                elif a.dim() == 2:      # (seq_len, d_model)
                    out.append(a[-1:, :])     # (1, d_model)
                else:                   # (d_model,)
                    out.append(a.unsqueeze(0))
            return torch.cat(out, dim=0)  # (N, d_model)

        forget_inputs[layer_idx] = last_token(post_forget)
        retain_inputs[layer_idx] = last_token(post_retain)

        logger.info(
            f"  Layer {layer_idx}: "
            f"forget {forget_inputs[layer_idx].shape}, "
            f"retain {retain_inputs[layer_idx].shape}"
        )

        del post_forget, post_retain
        torch.cuda.empty_cache()

    return all_forget, all_retain, forget_inputs, retain_inputs


# ======================================================================
# Build and train GMF v2 modules
# ======================================================================

def build_gmf_v2_modules(
    cfg,
    layer_idx_list: List[int],
    all_forget: Dict,
    all_retain: Dict,
    forget_inputs: Dict,
    retain_inputs: Dict,
    refusal_directions: Dict,
    hidden_size: int,
    device: str,
) -> Dict[int, GatedODEFlow]:
    """
    For each layer:
      1. Extract PCA submanifold + attractor
      2. (Optionally) train step_size scalar

    Returns: dict[layer_idx -> GatedODEFlow]
    """
    v2_config = GMFV2Config(
        pca_k=cfg.get('pca_k', 10),
        attractor_offset=cfg.get('attractor_offset', 2.0),
        sigma=cfg.get('sigma', 1.0),
        learnable_sigma=cfg.get('learnable_sigma', False),
        num_ode_steps=cfg.get('num_ode_steps', 5),
        step_size=cfg.get('step_size', 1.0),
        learnable_step=cfg.get('learnable_step', True),
        no_train=cfg.get('no_train', False),
        num_epochs=cfg.get('num_epochs', 10),
        batch_size=cfg.get('batch_size', 32),
        learning_rate=cfg.get('lr_v2', 1e-2),
        lambda_forget=cfg.get('lambda_forget', 1.0),
        lambda_retain=cfg.get('lambda_retain', 1.0),
        no_gate_mode=cfg.get('no_gate_mode', False),
        device=device,
    )

    modules = {}

    for layer_idx in layer_idx_list:
        logger.info(f"\n{'='*50}")
        logger.info(f"Building GMF v2 for layer {layer_idx}")
        logger.info(f"{'='*50}")

        trainer = GMFV2Trainer(
            hidden_size=hidden_size,
            config=v2_config,
            layer_idx=layer_idx,
        )

        # Phase 1: extract manifold
        trainer.phase1_extract(
            forget_activations=all_forget[layer_idx],
            retain_activations=all_retain[layer_idx],
            refusal_direction=refusal_directions[layer_idx],
        )

        # Phase 2: (optional) train step_size
        trainer.phase2_train(
            forget_inputs=forget_inputs[layer_idx],
            retain_inputs=retain_inputs[layer_idx],
        )

        modules[layer_idx] = trainer.get_module().cpu().eval()

    return modules


# ======================================================================
# Build inference hooks
# ======================================================================

def make_hook(gmf_module: GatedODEFlow, device: str):
    """
    Forward hook: intercepts layer output and applies GMF v2 transformation.
    Handles (batch, seq_len, d_model) activations (typical Transformer output).
    """
    gmf_module = gmf_module.to(device).eval()

    # Ensure all manifold buffers are on the correct device
    if gmf_module.gate.manifold_mu is not None:
        gmf_module.gate.manifold_mu = gmf_module.gate.manifold_mu.to(device)
        gmf_module.gate.manifold_U  = gmf_module.gate.manifold_U.to(device)
        gmf_module.gate.manifold_S  = gmf_module.gate.manifold_S.to(device)
    # Directional gate buffers
    if gmf_module.gate.dir_v is not None:
        gmf_module.gate.dir_v      = gmf_module.gate.dir_v.to(device)
        gmf_module.gate.dir_thresh = gmf_module.gate.dir_thresh.to(device)
        gmf_module.gate.dir_temp   = gmf_module.gate.dir_temp.to(device)
    if gmf_module.attractor_mu is not None:
        gmf_module.attractor_mu = gmf_module.attractor_mu.to(device)

    def hook_fn(module, input, output):
        activation = output[0] if isinstance(output, tuple) else output

        orig_dtype = activation.dtype

        if activation.dim() == 3:
            # (batch, seq_len, d_model) — only transform the last token position.
            # The PCA manifold was built from last-token activations, so applying
            # the gate to ALL positions would corrupt retain / padding tokens.
            last = activation[:, -1, :].float()   # (batch, d_model)
            with torch.no_grad():
                transformed_last, _ = gmf_module(last)
            result = activation.clone()
            result[:, -1, :] = transformed_last.to(orig_dtype)
        else:
            # 2-D fallback: (batch, d_model)
            with torch.no_grad():
                result, _ = gmf_module(activation.float())
            result = result.to(orig_dtype)

        if isinstance(output, tuple):
            return (result, *output[1:])
        return result

    return hook_fn


# ======================================================================
# Main
# ======================================================================

@hydra.main(version_base=None, config_path="config", config_name="forget_gmf_v2_tofu")
def run_gmf_v2(cfg: DictConfig):
    logger.info("=" * 60)
    logger.info("GMF v2  (PCA Subspace + ODE Transport)")
    logger.info("=" * 60)
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading model: {cfg.model_path}")
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    hidden_size = model_base.model.config.hidden_size

    # ------------------------------------------------------------------
    # Generate LUNAR-style refusal directions
    # ------------------------------------------------------------------
    logger.info("Generating refusal directions...")
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg, data_path,
        instructions_only=True,
        use_harmful=cfg.use_harmful,
        use_unverified=cfg.use_unverified,
    )
    candidate_directions = generate_candidate_directions(
        cfg, model_base, harmful_train, forget_train
    )

    layer_idx_list = list(cfg.layer_modified)
    positions = cfg.positions

    refusal_directions = {}
    for layer_idx in layer_idx_list:
        refusal_directions[layer_idx] = candidate_directions[positions, layer_idx + 1, :]
        logger.info(
            f"Layer {layer_idx}: "
            f"||direction||={refusal_directions[layer_idx].norm():.4f}"
        )

    # ------------------------------------------------------------------
    # Prepare datasets
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
    # Extract activations
    # ------------------------------------------------------------------
    logger.info("Extracting activations...")
    all_forget, all_retain, forget_inputs, retain_inputs = prepare_activations(
        layer_idx_list, model_base, forget_dataset, retain_dataset, device
    )

    # ------------------------------------------------------------------
    # Build GMF v2 modules (phase 1 + optional phase 2)
    # ------------------------------------------------------------------
    modules_cpu = build_gmf_v2_modules(
        cfg=cfg,
        layer_idx_list=layer_idx_list,
        all_forget=all_forget,
        all_retain=all_retain,
        forget_inputs=forget_inputs,
        retain_inputs=retain_inputs,
        refusal_directions=refusal_directions,
        hidden_size=hidden_size,
        device=str(device),
    )

    # ------------------------------------------------------------------
    # Free training memory
    # ------------------------------------------------------------------
    logger.info("Freeing training memory...")
    del model_base
    del all_forget, all_retain, forget_inputs, retain_inputs
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Load fresh model for evaluation
    # ------------------------------------------------------------------
    logger.info("Loading fresh model for evaluation...")
    eval_model = load_model(cfg.model_family, cfg.model_path, device)

    fwd_hooks = []
    for layer_idx in layer_idx_list:
        if layer_idx not in modules_cpu:
            continue
        hook = make_hook(modules_cpu[layer_idx], str(device))
        fwd_hooks.append((eval_model.model_block_modules[layer_idx], hook))

    logger.info(f"Registered {len(fwd_hooks)} hooks for layers {layer_idx_list}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    logger.info("Evaluating forget set...")
    eval_forget = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=eval_model.tokenizer,
        model=eval_model,
        eval_target="forget_edge",
        output_es_score=cfg.get('compute_es_score', False),
        fwd_hooks=fwd_hooks,
    )

    logger.info("Evaluating retain set...")
    eval_retain = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=eval_model.tokenizer,
        model=eval_model,
        eval_target="retained_edge",
        output_es_score=False,
        fwd_hooks=fwd_hooks,
    )

    eval_logs = {"forget": eval_forget, "retain": eval_retain}

    if cfg.get('if_eval_factual', False):
        logger.info("Evaluating factual data...")
        eval_factual = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=eval_model.tokenizer,
            model=eval_model,
            eval_target="factual_data",
            output_es_score=False,
            fwd_hooks=fwd_hooks,
        )
        eval_logs["factual"] = eval_factual

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    del eval_model
    torch.cuda.empty_cache()
    gc.collect()

    save_dir = cfg.get('save_path', 'run_results/gmf_v2')
    os.makedirs(save_dir, exist_ok=True)
    tag = "_".join(str(i) for i in layer_idx_list)
    save_file = os.path.join(save_dir, f"gmf_v2_{tag}.json")

    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)

    logger.info(f"\nResults saved to: {save_file}")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for split, logs in eval_logs.items():
        r1 = logs.get('rouge1_recall', 'N/A')
        rL = logs.get('rougeL_recall', 'N/A')
        ppl = logs.get('perplexity', 'N/A')
        if isinstance(r1, float):
            r1 = f"{r1:.4f}"
        if isinstance(rL, float):
            rL = f"{rL:.4f}"
        if isinstance(ppl, float):
            ppl = f"{ppl:.2f}"
        logger.info(f"  [{split}] rouge1={r1}, rougeL={rL}, ppl={ppl}")

    return eval_logs


if __name__ == "__main__":
    run_gmf_v2()
