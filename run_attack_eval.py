#!/usr/bin/env python3
"""
run_attack_eval.py
==================
One-stop robustness evaluation: runs 4 adversarial attacks against
LUNAR and GMF unlearning defenses on the TOFU dataset and prints a
comparison table.

Four attacks
------------
1. Paraphrase       (black-box)  – rephrase forget questions
2. Roleplay         (black-box)  – wrap in persona/jailbreak context
3. Few-shot Priming (black-box)  – prepend retain Q&A as in-context demos
4. Orthogonal SP    (white-box)  – optimised soft-prompt ⊥ LUNAR direction

Usage
-----
    cd /root/workspace/LUNAR-GMF          # repo root
    python run_attack_eval.py             # all defaults (see SETTINGS below)

    python run_attack_eval.py \\
        --layer 19 --n_forget 20 --n_retain 20 \\
        --gmf_ckpt attacks/gmf_ckpt.pt \\
        --output attacks/robustness_results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf

# ── Project imports ────────────────────────────────────────────────
from src.dataset_utils import (
    load_dataset_to_get_direction,
    prepare_estimated_net_list,
    prepare_trainset,
    split_raw_dataset_for_forget,
)
from src.estimated_net_utils import (
    ActivationDataset_multiple_layers,
    train_multiple_layers,
)
from src.eval_util import custom_evaluate
from src.generate_directions import generate_candidate_directions
from src.hook_for_unlearn import get_activations
from src.model_utils.model_loader import load_model
from src.utils.hook_utils import add_hooks

from gmf.trainer import GMFModule, GMFTrainer, GMFTrainerConfig
from gmf.manifold import ForgetManifold, AttractorManifold, ManifoldExtractor

from attacks.common import (
    evaluate_questions,
    load_tofu_data,
    print_results_table,
)
from attacks.attack1_paraphrase import ParaphraseAttack
from attacks.attack2_roleplay import RoleplayAttack
from attacks.attack3_fewshot import FewshotPrimingAttack
from attacks.attack4_softprompt import OrthogonalSoftPromptAttack

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
#  ★  SETTINGS  —  edit before running if needed
# ══════════════════════════════════════════════════════════════════
DEFAULT_LAYER        = 19
DEFAULT_COEFF        = 3.0
DEFAULT_MODEL_FAMILY = "llama2-7b"
DEFAULT_MODEL_PATH   = "models_finetune/tofu_llama2_7b"
DEFAULT_DATA_PATH    = "dataset/unlearning/tofu_full.json"
DEFAULT_FORGET_EDGE  = ["author_1"]
DEFAULT_GMF_CKPT     = "attacks/gmf_checkpoint.pt"
DEFAULT_N_FORGET     = 20
DEFAULT_N_RETAIN     = 20
DEFAULT_MAX_NEW_TOK  = 100
DEFAULT_SP_STEPS     = 200   # soft-prompt optimisation steps
DEFAULT_SP_TOKENS    = 20    # number of soft-prompt tokens
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Robustness evaluation: LUNAR vs GMF")
    p.add_argument("--layer",        type=int,   default=DEFAULT_LAYER)
    p.add_argument("--coeff",        type=float, default=DEFAULT_COEFF)
    p.add_argument("--model_family", default=DEFAULT_MODEL_FAMILY)
    p.add_argument("--model_path",   default=DEFAULT_MODEL_PATH)
    p.add_argument("--data_path",    default=DEFAULT_DATA_PATH)
    p.add_argument("--gmf_ckpt",     default=DEFAULT_GMF_CKPT,
                   help="Path to save/load GMF checkpoint (trained if not found)")
    p.add_argument("--n_forget",     type=int,   default=DEFAULT_N_FORGET)
    p.add_argument("--n_retain",     type=int,   default=DEFAULT_N_RETAIN)
    p.add_argument("--max_new_tokens", type=int, default=DEFAULT_MAX_NEW_TOK)
    p.add_argument("--sp_steps",     type=int,   default=DEFAULT_SP_STEPS,
                   help="Soft-prompt optimisation steps")
    p.add_argument("--sp_tokens",    type=int,   default=DEFAULT_SP_TOKENS,
                   help="Number of learnable soft-prompt tokens")
    p.add_argument("--sp_lambda",    type=float, default=3.0,
                   help="Weight for orthogonality penalty in soft-prompt attack")
    p.add_argument("--output",       default="attacks/robustness_results.csv")
    p.add_argument("--no_softprompt", action="store_true",
                   help="Skip soft-prompt attack (faster, skips GPU-intensive optimisation)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────
# Minimal config object (replaces Hydra DictConfig for utility calls)
# ──────────────────────────────────────────────────────────────────

class Cfg:
    """Minimal stand-in for Hydra DictConfig."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def get(self, key, default=None):
        return self.__dict__.get(key, default)


def make_cfg(args) -> Cfg:
    return Cfg(
        model_family                = args.model_family,
        model_path                  = args.model_path,
        data_name                   = "tofu_full",
        forget_edge                 = DEFAULT_FORGET_EDGE,
        layer_modified              = [args.layer],
        coeff_list                  = [f"+{args.coeff}"],
        use_harmful                 = True,
        use_unverified              = False,
        positions                   = -1,
        n_train                     = 128,
        n_test                      = 100,
        n_val                       = 32,
        max_new_tokens              = args.max_new_tokens,
        num_epochs                  = 20,
        lr                          = 1e-4,
        batch_size                  = 32,
        sigma                       = 1.0,
        learnable_sigma             = False,
        distance_method             = "mahalanobis",
        flow_hidden_dim             = 512,
        flow_num_layers             = 3,
        attractor_scale             = 1.0,
        lambda_attractor            = 2.0,
        lambda_retain               = 1.0,
        lambda_flow                 = 0.1,
        lambda_recoverability       = 0.1,
        eval_batch_size             = 4,
        eval_generation_max_length  = 256,
        eval_generation_max_new_tokens = args.max_new_tokens,
        if_eval_factual             = False,
        factual_data_path           = "dataset/unlearning/factual_data.json",
        compute_es_score            = False,
        save_unlearned_model        = False,
        use_different_retain_dataset= False,
    )


# ──────────────────────────────────────────────────────────────────
# LUNAR setup: compute direction + weight editing
# ──────────────────────────────────────────────────────────────────

def setup_lunar(
    cfg: Cfg,
    data_path: str,
    device: torch.device,
) -> Tuple[object, torch.Tensor]:
    """
    Run LUNAR unlearning on a fresh model copy.

    Returns:
        lunar_model:  model with modified down_proj weights
        direction:    LUNAR steering direction vector (d_model,)
    """
    logger.info("=" * 55)
    logger.info("Setting up LUNAR defense...")
    logger.info("=" * 55)

    # Load base model
    model_base = load_model(cfg.model_family, cfg.model_path, device)

    # ── 1. Generate refusal/forget directions ──────────────────
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg, data_path, instructions_only=True,
        use_harmful=cfg.use_harmful, use_unverified=cfg.use_unverified,
    )
    candidate_dirs = generate_candidate_directions(
        cfg, model_base, harmful_train, forget_train
    )  # (n_pos, n_layers, d_model)

    layer_idx = cfg.layer_modified[0]
    direction = candidate_dirs[cfg.positions, layer_idx + 1, :].clone()
    logger.info(f"  Direction norm: {direction.norm():.4f}")

    # ── 2. Prepare activation training sets ───────────────────
    forget_ds, retain_ds = split_raw_dataset_for_forget(
        cfg, data_path, model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )

    (
        forget_inputs, forget_targets,
        retain_inputs, retain_targets,
        _,
    ) = prepare_trainset(
        cfg.layer_modified, model_base,
        forget_ds, retain_ds,
        [direction],
        [float(str(cfg.coeff_list[0]).replace("+", ""))],
        device,
    )

    estimated_nets = prepare_estimated_net_list(
        device=device,
        layer_idx_list=cfg.layer_modified,
        model_base=model_base,
        init_model_list=None,
    )

    # ── 3. Train estimated networks ────────────────────────────
    from src.estimated_net_utils import ActivationDataset_multiple_layers
    from torch.utils.data import ConcatDataset, DataLoader

    train_ds = ConcatDataset([
        ActivationDataset_multiple_layers(forget_inputs, forget_targets),
        ActivationDataset_multiple_layers(retain_inputs, retain_targets),
    ])
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    optimizer  = optim.AdamW(
        chain(*[m.parameters() for m in estimated_nets]), lr=cfg.lr
    )
    scheduler  = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    updated_nets = train_multiple_layers(
        estimated_nets, loader, optimizer, scheduler,
        device=device, num_epochs=cfg.num_epochs,
    )

    # ── 4. Copy weights into fresh model ──────────────────────
    lunar_model = load_model(cfg.model_family, cfg.model_path, device)
    for i, lidx in enumerate(cfg.layer_modified):
        lunar_model.model_block_modules[lidx].mlp.down_proj.weight.data = (
            updated_nets[i].down_proj.weight.data
        )

    # Free training artefacts
    del model_base, estimated_nets, updated_nets
    torch.cuda.empty_cache(); gc.collect()

    logger.info("  LUNAR defense ready.")
    return lunar_model, direction.cpu()


# ──────────────────────────────────────────────────────────────────
# GMF setup: train / load module, build inference hook
# ──────────────────────────────────────────────────────────────────

def _activations_to_tensor(acts: list) -> torch.Tensor:
    """Convert list of activation tensors → (N, d_model)."""
    result = []
    for a in acts:
        a = a.cpu().float()
        if a.dim() == 3:
            result.append(a[:, -1, :])
        elif a.dim() == 2:
            result.append(a[-1:, :])
        else:
            result.append(a.unsqueeze(0))
    return torch.cat(result, dim=0)


def setup_gmf(
    cfg: Cfg,
    data_path: str,
    direction: torch.Tensor,
    device: torch.device,
    ckpt_path: str,
) -> Tuple[object, list]:
    """
    Train or load a GMF module and return (gmf_model, fwd_hooks).

    The GMF model is a fresh (unmodified) copy of the base model.
    The fwd_hooks list should be passed to custom_evaluate or evaluate_questions.
    """
    logger.info("=" * 55)
    logger.info("Setting up GMF defense...")
    logger.info("=" * 55)

    gmf_model = load_model(cfg.model_family, cfg.model_path, device)
    layer_idx = cfg.layer_modified[0]

    # ── Load from checkpoint if available ─────────────────────
    if os.path.isfile(ckpt_path):
        logger.info(f"  Loading GMF checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        trainer_cfg = GMFTrainerConfig(
            sigma           = cfg.sigma,
            distance_method = cfg.distance_method,
            flow_hidden_dim = cfg.flow_hidden_dim,
            flow_num_layers = cfg.flow_num_layers,
            device          = str(device),
        )
        trainer = GMFTrainer(
            hidden_size = gmf_model.model.config.hidden_size,
            config      = trainer_cfg,
        )
        trainer.gmf_module.load_state_dict(ckpt["gmf_module_state_dict"])
        trainer.forget_manifold = ForgetManifold(
            mu    = ckpt["forget_manifold_mu"],
            Sigma = ckpt["forget_manifold_sigma"],
        )
        trainer.attractor_manifold = AttractorManifold(
            mu        = ckpt["attractor_manifold_mu"],
            direction = ckpt["attractor_manifold_direction"],
        )
        trainer.gmf_module.set_manifold(trainer.forget_manifold)
        gmf_module = trainer.gmf_module.to(device)
        gmf_module.eval()
        forget_manifold = trainer.forget_manifold
        logger.info("  GMF checkpoint loaded.")
    else:
        # ── Train from scratch ─────────────────────────────────
        logger.info("  No checkpoint found — training GMF module from scratch.")

        forget_ds, retain_ds = split_raw_dataset_for_forget(
            cfg, data_path, gmf_model,
            forget_edge=cfg.forget_edge,
            instructions_only=True,
            torch_reformat=False,
        )

        logger.info("  Extracting activations...")
        (post_forget, post_retain, _, _, _, _) = get_activations(
            gmf_model, layer_idx, forget_ds, retain_ds,
            batch_size_forget=1, batch_size_remain=1,
        )

        forget_inputs = _activations_to_tensor(post_forget)
        retain_inputs = _activations_to_tensor(post_retain)
        logger.info(f"    forget {forget_inputs.shape}, retain {retain_inputs.shape}")

        trainer_cfg = GMFTrainerConfig(
            num_epochs      = cfg.num_epochs,
            sigma           = cfg.sigma,
            distance_method = cfg.distance_method,
            flow_hidden_dim = cfg.flow_hidden_dim,
            flow_num_layers = cfg.flow_num_layers,
            lambda_attractor= cfg.lambda_attractor,
            lambda_retain   = cfg.lambda_retain,
            lambda_flow     = cfg.lambda_flow,
            device          = str(device),
        )
        trainer = GMFTrainer(
            hidden_size = gmf_model.model.config.hidden_size,
            config      = trainer_cfg,
        )

        logger.info("  Phase 1: extracting manifolds...")
        trainer.phase1_extract_manifolds(
            forget_activations = post_forget,
            retain_activations = post_retain,
            refusal_direction  = direction.to(device),
        )

        logger.info("  Phase 2: training flow...")
        trainer.phase2_train(forget_inputs, retain_inputs)

        gmf_module     = trainer.gmf_module.to(device)
        forget_manifold = trainer.forget_manifold

        # Save checkpoint
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.save({
            "gmf_module_state_dict":   gmf_module.state_dict(),
            "forget_manifold_mu":      trainer.forget_manifold.mu,
            "forget_manifold_sigma":   trainer.forget_manifold.Sigma,
            "attractor_manifold_mu":   trainer.attractor_manifold.mu,
            "attractor_manifold_direction": trainer.attractor_manifold.direction,
        }, ckpt_path)
        logger.info(f"  GMF checkpoint saved to {ckpt_path}")

        del post_forget, post_retain
        torch.cuda.empty_cache(); gc.collect()

    # ── Build inference hook ───────────────────────────────────
    gmf_module.eval()
    # Move gate buffers to device
    gate = gmf_module.gate
    if hasattr(gate, "manifold_mu") and gate.manifold_mu is not None:
        gate.manifold_mu = gate.manifold_mu.to(device)
    if hasattr(gate, "manifold_sigma") and gate.manifold_sigma is not None:
        gate.manifold_sigma = gate.manifold_sigma.to(device)

    def make_hook(mod):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                act = output[0]
            else:
                act = output
            orig_shape = act.shape
            orig_dtype = act.dtype
            flat = act.view(-1, act.shape[-1])
            with torch.no_grad():
                transformed, _ = mod(flat.float().to(device))
            transformed = transformed.view(orig_shape).to(orig_dtype)
            return (transformed, *output[1:]) if isinstance(output, tuple) else transformed
        return hook_fn

    fwd_hooks = [(gmf_model.model_block_modules[layer_idx], make_hook(gmf_module))]
    logger.info("  GMF defense ready.")
    return gmf_model, fwd_hooks, forget_manifold


# ──────────────────────────────────────────────────────────────────
# Attack runner
# ──────────────────────────────────────────────────────────────────

def run_one_attack(
    attack_name: str,
    forget_qa: List[Tuple[str, str]],
    retain_qa: List[Tuple[str, str]],
    lunar_model,
    gmf_model,
    gmf_hooks: list,
    max_new_tokens: int,
    model_family: str,
) -> Dict:
    """Run one attack on both defenses; return metrics dict."""
    logger.info(f"\n  ── {attack_name} ──────────────────────────────────")
    results = {}

    for defense_name, model, hooks in [
        ("lunar", lunar_model, []),
        ("gmf",   gmf_model,  gmf_hooks),
    ]:
        logger.info(f"    [{defense_name.upper()}] forget ({len(forget_qa)} samples)...")
        forget_res = evaluate_questions(
            model, forget_qa, hooks, max_new_tokens, model_family
        )
        logger.info(
            f"      rouge1={forget_res['rouge1_recall']:.4f}  "
            f"rougeL={forget_res['rougeL_recall']:.4f}"
        )

        logger.info(f"    [{defense_name.upper()}] retain ({len(retain_qa)} samples)...")
        retain_res = evaluate_questions(
            model, retain_qa, [],  # retain always unmodified
            max_new_tokens, model_family
        )
        logger.info(
            f"      rouge1={retain_res['rouge1_recall']:.4f}  "
            f"rougeL={retain_res['rougeL_recall']:.4f}"
        )

        results[defense_name] = {
            "forget": forget_res,
            "retain": retain_res,
        }
    return results


# ──────────────────────────────────────────────────────────────────
# CSV export
# ──────────────────────────────────────────────────────────────────

def save_csv(all_results: Dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "attack", "defense",
            "forget_rouge1", "forget_rougeL", "forget_ppl",
            "retain_rouge1", "retain_rougeL", "retain_ppl",
            "score",
        ])
        for attack_name, v in all_results.items():
            for defense_name, d in v.items():
                fgt = d.get("forget", {})
                ret = d.get("retain", {})
                fr1 = fgt.get("rouge1_recall", float("nan"))
                rr1 = ret.get("rouge1_recall", float("nan"))
                w.writerow([
                    attack_name, defense_name,
                    fr1,
                    fgt.get("rougeL_recall", float("nan")),
                    fgt.get("perplexity",    float("nan")),
                    rr1,
                    ret.get("rougeL_recall", float("nan")),
                    ret.get("perplexity",    float("nan")),
                    rr1 - fr1,
                ])
    logger.info(f"\nCSV saved to: {path}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = make_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("\n" + "█" * 60)
    logger.info("  ADVERSARIAL ROBUSTNESS EVALUATION")
    logger.info(f"  Model : {args.model_family}  Layer : {args.layer}  Coeff : {args.coeff}")
    logger.info(f"  Forget: {DEFAULT_FORGET_EDGE}")
    logger.info(f"  Forget samples: {args.n_forget}  Retain samples: {args.n_retain}")
    logger.info("█" * 60 + "\n")

    data_path = args.data_path

    # ── Load data ─────────────────────────────────────────────────
    forget_qa, retain_qa = load_tofu_data(
        data_path, DEFAULT_FORGET_EDGE,
        n_forget=args.n_forget, n_retain=args.n_retain,
    )
    logger.info(f"Loaded {len(forget_qa)} forget / {len(retain_qa)} retain samples.")

    # ── Setup defenses ────────────────────────────────────────────
    lunar_model, direction = setup_lunar(cfg, data_path, device)
    gmf_model, gmf_hooks, forget_manifold = setup_gmf(
        cfg, data_path, direction, device, args.gmf_ckpt
    )

    all_results: Dict = {}

    # ── Baseline (no attack) ──────────────────────────────────────
    logger.info("\n" + "─" * 55)
    logger.info("BASELINE: No Attack")
    all_results["No Attack"] = run_one_attack(
        "No Attack", forget_qa, retain_qa,
        lunar_model, gmf_model, gmf_hooks,
        args.max_new_tokens, args.model_family,
    )

    # ── Attack 1: Paraphrase ──────────────────────────────────────
    logger.info("\n" + "─" * 55)
    logger.info("ATTACK 1: Paraphrase")
    atk1 = ParaphraseAttack()
    para_forget = atk1.attack(forget_qa)
    all_results["Paraphrase"] = run_one_attack(
        "Paraphrase", para_forget, retain_qa,
        lunar_model, gmf_model, gmf_hooks,
        args.max_new_tokens, args.model_family,
    )

    # ── Attack 2: Roleplay ────────────────────────────────────────
    logger.info("\n" + "─" * 55)
    logger.info("ATTACK 2: Roleplay / Jailbreak Context")
    atk2 = RoleplayAttack()
    # Use a single strong template (template index 0)
    rp_forget = atk2.attack_single_template(forget_qa, template_idx=0)
    all_results["Roleplay"] = run_one_attack(
        "Roleplay", rp_forget, retain_qa,
        lunar_model, gmf_model, gmf_hooks,
        args.max_new_tokens, args.model_family,
    )

    # ── Attack 3a: Few-shot Priming ───────────────────────────────
    logger.info("\n" + "─" * 55)
    logger.info("ATTACK 3a: Few-shot Priming")
    atk3 = FewshotPrimingAttack(retain_qa=retain_qa, n_shots=3)
    fs_forget = atk3.attack(forget_qa)
    all_results["Few-shot Priming"] = run_one_attack(
        "Few-shot Priming", fs_forget, retain_qa,
        lunar_model, gmf_model, gmf_hooks,
        args.max_new_tokens, args.model_family,
    )

    # ── Attack 3b: Multi-hop Reasoning ───────────────────────────
    logger.info("\n" + "─" * 55)
    logger.info("ATTACK 3b: Multi-hop Chain-of-Thought")
    mh_forget = atk3.attack_multihop(forget_qa)
    all_results["Multi-hop CoT"] = run_one_attack(
        "Multi-hop CoT", mh_forget, retain_qa,
        lunar_model, gmf_model, gmf_hooks,
        args.max_new_tokens, args.model_family,
    )

    # ── Attack 4: Orthogonal Soft-Prompt ─────────────────────────
    if not args.no_softprompt:
        logger.info("\n" + "─" * 55)
        logger.info("ATTACK 4: Orthogonal Soft-Prompt (white-box)")

        # --- Attack LUNAR: use the LUNAR model's direction --------
        logger.info("  [LUNAR target] Optimising soft-prompt vs LUNAR direction...")
        atk4_lunar = OrthogonalSoftPromptAttack(
            model_base  = lunar_model,
            direction   = direction,
            layer_idx   = args.layer,
            n_tokens    = args.sp_tokens,
            n_steps     = args.sp_steps,
            lambda_orth = args.sp_lambda,
        )
        sp_forget_lunar = atk4_lunar.attack(forget_qa, optimize_first=True)

        # Evaluate LUNAR with these prompts
        lunar_sp_res = {
            "forget": evaluate_questions(
                lunar_model, sp_forget_lunar, [],
                args.max_new_tokens, args.model_family
            ),
            "retain": evaluate_questions(
                lunar_model, retain_qa, [],
                args.max_new_tokens, args.model_family
            ),
        }

        # --- Attack GMF: same soft-prompt on GMF model ------------
        # Test whether the same prompts (optimised against LUNAR) also bypass GMF
        logger.info("  [GMF target] Testing same prompt against GMF defense...")
        sp_forget_gmf = [
            (sp_forget_lunar[i][0], forget_qa[i][1])
            for i in range(len(sp_forget_lunar))
        ]
        gmf_sp_res = {
            "forget": evaluate_questions(
                gmf_model, sp_forget_gmf, gmf_hooks,
                args.max_new_tokens, args.model_family
            ),
            "retain": evaluate_questions(
                gmf_model, retain_qa, [],
                args.max_new_tokens, args.model_family
            ),
        }

        all_results["Orthogonal SP (WB)"] = {
            "lunar": lunar_sp_res,
            "gmf":   gmf_sp_res,
        }
    else:
        logger.info("\nSoft-prompt attack skipped (--no_softprompt).")

    # ── Output ────────────────────────────────────────────────────
    print_results_table(all_results)
    save_csv(all_results, args.output)

    # ── Per-attack delta summary ──────────────────────────────────
    base = all_results.get("No Attack", {})
    base_lf = base.get("lunar", {}).get("forget", {}).get("rouge1_recall", 0)
    base_gf = base.get("gmf",   {}).get("forget", {}).get("rouge1_recall", 0)

    print("\n  ATTACK EFFECTIVENESS  (Δ forget_r1 vs baseline; higher = stronger attack)")
    print(f"  {'Attack':<26}  {'LUNAR Δ':>10}  {'GMF Δ':>10}  {'GMF more robust?':>18}")
    print("  " + "-" * 70)
    for name, v in all_results.items():
        if name == "No Attack":
            continue
        lf = v.get("lunar", {}).get("forget", {}).get("rouge1_recall", float("nan"))
        gf = v.get("gmf",   {}).get("forget", {}).get("rouge1_recall", float("nan"))
        d_lunar = lf - base_lf
        d_gmf   = gf - base_gf
        more_robust = "YES ✓" if d_lunar > d_gmf else "no"
        print(f"  {name:<26}  {d_lunar:>+10.4f}  {d_gmf:>+10.4f}  {more_robust:>18}")
    print()


if __name__ == "__main__":
    main()
