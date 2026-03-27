# attacks/common.py
# Shared utilities for all attack evaluations.

from __future__ import annotations

import json
import math
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

from src.utils.hook_utils import add_hooks


# ──────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────

def load_tofu_data(
    data_path: str,
    forget_edge: List[str],
    n_forget: int = 30,
    n_retain: int = 30,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Load forget and retain QA pairs from the TOFU dataset.

    Returns:
        forget_qa: list of (question, answer) for forget set
        retain_qa: list of (question, answer) for retain set
    """
    with open(data_path) as f:
        data = json.load(f)

    forget_qa = [
        (d["question"], d["answer"])
        for d in data if d["edge"] in forget_edge
    ][:n_forget]

    retain_qa = [
        (d["question"], d["answer"])
        for d in data if d["edge"] not in forget_edge
    ][:n_retain]

    return forget_qa, retain_qa


# ──────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────

def format_instruction(question: str, model_family: str = "llama2-7b") -> str:
    """Format a question as an instruction-tuned input string."""
    if model_family in ("llama2-7b", "llama2-7b-chat"):
        return f"[INST] {question} [/INST]"
    elif model_family == "llama3-8b-instruct":
        return (
            f"<|start_header_id|>user<|end_header_id|>\n{question}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        return f"[INST] {question} [/INST]"


# ──────────────────────────────────────────────────────────────────
# Core evaluation helper
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_questions(
    model_base,
    qa_pairs: List[Tuple[str, str]],
    fwd_hooks: list = [],
    max_new_tokens: int = 128,
    model_family: str = "llama2-7b",
    batch_size: int = 1,
) -> Dict[str, float]:
    """Generate completions for (question, answer) pairs and compute ROUGE.

    Args:
        model_base: loaded model wrapper (ModelBase)
        qa_pairs:   list of (question, ground_truth_answer)
        fwd_hooks:  list of (module, hook_fn) for add_hooks
        max_new_tokens: generation budget
        model_family:   string identifying the model type

    Returns:
        dict with rouge1_recall, rougeL_recall, perplexity (averages)
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    tokenizer = model_base.tokenizer
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    r1_list, rL_list, ppl_list = [], [], []

    for question, gt_answer in qa_pairs:
        input_str = format_instruction(question, model_family)

        inputs = tokenizer(
            [input_str],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            out = model_base._generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out[0][inputs.input_ids.shape[1]:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # ROUGE
        s = scorer.score(gt_answer, gen_text)
        r1_list.append(s["rouge1"].recall)
        rL_list.append(s["rougeL"].recall)

        # Perplexity via teacher-forced cross-entropy
        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            full_ids = tokenizer(
                input_str + " " + gt_answer,
                return_tensors="pt",
                add_special_tokens=True,
            ).input_ids.to("cuda")
            with torch.no_grad():
                labels = full_ids.clone()
                # mask prompt tokens
                prompt_len = inputs.input_ids.shape[1]
                labels[:, :prompt_len] = -100
                ce = model_base.model(input_ids=full_ids, labels=labels).loss
                ppl_list.append(math.exp(min(ce.item(), 20)))

    n = max(len(r1_list), 1)
    return {
        "rouge1_recall": sum(r1_list) / n,
        "rougeL_recall": sum(rL_list) / n,
        "perplexity":    sum(ppl_list) / n,
        "n_samples":     n,
    }


def print_results_table(results: Dict[str, Dict]) -> None:
    """Print a formatted comparison table of attack results.

    Expected structure:
        results = {
            "No Attack":   {"lunar": {...}, "gmf": {...}},
            "Paraphrase":  {"lunar": {...}, "gmf": {...}},
            ...
        }
    where each inner dict has keys rouge1_recall (forget + retain).
    """
    W = 110
    print("\n" + "=" * W)
    print("  ROBUSTNESS COMPARISON TABLE  (forget_r1↓ good,  retain_r1↑ good)")
    print("=" * W)
    print(
        f"  {'Attack':<26} │ "
        f"{'── LUNAR ──':^35} │ "
        f"{'── GMF ──':^35}"
    )
    print(
        f"  {'':26} │ "
        f"{'forget_r1':>10}  {'retain_r1':>10}  {'score':>8} │ "
        f"{'forget_r1':>10}  {'retain_r1':>10}  {'score':>8}"
    )
    print("  " + "-" * (W - 2))

    for attack_name, v in results.items():
        L  = v.get("lunar", {})
        G  = v.get("gmf", {})
        lf = L.get("forget", {}).get("rouge1_recall", float("nan"))
        lr = L.get("retain", {}).get("rouge1_recall", float("nan"))
        gf = G.get("forget", {}).get("rouge1_recall", float("nan"))
        gr = G.get("retain", {}).get("rouge1_recall", float("nan"))
        ls = lr - lf  # higher score = better defense
        gs = gr - gf
        print(
            f"  {attack_name:<26} │ "
            f"{lf:>10.4f}  {lr:>10.4f}  {ls:>+8.4f} │ "
            f"{gf:>10.4f}  {gr:>10.4f}  {gs:>+8.4f}"
        )

    print("=" * W)
    print("  score = retain_r1 - forget_r1  (higher = stronger defense)\n")
