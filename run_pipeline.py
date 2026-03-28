#!/usr/bin/env python3
"""
run_pipeline.py
===============
Two-stage pipeline:
  1. Run sweep_gmf.py  — parameter sweep, saves sweep_results.csv
  2. Parse sweep results, pick best layer/coeff that meets the target
  3. Run run_attack_eval.py with those best params

Safety: attack eval only starts after sweep_results.csv is confirmed saved
and non-empty.  If sweep fails or produces no results, the script exits.

Usage:
    cd /root/workspace/LUNAR-GMF
    python /path/to/run_pipeline.py

    # Override attack eval args:
    python /path/to/run_pipeline.py --n_forget 20 --n_retain 20 --no_softprompt
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
#  Settings
# ══════════════════════════════════════════════════════════════════
WORK_DIR        = "/root/workspace/LUNAR-GMF"
SWEEP_SCRIPT    = os.path.join(os.path.dirname(__file__), "sweep_gmf.py")
ATTACK_SCRIPT   = os.path.join(os.path.dirname(__file__), "run_attack_eval.py")
SWEEP_CSV       = os.path.join(WORK_DIR, "sweep_results.csv")

# Target criteria for best-param selection
FORGET_TARGET   = 0.19
RETAIN_TARGET   = 0.90
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────
# Step 1: run sweep
# ──────────────────────────────────────────────────────────────────

def run_sweep():
    print("\n" + "█" * 65)
    print("  STAGE 1 — Parameter Sweep (sweep_gmf.py)")
    print("█" * 65 + "\n")

    rc = subprocess.call(
        [sys.executable, SWEEP_SCRIPT],
        cwd=WORK_DIR,
    )
    if rc != 0:
        print(f"\n✗  sweep_gmf.py exited with code {rc}.  Aborting pipeline.")
        sys.exit(rc)

    # Confirm CSV was written and is non-empty
    if not os.path.isfile(SWEEP_CSV):
        print(f"\n✗  sweep_results.csv not found at {SWEEP_CSV}.  Aborting.")
        sys.exit(1)

    size = os.path.getsize(SWEEP_CSV)
    if size == 0:
        print(f"\n✗  sweep_results.csv is empty.  Aborting.")
        sys.exit(1)

    print(f"\n✓  sweep_results.csv saved ({size} bytes).")


# ──────────────────────────────────────────────────────────────────
# Step 2: parse sweep CSV → pick best params
# ──────────────────────────────────────────────────────────────────

def parse_best_params():
    """
    Read sweep_results.csv and return (layer, coeff) for the best experiment.

    Selection priority:
      1. Experiments that meet BOTH targets (forget≤0.19 AND retain≥0.90)
         → ranked by forget_rouge1 ascending (lowest forget wins)
      2. If none meet both, fall back to minimum combined miss:
             Δ = max(forget - 0.19, 0) + max(0.90 - retain, 0)
    """
    rows = []
    with open(SWEEP_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                forget = float(row["forget_rouge1"])
                retain = float(row["retain_rouge1"])
            except (ValueError, KeyError):
                continue   # FAIL rows
            # layers is stored as "[15]" or "[15, 19]" — extract first layer
            layers_str = row.get("layers", "[19]").strip("[]")
            try:
                first_layer = int(layers_str.split(",")[0].strip())
            except ValueError:
                first_layer = 19
            try:
                coeff = float(row["coeff"])
            except (ValueError, KeyError):
                coeff = 3.0
            rows.append({
                "name":   row["name"],
                "layer":  first_layer,
                "coeff":  coeff,
                "forget": forget,
                "retain": retain,
            })

    if not rows:
        print("✗  No valid rows in sweep_results.csv.  Using defaults layer=19, coeff=3.0")
        return 19, 3.0

    # Priority 1: both targets met
    hits = [r for r in rows if r["forget"] <= FORGET_TARGET and r["retain"] >= RETAIN_TARGET]
    if hits:
        hits.sort(key=lambda r: r["forget"])          # lowest forget first
        best = hits[0]
        print(f"\n  ★  {len(hits)} experiment(s) met both targets.")
    else:
        # Priority 2: closest miss (minimum combined Δ)
        rows.sort(key=lambda r: (
            max(r["forget"] - FORGET_TARGET, 0) + max(RETAIN_TARGET - r["retain"], 0)
        ))
        best = rows[0]
        delta = (max(best["forget"] - FORGET_TARGET, 0) +
                 max(RETAIN_TARGET - best["retain"], 0))
        print(f"\n  No experiment met both targets. "
              f"Best miss: Δ={delta:.4f}")

    print(f"  Best experiment : {best['name']}")
    print(f"  forget_rouge1   : {best['forget']:.4f}  "
          f"(target ≤ {FORGET_TARGET})")
    print(f"  retain_rouge1   : {best['retain']:.4f}  "
          f"(target ≥ {RETAIN_TARGET})")
    print(f"  → layer={best['layer']}, coeff={best['coeff']}")

    return best["layer"], best["coeff"]


# ──────────────────────────────────────────────────────────────────
# Step 3: run attack eval
# ──────────────────────────────────────────────────────────────────

def run_attack_eval(layer, coeff, extra_args):
    print("\n" + "█" * 65)
    print("  STAGE 2 — Robustness Evaluation (run_attack_eval.py)")
    print(f"  layer={layer}  coeff={coeff}")
    print("█" * 65 + "\n")

    cmd = [
        sys.executable, ATTACK_SCRIPT,
        "--layer",  str(layer),
        "--coeff",  str(coeff),
    ] + extra_args

    print("  Command:", " ".join(cmd), "\n")

    rc = subprocess.call(cmd, cwd=WORK_DIR)
    if rc != 0:
        print(f"\n✗  run_attack_eval.py exited with code {rc}.")
        sys.exit(rc)

    print("\n✓  Attack evaluation complete.")


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline: sweep_gmf → run_attack_eval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip_sweep", action="store_true",
                   help="Skip sweep and go straight to attack eval "
                        "(uses existing sweep_results.csv)")
    p.add_argument("--layer", type=int, default=None,
                   help="Override layer for attack eval (skip auto-detect)")
    p.add_argument("--coeff", type=float, default=None,
                   help="Override coeff for attack eval (skip auto-detect)")
    # Pass-through args for run_attack_eval.py
    p.add_argument("--n_forget",      type=int,   default=20)
    p.add_argument("--n_retain",      type=int,   default=20)
    p.add_argument("--no_softprompt", action="store_true")
    p.add_argument("--sp_steps",      type=int,   default=200)
    p.add_argument("--output",        default="attacks/robustness_results.csv")
    return p.parse_args()


def main():
    args = parse_args()

    # Stage 1
    if not args.skip_sweep:
        run_sweep()
    else:
        print("\n  --skip_sweep: using existing sweep_results.csv")
        if not os.path.isfile(SWEEP_CSV):
            print(f"✗  {SWEEP_CSV} not found.")
            sys.exit(1)

    # Stage 2a: determine best params
    if args.layer is not None and args.coeff is not None:
        layer, coeff = args.layer, args.coeff
        print(f"\n  Using manually specified layer={layer}, coeff={coeff}")
    else:
        layer, coeff = parse_best_params()
        if args.layer is not None:
            layer = args.layer
        if args.coeff is not None:
            coeff = args.coeff

    # Stage 2b: build pass-through args
    extra = [
        "--n_forget",  str(args.n_forget),
        "--n_retain",  str(args.n_retain),
        "--sp_steps",  str(args.sp_steps),
        "--output",    args.output,
    ]
    if args.no_softprompt:
        extra.append("--no_softprompt")

    # Stage 2
    run_attack_eval(layer, coeff, extra)


if __name__ == "__main__":
    main()
