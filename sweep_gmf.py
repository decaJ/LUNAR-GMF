#!/usr/bin/env python3
"""
GMF v1 Two-Phase Parameter Sweep
==================================
Phase 1  Layer Discovery
  Test every single layer (15-22) with fixed coeff=3.0.
  Automatically rank by (retain - forget) to find the best 1-2 layers.

Phase 2  Parameter Tuning  (auto-generated from Phase 1 results)
  Group A: coeff sweep on best layer
  Group B: multi-layer combinations of top layers
  Group C: sigma (gate bandwidth)
  Group D: lambda_attractor + attractor_scale
  Group E: lambda_retain
  Group F: num_epochs
  Group G: combined best-guess
  Group H: aggressive settings

Hardware: 2× A100  →  NUM_PARALLEL=4, GPUS=[0,0,1,1]

Usage:
    python sweep_gmf.py

Output:
    sweep_logs/<name>.log
    run_results/completions/llama2-7b/sweep/<name>/tofu_full/gmf_<layer>.json
    sweep_results.csv   (all phases combined)
"""

import os, json, time, subprocess, csv
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════
#  ★  SETTINGS  —  edit before running
# ══════════════════════════════════════════════════════════════════
WORK_DIR     = "/root/workspace/LUNAR-GMF"
NUM_PARALLEL = 3                   # cuda:0 × 1 + cuda:1 × 2
GPUS         = [0, 1, 1]
SCRIPT       = "run_gmf.py"
CONFIG       = "forget_gmf_tofu"
LOG_DIR      = "sweep_logs"

# Phase 1 settings
PHASE1_LAYERS    = [15, 16, 17, 18, 19, 20, 21, 22]
PHASE1_COEFF     = 3.0

# Phase 2 selection criteria
TOP_N_LAYERS     = 2      # how many best layers to carry forward
RETAIN_THRESHOLD = 0.60   # discard layers where retain_rouge1 < this
# ══════════════════════════════════════════════════════════════════

GPU_CYCLE = (GPUS * 4)[:NUM_PARALLEL]   # e.g. [0,0,1,1]


# ──────────────────────────────────────────────────────────────────
# Experiment builder
# ──────────────────────────────────────────────────────────────────

def exp(name, layers, coeff,
        lambda_a=1.0, sigma=1.0, attractor_scale=1.0,
        lambda_flow=0.1, num_epochs=20, lambda_retain=1.0):
    layers = sorted(layers)
    n = len(layers)
    layer_str = "[" + ",".join(map(str, layers)) + "]"
    coeff_str = "[" + ",".join([f"+{coeff}"] * n) + "]"
    save_path = f"run_results/completions/llama2-7b/sweep/{name}/tofu_full"
    return {
        "name": name,
        "display": {
            "layers":           layers,
            "coeff":            coeff,
            "lambda_attractor": lambda_a,
            "sigma":            sigma,
            "attractor_scale":  attractor_scale,
            "lambda_flow":      lambda_flow,
            "num_epochs":       num_epochs,
            "lambda_retain":    lambda_retain,
        },
        "overrides": [
            f"layer_modified={layer_str}",
            f"coeff_list={coeff_str}",
            f"lambda_attractor={lambda_a}",
            f"sigma={sigma}",
            f"attractor_scale={attractor_scale}",
            f"lambda_flow={lambda_flow}",
            f"num_epochs={num_epochs}",
            f"lambda_retain={lambda_retain}",
            f"save_path={save_path}",
        ],
        "result_file": os.path.join(
            WORK_DIR, save_path,
            "gmf_" + "_".join(map(str, layers)) + ".json"
        ),
    }


# ──────────────────────────────────────────────────────────────────
# Phase 1: single-layer discovery
# ──────────────────────────────────────────────────────────────────

PHASE1_EXPERIMENTS = [
    exp(f"P1_c3_L{l}", [l], coeff=PHASE1_COEFF)
    for l in PHASE1_LAYERS
]


# ──────────────────────────────────────────────────────────────────
# Phase 2: auto-generated from Phase 1 results
# ──────────────────────────────────────────────────────────────────

def build_phase2(best_layers):
    """
    Dynamically build all Phase 2 experiments.

    best_layers: list of ints, sorted best→worse, length >= 1.
    """
    L1 = best_layers[0]
    L2 = best_layers[1] if len(best_layers) >= 2 else None
    tag1 = str(L1)
    tag12 = f"{L1}_{L2}" if L2 else tag1

    exps = []

    # ── Group A: coeff sweep on best single layer ────────────────
    for c in [2.0, 3.0, 4.0, 5.0, 6.0]:
        exps.append(exp(f"A_c{c:.0f}_L{L1}", [L1], coeff=c))

    # ── Group B: multi-layer combinations ───────────────────────
    if L2 is not None:
        exps.append(exp(f"B1_c3_L{tag12}",       [L1, L2],     coeff=3.0))
        exps.append(exp(f"B2_c4_L{tag12}",       [L1, L2],     coeff=4.0))
        exps.append(exp(f"B3_c3_L{tag12}_s0.7",  [L1, L2],     coeff=3.0, sigma=0.7))
        # 3-layer: add a layer roughly halfway between L1 and L2 (or adjacent)
        span = sorted([L1, L2])
        mid  = (span[0] + span[1]) // 2
        if mid not in span:
            exps.append(exp(f"B4_c3_L{span[0]}_{mid}_{span[1]}",
                            [span[0], mid, span[1]], coeff=3.0))
            exps.append(exp(f"B5_c4_L{span[0]}_{mid}_{span[1]}",
                            [span[0], mid, span[1]], coeff=4.0))
        else:
            # layers are adjacent; extend outward
            L_lo = max(15, span[0] - 2)
            L_hi = min(22, span[1] + 2)
            exps.append(exp(f"B4_c3_L{L_lo}_{L1}_{L_hi}",
                            [L_lo, L1, L_hi], coeff=3.0))
            exps.append(exp(f"B5_c4_L{L_lo}_{L1}_{L_hi}",
                            [L_lo, L1, L_hi], coeff=4.0))
    else:
        # Only one good layer: try it with nearby layers
        L_lo = max(15, L1 - 3)
        L_hi = min(22, L1 + 3)
        exps.append(exp(f"B1_c3_L{L_lo}_{L1}",        [L_lo, L1],        coeff=3.0))
        exps.append(exp(f"B2_c3_L{L1}_{L_hi}",        [L1,   L_hi],      coeff=3.0))
        exps.append(exp(f"B3_c3_L{L_lo}_{L1}_{L_hi}", [L_lo, L1, L_hi], coeff=3.0))
        exps.append(exp(f"B4_c4_L{L_lo}_{L1}_{L_hi}", [L_lo, L1, L_hi], coeff=4.0))

    # ── Group C: sigma (gate bandwidth) ─────────────────────────
    for s in [0.5, 0.7, 1.5]:
        exps.append(exp(f"C_c3_L{L1}_s{s}",   [L1], coeff=3.0, sigma=s))
    exps.append(   exp(f"C_c4_L{L1}_s0.7",    [L1], coeff=4.0, sigma=0.7))

    # ── Group D: attractor strength ──────────────────────────────
    for la in [2.0, 3.0]:
        exps.append(exp(f"D_c3_L{L1}_a{la:.0f}",   [L1], coeff=3.0, lambda_a=la))
    for asc in [1.5, 2.0]:
        exps.append(exp(f"D_c3_L{L1}_as{asc}",     [L1], coeff=3.0, attractor_scale=asc))
    exps.append(    exp(f"D_c4_L{L1}_a2_as1.5",    [L1], coeff=4.0, lambda_a=2.0,
                                                          attractor_scale=1.5))

    # ── Group E: lambda_retain ───────────────────────────────────
    for lr in [0.5, 2.0, 3.0]:
        exps.append(exp(f"E_c3_L{L1}_lr{lr}",  [L1], coeff=3.0, lambda_retain=lr))
    exps.append(    exp(f"E_c4_L{L1}_lr2",     [L1], coeff=4.0, lambda_retain=2.0))

    # ── Group F: num_epochs ──────────────────────────────────────
    exps.append(exp(f"F_c3_L{L1}_e30",  [L1], coeff=3.0, num_epochs=30))
    exps.append(exp(f"F_c4_L{L1}_e30",  [L1], coeff=4.0, num_epochs=30))
    if L2 is not None:
        exps.append(exp(f"F_c3_L{tag12}_e30", [L1, L2], coeff=3.0, num_epochs=30))

    # ── Group G: combined best-guess ────────────────────────────
    exps.append(exp(f"G1_c4_L{L1}_s0.7_a2",
                    [L1], coeff=4.0, sigma=0.7, lambda_a=2.0))
    exps.append(exp(f"G2_c3_L{L1}_s0.7_lr2",
                    [L1], coeff=3.0, sigma=0.7, lambda_retain=2.0))
    exps.append(exp(f"G3_c4_L{L1}_s0.7_a2_lr2",
                    [L1], coeff=4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0))
    if L2 is not None:
        exps.append(exp(f"G4_c3_L{tag12}_s0.7_a2",
                        [L1, L2], coeff=3.0, sigma=0.7, lambda_a=2.0))
        exps.append(exp(f"G5_c4_L{tag12}_s0.7_lr2",
                        [L1, L2], coeff=4.0, sigma=0.7, lambda_retain=2.0))

    # ── Group H: aggressive forget + protect retain ──────────────
    exps.append(exp(f"H1_c5_L{L1}_s0.7_a2",
                    [L1], coeff=5.0, sigma=0.7, lambda_a=2.0))
    exps.append(exp(f"H2_c4_L{L1}_s0.7_a2_lr2",
                    [L1], coeff=4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0))
    exps.append(exp(f"H3_c4_L{L1}_s0.7_a2_lf0.05",
                    [L1], coeff=4.0, sigma=0.7, lambda_a=2.0, lambda_flow=0.05))
    exps.append(exp(f"H4_c5_L{L1}_s0.7_a2_lr2",
                    [L1], coeff=5.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0))
    if L2 is not None:
        exps.append(exp(f"H5_c4_L{tag12}_s0.7_a2_lr2",
                        [L1, L2], coeff=4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0))

    return exps


# ──────────────────────────────────────────────────────────────────
# Runner utilities
# ──────────────────────────────────────────────────────────────────

def launch(e, gpu_id, log_path):
    cmd = ["python", SCRIPT, f"--config-name={CONFIG}"] + e["overrides"]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_f = open(log_path, "w")
    proc  = subprocess.Popen(cmd, cwd=WORK_DIR, env=env,
                             stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def read_result(e):
    path = e["result_file"]
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        def get(split):
            v = d.get(split, d.get(
                "retained_edge" if split == "retain" else
                "factual_data"  if split == "factual" else split, {}))
            return (
                round(v.get("rouge1_recall", v.get("rouge1", float("nan"))), 4),
                round(v.get("rougeL_recall", v.get("rougeL", float("nan"))), 4),
                round(v.get("perplexity",    float("nan")), 1),
            )
        return {"forget": get("forget"), "retain": get("retain"), "factual": get("factual")}
    except Exception as ex:
        return {"error": str(ex)}


def run_experiments(experiments, phase_label, results):
    """Run experiments in parallel groups; append into results dict."""
    log_dir  = os.path.join(WORK_DIR, LOG_DIR)
    total    = len(experiments)
    n_grps   = (total + NUM_PARALLEL - 1) // NUM_PARALLEL

    print(f"\n{'='*65}")
    print(f"  {phase_label}")
    print(f"  {total} experiments  |  {NUM_PARALLEL} parallel  |  {n_grps} groups")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}\n")

    phase_start = time.time()

    for g_idx in range(n_grps):
        batch = experiments[g_idx * NUM_PARALLEL : (g_idx + 1) * NUM_PARALLEL]

        print(f"── Group {g_idx+1}/{n_grps} ──────────────────────────────────────")
        for i, e in enumerate(batch):
            print(f"   [{i+1}] {e['name']}  (GPU {GPU_CYCLE[i]})")
        print()

        procs = []
        for i, e in enumerate(batch):
            log_path = os.path.join(log_dir, f"{e['name']}.log")
            proc, log_f = launch(e, GPU_CYCLE[i], log_path)
            procs.append((proc, log_f, e))

        group_start = time.time()
        for proc, log_f, e in procs:
            proc.wait()
            log_f.close()
            r = read_result(e)
            results[e["name"]] = r
            status  = "✓" if r and "error" not in r else "✗"
            elapsed = str(timedelta(seconds=int(time.time() - group_start)))
            if r and "error" not in r:
                print(f"   {status} {e['name']:<48}  "
                      f"forget={r['forget'][0]:.4f}  "
                      f"retain={r['retain'][0]:.4f}  "
                      f"factual={r['factual'][0]:.4f}  [{elapsed}]")
            else:
                print(f"   {status} {e['name']:<48}  "
                      f"FAILED  (see {LOG_DIR}/{e['name']}.log)")

        elapsed_total = str(timedelta(seconds=int(time.time() - phase_start)))
        print(f"\n   Group {g_idx+1} done.  Phase elapsed: {elapsed_total}\n")


# ──────────────────────────────────────────────────────────────────
# Phase 1 analysis: select best layers
# ──────────────────────────────────────────────────────────────────

def select_best_layers(phase1_exps, results):
    """
    Rank Phase 1 layers.

    Primary criterion : retain_rouge1 >= RETAIN_THRESHOLD
    Secondary criterion: forget_rouge1 ascending (lower = better forgetting)
    Tie-break          : retain - forget score descending

    Falls back to best-available if no layer meets the retain threshold.
    """
    scored = []
    for e in phase1_exps:
        r = results.get(e["name"])
        if r and "error" not in r:
            layer     = e["display"]["layers"][0]
            forget_r1 = r["forget"][0]
            retain_r1 = r["retain"][0]
            scored.append((layer, forget_r1, retain_r1))

    if not scored:
        print("  WARNING: no valid Phase 1 results — defaulting to layer 19.")
        return [19]

    valid = [s for s in scored if s[2] >= RETAIN_THRESHOLD]
    if not valid:
        print(f"  WARNING: no layer has retain >= {RETAIN_THRESHOLD}. "
              "Using all layers ranked by (retain - forget).")
        valid = scored

    # Sort: forget ascending, then retain descending
    valid.sort(key=lambda s: (s[1], -s[2]))

    best = [s[0] for s in valid[:TOP_N_LAYERS]]
    return best


# ──────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────

def print_table(experiments, results):
    W = 140
    print("\n" + "=" * W)
    print("  FULL RESULTS TABLE")
    print("=" * W)
    hdr = (f"{'Experiment':<48} {'Layers':<14} {'c':>4} "
           f"{'λ_a':>4} {'σ':>5} {'aS':>4} {'λ_f':>5} {'λ_r':>4} {'Ep':>3} │ "
           f"{'forget_r1':>10} {'retain_r1':>10} {'factual_r1':>11} {'score':>8}")
    print(hdr)
    print("-" * W)
    for e in experiments:
        p = e["display"]
        r = results.get(e["name"])
        ls = str(p["layers"])
        if r and "error" not in r:
            f1 = r["forget"][0]; r1 = r["retain"][0]; a1 = r["factual"][0]
            score = r1 - f1
            print(f"{e['name']:<48} {ls:<14} {p['coeff']:>4} "
                  f"{p['lambda_attractor']:>4} {p['sigma']:>5} "
                  f"{p['attractor_scale']:>4} {p['lambda_flow']:>5} "
                  f"{p['lambda_retain']:>4} {p['num_epochs']:>3} │ "
                  f"{f1:>10.4f} {r1:>10.4f} {a1:>11.4f} {score:>+8.4f}")
        else:
            print(f"{e['name']:<48} {ls:<14} {p['coeff']:>4} "
                  f"{p['lambda_attractor']:>4} {p['sigma']:>5} "
                  f"{p['attractor_scale']:>4} {p['lambda_flow']:>5} "
                  f"{p['lambda_retain']:>4} {p['num_epochs']:>3} │  FAILED")
    print("=" * W)


def save_csv(experiments, results):
    csv_path = os.path.join(WORK_DIR, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "layers", "coeff", "lambda_attractor", "sigma",
                    "attractor_scale", "lambda_flow", "lambda_retain", "num_epochs",
                    "forget_rouge1", "retain_rouge1", "factual_rouge1",
                    "forget_rougeL", "retain_rougeL", "factual_rougeL",
                    "forget_ppl",    "retain_ppl",    "factual_ppl"])
        for e in experiments:
            p = e["display"]
            r = results.get(e["name"])
            base = [e["name"], str(p["layers"]), p["coeff"],
                    p["lambda_attractor"], p["sigma"], p["attractor_scale"],
                    p["lambda_flow"], p["lambda_retain"], p["num_epochs"]]
            if r and "error" not in r:
                row = base + [
                    r["forget"][0], r["retain"][0], r["factual"][0],
                    r["forget"][1], r["retain"][1], r["factual"][1],
                    r["forget"][2], r["retain"][2], r["factual"][2],
                ]
            else:
                row = base + ["FAIL"] * 9
            w.writerow(row)
    print(f"CSV saved to: {csv_path}\n")


# ──────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────

def _git(args, cwd=WORK_DIR, capture=True):
    """Run a git command; return (returncode, stdout+stderr)."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=capture,
        text=True,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def verify_git_push():
    """
    Dry-run verification that git push will succeed before experiments start.

    Steps:
      1. Confirm we are inside a git repo.
      2. Check there is a configured remote (origin).
      3. Attempt a dry-run push (--dry-run) to detect auth / permission issues.

    Returns True if all checks pass; exits with error otherwise.
    """
    print("\n── Git pre-flight check ────────────────────────────────────")

    # 1. Are we in a git repo?
    rc, out = _git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        print(f"  ✗ Not inside a git repository: {out}")
        print("    Please initialise git or set WORK_DIR correctly.")
        raise SystemExit(1)
    print("  ✓ Git repository detected.")

    # 2. Remote configured?
    rc, out = _git(["remote", "get-url", "origin"])
    if rc != 0:
        print("  ✗ No remote 'origin' configured.")
        print("    Run:  git remote add origin <url>")
        raise SystemExit(1)
    print(f"  ✓ Remote origin: {out}")

    # 3. Current branch
    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    print(f"  ✓ Branch: {branch}")

    # 4. Dry-run push
    print("  Attempting dry-run push ... ", end="", flush=True)
    rc, out = _git(["push", "--dry-run", "origin", branch])
    if rc != 0:
        print("FAILED")
        print(f"  ✗ Dry-run push failed:\n    {out}")
        print("    Fix git credentials / SSH key before running the sweep.")
        raise SystemExit(1)
    print("OK")
    print("  ✓ Git push pre-flight passed.\n")
    return True


def git_push_results(message: str):
    """
    Stage sweep_results.csv + sweep_logs/ + result JSON files, commit, push.
    Prints progress and warns on failure without aborting.
    """
    print("\n── Pushing results to remote ───────────────────────────────")

    # Stage files
    for pattern in [
        "sweep_results.csv",
        "sweep_logs/",
        "run_results/completions/llama2-7b/sweep/",
    ]:
        _git(["add", "-f", pattern])

    # Check if there is anything to commit
    rc, status = _git(["status", "--porcelain"])
    if not status.strip():
        print("  Nothing new to commit — results may already be up to date.")
        return

    # Commit
    rc, out = _git(["commit", "-m", message])
    if rc != 0:
        print(f"  ✗ Commit failed: {out}")
        return
    print(f"  ✓ Committed: {message}")

    # Push
    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    rc, out = _git(["push", "origin", branch])
    if rc != 0:
        print(f"  ✗ Push failed: {out}")
        print("    Results are committed locally; push manually when ready.")
    else:
        print(f"  ✓ Pushed to origin/{branch}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(WORK_DIR, LOG_DIR), exist_ok=True)

    # ── Pre-flight: verify git push works before wasting compute ──
    print("\n── Git pre-flight check ─────────────────────────────────────")
    verify_git_push()
    print("  Git push verified.  Starting experiments.\n")

    sweep_start   = time.time()
    all_results   = {}
    all_exps      = []

    # ── Phase 1: Layer Discovery ────────────────────────────────
    print("\n" + "█" * 65)
    print("  PHASE 1  —  Single-Layer Discovery")
    print(f"  Layers: {PHASE1_LAYERS}   coeff={PHASE1_COEFF}")
    print("█" * 65)

    run_experiments(PHASE1_EXPERIMENTS, "Phase 1: Layer Discovery", all_results)
    all_exps.extend(PHASE1_EXPERIMENTS)

    # ── Phase 1 summary ─────────────────────────────────────────
    print("\n── Phase 1 Summary ─────────────────────────────────────────")
    print(f"  {'Layer':<8} {'forget_r1':>10} {'retain_r1':>10} {'score (r-f)':>12}")
    print(f"  {'─'*44}")
    layer_scores = []
    for e in PHASE1_EXPERIMENTS:
        r = all_results.get(e["name"])
        l = e["display"]["layers"][0]
        if r and "error" not in r:
            f1 = r["forget"][0]; r1 = r["retain"][0]
            layer_scores.append((l, f1, r1))
            print(f"  Layer {l:<5} {f1:>10.4f} {r1:>10.4f} {r1-f1:>+12.4f}")
        else:
            print(f"  Layer {l:<5}  {'FAILED':>10}")

    # ── Select best layers ───────────────────────────────────────
    best_layers = select_best_layers(PHASE1_EXPERIMENTS, all_results)
    print(f"\n  ★  Best layer(s) selected for Phase 2: {best_layers}")
    for l in best_layers:
        match = next((s for s in layer_scores if s[0] == l), None)
        if match:
            _, f1, r1 = match
            print(f"     Layer {l}: forget={f1:.4f}, retain={r1:.4f}, "
                  f"score={r1-f1:+.4f}")

    # ── Phase 2: Parameter Tuning ────────────────────────────────
    phase2_exps = build_phase2(best_layers)

    print(f"\n{'█'*65}")
    print(f"  PHASE 2  —  Parameter Tuning  ({len(phase2_exps)} experiments)")
    print(f"  Using layer(s): {best_layers}")
    print(f"{'█'*65}")

    run_experiments(phase2_exps, "Phase 2: Parameter Tuning", all_results)
    all_exps.extend(phase2_exps)

    # ── Final output ─────────────────────────────────────────────
    total_time = str(timedelta(seconds=int(time.time() - sweep_start)))
    print(f"\n\nAll done in {total_time}.  "
          f"{len(all_exps)} experiments total "
          f"({len(PHASE1_EXPERIMENTS)} phase-1 + {len(phase2_exps)} phase-2).")

    print_table(all_exps, all_results)
    save_csv(all_exps, all_results)

    # ── Auto-push results to remote ──────────────────────────────
    commit_msg = (
        f"sweep results: {len(all_exps)} experiments "
        f"({len(PHASE1_EXPERIMENTS)} phase-1 + {len(all_exps)-len(PHASE1_EXPERIMENTS)} phase-2) "
        f"— {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    print(f"\n── Pushing results to git ───────────────────────────────────")
    git_push_results(commit_msg)
    print("  Done.")


if __name__ == "__main__":
    main()
