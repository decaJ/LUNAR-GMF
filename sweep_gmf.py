#!/usr/bin/env python3
"""
GMF v1 Targeted Parameter Sweep
================================
Goal: forget_rouge1 ≤ 0.19  AND  retain_rouge1 ≥ 0.90

Informed by prior sweep (sweep_results.csv):
  • sigma=0.5  → best retain preservation (C_c3_L15_s0.5: retain=0.888, forget=0.201)
  • sigma=0.7  → best forget suppression  (G3: forget=0.181, retain=0.840)
  • lambda_retain=2-4  → boosts retain without wrecking forget
  • dual-layer [15,X]  → forget↓↓ but retain drops to 0.7x
  Tension: no single experiment yet achieves both targets simultaneously.

Group layout (39 experiments total)
  A  sigma=0.5 sweep on L15               (6 exp)
  B  sigma=0.4 sweep on L15               (4 exp)
  C  sigma=0.7 + high lambda_retain L15   (5 exp)
  D  sigma=0.5/0.7 on L17/L19/L20/L21    (12 exp)
  E  dual-layer combos + high lambda_ret  (8 exp)
  F  epoch=30 on best candidate combos    (4 exp)

Hardware: 2× A100 → NUM_PARALLEL=3, GPUS=[0,1,1]
  cuda:0 runs 1 experiment per group
  cuda:1 runs 2 experiments per group

Usage:
    cd /root/workspace/LUNAR-GMF
    python sweep_gmf.py

Output:
    sweep_logs/<name>.log
    run_results/completions/llama2-7b/sweep/<name>/tofu_full/gmf_<layer>.json
    sweep_results.csv
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
# ══════════════════════════════════════════════════════════════════

GPU_CYCLE = (GPUS * 20)[:NUM_PARALLEL]   # [0,1,1]


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
# Experiment list  (39 experiments)
# ──────────────────────────────────────────────────────────────────
#
# Reading key:
#   c=coeff  s=sigma  a=lambda_attractor  lr=lambda_retain  e=epochs
#   L15 / L17_19 = layer(s)
#
# Prior baseline results for reference:
#   C_c3_L15_s0.5  forget=0.201  retain=0.888   ← sigma=0.5 champion
#   G3_c4_L15_s0.7_a2_lr2  forget=0.181  retain=0.840  ← sigma=0.7 champion

EXPERIMENTS = [

    # ── Group A: sigma=0.5, layer 15  (forget↓, retain best-preserved)
    # Baseline already run: C_c3_L15_s0.5 (forget=0.201, retain=0.888)
    # Goal: push forget below 0.19 while keeping retain ≥ 0.90
    exp("A1_c4_L15_s0.5",            [15], 4.0, sigma=0.5),
    exp("A2_c4_L15_s0.5_lr2",        [15], 4.0, sigma=0.5, lambda_retain=2.0),
    exp("A3_c3_L15_s0.5_lr2",        [15], 3.0, sigma=0.5, lambda_retain=2.0),
    exp("A4_c4_L15_s0.5_a2_lr2",     [15], 4.0, sigma=0.5, lambda_a=2.0, lambda_retain=2.0),
    exp("A5_c3_L15_s0.5_a2",         [15], 3.0, sigma=0.5, lambda_a=2.0),
    exp("A6_c5_L15_s0.5_lr2",        [15], 5.0, sigma=0.5, lambda_retain=2.0),

    # ── Group B: sigma=0.4, layer 15  (even tighter gate → forget↓↓)
    exp("B1_c3_L15_s0.4",            [15], 3.0, sigma=0.4),
    exp("B2_c3_L15_s0.4_lr2",        [15], 3.0, sigma=0.4, lambda_retain=2.0),
    exp("B3_c4_L15_s0.4_lr2",        [15], 4.0, sigma=0.4, lambda_retain=2.0),
    exp("B4_c4_L15_s0.4_a2_lr2",     [15], 4.0, sigma=0.4, lambda_a=2.0, lambda_retain=2.0),

    # ── Group C: sigma=0.7, high lambda_retain, layer 15
    # Baseline G3: forget=0.181, retain=0.840  → push retain to 0.90
    exp("C1_c4_L15_s0.7_a2_lr3",     [15], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),
    exp("C2_c4_L15_s0.7_a2_lr4",     [15], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=4.0),
    exp("C3_c3_L15_s0.7_a2_lr3",     [15], 3.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),
    exp("C4_c5_L15_s0.7_a2_lr3",     [15], 5.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),
    exp("C5_c4_L15_s0.7_a3_lr3",     [15], 4.0, sigma=0.7, lambda_a=3.0, lambda_retain=3.0),

    # ── Group D: promising combos on other layers (17 / 19 / 20 / 21)
    # Phase-1 results: L17 retain=0.901, L19 retain=0.916, L20 retain=0.929, L21 retain=0.928
    # These layers preserve retain better — pair with sigma↓ to suppress forget
    exp("D1_c3_L17_s0.5_lr2",        [17], 3.0, sigma=0.5, lambda_retain=2.0),
    exp("D2_c4_L17_s0.5_lr2",        [17], 4.0, sigma=0.5, lambda_retain=2.0),
    exp("D3_c4_L17_s0.7_a2_lr2",     [17], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0),
    exp("D4_c4_L17_s0.7_a2_lr3",     [17], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),

    exp("D5_c3_L19_s0.5_lr2",        [19], 3.0, sigma=0.5, lambda_retain=2.0),
    exp("D6_c4_L19_s0.5_lr2",        [19], 4.0, sigma=0.5, lambda_retain=2.0),
    exp("D7_c4_L19_s0.7_a2_lr2",     [19], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0),
    exp("D8_c4_L19_s0.7_a2_lr3",     [19], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),

    exp("D9_c3_L20_s0.5_lr2",        [20], 3.0, sigma=0.5, lambda_retain=2.0),
    exp("D10_c4_L20_s0.5_lr2",       [20], 4.0, sigma=0.5, lambda_retain=2.0),
    exp("D11_c4_L20_s0.7_a2_lr2",    [20], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0),

    exp("D12_c4_L21_s0.7_a2_lr2",    [21], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0),

    # ── Group E: dual-layer + sigma=0.5/0.7 + high lambda_retain
    # Single-layer [15] alone can't get forget far below 0.18.
    # Dual-layer adds forget pressure; sigma↓ + lr↑ fight retain drop.
    exp("E1_c3_L15_17_s0.7_lr2",     [15, 17], 3.0, sigma=0.7, lambda_retain=2.0),
    exp("E2_c4_L15_17_s0.7_lr2",     [15, 17], 4.0, sigma=0.7, lambda_retain=2.0),
    exp("E3_c4_L15_17_s0.7_a2_lr3",  [15, 17], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0),

    exp("E4_c3_L15_19_s0.5_lr2",     [15, 19], 3.0, sigma=0.5, lambda_retain=2.0),
    exp("E5_c4_L15_19_s0.5_lr2",     [15, 19], 4.0, sigma=0.5, lambda_retain=2.0),
    exp("E6_c4_L15_19_s0.5_a2_lr2",  [15, 19], 4.0, sigma=0.5, lambda_a=2.0, lambda_retain=2.0),

    exp("E7_c3_L17_19_s0.7_lr2",     [17, 19], 3.0, sigma=0.7, lambda_retain=2.0),
    exp("E8_c4_L17_19_s0.7_a2_lr2",  [17, 19], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=2.0),

    # ── Group F: epochs=30 on the most promising parameter combos
    # Longer training may let the flow network converge to tighter forget suppression
    exp("F1_c4_L15_s0.5_lr2_e30",    [15], 4.0, sigma=0.5, lambda_retain=2.0, num_epochs=30),
    exp("F2_c4_L15_s0.7_a2_lr3_e30", [15], 4.0, sigma=0.7, lambda_a=2.0, lambda_retain=3.0,
                                      num_epochs=30),
    exp("F3_c4_L17_s0.5_lr2_e30",    [17], 4.0, sigma=0.5, lambda_retain=2.0, num_epochs=30),
    exp("F4_c4_L19_s0.5_lr2_e30",    [19], 4.0, sigma=0.5, lambda_retain=2.0, num_epochs=30),
]


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
                hit = ("★ HIT" if r["forget"][0] <= 0.19 and r["retain"][0] >= 0.90
                       else "")
                print(f"   {status} {e['name']:<50}  "
                      f"forget={r['forget'][0]:.4f}  "
                      f"retain={r['retain'][0]:.4f}  "
                      f"factual={r['factual'][0]:.4f}  [{elapsed}] {hit}")
            else:
                print(f"   {status} {e['name']:<50}  "
                      f"FAILED  (see {LOG_DIR}/{e['name']}.log)")

        elapsed_total = str(timedelta(seconds=int(time.time() - phase_start)))
        print(f"\n   Group {g_idx+1} done.  Elapsed: {elapsed_total}\n")


# ──────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────

def print_table(experiments, results):
    W = 148
    print("\n" + "=" * W)
    print("  FULL RESULTS TABLE  (★ = forget≤0.19 AND retain≥0.90)")
    print("=" * W)
    hdr = (f"{'Experiment':<52} {'Layers':<12} {'c':>4} "
           f"{'λ_a':>4} {'σ':>5} {'λ_r':>4} {'Ep':>3} │ "
           f"{'forget_r1':>10} {'retain_r1':>10} {'factual_r1':>11} {'score':>8} {'':>6}")
    print(hdr)
    print("-" * W)
    hits = []
    for e in experiments:
        p = e["display"]
        r = results.get(e["name"])
        ls = str(p["layers"])
        if r and "error" not in r:
            f1 = r["forget"][0]; r1 = r["retain"][0]; a1 = r["factual"][0]
            score = r1 - f1
            tag = "★ HIT" if f1 <= 0.19 and r1 >= 0.90 else ""
            if tag:
                hits.append(e["name"])
            print(f"{e['name']:<52} {ls:<12} {p['coeff']:>4} "
                  f"{p['lambda_attractor']:>4} {p['sigma']:>5} "
                  f"{p['lambda_retain']:>4} {p['num_epochs']:>3} │ "
                  f"{f1:>10.4f} {r1:>10.4f} {a1:>11.4f} {score:>+8.4f} {tag:>6}")
        else:
            print(f"{e['name']:<52} {ls:<12} {p['coeff']:>4} "
                  f"{p['lambda_attractor']:>4} {p['sigma']:>5} "
                  f"{p['lambda_retain']:>4} {p['num_epochs']:>3} │  FAILED")
    print("=" * W)
    if hits:
        print(f"\n  ★  {len(hits)} experiment(s) hit target (forget≤0.19 & retain≥0.90):")
        for h in hits:
            print(f"     {h}")
    else:
        print("\n  No experiments hit both targets yet.")
        # Show closest misses
        candidates = []
        for e in experiments:
            r = results.get(e["name"])
            if r and "error" not in r:
                f1, r1 = r["forget"][0], r["retain"][0]
                miss = max(f1 - 0.19, 0) + max(0.90 - r1, 0)
                candidates.append((miss, e["name"], f1, r1))
        candidates.sort()
        print("  Closest misses:")
        for miss, name, f1, r1 in candidates[:5]:
            print(f"     {name:<52}  forget={f1:.4f}  retain={r1:.4f}  Δ={miss:+.4f}")


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
    result = subprocess.run(["git"] + args, cwd=cwd, capture_output=capture, text=True)
    output = (result.stdout + result.stderr).strip()
    return result.returncode, output


def verify_git_push():
    print("\n── Git pre-flight check ────────────────────────────────────")

    rc, out = _git(["rev-parse", "--is-inside-work-tree"])
    if rc != 0:
        print(f"  ✗ Not inside a git repository: {out}")
        raise SystemExit(1)
    print("  ✓ Git repository detected.")

    rc, out = _git(["remote", "get-url", "origin"])
    if rc != 0:
        print("  ✗ No remote 'origin' configured.")
        raise SystemExit(1)
    print(f"  ✓ Remote origin: {out}")

    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    print(f"  ✓ Branch: {branch}")

    print("  Attempting dry-run push ... ", end="", flush=True)
    rc, out = _git(["push", "--dry-run", "origin", branch])
    if rc != 0:
        print("FAILED")
        print(f"  ✗ {out}")
        raise SystemExit(1)
    print("OK")
    print("  ✓ Git push pre-flight passed.\n")
    return True


def git_push_results(message: str):
    print("\n── Pushing results to remote ───────────────────────────────")
    for pattern in ["sweep_results.csv", "sweep_logs/",
                    "run_results/completions/llama2-7b/sweep/"]:
        _git(["add", "-f", pattern])
    rc, status = _git(["status", "--porcelain"])
    if not status.strip():
        print("  Nothing new to commit.")
        return
    rc, out = _git(["commit", "-m", message])
    if rc != 0:
        print(f"  ✗ Commit failed: {out}")
        return
    print(f"  ✓ Committed: {message}")
    _, branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    rc, out = _git(["push", "origin", branch])
    if rc != 0:
        print(f"  ✗ Push failed: {out}")
    else:
        print(f"  ✓ Pushed to origin/{branch}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(WORK_DIR, LOG_DIR), exist_ok=True)

    verify_git_push()

    sweep_start = time.time()
    all_results = {}

    print("\n" + "█" * 65)
    print(f"  TARGETED SWEEP  —  {len(EXPERIMENTS)} experiments")
    print("  Target: forget_rouge1 ≤ 0.19  AND  retain_rouge1 ≥ 0.90")
    print("█" * 65)

    run_experiments(EXPERIMENTS, "Targeted Sweep", all_results)

    total_time = str(timedelta(seconds=int(time.time() - sweep_start)))
    print(f"\n\nAll done in {total_time}.  {len(EXPERIMENTS)} experiments.")

    print_table(EXPERIMENTS, all_results)
    save_csv(EXPERIMENTS, all_results)

    commit_msg = (
        f"targeted sweep: {len(EXPERIMENTS)} exps, goal forget≤0.19 retain≥0.90 "
        f"— {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    git_push_results(commit_msg)
    print("  Done.")


if __name__ == "__main__":
    main()
