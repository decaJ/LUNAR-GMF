#!/usr/bin/env python3
"""
GMF v1 Parameter Sweep
======================
Runs GMF v1 unlearning experiments in parallel groups and prints a result table.

Hardware assumed: 2× A100  →  NUM_PARALLEL=4, GPUS=[0,0,1,1]
  Each A100 runs 2 experiments simultaneously.
  Change NUM_PARALLEL / GPUS at the top if your setup differs.

Usage:
    python sweep_gmf.py

Output:
    • Per-experiment logs  →  sweep_logs/<name>.log
    • Results JSON         →  run_results/completions/llama2-7b/sweep/<name>/tofu_full/gmf_<layer>.json
    • Summary CSV          →  sweep_results.csv
    • Summary table        →  printed to stdout when all experiments finish
"""

import os, sys, json, time, subprocess, csv
from pathlib import Path
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════════
#  ★  SETTINGS  —  edit before running
# ══════════════════════════════════════════════════════════════════
WORK_DIR     = "/root/workspace/LUNAR-GMF"   # absolute path to repo on A100 machine
NUM_PARALLEL = 4                              # experiments per group (2 per A100)
GPUS         = [0, 0, 1, 1]                 # GPU IDs; 2 experiments share each A100
SCRIPT       = "run_gmf.py"
CONFIG       = "forget_gmf_tofu"
LOG_DIR      = "sweep_logs"
# ══════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────
# Experiment builder
# ──────────────────────────────────────────────────────────────────

def exp(name, layers, coeff,
        lambda_a=1.0, sigma=1.0, attractor_scale=1.0,
        lambda_flow=0.1, num_epochs=20, lambda_retain=1.0):
    n = len(layers)
    layer_str  = "[" + ",".join(map(str, layers)) + "]"
    coeff_str  = "[" + ",".join([f"+{coeff}"] * n) + "]"
    # Unique save path so experiments never overwrite each other
    save_path  = f"run_results/completions/llama2-7b/sweep/{name}/tofu_full"
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
# All experiments  (35 total → 18 groups of 2)
# ──────────────────────────────────────────────────────────────────

EXPERIMENTS = [

    # ── Group A: coeff variation, single layer 19 ───────────────
    exp("A1_c2_L19",              [19],        coeff=2.0),
    exp("A2_c3_L19",              [19],        coeff=3.0),
    exp("A3_c4_L19",              [19],        coeff=4.0),
    exp("A4_c5_L19",              [19],        coeff=5.0),
    exp("A5_c6_L19",              [19],        coeff=6.0),

    # ── Group B: multi-layer coverage ───────────────────────────
    exp("B1_c2_L19_22",           [19,22],     coeff=2.0),
    exp("B2_c3_L19_22",           [19,22],     coeff=3.0),
    exp("B3_c2_L15_19_22",        [15,19,22],  coeff=2.0),
    exp("B4_c3_L15_19_22",        [15,19,22],  coeff=3.0),
    exp("B5_c4_L15_19_22",        [15,19,22],  coeff=4.0),

    # ── Group C: attractor strength ─────────────────────────────
    exp("C1_c3_L19_a2",           [19],        coeff=3.0, lambda_a=2.0),
    exp("C2_c4_L19_a2",           [19],        coeff=4.0, lambda_a=2.0),
    exp("C3_c3_L19_as1.5",        [19],        coeff=3.0, attractor_scale=1.5),
    exp("C4_c3_L19_as2",          [19],        coeff=3.0, attractor_scale=2.0),
    exp("C5_c4_L15_19_22_a2",     [15,19,22],  coeff=4.0, lambda_a=2.0),

    # ── Group D: gate bandwidth (sigma) ─────────────────────────
    exp("D1_c3_L19_s0.7",         [19],        coeff=3.0, sigma=0.7),
    exp("D2_c3_L19_s0.5",         [19],        coeff=3.0, sigma=0.5),
    exp("D3_c4_L19_s0.7",         [19],        coeff=4.0, sigma=0.7),
    exp("D4_c3_L19_22_s0.7",      [19,22],     coeff=3.0, sigma=0.7),
    exp("D5_c4_L15_19_22_s0.7",   [15,19,22],  coeff=4.0, sigma=0.7),

    # ── Group E: flow regularisation ────────────────────────────
    exp("E1_c3_L19_lf0.05",       [19],        coeff=3.0, lambda_flow=0.05),
    exp("E2_c4_L19_lf0.05",       [19],        coeff=4.0, lambda_flow=0.05),
    exp("E3_c3_L15_19_22_lf0.05", [15,19,22],  coeff=3.0, lambda_flow=0.05),

    # ── Group F: more training epochs ───────────────────────────
    exp("F1_c3_L19_e30",          [19],        coeff=3.0, num_epochs=30),
    exp("F2_c4_L19_e30",          [19],        coeff=4.0, num_epochs=30),
    exp("F3_c3_L15_19_22_e30",    [15,19,22],  coeff=3.0, num_epochs=30),

    # ── Group G: combined best-guess settings ───────────────────
    exp("G1_c3_L19_s0.7_a2",      [19],        coeff=3.0, sigma=0.7, lambda_a=2.0),
    exp("G2_c4_L19_s0.7_a2",      [19],        coeff=4.0, sigma=0.7, lambda_a=2.0),
    exp("G3_c3_L19_22_s0.7_a2",   [19,22],     coeff=3.0, sigma=0.7, lambda_a=2.0),
    exp("G4_c4_L19_22_s0.7_a2",   [19,22],     coeff=4.0, sigma=0.7, lambda_a=2.0),
    exp("G5_c3_L15_19_22_s0.7_a2",[15,19,22],  coeff=3.0, sigma=0.7, lambda_a=2.0),

    # ── Group H: aggressive forget + protect retain ──────────────
    exp("H1_c4_L15_19_22_s0.7_a2_lf0.05",
                                  [15,19,22],  coeff=4.0, sigma=0.7, lambda_a=2.0,
                                               lambda_flow=0.05),
    exp("H2_c5_L19_s0.7_a2",      [19],        coeff=5.0, sigma=0.7, lambda_a=2.0),
    exp("H3_c4_L19_as2_s0.7",     [19],        coeff=4.0, attractor_scale=2.0, sigma=0.7),
    exp("H4_c3_L15_19_22_a2_e30", [15,19,22],  coeff=3.0, lambda_a=2.0, num_epochs=30),
    exp("H5_c4_L19_22_a2_e30",    [19,22],     coeff=4.0, lambda_a=2.0, num_epochs=30),
]


# ──────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────

def launch(e, gpu_id, log_path):
    """Start one experiment as a background subprocess."""
    cmd = (
        ["python", SCRIPT, f"--config-name={CONFIG}"]
        + e["overrides"]
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    log_f = open(log_path, "w")
    proc  = subprocess.Popen(cmd, cwd=WORK_DIR, env=env,
                             stdout=log_f, stderr=subprocess.STDOUT)
    return proc, log_f


def read_result(e):
    """Parse the saved JSON and return (forget_r1, retain_r1, factual_r1, forget_ppl)."""
    path = e["result_file"]
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        def get(split):
            v = d.get(split, d.get("retained_edge" if split=="retain" else
                                   "factual_data"  if split=="factual" else split, {}))
            return (
                round(v.get("rouge1_recall", v.get("rouge1", float("nan"))), 4),
                round(v.get("rougeL_recall", v.get("rougeL", float("nan"))), 4),
                round(v.get("perplexity",    float("nan")), 1),
            )
        return {"forget": get("forget"), "retain": get("retain"), "factual": get("factual")}
    except Exception as ex:
        return {"error": str(ex)}


def fmt(val):
    return f"{val:.4f}" if isinstance(val, float) and val == val else "ERR"


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    log_dir = os.path.join(WORK_DIR, LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    total   = len(EXPERIMENTS)
    n_grps  = (total + NUM_PARALLEL - 1) // NUM_PARALLEL
    gpu_cycle = GPUS * ((NUM_PARALLEL // len(GPUS)) + 1)  # cycle GPUs

    print(f"\n{'='*60}")
    print(f"  GMF Sweep  |  {total} experiments  |  {NUM_PARALLEL} parallel")
    print(f"  GPUs: {GPUS}   |   {n_grps} groups")
    print(f"  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    sweep_start = time.time()
    results     = {}   # name → result dict

    for g_idx in range(n_grps):
        batch = EXPERIMENTS[g_idx * NUM_PARALLEL : (g_idx + 1) * NUM_PARALLEL]

        print(f"── Group {g_idx+1}/{n_grps} ─────────────────────────────────")
        for i, e in enumerate(batch):
            print(f"   [{i+1}] {e['name']}  (GPU {gpu_cycle[i]})")
        print()

        procs = []
        for i, e in enumerate(batch):
            log_path = os.path.join(log_dir, f"{e['name']}.log")
            proc, log_f = launch(e, gpu_cycle[i], log_path)
            procs.append((proc, log_f, e))

        # Wait for all in this batch
        group_start = time.time()
        for proc, log_f, e in procs:
            proc.wait()
            log_f.close()
            rc = proc.returncode
            r  = read_result(e)
            results[e["name"]] = r
            status = "✓" if r and "error" not in r else "✗"
            elapsed = str(timedelta(seconds=int(time.time()-group_start)))
            if r and "error" not in r:
                fr = r["forget"][0]; rr = r["retain"][0]; fa = r["factual"][0]
                print(f"   {status} {e['name']:<38}  "
                      f"forget={fr:.4f}  retain={rr:.4f}  factual={fa:.4f}  [{elapsed}]")
            else:
                print(f"   {status} {e['name']:<38}  FAILED  (see {LOG_DIR}/{e['name']}.log)")

        elapsed_total = str(timedelta(seconds=int(time.time()-sweep_start)))
        print(f"\n   Group {g_idx+1} done.  Total elapsed: {elapsed_total}\n")

    # ── Final table ────────────────────────────────────────────────
    print("\n" + "="*130)
    print("  RESULTS TABLE")
    print("="*130)

    hdr = (f"{'Experiment':<40} {'Layers':<14} {'Coeff':>5} {'λ_a':>5} "
           f"{'σ':>5} {'aScale':>7} {'λ_f':>5} {'Ep':>3} │ "
           f"{'forget_r1':>10} {'retain_r1':>10} {'factual_r1':>11} {'forget_ppl':>11}")
    print(hdr)
    print("-" * 130)

    csv_rows = []
    for e in EXPERIMENTS:
        p = e["display"]
        r = results.get(e["name"])
        layers_str = str(p["layers"])

        if r and "error" not in r:
            f_r1, f_rl, f_ppl = r["forget"]
            r_r1, r_rl, r_ppl = r["retain"]
            a_r1, a_rl, a_ppl = r["factual"]
            row = (f"{e['name']:<40} {layers_str:<14} {p['coeff']:>5} "
                   f"{p['lambda_attractor']:>5} {p['sigma']:>5} "
                   f"{p['attractor_scale']:>7} {p['lambda_flow']:>5} "
                   f"{p['num_epochs']:>3} │ "
                   f"{f_r1:>10.4f} {r_r1:>10.4f} {a_r1:>11.4f} {f_ppl:>11.1f}")
            csv_rows.append([e["name"], layers_str,
                              p["coeff"], p["lambda_attractor"], p["sigma"],
                              p["attractor_scale"], p["lambda_flow"], p["num_epochs"],
                              f_r1, r_r1, a_r1, f_rl, r_rl, a_rl, f_ppl, r_ppl, a_ppl])
        else:
            row = (f"{e['name']:<40} {layers_str:<14} {p['coeff']:>5} "
                   f"{p['lambda_attractor']:>5} {p['sigma']:>5} "
                   f"{p['attractor_scale']:>7} {p['lambda_flow']:>5} "
                   f"{p['num_epochs']:>3} │  FAILED")
            csv_rows.append([e["name"], layers_str,
                              p["coeff"], p["lambda_attractor"], p["sigma"],
                              p["attractor_scale"], p["lambda_flow"], p["num_epochs"],
                              "FAIL","FAIL","FAIL","FAIL","FAIL","FAIL","FAIL","FAIL","FAIL"])
        print(row)

    print("="*130)
    total_time = str(timedelta(seconds=int(time.time()-sweep_start)))
    print(f"\nAll done in {total_time}.  {len(EXPERIMENTS)} experiments completed.")

    # ── Save CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(WORK_DIR, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name","layers","coeff","lambda_attractor","sigma",
                    "attractor_scale","lambda_flow","num_epochs",
                    "forget_rouge1","retain_rouge1","factual_rouge1",
                    "forget_rougeL","retain_rougeL","factual_rougeL",
                    "forget_ppl","retain_ppl","factual_ppl"])
        w.writerows(csv_rows)
    print(f"CSV saved to: {csv_path}\n")


if __name__ == "__main__":
    main()
