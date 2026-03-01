#!/usr/bin/env python3
"""
Aggregate accuracy results into global rollups per reproduce_plan_for_Nathan.md Section 8.2.
Outputs: results/bci2a/results_summary_acc.csv, results_subjectwise.csv
Run after run_benchmark.py has produced subjectwise.csv and summary.csv in subdirs.
"""
import os
import sys
import csv
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from config import DEFAULT_RESULTS_DIR, K_PER_CLASS_GRID, SEEDS


def _collect_subdirs(results_dir):
    """Find all accuracy subdirs: .../accuracy/{8ch|22ch}/{W|L|F|TTA}/{model}/seed_{s}/[K{k}/]"""
    base = Path(results_dir) / "bci2a" / "accuracy"
    if not base.exists():
        return []
    subdirs = []
    for ch in ("8ch", "22ch"):
        ch_dir = base / ch
        if not ch_dir.exists():
            continue
        for protocol in ("W", "L", "F", "TTA"):
            p_dir = ch_dir / protocol
            if not p_dir.exists():
                continue
            for model_dir in p_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model_name = model_dir.name
                for seed_dir in model_dir.glob("seed_*"):
                    if not seed_dir.is_dir():
                        continue
                    seed_str = seed_dir.name.replace("seed_", "")
                    try:
                        seed = int(seed_str)
                    except ValueError:
                        continue
                    sw_path = seed_dir / "subjectwise.csv"
                    if sw_path.exists():
                        subdirs.append({
                            "channels": ch,
                            "protocol": protocol,
                            "model": model_name,
                            "seed": seed,
                            "k_per_class": None,
                            "path": seed_dir,
                        })
                    # Protocol F may have K subdirs
                    for k_dir in seed_dir.iterdir():
                        if k_dir.is_dir() and k_dir.name.startswith("K"):
                            sw_k = k_dir / "subjectwise.csv"
                            if sw_k.exists():
                                try:
                                    k_val = int(k_dir.name[1:])
                                except ValueError:
                                    continue
                                subdirs.append({
                                    "channels": ch,
                                    "protocol": "F",
                                    "model": model_name,
                                    "seed": seed,
                                    "k_per_class": k_val,
                                    "path": k_dir,
                                })
    return subdirs


def _read_subjectwise(path):
    """Read subjectwise.csv into list of dicts."""
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: (float(v) if k in ("trialAcc", "macroF1", "kappa", "ITR") else v) for k, v in row.items()})
    return rows


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))
    subdirs = _collect_subdirs(results_dir)
    if not subdirs:
        print(f"No subjectwise.csv found under {results_dir}/bci2a/accuracy/")
        return

    summary_rows = []
    subjectwise_rows = []

    for sd in subdirs:
        sw_path = sd["path"] / "subjectwise.csv"
        rows = _read_subjectwise(sw_path)
        if not rows:
            continue
        ch, proto, model, seed = sd["channels"], sd["protocol"], sd["model"], sd["seed"]
        k = sd["k_per_class"]
        accs = [r["trialAcc"] for r in rows]
        f1s = [r["macroF1"] for r in rows]
        summary_rows.append({
            "channels": ch,
            "protocol": proto,
            "model": model,
            "seed": seed,
            "k_per_class": k if k is not None else "",
            "mean_trialAcc": float(np.mean(accs)),
            "std_trialAcc": float(np.std(accs)),
            "median_trialAcc": float(np.median(accs)),
            "iqr_trialAcc": float(np.percentile(accs, 75) - np.percentile(accs, 25)),
            "mean_macroF1": float(np.mean(f1s)),
            "std_macroF1": float(np.std(f1s)),
        })
        for r in rows:
            subjectwise_rows.append({
                "subject": r.get("subject", ""),
                "channels": ch,
                "protocol": proto,
                "model": model,
                "seed": seed,
                "k_per_class": k if k is not None else "",
                "trialAcc": r.get("trialAcc", ""),
                "macroF1": r.get("macroF1", ""),
                "kappa": r.get("kappa", ""),
                "ITR": r.get("ITR", ""),
            })

    out_dir = results_dir / "bci2a"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "results_summary_acc.csv"
    summary_fields = ["channels", "protocol", "model", "seed", "k_per_class", "mean_trialAcc", "std_trialAcc", "median_trialAcc", "iqr_trialAcc", "mean_macroF1", "std_macroF1"]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Wrote {summary_path} ({len(summary_rows)} rows)")

    subj_path = out_dir / "results_subjectwise.csv"
    subj_fields = ["subject", "channels", "protocol", "model", "seed", "k_per_class", "trialAcc", "macroF1", "kappa", "ITR"]
    with open(subj_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=subj_fields)
        w.writeheader()
        w.writerows(subjectwise_rows)
    print(f"Wrote {subj_path} ({len(subjectwise_rows)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
