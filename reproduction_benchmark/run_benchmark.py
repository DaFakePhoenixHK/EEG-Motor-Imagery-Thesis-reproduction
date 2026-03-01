#!/usr/bin/env python3
"""
Reproduction benchmark runner: BCI IV-2a protocols W, L, F, TTA.
Usage:
  python run_benchmark.py --data "/path/to/Four class motor imagery (001-2014)" --protocol W --model eegnetv4 --channels 8 --seed 0
"""
import os
import sys
import argparse
import csv
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
_PARENT = _BENCH.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from config import (
    DEFAULT_BCI2A_PATH, DEFAULT_RESULTS_DIR, K_PER_CLASS_GRID,
    MATRIX_8CH, MATRIX_22CH, DEFAULT_EPOCHS, SEEDS,
)
from protocols import protocol_W, protocol_L, protocol_F, protocol_TTA


def _normalize_path(p):
    if not p:
        return p
    p = os.path.expanduser(str(p))
    if os.path.sep == "/" and len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/").lstrip("/")
        return os.path.normpath(f"/mnt/{drive}/{rest}")
    return os.path.normpath(p.replace("\\", os.path.sep))


def _is_allowed(ch_label, protocol, model_name):
    """Check if (channels, protocol, model) is allowed per experiment matrix."""
    matrix = MATRIX_8CH if ch_label == "8ch" else MATRIX_22CH
    if protocol not in matrix:
        return False
    return model_name in matrix[protocol]


def run_protocol(protocol, data_path, n_channels, model_name, seed, k_per_class=None, epochs=None, batch_size=64):
    epochs = epochs if epochs is not None else DEFAULT_EPOCHS
    if protocol == "W":
        return protocol_W(data_path, n_channels, model_name, seed, epochs, batch_size)
    if protocol == "L":
        return protocol_L(data_path, n_channels, model_name, seed, epochs, batch_size)
    if protocol == "F":
        if k_per_class is None:
            k_per_class = K_PER_CLASS_GRID[0]
        return protocol_F(data_path, n_channels, model_name, seed, k_per_class, epochs, batch_size)
    if protocol == "TTA":
        return protocol_TTA(data_path, n_channels, model_name, seed, epochs, batch_size)
    raise ValueError(f"Unknown protocol: {protocol}")


def save_results(results_dir, ch_label, protocol, model_name, seed, results, k_per_class=None):
    subdir = results_dir / "bci2a" / "accuracy" / ch_label / protocol / model_name / f"seed_{seed}"
    if k_per_class is not None and protocol == "F":
        subdir = subdir / f"K{k_per_class}"
    subdir.mkdir(parents=True, exist_ok=True)

    # subjectwise.csv: exclude confusion_matrix (2D) and history (large dict)
    rows = [{k: v for k, v in r.items() if k not in ("confusion_matrix", "history")} for r in results]
    subjectwise_path = subdir / "subjectwise.csv"
    with open(subjectwise_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "trialAcc", "macroF1", "kappa", "ITR"])
        w.writeheader()
        w.writerows(rows)

    # confusion_{subject}.csv per subject
    for r in results:
        if "confusion_matrix" in r:
            cm = r["confusion_matrix"]
            subj = r.get("subject", "unknown")
            cm_path = subdir / f"confusion_{subj}.csv"
            np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    # training curves per subject (if history available)
    try:
        from main_TrainValTest import save_training_curves
        for r in results:
            if "history" in r and r["history"]:
                subj = r.get("subject", "unknown")
                save_training_curves(r["history"], str(subdir), f"subject_{subj}")
    except Exception:
        pass

    accs = [r["trialAcc"] for r in results]
    f1s = [r["macroF1"] for r in results]
    summary = {
        "mean_trialAcc": float(np.mean(accs)),
        "std_trialAcc": float(np.std(accs)),
        "median_trialAcc": float(np.median(accs)),
        "iqr_trialAcc": float(np.percentile(accs, 75) - np.percentile(accs, 25)),
        "mean_macroF1": float(np.mean(f1s)),
        "std_macroF1": float(np.std(f1s)),
    }
    summary_path = subdir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    return subdir, summary


def main():
    parser = argparse.ArgumentParser(description="BCI IV-2a reproduction benchmark")
    parser.add_argument("--data", type=str, default=str(DEFAULT_BCI2A_PATH), help="Path to BCI IV-2a folder")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--protocol", type=str, required=True, choices=["W", "L", "F", "TTA"])
    parser.add_argument("--model", type=str, required=True, choices=["eegnetv4", "shallow", "deep4", "conformer", "fbcsp_lda", "db_atcnet"])
    parser.add_argument("--channels", type=str, default="8", choices=["8", "22"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k_per_class", type=int, default=None, help="For Protocol F only; default first in grid")
    parser.add_argument("--epochs", type=int, default=None, help=f"Max epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_all_protocols", action="store_true", help="Run W,L,F,TTA in one call")
    parser.add_argument("--run_all_k", action="store_true", help="For Protocol F: run K=1,5,10,20")
    parser.add_argument("--run_all_seeds", action="store_true", help=f"Run seeds {SEEDS} per MD 5.1")
    args = parser.parse_args()

    data_path = _normalize_path(args.data)
    if not os.path.isdir(data_path):
        print(f"ERROR: --data must be an existing directory. Got: {data_path}")
        sys.exit(1)

    results_dir = Path(_normalize_path(args.results_dir)) if args.results_dir else DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    n_channels = int(args.channels)
    ch_label = f"{n_channels}ch"

    protocols_to_run = ["W", "L", "F", "TTA"] if args.run_all_protocols else [args.protocol]
    k_list = K_PER_CLASS_GRID if args.run_all_k else [args.k_per_class or K_PER_CLASS_GRID[0]]
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS
    seeds_to_run = SEEDS if args.run_all_seeds else [args.seed]

    for seed in seeds_to_run:
        for protocol in protocols_to_run:
            if not _is_allowed(ch_label, protocol, args.model):
                print(f"[SKIP] {ch_label}/{protocol}/{args.model} not in experiment matrix")
                continue
            if protocol == "F":
                for k in k_list:
                    print(f"[{protocol}] model={args.model} channels={n_channels} seed={seed} K={k}")
                    results = run_protocol(protocol, data_path, n_channels, args.model, seed, k, epochs, args.batch_size)
                    subdir, summary = save_results(results_dir, ch_label, protocol, args.model, seed, results, k)
                    print(f"  mean_trialAcc: {summary['mean_trialAcc']:.4f} ± {summary['std_trialAcc']:.4f}")
                    print(f"  -> {subdir}")
            else:
                print(f"[{protocol}] model={args.model} channels={n_channels} seed={seed}")
                results = run_protocol(protocol, data_path, n_channels, args.model, seed, None, epochs, args.batch_size)
                subdir, summary = save_results(results_dir, ch_label, protocol, args.model, seed, results)
                print(f"  mean_trialAcc: {summary['mean_trialAcc']:.4f} ± {summary['std_trialAcc']:.4f}")
                print(f"  -> {subdir}")

    print("Done.")


if __name__ == "__main__":
    main()
