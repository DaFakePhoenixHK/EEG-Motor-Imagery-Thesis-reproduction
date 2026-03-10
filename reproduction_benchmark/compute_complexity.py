#!/usr/bin/env python3
"""
Compute complexity metrics for all models (8ch and 22ch) per reproduce_plan_for_Nathan.md Section 6.
Outputs: results/bci2a/complexity/complexity_8ch.csv, complexity_22ch.csv
Columns: model, n_chans, n_times, params, macs_or_flops, peak_act_mem_bytes, pc_latency_ms, notes
"""
import os
import sys
import csv
import time
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
_PARENT = _BENCH.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from config import DEFAULT_RESULTS_DIR, IN_SAMPLES, MODELS


def _count_params_keras(model):
    """Trainable params for Keras model."""
    return int(model.count_params())


def _count_params_fbcsp():
    """Approximate params for FBCSP+LDA (sklearn)."""
    return -1  # N/A for sklearn


def _get_model_for_complexity(name, n_channels, n_times, n_classes=4):
    """Build model without training."""
    if name == "fbcsp_lda":
        return None
    from models_registry import get_model
    return get_model(name, n_channels, n_times, n_classes, seed=0)


def _measure_latency_keras(model, n_channels, n_times, n_warmup=5, n_repeat=50):
    """PC latency (ms) for one window, batch=1."""
    x = np.random.randn(1, 1, n_channels, n_times).astype(np.float32)
    for _ in range(n_warmup):
        _ = model.predict(x, verbose=0)
    start = time.perf_counter()
    for _ in range(n_repeat):
        _ = model.predict(x, verbose=0)
    elapsed = (time.perf_counter() - start) / n_repeat * 1000
    return round(elapsed, 3)


def _estimate_macs_flops(model, n_channels, n_times):
    """Rough estimate or placeholder. Full profiling needs tf.profiler/other tools."""
    try:
        import tensorflow as tf
        p = model.count_params()
        return p * 2
    except Exception:
        return -1


def _estimate_peak_activation_memory(model, n_channels, n_times):
    """Rough peak activation memory estimate (bytes). Placeholder."""
    return -1


def compute_one(model_name, n_channels, n_times):
    """Compute metrics for one model config."""
    model = _get_model_for_complexity(model_name, n_channels, n_times)
    if model is None:
        return {
            "model": model_name,
            "n_chans": n_channels,
            "n_times": n_times,
            "params": -1,
            "macs_or_flops": -1,
            "peak_act_mem_bytes": -1,
            "pc_latency_ms": -1,
            "notes": "sklearn FBCSP+LDA; N/A",
        }
    params = _count_params_keras(model)
    macs = _estimate_macs_flops(model, n_channels, n_times)
    peak_mem = _estimate_peak_activation_memory(model, n_channels, n_times)
    latency = _measure_latency_keras(model, n_channels, n_times)
    return {
        "model": model_name,
        "n_chans": n_channels,
        "n_times": n_times,
        "params": params,
        "macs_or_flops": macs,
        "peak_act_mem_bytes": peak_mem,
        "pc_latency_ms": latency,
        "notes": "batch=1",
    }


def main():
    results_dir = Path(os.environ.get("RESULTS_DIR", str(DEFAULT_RESULTS_DIR)))
    out_dir = results_dir / "bci2a" / "complexity"
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["model", "n_chans", "n_times", "params", "macs_or_flops", "peak_act_mem_bytes", "pc_latency_ms", "notes"]

    for n_ch in (8, 22):
        rows = []
        for m in MODELS:
            try:
                row = compute_one(m, n_ch, IN_SAMPLES)
                rows.append(row)
                print(f"  {m} {n_ch}ch: params={row['params']}, latency={row['pc_latency_ms']} ms")
            except Exception as e:
                rows.append({
                    "model": m, "n_chans": n_ch, "n_times": IN_SAMPLES,
                    "params": -1, "macs_or_flops": -1, "peak_act_mem_bytes": -1,
                    "pc_latency_ms": -1, "notes": str(e)[:80],
                })
                print(f"  {m} {n_ch}ch: ERROR {e}")
        path = out_dir / f"complexity_{n_ch}ch.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}")
    print("Done.")


if __name__ == "__main__":
    main()
