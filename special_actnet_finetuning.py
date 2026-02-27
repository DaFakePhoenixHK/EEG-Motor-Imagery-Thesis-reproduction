"""
ATCNet fine-tuning: pre-train on all subjects except one, then fine-tune on
fractions of that subject's train data and test on that subject's test data.

Steps:
  1. Pre-train ATCNet on all subjects except --test_subject (e.g. 1,3,4,5,6,7,8,9 if test_subject=2).
  2. Load the test subject's data; standardize with pre-train scalers.
  3. For each fraction: fine-tune until early stopping, then test on that subject's test set.
  Saves: pre-train + each fine-tune curves, confusion matrices, test metrics, and model weights.

Usage:
  python special_actnet_finetuning.py --data "/path/to/Four class motor imagery (001-2014)"
  python special_actnet_finetuning.py --data "/path/to/..." --test_subject 2   # train on 1,3-9; test/fine-tune on 2 (default)
  python special_actnet_finetuning.py --data "/path/to/..." --test_subject 1   # train on 2-9; test/fine-tune on 1
"""
import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))


def _normalize_path(path):
    if not path:
        return path
    path = os.path.expanduser(str(path))
    if os.path.sep == "/" and len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/").lstrip("/")
        return os.path.normpath(f"/mnt/{drive}/{rest}")
    return os.path.normpath(path.replace("\\", os.path.sep))


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
_log = logging.getLogger(__name__)

# Fine-tune fractions of target subject train data (5%, 15%, 20%, 25%)
FINETUNE_FRACTIONS = [0.05, 0.15, 0.20, 0.25]
SUBJECT_IDS = list(range(1, 10))  # BCI2a subjects 1-9
CLASS_LABELS = ["Left hand", "Right hand", "Foot", "Tongue"]
N_CLASSES = 4
N_CHANNELS = 22
IN_SAMPLES = 1125


def main():
    parser = argparse.ArgumentParser(description="Pre-train ATCNet on all subjects except one, fine-tune on % of that subject's train")
    parser.add_argument("--data", type=str, required=True, help="Path to raw BCI2a folder (e.g. 'Four class motor imagery (001-2014)' containing A01T.mat, A01E.mat, ...)")
    parser.add_argument("--test_subject", type=int, default=2, choices=SUBJECT_IDS, help="Subject to leave out for test/fine-tune (1-9). Pre-train on the other 8 subjects. Default: 2")
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--max_pretrain_epochs", type=int, default=500, help="Max epochs for pre-training (early stopping may stop earlier)")
    parser.add_argument("--max_finetune_epochs", type=int, default=300, help="Max epochs per fine-tune run (early stopping may stop earlier)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pretrain_lr", type=float, default=0.001)
    parser.add_argument("--finetune_lr", type=float, default=0.0003, help="Lower LR for fine-tuning")
    args = parser.parse_args()

    test_subject = args.test_subject
    pretrain_subject_ids = [s for s in SUBJECT_IDS if s != test_subject]

    data_path = _normalize_path(args.data)
    if not os.path.isdir(data_path):
        parser.error(f"--data must be an existing directory. Provided: {data_path!r}. Use the full path to the BCI2a folder (e.g. containing A01T.mat, A01E.mat).")
    results_dir = Path(_normalize_path(args.results_dir)) if args.results_dir else _here / "results_finetune"
    results_dir.mkdir(parents=True, exist_ok=True)

    from preprocess import get_data, standardize_fit_train_return_scalers, standardize_apply_scalers
    from main_TrainValTest import getModel, _make_early_stopping, save_training_curves, save_confusion_matrix

    # History callback for saving curves
    class HistoryTracker(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        def on_epoch_end(self, epoch, logs=None):
            if logs:
                for k in self.history:
                    if k in logs:
                        self.history[k].append(float(logs[k]))

    dataset = "BCI2a"
    dataset_conf = {
        "n_classes": N_CLASSES, "n_channels": N_CHANNELS, "in_samples": IN_SAMPLES,
        "cl_labels": ["Left hand", "Right hand", "Foot", "Tongue"],
    }

    # ---------- 1. Pre-train on all subjects except test_subject ----------
    _log.info("Pre-train on subjects %s (excluding subject %d for test/fine-tune)", pretrain_subject_ids, test_subject)
    X_list, y_list = [], []
    for sub_id in pretrain_subject_ids:
        sub_idx = sub_id - 1
        X_s, _, y_s_onehot, _, _, _ = get_data(data_path, sub_idx, dataset, isStandard=False)
        X_list.append(X_s)
        y_list.append(y_s_onehot)
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    X_pt_train, X_pt_val, y_pt_train, y_pt_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    n_ch = X_pt_train.shape[2]
    scalers = standardize_fit_train_return_scalers(X_pt_train, n_ch)
    standardize_apply_scalers(X_pt_train, scalers, n_ch)
    standardize_apply_scalers(X_pt_val, scalers, n_ch)

    _log.info("Pre-training ATCNet on subjects %s: train=%d, val=%d", pretrain_subject_ids, len(X_pt_train), len(X_pt_val))
    base = getModel("ATCNet", dataset_conf)
    base.build((None, 1, N_CHANNELS, IN_SAMPLES))
    base.compile(
        loss=CategoricalCrossentropy(from_logits=False),
        optimizer=Adam(learning_rate=args.pretrain_lr),
        metrics=["accuracy"],
    )
    pretrain_weights = results_dir / f"pretrain_except_{test_subject}.weights.h5"
    saved_models_dir = results_dir / "saved_models"
    saved_models_dir.mkdir(exist_ok=True)
    pt_history = HistoryTracker()
    base.fit(
        X_pt_train, y_pt_train,
        validation_data=(X_pt_val, y_pt_val),
        epochs=args.max_pretrain_epochs,
        batch_size=args.batch_size,
        callbacks=[
            ModelCheckpoint(str(pretrain_weights), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=20, min_lr=1e-5, verbose=0),
            _make_early_stopping(start_epoch=1, patience=50),
            pt_history,
        ],
        verbose=1,
    )
    base.load_weights(str(pretrain_weights))
    pretrain_curves_prefix = f"pretrain_except_{test_subject}"
    if pt_history.history.get("loss"):
        save_training_curves(pt_history.history, str(results_dir), pretrain_curves_prefix)
        _log.info("Saved pre-train curves to %s/%s_curves.png", results_dir, pretrain_curves_prefix)
    _log.info("Pre-training done. Weights saved to %s", pretrain_weights)

    # ---------- 2. Load test subject data (standardize with pre-train scalers) ----------
    target_idx = test_subject - 1
    _log.info("Loading subject %d data (for fine-tune and test)...", test_subject)
    X_target_train, _, y_target_train_onehot, X_target_test, _, y_target_test_onehot = get_data(
        data_path, target_idx, dataset, isStandard=False
    )
    standardize_apply_scalers(X_target_train, scalers, n_ch)
    standardize_apply_scalers(X_target_test, scalers, n_ch)
    y_target_test = np.argmax(y_target_test_onehot, axis=1)
    _log.info("Subject %d: train=%d, test=%d", test_subject, len(X_target_train), len(X_target_test))

    # ---------- 3. Fine-tune on fractions of target subject train; test on target subject test ----------
    results = []
    for frac in FINETUNE_FRACTIONS:
        _log.info("=" * 60)
        _log.info("Fine-tuning with %.0f%% of subject %d train (%d samples)", frac * 100, test_subject, int(len(X_target_train) * frac))
        X_ft, _, y_ft, _ = train_test_split(
            X_target_train, y_target_train_onehot, train_size=frac, random_state=42, stratify=np.argmax(y_target_train_onehot, axis=1)
        )
        if len(X_ft) < 10:
            _log.warning("Too few samples (%.0f%% = %d); skipping", frac * 100, len(X_ft))
            results.append({"fraction": frac, "train_samples": len(X_ft), "test_accuracy": None, "test_metrics": None})
            continue

        model_ft = getModel("ATCNet", dataset_conf)
        model_ft.build((None, 1, N_CHANNELS, IN_SAMPLES))
        model_ft.load_weights(str(pretrain_weights))
        model_ft.compile(
            loss=CategoricalCrossentropy(from_logits=False),
            optimizer=Adam(learning_rate=args.finetune_lr),
            metrics=["accuracy"],
        )
        frac_pct = int(round(frac * 100))
        ft_weights = saved_models_dir / f"finetune_subject{test_subject}_{frac_pct}pct.weights.h5"
        ft_history = HistoryTracker()
        model_ft.fit(
            X_ft, y_ft,
            validation_split=0.2,
            epochs=args.max_finetune_epochs,
            batch_size=args.batch_size,
            callbacks=[
                ModelCheckpoint(str(ft_weights), monitor="val_loss", save_best_only=True, save_weights_only=True, mode="min", verbose=0),
                ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=15, min_lr=1e-5, verbose=0),
                _make_early_stopping(start_epoch=1, patience=50),
                ft_history,
            ],
            verbose=1,
        )
        model_ft.load_weights(str(ft_weights))

        if ft_history.history.get("loss"):
            save_training_curves(ft_history.history, str(results_dir), f"finetune_subject{test_subject}_{frac_pct}pct")

        y_pred_proba = model_ft.predict(X_target_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc = float(accuracy_score(y_target_test, y_pred))
        prec = float(precision_score(y_target_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_target_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_target_test, y_pred, average="weighted", zero_division=0))
        cm = confusion_matrix(y_target_test, y_pred)

        save_confusion_matrix(cm, CLASS_LABELS, str(results_dir), f"finetune_subject{test_subject}_{frac_pct}pct_test")
        test_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "confusion_matrix": cm.tolist()}
        results.append({
            "fraction": frac, "train_samples": len(X_ft),
            "test_accuracy": acc, "test_metrics": test_metrics,
        })
        _log.info("Fraction %.0f%% -> subject %d test accuracy: %.4f (weights: %s)", frac * 100, test_subject, acc, ft_weights.name)

    # ---------- Save summary ----------
    summary_data = {"pretrain_subjects": pretrain_subject_ids, "target_subject": test_subject, "results": []}
    for r in results:
        row = {"fraction": r["fraction"], "train_samples": r["train_samples"], "test_accuracy": r["test_accuracy"]}
        if r.get("test_metrics"):
            row["test_precision"] = r["test_metrics"].get("precision")
            row["test_recall"] = r["test_metrics"].get("recall")
            row["test_f1_score"] = r["test_metrics"].get("f1_score")
            row["confusion_matrix"] = r["test_metrics"].get("confusion_matrix")
        summary_data["results"].append(row)
    summary_path = results_dir / "finetune_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    _log.info("Summary saved to %s", summary_path)
    _log.info("=" * 60)
    _log.info("Fine-tune summary (subject %d test accuracy):", test_subject)
    for r in results:
        _log.info("  %.0f%% subject %d train -> test acc: %s", r["fraction"] * 100, test_subject, f"{r['test_accuracy']:.4f}" if r["test_accuracy"] is not None else "N/A")
    _log.info("Outputs: pre-train + fine-tune curves, confusion matrices, weights in saved_models/, finetune_summary.json")
    return results


if __name__ == "__main__":
    main()
