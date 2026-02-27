# Replication workspace (Thesis)

This folder is the **replication workspace** for the thesis. All replication-related scripts, models, and documentation live here (changes are made here instead of `files2`).

## Contents

- **special_actnet_finetuning.py** — ATCNet pre-train on subjects 2–9, fine-tune on 5%/15%/20%/25% of subject 1 train, test on subject 1. Saves curves, confusion matrices, test metrics, and weights.
- **main_TrainValTest.py** — Train–val–test pipeline; provides `getModel`, early stopping, `save_training_curves`, `save_confusion_matrix`.
- **preprocess.py** — BCI2a loading and standardization.
- **models.py** — ATCNet and other EEG models.
- **attention_models.py** — Attention blocks used by ATCNet.
- **\*.md** — Documentation (preprocessing, ATCNet vs Braindecode, etc.).

## Running the fine-tuning script

**Requirement:** `--data` must be the **full path to an existing directory** containing the BCI2a raw data (e.g. the folder that contains `A01T.mat`, `A01E.mat`, …, `A09T.mat`, `A09E.mat`). Typical folder name: `Four class motor imagery (001-2014)`.

**Subject selection:** Use `--test_subject N` (1–9) to choose which subject is left out for **test and fine-tune**. Pre-training uses the other 8 subjects. Default is `--test_subject 2` (train on 1,3,4,5,6,7,8,9; test and fine-tune on subject 2).

**Examples:**

- **Default (test on subject 2):** train on 1,3–9; fine-tune and test on 2:
  ```bash
  python special_actnet_finetuning.py --data "/mnt/c/Users/User/.../Four class motor imagery (001-2014)" --results_dir results_finetune
  ```
- **Test on subject 1:** train on 2–9; fine-tune and test on 1:
  ```bash
  python special_actnet_finetuning.py --data "/path/to/Four class motor imagery (001-2014)" --test_subject 1 --results_dir results_finetune
  ```
- **Test on subject 5:**
  ```bash
  python special_actnet_finetuning.py --data "/path/to/..." --test_subject 5 --results_dir results_finetune
  ```

If you pass a non-existent path, the script will exit with an error that shows the resolved path so you can correct it.

## Outputs

- **results_finetune/** (or `--results_dir`): pre-train and fine-tune curves (PNG), confusion matrices (PNG), **saved_models/** (pre-train + per-fraction fine-tune weights), **finetune_summary.json** (test accuracy, precision, recall, F1, confusion matrix per fraction).
