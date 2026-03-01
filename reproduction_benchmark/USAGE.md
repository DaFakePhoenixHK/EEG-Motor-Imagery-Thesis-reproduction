# Reproduction Benchmark — Usage

BCI IV-2a reproduction benchmark (Protocol W, L, F, TTA per `reproduce_plan_for_Nathan.md`).

## Prerequisites

- Python 3.8+ with TensorFlow 2.x, scikit-learn, numpy, mne (for FBCSP+LDA)
- BCI Competition IV-2a data in `Four class motor imagery (001-2014)` format (A01T.mat, A01E.mat, etc.)

## Protocol–Model Matrix (Experiment Matrix)

Only (channels, protocol, model) combinations in the matrix are run; others are skipped.

**8ch:**
- W, L: fbcsp_lda, eegnetv4, shallow, deep4, conformer, db_atcnet
- F, TTA: eegnetv4, shallow, conformer, db_atcnet

**22ch (no W):**
- L, F: shallow, conformer, db_atcnet
- TTA: conformer only

## Command-line options

```text
--protocol W | L | F | TTA
--model eegnetv4 | shallow | deep4 | conformer | fbcsp_lda | db_atcnet
--channels 8 | 22
--seed 0
```

**Additional options:**
- `--data PATH` — BCI IV-2a folder (default: `../files2/Four class motor imagery (001-2014)`)
- `--results_dir PATH` — Output directory (default: `./results`)
- `--k_per_class N` — For Protocol F only; K trials per class (default: 1)
- `--run_all_protocols` — Run W, L, F, TTA in one call (F uses default K)
- `--run_all_k` — For Protocol F: run K=1,5,10,20
- `--epochs N` — Max epochs (default: 500; early stopping: start_epoch=100, patience=80)
- `--batch_size N` — Batch size (default: 64)

## Example commands

**Protocol W (within-subject), EEGNetv4, 8ch, seed 0:**
```bash
python run_benchmark.py --protocol W --model eegnetv4 --channels 8 --seed 0
```

**Protocol L (LOSO), Shallow, 22ch:**
```bash
python run_benchmark.py --protocol L --model shallow --channels 22 --seed 0
```

**Protocol F, K=5 per class:**
```bash
python run_benchmark.py --protocol F --model eegnetv4 --channels 8 --k_per_class 5 --seed 0
```

**Protocol F, all K values (1,5,10,20):**
```bash
python run_benchmark.py --protocol F --model eegnetv4 --channels 8 --run_all_k --seed 0
```

**All protocols (W, L, F, TTA), EEGNetv4:**
```bash
python run_benchmark.py --protocol W --model eegnetv4 --channels 8 --run_all_protocols --seed 0
```
(Use `--protocol W` as placeholder when `--run_all_protocols`; all four run.)

**Custom data path (WSL):**
```bash
python run_benchmark.py --data "/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" --protocol W --model eegnetv4 --channels 8 --seed 0
```

## Output layout

Per run:
```
results/bci2a/accuracy/{8ch|22ch}/{W|L|F|TTA}/{model}/seed_{s}/
  subjectwise.csv      # one row per subject: subject, trialAcc, macroF1, kappa, ITR
  summary.csv          # mean_trialAcc, std_trialAcc, median_trialAcc, iqr_trialAcc, mean_macroF1, std_macroF1
  confusion_{1..9}.csv # per-subject confusion matrix (optional)
```

For Protocol F with K:
```
results/bci2a/accuracy/8ch/F/eegnetv4/seed_0/K5/
  subjectwise.csv
  summary.csv
  confusion_{1..9}.csv
```

**Global rollups** (after running `aggregate_results.py`):
```
results/bci2a/results_summary_acc.csv   # channels, protocol, model, seed, mean_trialAcc, std_trialAcc, ...
results/bci2a/results_subjectwise.csv   # subject, channels, protocol, model, seed, trialAcc, macroF1, ...
```

**Complexity metrics** (after running `compute_complexity.py`):
```
results/bci2a/complexity/complexity_8ch.csv
results/bci2a/complexity/complexity_22ch.csv
```

## How it works

1. **data_loader.py** — Loads BCI IV-2a via `preprocess.load_BCI2a_data`; 8ch uses a motor-cortex subset.
2. **protocols.py** — Implements W (train session1, test session2), L (LOSO), F (few-shot calibration), TTA (causal EA). Uses 500 epochs + early stopping (start_epoch=100, patience=80).
3. **models_registry.py** — Builds models: EEGNet, ShallowConvNet, DeepConvNet, FBCSP+LDA, DB-ATCNet. `conformer` falls back to ShallowConvNet (EEGConformer requires PyTorch/Braindecode).
4. **run_benchmark.py** — CLI entry point: parses args, enforces protocol–model matrix, runs protocol, writes results + confusion matrices.
5. **compute_complexity.py** — Computes params, MACs, latency for each model (8ch/22ch).
6. **aggregate_results.py** — Aggregates all runs into `results_summary_acc.csv` and `results_subjectwise.csv`.

## Protocol summary

| Protocol | Train | Calibration | Test |
|----------|-------|-------------|------|
| W | Subject session1 | — | Subject session2 |
| L | Others session1 | — | Target session2 |
| F | Others session1 | Target session1, K per class | Target session2 |
| TTA | Others session1 | — | Target session2 (causal EA) |

## Model links (references)

- **FBCSP+LDA**: https://github.com/orvindemsy/BCICIV2a-FBCSP — Filter Bank CSP + One-vs-Rest LDA.
- **DB-ATCNet**: https://github.com/zk-xju/DB-ATCNet — Dual-Branch Convolution + ECA attention (TensorFlow).
- **EEGConformer**: https://github.com/eeyhsong/EEG-Conformer — Original repo. Braindecode: `braindecode.models.EEGConformer` (PyTorch). Not integrated in this benchmark (requires PyTorch training loop).

## Validation

With 500 epochs + early stopping (start_epoch=100, patience=80):
- BCI IV-2a within-subject (Protocol W) typically reaches ~60–80% with sufficient training.
- Use `--epochs 100` for quicker validation runs.
