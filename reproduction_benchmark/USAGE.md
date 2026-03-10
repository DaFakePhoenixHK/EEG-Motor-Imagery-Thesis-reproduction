# Reproduction Benchmark Command Guide

This guide explains all practical commands for running the benchmark in WSL, including seed/protocol control and pause/resume/stop for long runs.

## 1) Go to the benchmark folder

```bash
cd /mnt/c/Users/User/Desktop/Thesis/files_replication/reproduction_benchmark
```

## 2) Data path behavior

- Default data path is taken from `run_8ch_only.sh`:
  - `../../files2/Four class motor imagery (001-2014)`
- If your dataset is elsewhere, set `BCI_DATA`:

```bash
BCI_DATA="/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" bash run_8ch_only.sh
```

## 3) Current 8-channel set

From `config.py`:
- `EIGHT_CH_INDICES = [3, 7, 8, 9, 10, 11, 14, 16]`
- Channels: `FCz, C3, C1, Cz, C2, C4, CP1, CP2`

This includes the required core MI channels: `C3`, `Cz`, `C4`.

## 4) Main script: `run_8ch_only.sh`

This is the recommended script for 8ch experiments. It supports:
- protocol selection
- seed selection
- F-protocol K-value selection
- graceful stop point
- pause/resume/stop via control files

### 4.1 Run all 8ch protocols and all seeds

```bash
bash run_8ch_only.sh
```

Defaults:
- `PROTOCOLS="W L F TTA"`
- `SEEDS="0 1 2 3 4"`
- `K_VALUES="1 5 10 20"`

### 4.2 Run only one seed

```bash
SEEDS="0" bash run_8ch_only.sh
```

### 4.3 Run only selected protocols

```bash
PROTOCOLS="W L" SEEDS="0 1" bash run_8ch_only.sh
```

### 4.4 Run only Protocol TTA for seeds 0 and 1

```bash
PROTOCOLS="TTA" SEEDS="0 1" bash run_8ch_only.sh
```

### 4.5 Run only Protocol F with custom K grid

```bash
PROTOCOLS="F" SEEDS="0 1" K_VALUES="1 5 10 20" bash run_8ch_only.sh
```

### 4.6 Stop automatically after a seed boundary

Example: stop after finishing `Protocol F`, `seed 1`:

```bash
STOP_AFTER_PROTOCOL=F STOP_AFTER_SEED=1 bash run_8ch_only.sh
```

This exits cleanly after the script completes all runs for that protocol/seed.

## 5) Pause / Resume / Stop during a long run

`run_8ch_only.sh` reads control files under:
- `.run_control/pause`
- `.run_control/stop`

Important: use a **second terminal** while training is running.

### 5.1 Terminal A (start run)

```bash
cd /mnt/c/Users/User/Desktop/Thesis/files_replication/reproduction_benchmark
bash run_8ch_only.sh
```

### 5.2 Terminal B (control run)

```bash
cd /mnt/c/Users/User/Desktop/Thesis/files_replication/reproduction_benchmark
```

Pause:

```bash
touch .run_control/pause
```

Resume:

```bash
rm -f .run_control/pause
```

Graceful stop (after current `run_benchmark.py` call finishes):

```bash
touch .run_control/stop
```

If you are in one terminal only, `Ctrl+C` is the only immediate control.

## 6) Direct single-command runs (`run_benchmark.py`)

### 6.1 One model, one protocol, one seed

```bash
python3 run_benchmark.py \
  --data "/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" \
  --protocol W \
  --model eegnetv4 \
  --channels 8 \
  --seed 0
```

### 6.2 Protocol F with a specific K

```bash
python3 run_benchmark.py \
  --data "/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" \
  --protocol F \
  --model deep4 \
  --channels 8 \
  --seed 1 \
  --k_per_class 10
```

### 6.3 Protocol TTA for one model, seed 0

```bash
python3 run_benchmark.py \
  --data "/mnt/c/Users/User/Desktop/Thesis/files2/Four class motor imagery (001-2014)" \
  --protocol TTA \
  --model db_atcnet \
  --channels 8 \
  --seed 0
```

## 7) Existing helper script: `run_all_single_seed.sh`

This script runs a mixed 8ch+22ch matrix with seed 0.

```bash
bash run_all_single_seed.sh
```

Note:
- Use `run_8ch_only.sh` for full 8ch control and pause/resume/stop features.

## 8) Results location

Outputs are written under:

```text
results/bci2a/accuracy/{8ch|22ch}/{W|L|F|TTA}/{model}/seed_{s}/
```

For Protocol F:

```text
.../F/{model}/seed_{s}/K{k}/
```

## 9) Preserve old results before a new run

```bash
mv results results_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p results
```

or simply rename:

```bash
mv results results_old
```

## 10) Aggregate results after runs

```bash
python3 aggregate_results.py
```

Generates:
- `results/bci2a/results_summary_acc.csv`
- `results/bci2a/results_subjectwise.csv`

## 11) Compute model complexity

```bash
python3 compute_complexity.py
```

Generates:
- `results/bci2a/complexity/complexity_8ch.csv`
- `results/bci2a/complexity/complexity_22ch.csv`

## 12) Quick command cookbook

Run only TTA, seeds 0 and 1:

```bash
PROTOCOLS="TTA" SEEDS="0 1" bash run_8ch_only.sh
```

Run W+L only, seed 0:

```bash
PROTOCOLS="W L" SEEDS="0" bash run_8ch_only.sh
```

Run F only, all seeds:

```bash
PROTOCOLS="F" bash run_8ch_only.sh
```

Run all protocols, stop after F seed 1:

```bash
STOP_AFTER_PROTOCOL=F STOP_AFTER_SEED=1 bash run_8ch_only.sh
```

