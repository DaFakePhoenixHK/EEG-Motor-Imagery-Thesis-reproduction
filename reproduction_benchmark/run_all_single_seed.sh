#!/bin/bash
# Run full experiment matrix with single seed (0) per cosupervisor suggestion.
# WSL: run from reproduction_benchmark or set BENCH_DIR.

set -e

# Paths (WSL format)
BENCH_DIR="${BENCH_DIR:-$(dirname "$0")}"
DATA_PATH="${BCI_DATA:-${BENCH_DIR}/../../files2/Four class motor imagery (001-2014)}"

cd "$BENCH_DIR"

echo "=== Benchmark: single seed (0) ==="
echo "Data: $DATA_PATH"
echo "Results: ${BENCH_DIR}/results"
echo ""

run_one() {
  local protocol=$1
  local model=$2
  local channels=$3
  local k=$4
  echo "[$(date +%H:%M:%S)] $channels $protocol $model ${k:+K=$k}"
  if [ -n "$k" ]; then
    python3 run_benchmark.py --data "$DATA_PATH" --protocol "$protocol" --model "$model" --channels "$channels" --seed 0 --k_per_class "$k"
  else
    python3 run_benchmark.py --data "$DATA_PATH" --protocol "$protocol" --model "$model" --channels "$channels" --seed 0
  fi
  echo ""
}

# --- 8ch ---
echo "=== 8ch: Protocol W ==="
for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do run_one W "$m" 8; done

echo "=== 8ch: Protocol L ==="
for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do run_one L "$m" 8; done

echo "=== 8ch: Protocol F (K=1,5,10,20) ==="
for m in eegnetv4 shallow conformer db_atcnet; do
  for k in 1 5 10 20; do run_one F "$m" 8 "$k"; done
done

echo "=== 8ch: Protocol TTA ==="
for m in eegnetv4 shallow conformer db_atcnet; do run_one TTA "$m" 8; done

# --- 22ch (no W) ---
echo "=== 22ch: Protocol L ==="
for m in shallow conformer db_atcnet; do run_one L "$m" 22; done

echo "=== 22ch: Protocol F (K=1,5,10,20) ==="
for m in shallow conformer db_atcnet; do
  for k in 1 5 10 20; do run_one F "$m" 22 "$k"; done
done

echo "=== 22ch: Protocol TTA ==="
run_one TTA conformer 22

echo "=== Done ==="
