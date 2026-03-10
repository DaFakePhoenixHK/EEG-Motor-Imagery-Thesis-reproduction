#!/bin/bash
# Run 8ch only benchmark with controls for long runs.
# Uses new channel set: FCz, C3, C1, Cz, C2, C4, CP1, CP2 (includes C3, Cz, C4).
# WSL: run from reproduction_benchmark (cd there first).
#
# Seeds: set SEEDS="0" for single seed, or SEEDS="0 1 2 3 4" for all (default).
SEEDS="${SEEDS:-0 1 2 3 4}"
PROTOCOLS="${PROTOCOLS:-W L F TTA}"
K_VALUES="${K_VALUES:-1 5 10 20}"

set -e

BENCH_DIR="${BENCH_DIR:-$(dirname "$0")}"
DATA_PATH="${BCI_DATA:-${BENCH_DIR}/../../files2/Four class motor imagery (001-2014)}"
CONTROL_DIR="${CONTROL_DIR:-${BENCH_DIR}/.run_control}"
PAUSE_FILE="${CONTROL_DIR}/pause"
STOP_FILE="${CONTROL_DIR}/stop"

# Optional graceful stop point:
# Example: STOP_AFTER_PROTOCOL=F STOP_AFTER_SEED=1 bash run_8ch_only.sh
STOP_AFTER_PROTOCOL="${STOP_AFTER_PROTOCOL:-}"
STOP_AFTER_SEED="${STOP_AFTER_SEED:-}"

cd "$BENCH_DIR"
mkdir -p "$CONTROL_DIR"

echo "=== 8ch only ==="
echo "Protocols: $PROTOCOLS"
echo "Seeds: $SEEDS"
echo "Data: $DATA_PATH"
echo "Results: ${BENCH_DIR}/results"
echo "Control files:"
echo "  Pause : touch \"$PAUSE_FILE\""
echo "  Resume: rm -f \"$PAUSE_FILE\""
echo "  Stop  : touch \"$STOP_FILE\"   (stops after current run)"
echo ""

wait_if_paused() {
  while [ -f "$PAUSE_FILE" ]; do
    echo "[$(date +%H:%M:%S)] Paused. Remove $PAUSE_FILE to resume."
    sleep 5
  done
}

check_stop_requested() {
  if [ -f "$STOP_FILE" ]; then
    echo "[$(date +%H:%M:%S)] Stop requested via $STOP_FILE. Exiting."
    exit 0
  fi
}

check_stop_after_seed() {
  local protocol=$1
  local seed=$2
  if [ -n "$STOP_AFTER_PROTOCOL" ] && [ -n "$STOP_AFTER_SEED" ] && \
     [ "$protocol" = "$STOP_AFTER_PROTOCOL" ] && [ "$seed" = "$STOP_AFTER_SEED" ]; then
    echo "[$(date +%H:%M:%S)] Reached STOP_AFTER_PROTOCOL=$STOP_AFTER_PROTOCOL STOP_AFTER_SEED=$STOP_AFTER_SEED. Exiting."
    exit 0
  fi
}

run_one() {
  local protocol=$1
  local model=$2
  local k=$3
  local seed=$4
  wait_if_paused
  check_stop_requested
  echo "[$(date +%H:%M:%S)] 8ch $protocol $model ${k:+K=$k} seed=$seed"
  if [ -n "$k" ]; then
    python3 run_benchmark.py --data "$DATA_PATH" --protocol "$protocol" --model "$model" --channels 8 --seed "$seed" --k_per_class "$k"
  else
    python3 run_benchmark.py --data "$DATA_PATH" --protocol "$protocol" --model "$model" --channels 8 --seed "$seed"
  fi
  echo ""
}

for p in $PROTOCOLS; do
  case "$p" in
    W)
      echo "=== 8ch: Protocol W ==="
      for s in $SEEDS; do
        for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do run_one W "$m" "" "$s"; done
        check_stop_after_seed W "$s"
      done
      ;;
    L)
      echo "=== 8ch: Protocol L ==="
      for s in $SEEDS; do
        for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do run_one L "$m" "" "$s"; done
        check_stop_after_seed L "$s"
      done
      ;;
    F)
      echo "=== 8ch: Protocol F (K=$K_VALUES) ==="
      for s in $SEEDS; do
        for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do
          for k in $K_VALUES; do run_one F "$m" "$k" "$s"; done
        done
        check_stop_after_seed F "$s"
      done
      ;;
    TTA)
      echo "=== 8ch: Protocol TTA ==="
      for s in $SEEDS; do
        for m in fbcsp_lda eegnetv4 shallow deep4 conformer db_atcnet; do run_one TTA "$m" "" "$s"; done
        check_stop_after_seed TTA "$s"
      done
      ;;
    *)
      echo "Unknown protocol in PROTOCOLS: $p"
      exit 1
      ;;
  esac
done

echo "=== Done ==="
