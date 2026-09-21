#!/usr/bin/env bash
# Real-GPU validation matrix. DRAFT: written without GPU access.
#   NGPU=2 REPEATS=3 SECONDS_PER_RUN=120 MAX_MINUTES=90 bench/run_matrix.sh
# Refuses to start a run that would exceed MAX_MINUTES: that is the spend cap.
set -euo pipefail
NGPU=${NGPU:-2}; REPEATS=${REPEATS:-3}; SECS=${SECONDS_PER_RUN:-120}; MAX_MIN=${MAX_MINUTES:-90}
MODES=${MODES:-"clean host_stall contention power_cap clock_lock"}
OUT=${OUT:-runs}; GFR=${GFR:-bin/gfr}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000} TORCH_NCCL_ENABLE_TIMING=1
mkdir -p "$OUT"; START=$(date +%s)
PROBE=$(python bench/inject_fault.py probe); echo "$PROBE" | tee "$OUT/probe.json"
supported() { [[ $1 == clean ]] || echo "$PROBE" | python -c "import sys,json;sys.exit(0 if json.load(sys.stdin).get('$1') is True else 1)"; }
for mode in $MODES; do
  supported "$mode" || { echo "skip $mode: not permitted on this host"; continue; }
  for i in $(seq 1 "$REPEATS"); do
    (( ($(date +%s) - START + SECS + 30) > MAX_MIN * 60 )) && { echo "budget cap reached before $mode-$i; stopping"; exit 0; }
    D="$OUT/$mode-$i"; mkdir -p "$D"
    GPU=$(( RANDOM % NGPU ))
    if [[ $mode != clean ]]; then
      python bench/inject_fault.py run --mode "$mode" --gpu "$GPU" --start-after $(( SECS / 2 )) \
        --duration $(( SECS / 2 )) --truth "$D/truth.json" --control "$D/fault.json" &
      INJ=$!
    fi
    torchrun --nproc_per_node "$NGPU" bench/train_ddp.py --seconds "$SECS" --out-dir "$D" --control "$D/fault.json"
    [[ $mode != clean ]] && wait "$INJ"
    python bench/score_real.py "$D" --gfr "$GFR" --results "$OUT/results.jsonl"
  done
done
echo "elapsed: $(( ($(date +%s) - START) / 60 )) min"
