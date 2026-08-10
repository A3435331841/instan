#!/usr/bin/env bash
# Queue paired UETrack baseline/ERP-wrap batches while 360VOT data arrives.
set -euo pipefail

if [ "$#" -lt 5 ]; then
  echo "usage: $0 PYTHON WORKSPACE DATA GPU SEQ_GROUP [SEQ_GROUP ...]" >&2
  echo "example SEQ_GROUP: 0030,0032,0034,0036,0038" >&2
  exit 2
fi

PYTHON_BIN="$1"
WORKSPACE="$2"
DATA_ROOT="$3"
GPU="$4"
shift 4
ADAPTER_ROOT="$(cd "$(dirname "$0")/../integrations/uetrack" && pwd)"
INCOMING_ROOT="${GRT360_INCOMING_ROOT:-/data/incoming}"

sequence_ready() {
  local sequence="$1"
  local root="$DATA_ROOT/$sequence"
  local direct="$root/label.json"
  local nested="$root/$sequence/label.json"
  [ ! -e "$INCOMING_ROOT/$sequence.tar.zst" ] \
    && { [ -f "$direct" ] || [ -f "$nested" ]; }
}

wait_for_gpu_queue() {
  while pgrep -f "run_erp.py.*--gpu $GPU" >/dev/null; do
    echo "WAIT_GPU $GPU" >&2
    sleep 15
  done
}

for group in "$@"; do
  IFS=',' read -r -a sequences <<< "$group"
  for sequence in "${sequences[@]}"; do
    while ! sequence_ready "$sequence"; do
      echo "WAIT_DATA $sequence" >&2
      sleep 30
    done
    echo "READY $sequence" >&2
  done

  wait_for_gpu_queue
  "$PYTHON_BIN" "$ADAPTER_ROOT/run_erp.py" \
    --workspace "$WORKSPACE" --data "$DATA_ROOT" --seqs "$group" \
    --gpu "$GPU" --skip-existing
  "$PYTHON_BIN" "$ADAPTER_ROOT/run_erp.py" \
    --workspace "$WORKSPACE" --data "$DATA_ROOT" --seqs "$group" \
    --gpu "$GPU" --erp-wrap --result-tag erpwrapfast --skip-existing
  echo "PAIR_DONE $group" >&2
done

echo "QUEUE_DONE gpu=$GPU" >&2
