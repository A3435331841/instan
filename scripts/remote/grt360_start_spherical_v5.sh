#!/usr/bin/env bash
set -euo pipefail

PY=/home/wuyou/grt_env/bin/python
MANIFEST=/data/runs/spherical_training_manifest_20260826/training_manifest.jsonl
SAVE=/data/training_spherical_v5_20260826
LOG=/data/finetune_spherical_v5_20260826.log

if pgrep -af 'run_training.py.*finetune_spherical_v5' | grep -v grep >/dev/null; then
  echo "finetune_spherical_v5 already running"
  exit 0
fi
cd /data/odtrack/lib/train
GRT360_TRAIN_MANIFEST="$MANIFEST" CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
nohup "$PY" -u run_training.py \
  --script odtrack --config finetune_spherical_v5 --save_dir "$SAVE" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > /data/finetune_spherical_v5_20260826.pid
echo "started pid=$!"
