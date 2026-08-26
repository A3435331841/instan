#!/usr/bin/env bash
set -euo pipefail

PY=/home/wuyou/grt_env/bin/python
ROOT=/data/pano360
DATA=/data/traindata/train
PID_FILE=/data/finetune_spherical_v5_20260826.pid
CKPT_DIR=/data/training_spherical_v5_20260826/checkpoints/train/odtrack/finetune_spherical_v5
OUT=/data/runs/finetune_spherical_v5_gpu0_20260826
LOG=/data/grt360_v5_gpu0_eval_after_train_20260826.log
SEQ_FILE=/data/runs/representative_leaderboard_20260825/medium_validation_sequences.txt
REP_SEQ="$(head -9 "$SEQ_FILE" | paste -sd, -)"
MED_SEQ="$(paste -sd, "$SEQ_FILE")"

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

while true; do
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! ps -p "$pid" -o cmd= 2>/dev/null | grep -q 'finetune_spherical_v5'; then
    break
  fi
  sleep 60
done

mkdir -p "$OUT"
for ep in 1 2 3 4 5 6; do
  ckpt="$(printf '%s/ODTrack_ep%04d.pth.tar' "$CKPT_DIR" "$ep")"
  [[ -s "$ckpt" ]] || continue
  if find "$OUT/ep${ep}_rep9" \
      "/data/runs/finetune_spherical_v5_20260826/ep${ep}_rep9" \
      -name summary.json 2>/dev/null | grep -q .; then continue; fi
  log "evaluate v5 ep$ep representative9 on released GPU0"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
    --data "$DATA" --out "$OUT/ep${ep}_rep9" --seqs "$REP_SEQ" --gpu 0 \
    --tracker odtrack --odtrack-workspace /data/odtrack \
    --odtrack-config finetune_spherical_v5 --odtrack-ckpt "$ckpt" \
    > "/data/v5_gpu0_ep${ep}_rep9_20260826.log" 2>&1
done

best_ckpt="$($PY - <<'PY'
import glob,json,re
rows=[]
patterns=('/data/runs/finetune_spherical_v5_gpu0_20260826/ep*_rep9/*/summary.json',
          '/data/runs/finetune_spherical_v5_20260826/ep*_rep9/*/summary.json')
for pattern in patterns:
    for p in glob.glob(pattern):
        d=json.load(open(p)); ep=int(re.search(r'/ep(\d+)_rep9/',p).group(1)); rows.append((d['auc'],ep))
if not rows: raise SystemExit(1)
print('/data/training_spherical_v5_20260826/checkpoints/train/odtrack/finetune_spherical_v5/ODTrack_ep%04d.pth.tar'%max(rows)[1])
PY
)"
printf '%s\n' "$best_ckpt" > /data/grt360_v5_best_checkpoint_20260826.txt
log "best v5 checkpoint=$best_ckpt; evaluate medium30 on GPU0"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
  --data "$DATA" --out "$OUT/best_medium30" --seqs "$MED_SEQ" --gpu 0 \
  --tracker odtrack --odtrack-workspace /data/odtrack \
  --odtrack-config finetune_spherical_v5 --odtrack-ckpt "$best_ckpt" \
  > /data/v5_gpu0_best_medium30_20260826.log 2>&1
touch /data/grt360_v5_gpu0_eval_after_train_20260826.done
log "v5 gpu0 evaluation complete"
