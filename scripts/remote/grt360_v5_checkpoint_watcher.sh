#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/pano360
PY=/home/wuyou/grt_env/bin/python
DATA=/data/traindata/train
CKPT_DIR=/data/training_spherical_v5_20260826/checkpoints/train/odtrack/finetune_spherical_v5
OUT=/data/runs/finetune_spherical_v5_20260826
LOG=/data/grt360_v5_checkpoint_watcher_20260826.log
SEQ_FILE=/data/runs/representative_leaderboard_20260825/medium_validation_sequences.txt
REP_SEQ="$(head -9 "$SEQ_FILE" | paste -sd, -)"
MED_SEQ="$(paste -sd, "$SEQ_FILE")"

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

while [[ ! -f /data/grt360_phase2_gpu1_queue_20260826.done ]]; do sleep 60; done
mkdir -p "$OUT"

# Micro gate for the online specialist episode: full hard sequence, with the
# already-completed T224/B224/ODTrack runs serving as the two fixed baselines.
if [[ ! -f /data/adaptive_s_episode_probe_20260826.done ]]; then
  log "run adaptive S expert-episode probe on train_sim/seq_0046"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
    --data "$DATA" --out /data/runs/adaptive_s_episode_probe_20260826 \
    --seqs train_sim/seq_0046 --gpu 0 --tracker adaptive_spherical \
    --sutrack-workspace /data/sutrack_src_20260825/SUTrack \
    --adaptive-main sutrack_t224 \
    --adaptive-main-ckpt /data/weights/SUTRACK_t224_ep0180.pth.tar \
    --adaptive-expert sutrack_b224 \
    --adaptive-expert-ckpt /data/weights/SUTRACK_b224_ep0180.pth.tar \
    --adaptive-expert-max-fraction 0.20 --adaptive-target-frame-ms 33.333 \
    --adaptive-expert-episode-frames 15 --adaptive-no-global-redetect \
    > /data/adaptive_s_episode_probe_20260826.log 2>&1
  touch /data/adaptive_s_episode_probe_20260826.done
fi

for ep in 1 2 3 4 5 6; do
  ckpt="$(printf '%s/ODTrack_ep%04d.pth.tar' "$CKPT_DIR" "$ep")"
  while true; do
    while [[ ! -s "$ckpt" ]]; do sleep 60; done
    size1="$(stat -c %s "$ckpt")"; sleep 30; size2="$(stat -c %s "$ckpt")"
    [[ "$size1" == "$size2" ]] && break
  done
  log "evaluate v5 ep$ep representative9"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
    --data "$DATA" --out "$OUT/ep${ep}_rep9" --seqs "$REP_SEQ" --gpu 0 \
    --tracker odtrack --odtrack-workspace /data/odtrack \
    --odtrack-config finetune_spherical_v5 --odtrack-ckpt "$ckpt" \
    > "/data/v5_ep${ep}_rep9_20260826.log" 2>&1
done

best_ckpt="$($PY - <<'PY'
import glob,json,re
rows=[]
for p in glob.glob('/data/runs/finetune_spherical_v5_20260826/ep*_rep9/*/summary.json'):
 d=json.load(open(p)); ep=int(re.search(r'/ep(\d+)_rep9/',p).group(1)); rows.append((d['auc'],ep))
print('/data/training_spherical_v5_20260826/checkpoints/train/odtrack/finetune_spherical_v5/ODTrack_ep%04d.pth.tar'%max(rows)[1])
PY
)"
log "best representative checkpoint: $best_ckpt; start medium30"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
  --data "$DATA" --out "$OUT/best_medium30" --seqs "$MED_SEQ" --gpu 0 \
  --tracker odtrack --odtrack-workspace /data/odtrack \
  --odtrack-config finetune_spherical_v5 --odtrack-ckpt "$best_ckpt" \
  > /data/v5_best_medium30_20260826.log 2>&1
touch /data/grt360_v5_checkpoint_watcher_20260826.done
log "v5 checkpoint watcher complete"
