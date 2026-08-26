#!/usr/bin/env bash
set -euo pipefail

PID_FILE=/data/sutrack_b224_all130_20260826.pid
LOG=/data/grt360_resume_v5_watcher_after_b224_20260826.log

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

while true; do
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! ps -p "$pid" -o cmd= 2>/dev/null | grep -q 'sutrack_b224_all130'; then
    break
  fi
  sleep 60
done

log "B224 all130 finished; resume v5 checkpoint watcher on GPU1"
if [[ ! -f /data/sutrack_b224_amp_speed_probe_20260826.done ]]; then
  log "run B224 AMP precision/speed A-B probe"
  cd /data/pano360
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /home/wuyou/grt_env/bin/python -u scripts/eval_official.py \
      --data /data/traindata/train \
      --out /data/runs/sutrack_b224_amp_speed_probe_20260826 \
      --seqs train_real/seq_0001,train_real/seq_0016,train_real/seq_0022,train_sim/seq_0011,train_sim/seq_0046 \
      --gpu 0 --tracker sutrack \
      --sutrack-workspace /data/sutrack_src_20260825/SUTrack \
      --sutrack-config sutrack_b224 --sutrack-ckpt /data/weights/SUTRACK_b224_ep0180.pth.tar \
      --sutrack-amp > /data/sutrack_b224_amp_speed_probe_20260826.log 2>&1
  touch /data/sutrack_b224_amp_speed_probe_20260826.done
fi
if [[ ! -f /data/odtrack_tangent_polar_probe_20260826.done ]]; then
  log "run fixed-tangent ODTrack polar probe before v5 checkpoint evaluations"
  cd /data/pano360
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /home/wuyou/grt_env/bin/python -u scripts/eval_official.py \
      --data /data/traindata/train \
      --out /data/runs/odtrack_tangent_polar_probe_20260826 \
      --seqs train_real/seq_0016,train_sim/seq_0046,train_sim/seq_0011,train_sim/seq_0044 \
      --gpu 0 --tracker odtrack_tangent \
      --odtrack-workspace /data/odtrack --odtrack-ckpt /data/weights/ODTrack_ep0300.pth.tar \
      --tangent-patch-size 720 --tangent-fov-deg 120 --tangent-context 3.5 \
      > /data/odtrack_tangent_polar_probe_20260826.log 2>&1
  touch /data/odtrack_tangent_polar_probe_20260826.done
fi
if [[ ! -f /data/odtrack_geo_medium30_20260826.done ]] && \
   /home/wuyou/grt_env/bin/python - <<'PY'; then
import json
from pathlib import Path
troot = Path('/data/runs/odtrack_tangent_polar_probe_20260826')
broot = Path('/data/runs/_all130_gpu0/odtrack_20260826_021411')
def load(root):
    rows = {}
    for path in root.rglob('metrics.json'):
        row = json.load(open(path))
        rows[row['sequence']] = row
    return rows
tangent, baseline = load(troot), load(broot)
seqs = sorted(set(tangent) & set(baseline))
if not seqs:
    raise SystemExit(1)
delta = sum(tangent[s]['auc'] - baseline[s]['auc'] for s in seqs) / len(seqs)
print('tangent_polar_delta', delta)
raise SystemExit(0 if delta >= .02 else 1)
PY
  log "tangent probe passed; evaluate unified geometry adapter on medium30"
  cd /data/pano360
  MED_SEQ="$(paste -sd, /data/runs/representative_leaderboard_20260825/medium_validation_sequences.txt)"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    /home/wuyou/grt_env/bin/python -u scripts/eval_official.py \
      --data /data/traindata/train --out /data/runs/odtrack_geo_medium30_20260826 \
      --seqs "$MED_SEQ" --gpu 0 --tracker odtrack_geo \
      --odtrack-workspace /data/odtrack --odtrack-ckpt /data/weights/ODTrack_ep0300.pth.tar \
      --tangent-patch-size 720 --tangent-fov-deg 120 --tangent-context 3.5 \
      > /data/odtrack_geo_medium30_20260826.log 2>&1
  touch /data/odtrack_geo_medium30_20260826.done
fi
if [[ -f /data/grt360_v5_gpu0_eval_after_train_20260826.done ]]; then
  log "v5 checkpoint evaluation already completed on GPU0"
  exit 0
fi
if pgrep -af 'grt360_v5_checkpoint_watcher_20260826.sh' | grep -v grep >/dev/null; then
  log "v5 watcher already active"
  exit 0
fi
nohup bash /data/grt360_v5_checkpoint_watcher_20260826.sh \
  > /data/grt360_v5_checkpoint_watcher_20260826.nohup 2>&1 < /dev/null &
echo $! > /data/grt360_v5_checkpoint_watcher_20260826.pid
log "started watcher pid=$!"
