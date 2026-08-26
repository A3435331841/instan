#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/pano360
PY=/home/wuyou/grt_env/bin/python
DATA=/data/traindata/train
SUTRACK=/data/sutrack_src_20260825/SUTrack
OUT=/data/runs/phase2_sutrack_20260826
LOG=/data/grt360_phase2_gpu1_queue_20260826.log

log() { echo "[$(date)] $*" | tee -a "$LOG"; }

run_valid() {
  local name="$1" ckpt="$2"
  log "start valid35 $name"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
    --data "$DATA" --out "$OUT/${name}_valid35" --split valid --gpu 0 \
    --tracker sutrack --sutrack-workspace "$SUTRACK" \
    --sutrack-config "$name" --sutrack-ckpt "$ckpt" \
    > "/data/${name}_valid35_20260826.log" 2>&1
  log "finish valid35 $name"
}

run_full() {
  local name="$1" ckpt="$2"
  log "start all130 $name"
  cd "$ROOT"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u scripts/eval_official.py \
    --data "$DATA" --out "$OUT/${name}_all130" --split all --gpu 0 \
    --tracker sutrack --sutrack-workspace "$SUTRACK" \
    --sutrack-config "$name" --sutrack-ckpt "$ckpt" \
    > "/data/${name}_all130_20260826.log" 2>&1
  log "finish all130 $name"
}

mkdir -p "$OUT"
run_valid sutrack_t224 /data/weights/SUTRACK_t224_ep0180.pth.tar
run_valid sutrack_b224 /data/weights/SUTRACK_b224_ep0180.pth.tar

winner="$($PY - <<'PY'
import glob,json,math
rows=[]
for name in ('sutrack_t224','sutrack_b224'):
 p=glob.glob(f'/data/runs/phase2_sutrack_20260826/{name}_valid35/*/summary.json')
 if not p: continue
 d=json.load(open(sorted(p)[-1])); rows.append((name,d))
eligible=[r for r in rows if r[1].get('e2e_fps',0)>30]
pool=eligible or rows
print(max(pool,key=lambda r:r[1]['auc'])[0])
PY
)"
if [[ "$winner" == "sutrack_t224" ]]; then
  winner_ckpt=/data/weights/SUTRACK_t224_ep0180.pth.tar
else
  winner_ckpt=/data/weights/SUTRACK_b224_ep0180.pth.tar
fi
log "valid35 winner under speed constraint: $winner"
run_full "$winner" "$winner_ckpt"
touch /data/grt360_phase2_gpu1_queue_20260826.done
log "gpu1 SUTRACK queue complete"
