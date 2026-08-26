#!/usr/bin/env bash
set -euo pipefail

PY=/home/wuyou/grt_env/bin/python
ROOT=/data/pano360
DONE=/data/grt360_v5_gpu0_eval_after_train_20260826.done
CKPT_FILE=/data/grt360_v5_best_checkpoint_20260826.txt
TEMPLATE=/data/pano360/integrations/odtrack/finetune_spherical_v6_tiled.yaml.in
CONFIG=/data/odtrack/experiments/odtrack/finetune_spherical_v6_tiled.yaml
SAVE=/data/training_spherical_v6_tiled_20260826
LOG=/data/finetune_spherical_v6_tiled_20260826.log

while [[ ! -f "$DONE" ]]; do sleep 60; done
if pgrep -af 'run_training.py.*finetune_spherical_v6_tiled' | grep -v grep >/dev/null; then exit 0; fi

# v6 is an explicit ablation: only start when v5 does not clear the +0.01
# medium gain gate on the same sequences as the frozen ODTrack baseline.
if "$PY" - <<'PY'
import json
from pathlib import Path
base=Path('/data/runs/_all130_gpu0/odtrack_20260826_021411')
v5=Path('/data/runs/finetune_spherical_v5_gpu0_20260826/best_medium30')
def load(root):
 d={}
 for p in root.rglob('metrics.json'):
  x=json.load(open(p)); d[x['sequence']]=x
 return d
a,b=load(base),load(v5); common=sorted(set(a)&set(b))
if not common: raise SystemExit(1)
delta=sum(b[s]['auc']-a[s]['auc'] for s in common)/len(common)
print('v5_medium_delta',delta)
raise SystemExit(0 if delta >= .01 else 1)
PY
then
  echo "v5 passed medium gate; no v6 launch"
  exit 0
fi

ckpt="$(cat "$CKPT_FILE")"
sed "s|__PRETRAIN_FILE__|$ckpt|g" "$TEMPLATE" > "$CONFIG"
cd /data/odtrack/lib/train
GRT360_TRAIN_MANIFEST=/data/runs/spherical_training_manifest_20260826/training_manifest.jsonl \
GRT360_SPHERICAL_TILE_PROB=0.45 CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
nohup "$PY" -u run_training.py --script odtrack --config finetune_spherical_v6_tiled \
  --save_dir "$SAVE" > "$LOG" 2>&1 < /dev/null &
echo $! > /data/finetune_spherical_v6_tiled_20260826.pid
echo "started v6 tiled pid=$! from $ckpt"
