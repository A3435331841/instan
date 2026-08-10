#!/usr/bin/env bash
set -euo pipefail
wait_pid="$1"
gpu="$2"
seqs="$3"
out="$4"
log="$5"
while kill -0 "$wait_pid" 2>/dev/null; do
  sleep 30
done
cd /data/projects/instan_grt360
export CUDA_VISIBLE_DEVICES="$gpu"
exec /opt/miniconda3/envs/uetrack/bin/python -u scripts/odtrack_360vot.py \
  --odtrack-root /data/projects/instan_check/odtrack \
  --data /data/projects/instan/data360 \
  --checkpoint /data/projects/instan_check/odtrack/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar \
  --config baseline --seqs "$seqs" --gpu 0 --downscale 1.0 --out "$out"

