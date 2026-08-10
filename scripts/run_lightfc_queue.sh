#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
seqs_csv="$2"
out="$3"
mkdir -p "$out"
IFS=',' read -r -a seqs <<< "$seqs_csv"
for seq in "${seqs[@]}"; do
  if [[ -s "$out/$seq/metrics.json" && -s "$out/$seq/results.txt" ]]; then
    echo "SKIP $seq (complete result already present)"
    continue
  fi
  echo "RUN $seq on gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" /opt/miniconda3/envs/panotrack/bin/python \
    scripts/eval_360vot.py \
    --data /data/projects/instan/data360 \
    --seqs "$seq" --downscale 1.0 --out "$out" \
    --config configs/lightfc_gpu.json
done
echo "QUEUE_DONE gpu=$gpu"

