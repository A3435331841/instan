#!/bin/bash
# B224 速度 A/B 测试：baseline vs AMP+no_grad
# 在 GPU1 上运行，用 5 条 representative 序列测速
set -euo pipefail

DATA=/data/traindata/train
SUTRACK=/data/sutrack_src_20260825/SUTrack
CKPT=/data/weights/SUTRACK_b224_ep0180.pth.tar
PYTHON=/home/wuyou/grt_env/bin/python
OUT=/data/runs/b224_speed_ab_$(date +%Y%m%d_%H%M%S)
SEQS="train_sim/seq_0002,train_sim/seq_0003,train_real/seq_0004,train_sim/seq_0046,train_real/seq_0026"

mkdir -p "$OUT"

echo "=== B224 Speed A/B Test ==="
echo "Output: $OUT"
echo "Sequences: $SEQS"
echo ""

# Test 1: Baseline (no AMP, no no_grad optimization)
echo "--- Test 1: Baseline (no AMP) ---"
CUDA_VISIBLE_DEVICES=1 $PYTHON -u scripts/eval_official.py \
  --data "$DATA" --out "$OUT/baseline" --gpu 0 \
  --tracker sutrack --sutrack-workspace "$SUTRACK" \
  --sutrack-config sutrack_b224 --sutrack-ckpt "$CKPT" \
  --no-sutrack-amp --seqs "$SEQS" 2>&1 | tee "$OUT/baseline.log"

echo ""
echo "--- Test 2: AMP + no_grad (optimized) ---"
CUDA_VISIBLE_DEVICES=1 $PYTHON -u scripts/eval_official.py \
  --data "$DATA" --out "$OUT/optimized" --gpu 0 \
  --tracker sutrack --sutrack-workspace "$SUTRACK" \
  --sutrack-config sutrack_b224 --sutrack-ckpt "$CKPT" \
  --sutrack-amp --seqs "$SEQS" 2>&1 | tee "$OUT/optimized.log"

echo ""
echo "=== Results ==="
echo "Baseline:"
grep -E "FPS|平均" "$OUT/baseline.log" || true
echo ""
echo "Optimized:"
grep -E "FPS|平均" "$OUT/optimized.log" || true
