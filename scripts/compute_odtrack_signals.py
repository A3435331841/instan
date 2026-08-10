#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线判丢信号计算器（Step 2，纯 CPU）。

对 120 条已有 ODTrack 逐帧结果，计算每帧的判丢代理信号（不依赖模型内部值，
不需要 GPU 重跑）：

  C_motion   预测框中心与圆周恒速外推中心的偏差（归一化到框对角线）
  C_scale    框面积 log 变化 vs 滑动历史 EMA
  geometry   ERP 极区风险 + 接缝风险（复用 causal_dtp 公式）
  anchor_ncc 预测框 crop 与首帧 GT crop 的 NCC 相似度（可选，--with-ncc，
             需解码帧，较慢）

输出：<out>/<seq>/signals.csv（逐帧 signal 列）+ 汇总 JSON。
后续由 `scripts/score_offline_gate.py` 用它做 60/60 留出标定与判丢评估。
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import load_vot360_annotations  # noqa: E402


def _center(box, width):
    return (float(box[0]) + 0.5 * float(box[2])) % float(width)


def _circ_delta(a, b, width):
    return ((float(a) - float(b) + width / 2.0) % width) - width / 2.0


def _circ_dist(a, b, width):
    return abs(_circ_delta(a, b, width))


def _geometry_risk(box, width, height):
    """极区/接缝几何风险（与 causal_dtp._geometry_risk 一致）。"""
    cy = float(box[1]) + 0.5 * float(box[3])
    lat = abs(90.0 - 180.0 * np.clip(cy, 0.0, height) / height)
    pole = np.clip((lat - 55.0) / 35.0, 0.0, 1.0)
    cx = _center(box, width)
    seam_dist = min(cx, width - cx)
    seam = 1.0 - np.clip(seam_dist / max(1.0, 0.12 * width), 0.0, 1.0)
    aspect = abs(np.log(max(float(box[2]), 1.0) / max(float(box[3]), 1.0)))
    return float(np.clip(0.50 * pole + 0.35 * seam + 0.15 * np.clip(aspect / 4.0, 0.0, 1.0), 0.0, 1.0))


def _to_gray_f32(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return (arr[..., 0].astype(np.float32) * 0.299
                + arr[..., 1].astype(np.float32) * 0.587
                + arr[..., 2].astype(np.float32) * 0.114)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resize_gray(gray, out_w, out_h):
    from PIL import Image
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((int(out_w), int(out_h)), Image.BILINEAR))


def _ncc(a, b, size=32):
    """两个灰度图 resize 到固定尺寸后的归一化互相关。"""
    a = _resize_gray(_to_gray_f32(a), size, size).astype(np.float64)
    b = _resize_gray(_to_gray_f32(b), size, size).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _crop_wrap(frame, box):
    """按跨界约定裁剪框区域（x 回绕）。"""
    H, W = frame.shape[:2]
    x, y, w, h = (float(v) for v in box)
    iw, ih = max(2, int(round(w))), max(2, int(round(h)))
    ix, iy = int(round(x)), int(round(y))
    iy = int(np.clip(iy, 0, max(0, H - ih)))
    cols = np.mod(ix + np.arange(iw), W)
    return np.ascontiguousarray(frame[iy:iy + ih][:, cols])


def compute_frame_signals(pred, width, height):
    """纯框信号：逐帧 (c_motion, c_scale, geometry_risk)。

    c_motion: 与恒速外推的偏差（0 完美 -> 大偏差小）；c_scale: 面积平稳度
    （1 平稳 -> 0 突变）；geometry_risk: 0~1 风险。
    """
    n = len(pred)
    c_motion = np.zeros(n, dtype=float)
    c_scale = np.ones(n, dtype=float)
    geometry = np.zeros(n, dtype=float)
    if n < 2:
        return c_motion, c_scale, geometry
    # 恒速外推：用最近 5 帧的中心速度（圆周 x + 线性 y）
    prev_center = None
    prev_vel = np.zeros(2, dtype=float)
    scale_ema = float(pred[0][2] * pred[0][3])
    for i in range(n):
        box = pred[i]
        diag = max(2.0, float(np.hypot(box[2], box[3])))
        center = np.array([_center(box, width), float(box[1]) + 0.5 * float(box[3])])
        if prev_center is not None and i > 1:
            dx = _circ_delta(center[0], prev_center[0], width)
            dist = float(np.hypot(dx, center[1] - prev_center[1]))
            # 预测点 = 上一帧中心 + 平均速度（用最近至多 5 帧速度 EMA）
            pred_center = prev_center + prev_vel
            pred_center[0] = _circ_delta(pred_center[0], 0.0, width)
            err = float(np.hypot(_circ_delta(center[0], pred_center[0], width),
                                 center[1] - pred_center[1]))
            c_motion[i] = float(np.clip(1.0 - err / diag, 0.0, 1.0))
        if prev_center is not None:
            vel = np.array([
                _circ_delta(center[0], prev_center[0], width),
                center[1] - prev_center[1]])
            prev_vel = 0.7 * prev_vel + 0.3 * vel
        prev_center = center
        area = max(4.0, float(box[2] * box[3]))
        ratio = area / scale_ema
        c_scale[i] = float(np.clip(np.exp(-abs(np.log(ratio)) / 0.4), 0.0, 1.0))
        scale_ema = 0.9 * scale_ema + 0.1 * area
        geometry[i] = _geometry_risk(box, width, height)
    return c_motion, c_scale, geometry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='360VOT 数据根目录')
    p.add_argument('--result-root', required=True,
                   help='ODTrack 结果根目录（<seq>/results.txt）')
    p.add_argument('--seqs', default='all')
    p.add_argument('--out', required=True, help='输出目录')
    p.add_argument('--with-ncc', action='store_true',
                   help='额外计算 anchor NCC（需解码帧，较慢）')
    args = p.parse_args(argv)

    from PIL import Image
    from panotrack.data.vot360 import find_sequences

    seq_dirs = find_sequences(args.data)
    if args.seqs.lower() != 'all':
        wanted = {s.strip().zfill(4) for s in args.seqs.split(',') if s.strip()}
        seq_dirs = [d for d in seq_dirs if d.name.zfill(4) in wanted]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {'seqs': args.seqs, 'with_ncc': args.with_ncc, 'sequences': []}
    t0 = time.perf_counter()
    for seq_dir in sorted(seq_dirs):
        seq = seq_dir.name
        res_path = Path(args.result_root) / seq / 'results.txt'
        if not res_path.is_file():
            print(f'skip {seq}: no results', file=sys.stderr)
            continue
        _, gt = load_vot360_annotations(seq_dir)
        pred = np.loadtxt(res_path, delimiter=',', dtype=float).reshape(-1, 4)
        if len(pred) != len(gt):
            print(f'skip {seq}: rows {len(pred)} != GT {len(gt)}', file=sys.stderr)
            continue
        images = sorted(pp for pp in (Path(seq_dir) / 'image').glob('*')
                        if pp.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        if not images:
            print(f'skip {seq}: no images', file=sys.stderr)
            continue
        with Image.open(images[0]) as image:
            width, height = image.size
        c_motion, c_scale, geometry = compute_frame_signals(pred, width, height)
        rows = list(zip(c_motion, c_scale, geometry))
        ncc = np.full(len(pred), np.nan, dtype=float)
        if args.with_ncc:
            with Image.open(images[0]) as image:
                anchor_crop = _crop_wrap(np.asarray(image.convert('RGB')), pred[0])
            for i, img_path in enumerate(images):
                with Image.open(img_path) as image:
                    frame = np.asarray(image.convert('RGB'))
                crop = _crop_wrap(frame, pred[i])
                ncc[i] = _ncc(crop, anchor_crop)
            rows = [(*r, float(v)) for r, v in zip(rows, ncc)]
        dst = out_root / seq
        dst.mkdir(parents=True, exist_ok=True)
        cols = ['c_motion', 'c_scale', 'geometry'] + (['anchor_ncc'] if args.with_ncc else [])
        with open(dst / 'signals.csv', 'w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(cols)
            writer.writerows(rows)
        meta['sequences'].append({'sequence': seq, 'n_frames': len(pred)})
        elapsed = time.perf_counter() - t0
        print(f'{seq} done ({elapsed:.0f}s)', flush=True)
    meta['elapsed_sec'] = float(time.perf_counter() - t0)
    with open(out_root / 'meta.json', 'w', encoding='utf-8') as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f'DONE {len(meta["sequences"])} sequences in {meta["elapsed_sec"]:.0f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
