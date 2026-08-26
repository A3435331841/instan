#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""微调数据准备：官方训练集 -> GOT-10k 格式训练数据（P0-C，2026-08-24）。

输入：train/{train_real,train_sim}/seq_*/  (video.mp4 + groundtruth.txt BFoV)
输出（GOT-10k 风格，ODTrack 训练器直接可读）：
  <out>/list.txt                       # 子序列名列表
  <out>/<subseq>/%08d.jpg              # 连续有效帧（GOT-10k 标准 1-based）
  <out>/<subseq>/groundtruth.txt       # ERP xywh（与帧一一对应）
  <out>/<subseq>/absence.label         # 0=可见
  <out>/<subseq>/cover.label           # 8=完全可见

关键处理：
  - GT 0,0,0,0（目标消失帧）剔除；序列按消失段切分为连续子序列
    （ODTrack 训练采样器按帧间隔配对，必须保证时间连续性）；
  - 长度 < min_len 的碎片丢弃；
  - BFoV -> ERP xywh 用 panotrack 官方实现（跨界框 x+w 可超 W，训练 crop 需环绕填充，
    由 ODTrack 的 crop 处理——见注意事项）；
  - 只处理 official_split 的 train 集（95 条），valid 35 条绝不进入。

用法:
  python scripts/prepare_finetune_data.py --data /data/traindata/train \
      --out /data/finetune/official_got10k [--workers 4]
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov  # noqa: E402

try:
    import cv2 as cv
except ImportError:
    cv = None

SPLIT_DIR = PROJECT_ROOT / "data360" / "official_split"
MIN_RUN_LEN = 40          # 碎片子序列最小长度（帧）
JPEG_QUALITY = 92
SEAM_MARGIN = 24          # 距边多少像素内视为跨界，触发回滚


def roll_to_center(frame, box, W):
    """跨界框处理：水平平移图像使目标居中，返回 (新帧, 新框)。

    与推理侧三平铺的作用一致——保证目标周围 crop 完整、无黑边。
    """
    x, y, w, h = box
    cx = (x + w / 2.0) % W
    shift = int(cx - W / 2.0)
    if shift == 0:
        return frame, box
    rolled = np.roll(frame, -shift, axis=1)
    nx = x - shift
    if nx < -SEAM_MARGIN or nx + w > W + SEAM_MARGIN:
        nx = nx % W
    return rolled, (nx, y, w, h)


def load_gt_bfov(path: Path):
    rows, valid = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        f = [float(v) for v in ln.replace(",", " ").split()]
        rows.append(f)
        valid.append(f[2] > 0.0 and f[3] > 0.0)
    return rows, valid


def contiguous_runs(valid):
    """布尔序列 -> [(start, end)] 连续 True 区段（end exclusive）。"""
    runs, start = [], None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(valid)))
    return runs


def process_sequence(seq_dir: Path, out_root: Path, seq_tag: str) -> list:
    """处理一条序列：抽帧+转框+分段写出。返回写出的子序列名列表。"""
    gt_rows, valid = load_gt_bfov(seq_dir / "groundtruth.txt")
    runs = [r for r in contiguous_runs(valid) if r[1] - r[0] >= MIN_RUN_LEN]
    if not runs:
        return []

    cap = cv.VideoCapture(str(seq_dir / "video.mp4"))
    try:
        ok, first = cap.read()
        if not ok or first is None:
            return []
        H, W = first.shape[:2]

        # 预转所有有效帧的框
        boxes = {}
        for i, f in enumerate(gt_rows):
            if valid[i]:
                boxes[i] = erp_bbox_from_bfov(
                    BFoV(lon=f[0], lat=f[1], fov_h=f[2], fov_v=f[3]), W, H)

        written = []
        for run_id, (s, e) in enumerate(runs):
            sub_name = f"{seq_tag}_r{run_id:02d}"
            sub_dir = out_root / sub_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            cap.set(cv.CAP_PROP_POS_FRAMES, s)
            gt_lines = []
            for i in range(s, e):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                box = boxes[i]
                x, y, w, h = box
                if x < SEAM_MARGIN or x + w > W - SEAM_MARGIN:
                    frame, (x, y, w, h) = roll_to_center(frame, box, W)
                rel = i - s + 1
                out_jpg = sub_dir / f"{rel:08d}.jpg"
                cv.imwrite(str(out_jpg), frame, [cv.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                gt_lines.append(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}")
            if gt_lines:
                (sub_dir / "groundtruth.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
                (sub_dir / "absence.label").write_text("0\n" * len(gt_lines), encoding="utf-8")
                (sub_dir / "cover.label").write_text("8\n" * len(gt_lines), encoding="utf-8")
                written.append((sub_name, len(gt_lines)))
            else:
                sub_dir.rmdir()
        return written
    finally:
        cap.release()


def process_entry(task):
    idx, total, entry, data_root_s, out_root_s = task
    data_root = Path(data_root_s)
    out_root = Path(out_root_s)
    block, seq = entry.split("/")
    seq_dir = data_root / block / seq
    if not seq_dir.is_dir():
        return idx, entry, [], "目录不存在"
    tag = f"{block}_{seq}"
    written = process_sequence(seq_dir, out_root, tag)
    return idx, entry, written, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="官方训练集根目录（含 train_real/ train_sim/）")
    ap.add_argument("--out", required=True, help="输出根目录（GOT-10k 风格）")
    ap.add_argument("--split-file", default=str(SPLIT_DIR / "seqlist_official_train.txt"))
    ap.add_argument("--workers", type=int, default=1, help="并行处理序列数")
    args = ap.parse_args()

    if cv is None:
        raise SystemExit("需要 cv2")

    data_root = Path(args.data)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = [ln.strip() for ln in Path(args.split_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"[prep] 训练集 {len(entries)} 条序列 -> {out_root}")

    all_names, total_frames = [], 0
    tasks = [(n, len(entries), entry, str(data_root), str(out_root))
             for n, entry in enumerate(entries, 1)]
    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(process_entry, tasks)
            for n, entry, written, err in results:
                if err:
                    print(f"  [skip] {entry}: {err}")
                    continue
                for name, cnt in written:
                    all_names.append(name)
                    total_frames += cnt
                print(f"  [{n}/{len(entries)}] {entry}: {len(written)} 子序列 "
                      f"({sum(c for _, c in written)} 帧)", flush=True)
    else:
        results = [process_entry(task) for task in tasks]

    if args.workers <= 1:
        for n, entry, written, err in results:
            if err:
                print(f"  [skip] {entry}: {err}")
                continue
            for name, cnt in written:
                all_names.append(name)
                total_frames += cnt
            print(f"  [{n}/{len(entries)}] {entry}: {len(written)} 子序列 "
                  f"({sum(c for _, c in written)} 帧)")

    (out_root / "list.txt").write_text("\n".join(all_names) + "\n", encoding="utf-8")
    print(f"[prep] 完成: {len(all_names)} 子序列 / {total_frames} 帧 -> list.txt")


if __name__ == "__main__":
    main()
