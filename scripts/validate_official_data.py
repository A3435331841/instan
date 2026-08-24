#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方训练集完整性校验：video.mp4 帧数 == groundtruth.txt 行数。

用法:
    python scripts/validate_official_data.py                # 校验全部 130 条
    python scripts/validate_official_data.py --quick 12    # 每块只抽 12 条
输出: 逐条 OK/MISMATCH + 汇总；MISMATCH 序列写入
      data360/official_split/data_issues.json（供训练加载器跳过）。
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(r"D:\instan\初赛数据\train")
OUT = Path(__file__).resolve().parents[1] / "data360" / "official_split" / "data_issues.json"


def n_frames(video: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0, help=">0 时每块只抽 N 条")
    args = ap.parse_args()

    issues, rows = [], []
    for block in ("train_real", "train_sim"):
        block_dir = ROOT / block
        seqs = sorted(d for d in block_dir.iterdir() if d.is_dir())
        if args.quick:
            seqs = seqs[: args.quick]
        for seq_dir in seqs:
            gt = seq_dir / "groundtruth.txt"
            video = seq_dir / "video.mp4"
            n_gt = len(gt.read_text(encoding="utf-8").strip().splitlines())
            n_cv = n_frames(video)
            ok = n_gt == n_cv
            w, h = None, None
            if ok:
                cap = cv2.VideoCapture(str(video))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
            rows.append((f"{block}/{seq_dir.name}", n_cv, n_gt, w, h, ok))
            if not ok:
                issues.append({"seq": f"{block}/{seq_dir.name}",
                               "frames": n_cv, "gt_lines": n_gt})
            print(f"{'OK ' if ok else 'BAD'} {block}/{seq_dir.name}: "
                  f"frames={n_cv} gt={n_gt}" + ("" if ok else "  <-- MISMATCH"))

    print(f"\n共 {len(rows)} 条，MISMATCH {len(issues)} 条")
    OUT.write_text(json.dumps(issues, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"issues -> {OUT}")


if __name__ == "__main__":
    main()
