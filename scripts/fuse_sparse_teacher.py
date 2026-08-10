#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稀疏教师调用实验：UETrack 每帧，ODTrack 每 K 帧做因果校正。

这是 GRT360-Causal-DTP-ERP 的第一轮速度实验，不读取真值。它用最近一次
ODTrack 教师框和当前 UETrack 学生框的圆周中心偏差做短期校正，并将校正量
指数衰减到下一次教师帧。结果必须和独立 UETrack、ODTrack 同时比较。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import find_sequences, load_vot360_annotations  # noqa: E402
from scripts.score_external_results import find_result, parse_boxes  # noqa: E402


def _center(box, width):
    return (float(box[0]) + 0.5 * float(box[2])) % float(width)


def _delta(a, b, width):
    return ((float(a) - float(b) + width / 2.0) % width) - width / 2.0


def _correct(student, teacher, width, decay):
    student = np.asarray(student, dtype=float)
    teacher = np.asarray(teacher, dtype=float)
    alpha = float(np.clip(decay, 0.0, 1.0))
    c = (_center(student, width)
         + alpha * _delta(_center(teacher, width), _center(student, width), width)) % width
    ywh = student[1:] + alpha * (teacher[1:] - student[1:])
    return np.array([c - 0.5 * ywh[1], ywh[0], ywh[1], ywh[2]], dtype=float)


def fuse_sequence(od, ue, width, interval=5, decay=0.35):
    od, ue = np.asarray(od, dtype=float), np.asarray(ue, dtype=float)
    if od.shape != ue.shape or od.ndim != 2 or od.shape[1] != 4:
        raise ValueError('OD/UE shape mismatch')
    interval = max(1, int(interval))
    out = np.empty_like(ue)
    teacher_used = np.zeros(len(ue), dtype=bool)
    correction = np.zeros(4, dtype=float)
    for i in range(len(ue)):
        if i % interval == 0:
            out[i] = od[i]
            correction = od[i] - ue[i]
            # Store x as a circular center offset instead of a raw left-edge
            # difference, otherwise a seam crossing looks like a 360-degree jump.
            correction[0] = _delta(_center(od[i], width), _center(ue[i], width), width)
            teacher_used[i] = True
        else:
            base = ue[i].copy()
            c = (_center(base, width) + correction[0]) % width
            ywh = base[1:] + correction[1:]
            out[i] = np.array([c - 0.5 * ywh[1], ywh[0], ywh[1], ywh[2]])
            correction *= float(np.clip(1.0 - decay, 0.0, 1.0))
    return out, teacher_used


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--od-root', required=True)
    p.add_argument('--ue-root', required=True)
    p.add_argument('--seqs', default='all')
    p.add_argument('--interval', type=int, default=5)
    p.add_argument('--decay', type=float, default=0.35)
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)
    wanted = None if args.seqs.lower() == 'all' else {
        x.strip().zfill(4) for x in args.seqs.split(',') if x.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'router_config.json', 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in vars(args).items() if k not in ('data', 'out', 'seqs')},
                  f, ensure_ascii=False, indent=2)
    for seq_dir in find_sequences(args.data):
        seq = seq_dir.name.zfill(4)
        if wanted is not None and seq not in wanted:
            continue
        frames, gt = load_vot360_annotations(seq_dir)
        od = parse_boxes(find_result(args.od_root, seq))
        ue = parse_boxes(find_result(args.ue_root, seq))
        if len(od) != len(gt) or len(ue) != len(gt):
            raise ValueError(f'{seq}: prediction/GT length mismatch')
        with Image.open(frames[0]) as image:
            width = image.size[0]
        fused, used = fuse_sequence(od, ue, width, args.interval, args.decay)
        dst = out / seq
        dst.mkdir(parents=True, exist_ok=True)
        np.savetxt(dst / 'results.txt', fused, delimiter=',', fmt='%.9f')
        np.savetxt(dst / 'teacher_used.txt', used.astype(int), fmt='%d')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
