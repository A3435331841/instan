#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ODTrack last_pred_iou 置信度与失锁的相关性分析（Step 1 验证）。

对已有 ODTrack 逐帧结果（results.txt + confidence.txt + GT）做三件事：
1. 帧级 Pearson 相关系数（confidence vs IoU）；
2. 失锁段（IoU<0.5 连续段）与正常段的置信度分布对比；
3. 把 confidence 当"判丢器"的区分能力：以 IoU<0.5 为真失锁标签，算 ROC-AUC。

输出逐序列与汇总两类指标，用于判定 last_pred_iou 是否可以作为
OdtrackRecaptureTracker 的 C_visual 判丢信号。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import load_vot360_annotations  # noqa: E402
from panotrack.evaluation.metrics import iou_xywh  # noqa: E402


def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _roc_auc(scores, labels):
    """以 scores 判 labels(1=失锁) 的 ROC-AUC。

    注意方向约定：判丢信号应当"低置信度 -> 失锁"，因此期望 AUC < 0.5
    （正例分数低于负例）；|AUC-0.5| 越大区分越强，1-AUC 即归一化区分度。
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    ranked = labels[order]
    pos = int(ranked.sum())
    neg = ranked.size - pos
    tpr = np.cumsum(ranked) / pos
    fpr = np.cumsum(1 - ranked) / neg
    # 梯形积分
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(tpr, fpr))


def analyze_sequence(seq_dir, result_root, width):
    """单序列分析：返回逐帧 IoU 与 confidence 的统计。"""
    sequence = Path(seq_dir).name
    conf_path = Path(result_root) / sequence / 'confidence.txt'
    res_path = Path(result_root) / sequence / 'results.txt'
    if not conf_path.is_file():
        raise FileNotFoundError(f'missing confidence: {conf_path}')
    _, gt = load_vot360_annotations(seq_dir)
    pred = np.loadtxt(res_path, delimiter=',', dtype=float).reshape(-1, 4)
    conf = np.loadtxt(conf_path, dtype=float).reshape(-1)
    if len(pred) != len(gt) or len(conf) != len(gt):
        raise ValueError(
            f'{sequence}: pred={len(pred)} gt={len(gt)} conf={len(conf)} 行数不一致')
    # 首帧只初始化，不计入（与 OPE 协议一致）
    ious = [iou_xywh(p, g) for p, g in zip(pred[1:], gt[1:])]
    confs = conf[1:]
    ious = np.asarray(ious, dtype=float)
    confs = np.asarray(confs, dtype=float)
    lost = (ious < 0.5).astype(int)
    # 失锁段：连续 lost 的帧段（至少 3 帧）
    runs = []
    start = None
    for i, flag in enumerate(lost):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 3:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(lost) - start >= 3:
        runs.append((start, len(lost) - 1))
    return {
        'sequence': sequence,
        'n_frames': int(len(gt) - 1),
        'mean_iou': float(ious.mean()),
        'sr': float(np.mean(~lost.astype(bool))),
        'pearson_conf_iou': _pearson(confs, ious),
        'roc_auc_conf_lost': _roc_auc(confs, lost),
        'conf_mean_ok': float(confs[~lost.astype(bool)].mean()) if (~lost.astype(bool)).any() else float('nan'),
        'conf_mean_lost': float(confs[lost.astype(bool)].mean()) if lost.any() else float('nan'),
        'conf_p25_lost': float(np.percentile(confs[lost.astype(bool)], 25)) if lost.any() else float('nan'),
        'n_lost_runs': len(runs),
        'lost_run_frames': sum(hi - lo + 1 for lo, hi in runs),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='360VOT 数据根目录')
    p.add_argument('--result-root', required=True,
                   help='含 <seq>/results.txt + confidence.txt 的结果根目录')
    p.add_argument('--seqs', default='all')
    p.add_argument('--out', required=True, help='输出 JSON')
    args = p.parse_args(argv)

    from panotrack.data.vot360 import find_sequences
    seq_dirs = find_sequences(args.data)
    if args.seqs.lower() != 'all':
        wanted = {s.strip().zfill(4) for s in args.seqs.split(',') if s.strip()}
        seq_dirs = [d for d in seq_dirs if d.name.zfill(4) in wanted]

    from PIL import Image
    rows = []
    for seq_dir in sorted(seq_dirs):
        images = sorted(p for p in (Path(seq_dir) / 'image').glob('*')
                        if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'))
        if not images:
            print(f'skip {Path(seq_dir).name}: no images', file=sys.stderr)
            continue
        with Image.open(images[0]) as image:
            width = int(image.size[0])
        try:
            row = analyze_sequence(seq_dir, args.result_root, width)
        except (FileNotFoundError, ValueError) as exc:
            print(f'skip {Path(seq_dir).name}: {exc}', file=sys.stderr)
            continue
        rows.append(row)
        print(f"{row['sequence']} n={row['n_frames']:>5d} "
              f"pearson={row['pearson_conf_iou']:+.3f} "
              f"roc_auc={row['roc_auc_conf_lost']:.3f} "
              f"conf_ok={row['conf_mean_ok']:.3f} conf_lost={row['conf_mean_lost']:.3f}")

    if not rows:
        raise SystemExit('no sequences analyzed')
    keys = ['pearson_conf_iou', 'roc_auc_conf_lost']
    summary = {
        'n_sequences': len(rows),
        'mean_pearson_conf_iou': float(np.nanmean(
            [r['pearson_conf_iou'] for r in rows])),
        'mean_roc_auc': float(np.nanmean(
            [r['roc_auc_conf_lost'] for r in rows])),
        'mean_sr': float(np.mean([r['sr'] for r in rows])),
        'mean_conf_ok': float(np.nanmean([r['conf_mean_ok'] for r in rows])),
        'mean_conf_lost': float(np.nanmean([r['conf_mean_lost'] for r in rows])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'rows': rows}, handle,
                  ensure_ascii=False, indent=2)
    print('---SUMMARY---')
    for key, value in summary.items():
        print(f'{key}: {value}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
