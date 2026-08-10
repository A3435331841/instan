#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线判丢门控评估器（Step 2 离线版本，纯 CPU）。

输入 compute_odtrack_signals.py 产出的逐帧信号 + ODTrack 结果 + GT，做：

1. 60/60 留出划分（序列号奇偶，划分落盘到输出 JSON）——拿全部 120 条
   调参再报告成绩属于自欺欺人，这条纪律是从 LightFC 全量翻车的教训里来的；
2. 标定集上扫阈值，选 lost_run_recall / false_alarm 的 F1 最优门控
   （连续 run_len 帧可靠性 R < 阈值 -> 判 lost）；
3. 验证集上报告：帧级 ROC-AUC、失锁段召回率、正常帧误报率；
4. 输出标定参数 JSON，供 recapture.py 运行时加载。

注意：这是离线可交付的最大验证（不依赖 GPU 与模型内部值）；端到端
重捕获效果仍需队友在服务器上跑 recapture.py 全量 120 条。
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


def _roc_auc(scores, labels):
    """AUC<0.5 表示低分->正例（失锁），1-AUC 为区分度。"""
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
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, 'trapezoid') \
        else float(np.trapz(tpr, fpr))


def _load_signals(signals_csv):
    with open(signals_csv, encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        cols = reader.fieldnames
        data = {c: [] for c in cols}
        for row in reader:
            for c in cols:
                data[c].append(float(row[c]))
    return {c: np.asarray(v, dtype=float) for c, v in data.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _reliability(sig, w_motion, w_scale, w_ncc, geometry_penalty):
    """融合可靠性 R ∈ [0,1]（低 R = 判丢）。"""
    n = len(sig['c_motion'])
    ncc = sig.get('anchor_ncc', np.full(n, np.nan))
    ncc_ok = np.where(np.isnan(ncc), 0.5, ncc)  # 无 NCC 时取中性 0.5
    logit = (w_motion * (sig['c_motion'] - 0.5)
             + w_scale * (sig['c_scale'] - 0.5)
             + w_ncc * (ncc_ok - 0.5)
             - geometry_penalty * sig['geometry'])
    return _sigmoid(logit)


def _gate_flag(reliability, threshold, run_len):
    """连续 run_len 帧低 R -> 该帧起标记为 lost。"""
    flag = (reliability < threshold).astype(int)
    gate = np.zeros_like(flag)
    cnt = 0
    for i in range(len(flag)):
        cnt = cnt + 1 if flag[i] else 0
        if cnt >= run_len:
            gate[i] = 1
    return gate


def _run_stats(pool, threshold, run_len):
    """失锁段召回率 + 正常帧误报率。

    - lost_run_recall: 长度 >= run_len 的真实失锁连续段中，
      被门控覆盖至少 run_len 帧的比例；
    - false_alarm: 正常帧（IoU>=0.5）中被门控标记的比例。
    """
    runs = 0
    hits = 0
    normal_frames = 0
    fa_frames = 0
    for s in pool:
        gate = _gate_flag(s['reliability'], threshold, run_len)
        lost = s['lost']
        normal_frames += int((1 - lost).sum())
        fa_frames += int((gate == 1).sum() - ((gate == 1) & (lost == 1)).sum())
        i = 0
        while i < len(lost):
            if lost[i]:
                j = i
                while j < len(lost) and lost[j]:
                    j += 1
                if j - i >= run_len:
                    runs += 1
                    if int(gate[i:j].sum()) >= run_len:
                        hits += 1
                i = j
            else:
                i += 1
    recall = hits / runs if runs else float('nan')
    false_alarm = fa_frames / normal_frames if normal_frames else float('nan')
    return recall, false_alarm


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--result-root', required=True,
                   help='ODTrack 结果根目录（<seq>/results.txt）')
    p.add_argument('--signal-root', required=True,
                   help='compute_odtrack_signals.py 输出根目录')
    p.add_argument('--seqs', default='all')
    p.add_argument('--out', required=True, help='输出 JSON')
    p.add_argument('--w-motion', type=float, default=1.0)
    p.add_argument('--w-scale', type=float, default=1.0)
    p.add_argument('--w-ncc', type=float, default=0.0)
    p.add_argument('--w-geom', type=float, default=1.0)
    p.add_argument('--run-len', type=int, default=5,
                   help='连续低可靠判定 lost 所需帧数')
    args = p.parse_args(argv)

    from panotrack.data.vot360 import find_sequences
    seq_dirs = find_sequences(args.data)
    if args.seqs.lower() != 'all':
        wanted = {s.strip().zfill(4) for s in args.seqs.split(',') if s.strip()}
        seq_dirs = [d for d in seq_dirs if d.name.zfill(4) in wanted]

    pool = []
    for seq_dir in sorted(seq_dirs):
        seq = seq_dir.name
        res_path = Path(args.result_root) / seq / 'results.txt'
        sig_path = Path(args.signal_root) / seq / 'signals.csv'
        if not res_path.is_file() or not sig_path.is_file():
            continue
        _, gt = load_vot360_annotations(seq_dir)
        pred = np.loadtxt(res_path, delimiter=',', dtype=float).reshape(-1, 4)
        if len(pred) != len(gt):
            continue
        ious = np.asarray([iou_xywh(p, g) for p, g in zip(pred[1:], gt[1:])],
                          dtype=float)
        sig = _load_signals(sig_path)
        if len(sig['c_motion']) != len(gt):
            continue
        # 首帧只初始化，不计入
        for key in sig:
            sig[key] = sig[key][1:]
        lost = (ious < 0.5).astype(int)
        reliability = _reliability(sig, args.w_motion, args.w_scale,
                                   args.w_ncc, args.w_geom)
        pool.append({'sequence': seq, 'reliability': reliability,
                     'lost': lost, 'iou': ious})

    if not pool:
        raise SystemExit('no sequences with signals found')

    # 60/60 留出：序列号奇偶
    calib = [s for s in pool if int(s['sequence']) % 2 == 1]
    valid = [s for s in pool if int(s['sequence']) % 2 == 0]
    print(f'sequences: total={len(pool)} calib(odd)={len(calib)} '
          f'valid(even)={len(valid)}')

    def _frame_auc(part):
        rel = np.concatenate([s['reliability'] for s in part])
        lost = np.concatenate([s['lost'] for s in part])
        return _roc_auc(rel, lost)

    calib_auc = _frame_auc(calib)
    valid_auc = _frame_auc(valid)
    print(f'frame ROC-AUC (低 R -> 失锁): calib={calib_auc:.4f} '
          f'valid={valid_auc:.4f} (区分度 1-AUC: {1-calib_auc:.4f}/'
          f'{1-valid_auc:.4f})')

    # 标定集阈值扫描（F1 最优）
    best = None
    for threshold in np.linspace(0.2, 0.8, 13):
        recall, fa = _run_stats(calib, threshold, args.run_len)
        if np.isnan(recall):
            continue
        denom = recall + (1.0 - fa)
        f1 = 2.0 * recall * (1.0 - fa) / denom if denom > 0 else 0.0
        if best is None or f1 > best[3]:
            best = (threshold, recall, fa, f1)
    if best is None:
        raise SystemExit('calibration failed: no lost runs in calib set')
    threshold, calib_recall, calib_fa, calib_f1 = best
    valid_recall, valid_fa = _run_stats(valid, threshold, args.run_len)
    print(f'calibrated threshold={threshold:.2f} run_len={args.run_len} '
          f'(calib F1={calib_f1:.3f})')
    print(f'calib : lost_run_recall={calib_recall:.3f} '
          f'false_alarm={calib_fa:.4f}')
    print(f'valid : lost_run_recall={valid_recall:.3f} '
          f'false_alarm={valid_fa:.4f}')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as handle:
        json.dump({
            'args': vars(args),
            'split': {'calib': [s['sequence'] for s in calib],
                      'valid': [s['sequence'] for s in valid]},
            'frame_auc': {'calib': calib_auc, 'valid': valid_auc,
                          'calib_discriminative': 1.0 - calib_auc,
                          'valid_discriminative': 1.0 - valid_auc},
            'gate': {'threshold': threshold, 'run_len': args.run_len,
                     'calib': {'lost_run_recall': calib_recall,
                               'false_alarm': calib_fa, 'f1': calib_f1},
                     'valid': {'lost_run_recall': valid_recall,
                               'false_alarm': valid_fa}},
        }, handle, ensure_ascii=False, indent=2)
    print(f'saved {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
