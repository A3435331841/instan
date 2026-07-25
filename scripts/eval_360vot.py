# -*- coding: utf-8 -*-
"""360VOT 本地批量评测：PanoTracker OPE -> 普通/dual 双口径指标汇总。

用法（工作目录 D:\\instan\\pano360）：
  python scripts/eval_360vot.py --data data360 --seqs all --downscale 0.5 --out runs/360vot
  python scripts/eval_360vot.py --seqs 001,003 --max-frames 100     # 调试

输出（--out 目录下）：
  <seq>/results.txt   逐帧跟踪框 x,y,w,h（2 位小数，首帧为初始化 GT 框）
  <seq>/metrics.json  单序列指标（sr/auc 普通+dual、fps、丢失/找回统计）
  summary.csv         全部序列汇总（sequence,n_frames,sr,sr_dual,auc,auc_dual,fps + MEAN 行）
终端 stdout 打印汇总表（含均值行）；过程日志一律走 stderr。

说明：--downscale 将帧与 GT 同步缩放（IoU 尺度不变，dual IoU 的 width 用缩放后
帧宽），4K 原始分辨率内存与耗时都很大，建议 0.5。
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

from panotrack.data.vot360 import find_sequences, iter_vot360_sequence
from panotrack.evaluation.metrics import ope_evaluate
from panotrack.pipeline.pipeline import PanoTracker

SUMMARY_COLS = ('sequence', 'n_frames', 'sr', 'sr_dual', 'auc', 'auc_dual', 'fps')


def _log(msg):
    """日志一律走 stderr（私有）。"""
    print(msg, file=sys.stderr, flush=True)


def eval_sequence(seq_dir, out_dir, downscale=1.0, max_frames=None, config=None):
    """单序列 OPE 评测，并落盘 results.txt 与 metrics.json。

    参数：seq_dir —— 360VOT 序列目录；out_dir —— 输出根目录（按序列名建子目录）；
         downscale —— 帧与 GT 同步缩放比例；max_frames —— 调试截断帧数；
         config —— PanoTracker 配置 dict（None 全默认）。
    返回：指标 dict，含 sequence/n_frames/width/height/downscale/
         sr/sr_dual/auc/auc_dual/fps/n_lost/n_recovered。
    """
    seq_dir = Path(seq_dir)
    name = seq_dir.name

    # 流式逐帧评测（大序列不全量入内存）；跨界 GT 的 x1 归一化到 [0, W)
    it = iter_vot360_sequence(seq_dir, downscale=downscale, max_frames=max_frames)
    try:
        _, first_frame, first_gt = next(it)
    except StopIteration:
        raise ValueError(f'序列为空: {seq_dir}')
    height, width = first_frame.shape[:2]

    def _norm_x(box):
        b = [float(v) for v in box]
        b[0] = b[0] % width
        return tuple(b)

    tracker = PanoTracker(config)
    tracker.init(first_frame, _norm_x(first_gt))
    preds = [_norm_x(first_gt)]
    gts = [_norm_x(first_gt)]
    statuses = ['ok']
    t0 = time.perf_counter()
    for _, frame, row in it:
        out = tracker.update(frame)
        preds.append(tuple(float(v) for v in out['bbox']))
        gts.append(_norm_x(row))
        statuses.append(str(out.get('status', 'ok')))
    elapsed = time.perf_counter() - t0
    n = len(preds)
    fps = (n - 1) / max(elapsed, 1e-9) if n > 1 else 0.0

    m = ope_evaluate(np.asarray(preds), np.asarray(gts, dtype=float), width)

    dst = Path(out_dir) / name
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / 'results.txt', 'w', encoding='utf-8') as f:
        for b in preds:
            f.write(f'{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n')
    metrics = {
        'sequence': name, 'n_frames': n,
        'width': int(width), 'height': int(height),
        'downscale': float(downscale),
        'sr': m['sr'], 'sr_dual': m['sr_dual'],
        'auc': m['auc'], 'auc_dual': m['auc_dual'], 'fps': fps,
        'n_lost': sum(1 for s in statuses if s == 'lost'),
        'n_recovered': sum(1 for s in statuses if s == 'recovered'),
    }
    with open(dst / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def write_summary(rows, out_dir):
    """写 summary.csv（含 MEAN 行）并生成终端汇总表格文本。

    参数：rows —— eval_sequence 返回的指标 dict 列表；out_dir —— 输出目录。
    返回：(csv_path, table_text)；table_text 含表头、各序列行与 MEAN 行。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'summary.csv'
    mean = {k: float(np.mean([r[k] for r in rows]))
            for k in ('sr', 'sr_dual', 'auc', 'auc_dual', 'fps')}
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLS)
        for r in rows:
            w.writerow([r['sequence'], r['n_frames'],
                        f"{r['sr']:.4f}", f"{r['sr_dual']:.4f}",
                        f"{r['auc']:.4f}", f"{r['auc_dual']:.4f}",
                        f"{r['fps']:.2f}"])
        w.writerow(['MEAN', '', f"{mean['sr']:.4f}", f"{mean['sr_dual']:.4f}",
                    f"{mean['auc']:.4f}", f"{mean['auc_dual']:.4f}",
                    f"{mean['fps']:.2f}"])

    header = (f"{'SEQ':<12s}{'FRAMES':>8s}{'SR':>8s}{'SR_dual':>9s}"
              f"{'AUC':>8s}{'AUC_dual':>10s}{'FPS':>8s}")
    sep = '-' * len(header)
    lines = [header, sep]
    for r in rows:
        lines.append(f"{r['sequence']:<12s}{r['n_frames']:>8d}"
                     f"{r['sr']:>8.3f}{r['sr_dual']:>9.3f}{r['auc']:>8.3f}"
                     f"{r['auc_dual']:>10.3f}{r['fps']:>8.1f}")
    lines.append(sep)
    lines.append(f"{'MEAN(' + str(len(rows)) + ')':<12s}{'':>8s}"
                 f"{mean['sr']:>8.3f}{mean['sr_dual']:>9.3f}{mean['auc']:>8.3f}"
                 f"{mean['auc_dual']:>10.3f}{mean['fps']:>8.1f}")
    return csv_path, '\n'.join(lines)


def select_sequences(data_root, seqs_arg):
    """按 --seqs 参数筛选序列目录（私有）。

    参数：data_root —— 数据集根目录；seqs_arg —— 'all' 或逗号分隔序列名
         （数字名忽略前导零匹配，如 '1' 可匹配 '001'）。
    返回：list[Path]（按序列名排序）。
    """
    seq_dirs = find_sequences(data_root)
    if seqs_arg.strip().lower() == 'all':
        return seq_dirs
    wanted = [s.strip() for s in seqs_arg.split(',') if s.strip()]
    norm = {(w.lstrip('0') or '0') for w in wanted}
    return [d for d in seq_dirs
            if d.name in wanted or (d.name.lstrip('0') or '0') in norm]


def main(argv=None):
    """批量评测入口：发现序列 -> 逐序列 OPE -> 汇总表与 summary.csv。

    参数：argv 命令行参数（None 取 sys.argv）。
    返回：退出码（0 正常；1 无成功序列；2 数据目录无序列/匹配为空）。
    """
    p = argparse.ArgumentParser(
        description='360VOT 本地批量评测（PanoTracker OPE，普通/dual 双口径）')
    p.add_argument('--data', default=str(PROJECT_ROOT / 'data360'),
                   help='数据集根目录（默认 <项目>/data360）')
    p.add_argument('--seqs', default='all',
                   help="'all' 或逗号分隔序列名，如 001,003（默认 all）")
    p.add_argument('--downscale', type=float, default=1.0,
                   help='帧与 GT 同步缩放比例（如 0.5 提速；IoU 尺度不变，默认 1.0）')
    p.add_argument('--max-frames', type=int, default=None,
                   help='每序列最多评测帧数（调试用，默认全量）')
    p.add_argument('--out', default=str(PROJECT_ROOT / 'runs' / '360vot'),
                   help='输出目录（默认 <项目>/runs/360vot）')
    p.add_argument('--config', default=None,
                   help='PanoTracker 配置 JSON 路径（覆盖默认键，如 {"refine": false}）')
    args = p.parse_args(argv)

    config = None
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        _log(f'使用配置: {args.config} -> {config}')

    seq_dirs = select_sequences(args.data, args.seqs)
    if not seq_dirs:
        _log(f'未在 {args.data} 发现匹配的 360VOT 序列（--seqs {args.seqs}）。\n'
             f'请先运行：python scripts/download_360vot.py --extract')
        return 2
    _log(f'共 {len(seq_dirs)} 个序列：' + ', '.join(d.name for d in seq_dirs))

    rows = []
    for k, d in enumerate(seq_dirs, 1):
        _log(f'[{k}/{len(seq_dirs)}] {d.name} 评测中...')
        try:
            r = eval_sequence(d, args.out, downscale=args.downscale,
                              max_frames=args.max_frames, config=config)
        except (FileNotFoundError, ValueError) as e:
            _log(f'  跳过 {d.name}: {e}')
            continue
        rows.append(r)
        _log(f"  {d.name}: frames={r['n_frames']} sr={r['sr']:.3f} "
             f"sr_dual={r['sr_dual']:.3f} auc={r['auc']:.3f} "
             f"auc_dual={r['auc_dual']:.3f} fps={r['fps']:.1f}")
    if not rows:
        _log('没有成功评测的序列')
        return 1
    csv_path, table = write_summary(rows, args.out)
    print(table)
    _log(f'汇总已写入 {csv_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
