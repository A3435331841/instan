#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score external 360VOT tracker outputs with the repository OPE protocol."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import (  # noqa: E402
    find_sequences,
    load_vot360_annotations,
)
from panotrack.evaluation.metrics import ope_evaluate  # noqa: E402


CSV_COLUMNS = (
    'tracker', 'sequence', 'n_frames', 'sr', 'sr_dual', 'auc',
    'auc_dual', 'fps', 'first_frame_linf', 'result_path',
)


def parse_boxes(path):
    """Parse a comma/tab/space separated external ``x,y,w,h`` result file."""
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            fields = line.replace(',', ' ').replace('\t', ' ').split()
            if len(fields) < 4:
                raise ValueError(f'{path}:{line_no}: expected at least 4 columns')
            try:
                row = [float(value) for value in fields[:4]]
            except ValueError as exc:
                raise ValueError(f'{path}:{line_no}: non-numeric box') from exc
            if not np.all(np.isfinite(row)):
                raise ValueError(f'{path}:{line_no}: non-finite box')
            rows.append(row)
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def find_result(root, sequence):
    """Locate the two repository-supported external result layouts."""
    root = Path(root)
    direct = (root / sequence / 'results.txt', root / f'{sequence}.txt')
    for candidate in direct:
        if candidate.is_file():
            return candidate
    matches = [p for p in root.rglob(f'{sequence}.txt')
               if not p.name.endswith('_time.txt')]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f'ambiguous results for {sequence} under {root}: '
            + ', '.join(str(p) for p in matches))
    raise FileNotFoundError(f'no results for {sequence} under {root}')


def infer_fps(result_path, n_frames):
    """Read native tracker timing without mixing it into accuracy scoring."""
    result_path = Path(result_path)
    metrics_path = result_path.parent / 'metrics.json'
    if metrics_path.is_file():
        with open(metrics_path, 'r', encoding='utf-8') as handle:
            value = json.load(handle).get('fps')
        if value is not None:
            return float(value)

    timing_path = result_path.with_name(result_path.stem + '_time.txt')
    if timing_path.is_file():
        times = np.loadtxt(timing_path, dtype=float).reshape(-1)
        if len(times) not in (n_frames, n_frames - 1):
            raise ValueError(
                f'timing length {len(times)} does not match {n_frames} frames: '
                f'{timing_path}')
        if len(times) == n_frames:
            times = times[1:]
        elapsed = float(np.sum(times))
        return (n_frames - 1) / elapsed if elapsed > 0.0 else 0.0
    return float('nan')


def score_sequence(tracker, result_root, seq_dir):
    """Score one sequence, strictly enforcing the official OPE frame count."""
    sequence = Path(seq_dir).name
    frame_paths, gt = load_vot360_annotations(seq_dir)
    result_path = find_result(result_root, sequence)
    pred = parse_boxes(result_path)
    if len(pred) != len(gt):
        raise ValueError(
            f'{tracker}/{sequence}: prediction rows {len(pred)} != GT rows {len(gt)}')

    with Image.open(frame_paths[0]) as image:
        width = int(image.size[0])
    gt = gt.copy()
    pred = pred.copy()
    gt[:, 0] %= width
    pred[:, 0] %= width
    first_frame_linf = float(np.max(np.abs(pred[0] - gt[0])))
    metrics = ope_evaluate(pred, gt, width)
    return {
        'tracker': tracker,
        'sequence': sequence,
        'n_frames': int(len(gt)),
        'sr': metrics['sr'],
        'sr_dual': metrics['sr_dual'],
        'auc': metrics['auc'],
        'auc_dual': metrics['auc_dual'],
        'fps': infer_fps(result_path, len(gt)),
        'first_frame_linf': first_frame_linf,
        'result_path': str(result_path.resolve()),
    }


def aggregate(rows):
    """Return macro averages, one equal-weight vote per sequence."""
    def finite_mean(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float('nan')

    grouped = {}
    for row in rows:
        grouped.setdefault(row['tracker'], []).append(row)
    summary = []
    for tracker, tracker_rows in sorted(grouped.items()):
        summary.append({
            'tracker': tracker,
            'n_sequences': len(tracker_rows),
            **{key: finite_mean([r[key] for r in tracker_rows])
               for key in ('sr', 'sr_dual', 'auc', 'auc_dual', 'fps')},
        })
    summary.sort(key=lambda row: (row['auc'], row['sr']), reverse=True)
    return summary


def write_outputs(rows, out_dir, tracker_roots, selected_sequences):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'scores.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = aggregate(rows)
    protocol = {
        'dataset': '360VOT test',
        'box_format': 'x,y,w,h',
        'first_frame_excluded': True,
        'strict_prediction_length': True,
        'success_threshold': 0.5,
        'auc_thresholds': [round(v, 2) for v in np.linspace(0.0, 1.0, 21)],
        'ordinary_iou': True,
        'dual_iou_horizontal_shifts': ['-W', '0', '+W'],
        'macro_average': 'equal weight per sequence',
        'x_normalization': 'prediction and GT x modulo ERP width',
        'sequences': list(selected_sequences),
        'tracker_roots': {name: str(Path(root).resolve())
                          for name, root in tracker_roots.items()},
    }
    payload = {
        'protocol': protocol,
        'winner_by_ordinary_auc': summary[0]['tracker'] if summary else None,
        'summary': summary,
        'rows': rows,
    }
    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        return value
    with open(out_dir / 'bakeoff.json', 'w', encoding='utf-8') as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False,
                  indent=2, allow_nan=False)
    return payload


def select_sequence_dirs(data_root, requested):
    found = find_sequences(data_root)
    by_name = {path.name.zfill(4): path for path in found}
    names = sorted(by_name) if requested.lower() == 'all' else [
        value.strip().zfill(4) for value in requested.split(',') if value.strip()
    ]
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError('missing data sequences: ' + ','.join(missing))
    return [(name, by_name[name]) for name in names]


def parse_tracker_specs(specs):
    trackers = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError(f'--tracker must be NAME=RESULT_ROOT, got {spec!r}')
        name, root = spec.split('=', 1)
        name, root = name.strip(), root.strip()
        if not name or not root or name in trackers:
            raise ValueError(f'invalid or duplicate tracker spec: {spec!r}')
        trackers[name] = Path(root)
    return trackers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, help='360VOT data root')
    parser.add_argument('--tracker', action='append', required=True,
                        help='NAME=RESULT_ROOT; repeat for each tracker')
    parser.add_argument('--seqs', default='all',
                        help="comma-separated sequence ids or 'all'")
    parser.add_argument('--out', required=True, help='output directory')
    args = parser.parse_args(argv)

    trackers = parse_tracker_specs(args.tracker)
    sequence_dirs = select_sequence_dirs(args.data, args.seqs)
    rows = []
    for tracker, result_root in trackers.items():
        for sequence, seq_dir in sequence_dirs:
            row = score_sequence(tracker, result_root, seq_dir)
            rows.append(row)
            print(f"{tracker:16s} {sequence} AUC={row['auc']:.4f} "
                  f"SR={row['sr']:.4f} FPS={row['fps']:.2f}")

    payload = write_outputs(
        rows, args.out, trackers, [name for name, _ in sequence_dirs])
    for row in payload['summary']:
        print(f"MEAN {row['tracker']:11s} AUC={row['auc']:.4f} "
              f"SR={row['sr']:.4f} FPS={row['fps']:.2f}")
    print('WINNER', payload['winner_by_ordinary_auc'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
