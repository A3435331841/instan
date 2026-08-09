#!/usr/bin/env python3
"""Audit 360VOT sequences for seam, pole, scale, and motion exposure."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import find_sequences, load_vot360_annotations  # noqa: E402


def audit_sequence(seq_dir, search_factor=4.0):
    paths, boxes = load_vot360_annotations(seq_dir)
    with Image.open(paths[0]) as image:
        width, height = image.size
    x, y, box_width, box_height = boxes.T
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    crop = np.ceil(np.sqrt(np.maximum(0.0, box_width * box_height)) * search_factor)
    seam_box = (x < 0.0) | (x + box_width > width)
    seam_search = ((center_x - crop / 2.0 < 0.0)
                   | (center_x + crop / 2.0 > width))
    pole_search = ((center_y - crop / 2.0 < 0.0)
                   | (center_y + crop / 2.0 > height))
    latitude = 90.0 - 180.0 * center_y / float(height)

    dx = np.diff(center_x)
    dx = (dx + width / 2.0) % width - width / 2.0
    dy = np.diff(center_y)
    angular_motion = np.hypot(dx * 360.0 / width, dy * 180.0 / height)
    return {
        'sequence': Path(seq_dir).name.zfill(4),
        'n_frames': int(len(boxes)),
        'width': int(width),
        'height': int(height),
        'seam_box_fraction': float(np.mean(seam_box)),
        'seam_search_fraction': float(np.mean(seam_search)),
        'pole_search_fraction': float(np.mean(pole_search)),
        'mean_abs_latitude_deg': float(np.mean(np.abs(latitude))),
        'p95_angular_motion_deg': (float(np.percentile(angular_motion, 95))
                                   if len(angular_motion) else 0.0),
        'median_area_fraction': float(np.median(box_width * box_height)
                                      / float(width * height)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--seqs', default='all')
    parser.add_argument('--search-factor', type=float, default=4.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    requested = None if args.seqs.lower() == 'all' else {
        value.strip().zfill(4) for value in args.seqs.split(',') if value.strip()}
    seq_dirs = [path for path in find_sequences(args.data)
                if requested is None or path.name.zfill(4) in requested]
    rows = [audit_sequence(path, args.search_factor) for path in seq_dirs]
    rows.sort(key=lambda row: row['sequence'])
    if requested is not None:
        found = {row['sequence'] for row in rows}
        if found != requested:
            raise FileNotFoundError('missing sequences: ' + ','.join(sorted(requested - found)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with open(out_dir / 'sequence_geometry.csv', 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        'search_factor': args.search_factor,
        'n_sequences': len(rows),
        'macro_mean': {
            key: float(np.mean([row[key] for row in rows])) if rows else math.nan
            for key in ('seam_box_fraction', 'seam_search_fraction',
                        'pole_search_fraction', 'mean_abs_latitude_deg',
                        'p95_angular_motion_deg', 'median_area_fraction')
        },
        'rows': rows,
    }
    with open(out_dir / 'sequence_geometry.json', 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    for row in sorted(rows, key=lambda item: item['seam_search_fraction'], reverse=True):
        print(f"{row['sequence']} seam_search={row['seam_search_fraction']:.3f} "
              f"pole={row['pole_search_fraction']:.3f} "
              f"motion95={row['p95_angular_motion_deg']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
