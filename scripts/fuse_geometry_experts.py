#!/usr/bin/env python3
"""Fuse baseline and seam-aware tracker outputs with a geometry-only soft router."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import find_sequences, load_vot360_annotations  # noqa: E402
from scripts.score_external_results import find_result, parse_boxes  # noqa: E402


def seam_weight(box, panorama_width, search_factor=4.0, transition=0.25):
    """Return a smooth seam-expert weight from predicted search geometry."""
    x, _, width, height = (float(value) for value in box)
    center = (x + width / 2.0) % panorama_width
    distance = min(center, panorama_width - center)
    half_crop = max(1.0, np.sqrt(max(0.0, width * height))
                    * float(search_factor) / 2.0)
    band = max(1.0, half_crop * float(transition))
    # Weight is one when the search crop crosses the seam by at least ``band``
    # and zero when it stays at least ``band`` away.  The middle is linear and
    # therefore auditable; it does not use image content or ground truth.
    return float(np.clip((half_crop + band - distance) / (2.0 * band), 0.0, 1.0))


def circular_blend_box(baseline, seam, alpha, panorama_width):
    """Blend box centers on S1 longitude and the other fields linearly."""
    baseline = np.asarray(baseline, dtype=float)
    seam = np.asarray(seam, dtype=float)
    center_a = (baseline[0] + baseline[2] / 2.0) % panorama_width
    center_b = (seam[0] + seam[2] / 2.0) % panorama_width
    angle_a = center_a * (2.0 * np.pi / panorama_width)
    angle_b = center_b * (2.0 * np.pi / panorama_width)
    vector = ((1.0 - alpha) * np.array([np.cos(angle_a), np.sin(angle_a)])
              + alpha * np.array([np.cos(angle_b), np.sin(angle_b)]))
    if np.linalg.norm(vector) < 1e-9:
        center_x = center_b if alpha >= 0.5 else center_a
    else:
        center_x = (np.arctan2(vector[1], vector[0]) % (2.0 * np.pi)) \
            * panorama_width / (2.0 * np.pi)
    width = (1.0 - alpha) * baseline[2] + alpha * seam[2]
    y = (1.0 - alpha) * baseline[1] + alpha * seam[1]
    height = (1.0 - alpha) * baseline[3] + alpha * seam[3]
    return np.array([(center_x - width / 2.0) % panorama_width,
                     y, width, height], dtype=float)


def fuse_sequence(baseline, seam, panorama_width, search_factor=4.0,
                  transition=0.25):
    if baseline.shape != seam.shape:
        raise ValueError(f'expert shape mismatch: {baseline.shape} != {seam.shape}')
    fused = np.empty_like(baseline, dtype=float)
    weights = np.zeros(len(baseline), dtype=float)
    fused[0] = baseline[0]
    for index in range(1, len(baseline)):
        # Routing uses the previous fused state, exactly the information that
        # is available before extracting the next search crop online.
        alpha = seam_weight(
            fused[index - 1], panorama_width, search_factor, transition)
        weights[index] = alpha
        fused[index] = circular_blend_box(
            baseline[index], seam[index], alpha, panorama_width)
    return fused, weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--baseline-root', required=True)
    parser.add_argument('--seam-root', required=True)
    parser.add_argument('--seqs', default='all')
    parser.add_argument('--search-factor', type=float, default=4.0)
    parser.add_argument('--transition', type=float, default=0.25)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    wanted = None if args.seqs.lower() == 'all' else {
        value.strip().zfill(4) for value in args.seqs.split(',') if value.strip()}
    seq_dirs = [path for path in find_sequences(args.data)
                if wanted is None or path.name.zfill(4) in wanted]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for seq_dir in seq_dirs:
        sequence = seq_dir.name.zfill(4)
        frame_paths, gt = load_vot360_annotations(seq_dir)
        baseline = parse_boxes(find_result(args.baseline_root, sequence))
        seam = parse_boxes(find_result(args.seam_root, sequence))
        if len(baseline) != len(gt) or len(seam) != len(gt):
            raise ValueError(f'{sequence}: expert/GT length mismatch')
        with Image.open(frame_paths[0]) as image:
            panorama_width = int(image.size[0])
        fused, weights = fuse_sequence(
            baseline, seam, panorama_width, args.search_factor, args.transition)
        seq_out = out_root / sequence
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(seq_out / 'results.txt', fused, delimiter=',', fmt='%.9f')
        np.savetxt(seq_out / 'router_weights.txt', weights, fmt='%.9f')
        print(sequence, f'mean_seam_weight={weights.mean():.4f}',
              f'active_fraction={np.mean(weights > 0.0):.4f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
