#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse external 360VOT predictions with a causal, seam-aware expert router.

The router is deliberately prediction-only: it never reads ground truth.  The
ODTrack stream is the accuracy expert and UETrack ERP-wrap is the recovery
expert.  A candidate is penalized for an implausible jump, abrupt scale change,
and disagreement with the other experts.  Hysteresis prevents frame-to-frame
expert oscillation near the ERP seam.
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
    return (float(box[0]) + float(box[2]) / 2.0) % width


def _circ_dist(a, b, width):
    d = abs(float(a) - float(b)) % width
    return min(d, width - d)


def _blend(a, b, alpha, width):
    """Blend two boxes using circular longitude for x center."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ca, cb = _center(a, width), _center(b, width)
    da = ((cb - ca + width / 2.0) % width) - width / 2.0
    center = (ca + float(alpha) * da) % width
    size = (1.0 - float(alpha)) * a[2:] + float(alpha) * b[2:]
    y = (1.0 - float(alpha)) * a[1] + float(alpha) * b[1]
    return np.array([(center - size[0] / 2.0) % width, y, size[0], size[1]])


def _candidate_cost(candidate, other, previous, width, jump_scale,
                    scale_scale, disagreement_scale):
    prev_diag = max(1.0, float(np.sqrt(max(previous[2] * previous[3], 1.0))))
    jump = _circ_dist(_center(candidate, width), _center(previous, width), width)
    jump /= prev_diag
    # Log scale change is symmetric and robust to growing/shrinking boxes.
    old_area = max(1.0, float(previous[2] * previous[3]))
    new_area = max(1.0, float(candidate[2] * candidate[3]))
    scale_change = abs(np.log(new_area / old_area))
    disagreement = _circ_dist(_center(candidate, width), _center(other, width), width)
    disagreement /= prev_diag
    # Lower is better.  The constants are exposed as CLI knobs for screening.
    return (jump / max(jump_scale, 1e-6)
            + scale_change / max(scale_scale, 1e-6)
            + disagreement / max(disagreement_scale, 1e-6))


def fuse_sequence(od, ue, lightfc, width, jump_scale=3.0,
                  scale_scale=0.70, disagreement_scale=5.0,
                  switch_margin=1.0, hold_frames=3, blend_alpha=0.20,
                  lightfc_penalty=6.0, od_confidence=None,
                  confidence_threshold=None, min_low_confidence_run=1):
    """Return fused boxes and integer expert ids (0=OD, 1=UE, 2=LightFC)."""
    arrays = [np.asarray(v, dtype=float) for v in (od, ue, lightfc)]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError('expert shape mismatch')
    if arrays[0].ndim != 2 or arrays[0].shape[1] != 4:
        raise ValueError('experts must be [frames,4]')
    fused = np.empty_like(arrays[0])
    chosen = np.zeros(len(fused), dtype=np.int8)
    if od_confidence is not None:
        od_confidence = np.asarray(od_confidence, dtype=float).reshape(-1)
        if len(od_confidence) != len(fused):
            raise ValueError('OD confidence length mismatch')
    low_run = 0
    fused[0] = arrays[0][0]
    active = 0
    hold = 0
    for i in range(1, len(fused)):
        previous = fused[i - 1]
        # Pair each expert with the strongest independent alternative.
        costs = np.array([
            _candidate_cost(arrays[0][i], arrays[1][i], previous, width,
                            jump_scale, scale_scale, disagreement_scale),
            _candidate_cost(arrays[1][i], arrays[0][i], previous, width,
                            jump_scale, scale_scale, disagreement_scale),
            _candidate_cost(arrays[2][i], arrays[0][i], previous, width,
                            jump_scale, scale_scale, disagreement_scale),
        ])
        # LightFC is a low-cost scout, not the default accuracy expert.  Keep
        # it available only for catastrophic disagreement of both main experts.
        costs[2] += float(lightfc_penalty)
        if (od_confidence is not None and confidence_threshold is not None
                and od_confidence[i] < float(confidence_threshold)):
            low_run += 1
        else:
            low_run = 0
        best = int(np.argmin(costs))
        # OD is the default accuracy expert.  Switch only when it is clearly
        # less plausible, and hold that decision for a few frames.
        confidence_recovery = (active == 0 and low_run >= int(min_low_confidence_run)
                               and costs[1] <= costs[0] + float(switch_margin))
        if active == 0 and (costs[best] + switch_margin < costs[0]
                            or confidence_recovery):
            active = best
            hold = int(hold_frames)
        elif active != 0:
            hold = max(0, hold - 1)
            if hold == 0 and costs[0] + switch_margin < costs[active]:
                active = 0
        elif best == 0:
            active = 0
        chosen[i] = active
        primary = arrays[active][i]
        # A small causal correction toward OD is useful after recovery, but is
        # disabled when experts strongly disagree (to avoid averaging a jump).
        if active != 0:
            od_dist = _circ_dist(_center(arrays[0][i], width),
                                 _center(primary, width), width)
            if od_dist / max(1.0, np.sqrt(max(primary[2] * primary[3], 1.0))) < 2.0:
                primary = _blend(primary, arrays[0][i], blend_alpha, width)
        fused[i] = primary
    return fused, chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--od-root', required=True)
    parser.add_argument('--ue-root', required=True)
    parser.add_argument('--lightfc-root', required=True)
    parser.add_argument('--seqs', default='all')
    parser.add_argument('--jump-scale', type=float, default=3.0)
    parser.add_argument('--scale-scale', type=float, default=0.70)
    parser.add_argument('--disagreement-scale', type=float, default=5.0)
    parser.add_argument('--switch-margin', type=float, default=1.0)
    parser.add_argument('--hold-frames', type=int, default=3)
    parser.add_argument('--blend-alpha', type=float, default=0.20)
    parser.add_argument('--lightfc-penalty', type=float, default=6.0)
    parser.add_argument('--od-confidence-root', default=None,
                        help='optional root with per-sequence confidence.txt')
    parser.add_argument('--confidence-threshold', type=float, default=None)
    parser.add_argument('--min-low-confidence-run', type=int, default=2)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    wanted = None if args.seqs.lower() == 'all' else {
        value.strip().zfill(4) for value in args.seqs.split(',') if value.strip()}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    configs = vars(args).copy()
    configs.pop('data', None)
    configs.pop('out', None)
    configs.pop('seqs', None)
    with open(out_root / 'router_config.json', 'w', encoding='utf-8') as handle:
        json.dump(configs, handle, ensure_ascii=False, indent=2)
    for seq_dir in find_sequences(args.data):
        sequence = seq_dir.name.zfill(4)
        if wanted is not None and sequence not in wanted:
            continue
        frame_paths, gt = load_vot360_annotations(seq_dir)
        od = parse_boxes(find_result(args.od_root, sequence))
        ue = parse_boxes(find_result(args.ue_root, sequence))
        lightfc = parse_boxes(find_result(args.lightfc_root, sequence))
        confidence = None
        if args.od_confidence_root:
            confidence_path = Path(args.od_confidence_root) / sequence / 'confidence.txt'
            if not confidence_path.is_file():
                raise FileNotFoundError(f'missing OD confidence: {confidence_path}')
            confidence = np.loadtxt(confidence_path, dtype=float).reshape(-1)
        if len(od) != len(gt) or len(ue) != len(gt) or len(lightfc) != len(gt):
            raise ValueError(f'{sequence}: expert/GT length mismatch')
        with Image.open(frame_paths[0]) as image:
            width = int(image.size[0])
        fused, chosen = fuse_sequence(
            od, ue, lightfc, width, args.jump_scale, args.scale_scale,
            args.disagreement_scale, args.switch_margin, args.hold_frames,
            args.blend_alpha, args.lightfc_penalty, confidence,
            args.confidence_threshold, args.min_low_confidence_run)
        seq_out = out_root / sequence
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(seq_out / 'results.txt', fused, delimiter=',', fmt='%.9f')
        np.savetxt(seq_out / 'expert_ids.txt', chosen, fmt='%d')
        print(sequence, 'od_fraction=%.4f ue_fraction=%.4f lightfc_fraction=%.4f' % (
            np.mean(chosen == 0), np.mean(chosen == 1), np.mean(chosen == 2)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
