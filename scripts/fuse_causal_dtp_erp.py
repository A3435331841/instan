#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the causal DTP-ERP router over three existing tracker streams."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.vot360 import find_sequences, load_vot360_annotations  # noqa: E402
from panotrack.geometry.causal_dtp import CausalDTPRouter  # noqa: E402
from scripts.score_external_results import find_result, parse_boxes  # noqa: E402


def fuse_sequence(od, ue, lightfc, width, height, **kwargs):
    arrays = [np.asarray(v, dtype=float) for v in (od, ue, lightfc)]
    if not (arrays[0].shape == arrays[1].shape == arrays[2].shape):
        raise ValueError('expert shape mismatch')
    router = CausalDTPRouter(width, height, **kwargs)
    fused = np.empty_like(arrays[0])
    chosen = np.empty(len(fused), dtype=np.int8)
    reliability = np.empty((len(fused), 3), dtype=np.float64)
    for i in range(len(fused)):
        fused[i], chosen[i], reliability[i] = router.update([a[i] for a in arrays])
    return fused, chosen, reliability


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--od-root', required=True)
    p.add_argument('--ue-root', required=True)
    p.add_argument('--lightfc-root', required=True)
    p.add_argument('--seqs', default='all')
    p.add_argument('--out', required=True)
    p.add_argument('--hold-frames', type=int, default=3)
    p.add_argument('--blend-alpha', type=float, default=0.18)
    # Conservative by default: the current implementation is a prediction-
    # level router, so it must not replace the accuracy teacher without a
    # large reliability gap.  Training can later lower these thresholds.
    p.add_argument('--teacher-margin', type=float, default=0.90)
    p.add_argument('--recovery-margin', type=float, default=0.20)
    p.add_argument('--reliability-decay', type=float, default=18.0)
    p.add_argument('--geometry-penalty', type=float, default=0.35)
    args = p.parse_args(argv)
    wanted = None if args.seqs.lower() == 'all' else {
        x.strip().zfill(4) for x in args.seqs.split(',') if x.strip()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('data', 'out', 'seqs')}
    with open(out / 'router_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    for seq_dir in find_sequences(args.data):
        seq = seq_dir.name.zfill(4)
        if wanted is not None and seq not in wanted:
            continue
        frame_paths, gt = load_vot360_annotations(seq_dir)
        od = parse_boxes(find_result(args.od_root, seq))
        ue = parse_boxes(find_result(args.ue_root, seq))
        lightfc = parse_boxes(find_result(args.lightfc_root, seq))
        if len(od) != len(gt) or len(ue) != len(gt) or len(lightfc) != len(gt):
            raise ValueError(f'{seq}: expert/GT length mismatch')
        with Image.open(frame_paths[0]) as image:
            width, height = image.size
        fused, chosen, reliability = fuse_sequence(
            od, ue, lightfc, width, height,
            hold_frames=args.hold_frames, blend_alpha=args.blend_alpha,
            teacher_margin=args.teacher_margin,
            recovery_margin=args.recovery_margin,
            reliability_decay=args.reliability_decay,
            geometry_penalty=args.geometry_penalty)
        seq_out = out / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(seq_out / 'results.txt', fused, delimiter=',', fmt='%.9f')
        np.savetxt(seq_out / 'expert_ids.txt', chosen, fmt='%d')
        np.savetxt(seq_out / 'reliability.txt', reliability, delimiter=',', fmt='%.6f')
        print(seq, 'teacher=%.4f student=%.4f scout=%.4f mean_rel=%.4f' % (
            np.mean(chosen == 0), np.mean(chosen == 1), np.mean(chosen == 2),
            float(np.mean(reliability))))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
