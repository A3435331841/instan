#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official ODTrack checkpoint on the local 360VOT protocol.

ODTrack's upstream evaluator expects ordinary planar SOT datasets.  This
adapter keeps the upstream model and scorer, but tiles each ERP frame three
times horizontally so crops crossing the seam remain valid.  Predictions are
mapped back modulo the original ERP width before being written.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _patch_torch_six() -> None:
    """Compatibility shim for ODTrack's pre-PyTorch-2.0 import."""
    try:
        import torch._six  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        six = types.ModuleType("torch._six")
        six.string_classes = (str,)
        six.int_classes = (int,)
        sys.modules["torch._six"] = six
    # The upstream tracker imports its optional Visdom debug UI even when
    # debug mode is disabled.  Keep evaluation headless without pulling in
    # the full web UI dependency.
    if "visdom" not in sys.modules:
        visdom = types.ModuleType("visdom")
        visdom.__path__ = []
        visdom.Visdom = object
        server = types.ModuleType("visdom.server")
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = server
    if "lib.vis.visdom_cus" not in sys.modules:
        visdom_cus = types.ModuleType("lib.vis.visdom_cus")
        visdom_cus.Visdom = type("Visdom", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["lib.vis.visdom_cus"] = visdom_cus


def _parse_seqs(value: str, find_sequences):
    if value.strip().lower() == "all":
        return find_sequences
    wanted = {item.strip().zfill(4) for item in value.split(",") if item.strip()}
    return [path for path in find_sequences if path.name.zfill(4) in wanted]


def _build_params(cfg, checkpoint):
    from lib.test.utils.params import TrackerParams

    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(checkpoint)
    params.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0
    return params


def _tile_box(gt, width):
    x, y, w, h = (float(v) for v in gt)
    return [x % width + width, y, w, h]


def run_sequence(seq_dir, tracker, iter_vot360_sequence, ope_evaluate,
                 out_dir, downscale=1.0, max_frames=None, gpu=0):
    it = iter_vot360_sequence(seq_dir, downscale=downscale,
                               max_frames=max_frames)
    _, frame, first_gt = next(it)
    height, width = frame.shape[:2]
    tiled = np.concatenate((frame, frame, frame), axis=1)
    tracker.initialize(tiled, {"init_bbox": _tile_box(first_gt, width)})
    preds = [[float(first_gt[0]) % width, float(first_gt[1]),
              float(first_gt[2]), float(first_gt[3])]]
    gts = [np.asarray(first_gt, dtype=float)]
    import torch
    torch.cuda.synchronize(gpu)
    t0 = time.perf_counter()
    for _, image, gt in it:
        tiled = np.concatenate((image, image, image), axis=1)
        out = tracker.track(tiled)
        pred = np.asarray(out["target_bbox"], dtype=float).reshape(4)
        pred[0] %= width
        preds.append(pred.tolist())
        gts.append(np.asarray(gt, dtype=float))
    torch.cuda.synchronize(gpu)
    elapsed = time.perf_counter() - t0
    pred_arr = np.asarray(preds, dtype=float)
    gt_arr = np.asarray(gts, dtype=float)
    pred_arr[:, 0] %= width
    gt_arr[:, 0] %= width
    metrics = ope_evaluate(pred_arr, gt_arr, width)
    fps = (len(preds) - 1) / max(elapsed, 1e-9)

    dst = Path(out_dir) / Path(seq_dir).name
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "results.txt", "w", encoding="utf-8") as handle:
        for row in pred_arr:
            handle.write(",".join(f"{v:.6f}" for v in row) + "\n")
    payload = {
        "sequence": Path(seq_dir).name,
        "n_frames": len(preds),
        "width": int(width),
        "height": int(height),
        "downscale": float(downscale),
        "sr": float(metrics["sr"]),
        "sr_dual": float(metrics["sr_dual"]),
        "auc": float(metrics["auc"]),
        "auc_dual": float(metrics["auc_dual"]),
        "fps": float(fps),
        "n_lost": 0,
        "n_recovered": 0,
    }
    with open(dst / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--odtrack-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="baseline")
    parser.add_argument("--seqs", default="all")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--downscale", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    _patch_torch_six()
    import torch
    torch.cuda.set_device(args.gpu)
    root = Path(args.odtrack_root).resolve()
    sys.path.insert(0, str(root))

    from panotrack.data.vot360 import find_sequences, iter_vot360_sequence
    from panotrack.evaluation.metrics import ope_evaluate
    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack

    update_config_from_file(root / "experiments" / "odtrack" / f"{args.config}.yaml")
    params = _build_params(cfg, args.checkpoint)
    sequences = _parse_seqs(args.seqs, find_sequences(args.data))
    if not sequences:
        raise SystemExit("no matching 360VOT sequences")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seq_dir in enumerate(sequences, 1):
        print(f"[{index}/{len(sequences)}] {seq_dir.name}", flush=True)
        tracker = ODTrack(params)
        row = run_sequence(seq_dir, tracker, iter_vot360_sequence,
                           ope_evaluate, out_dir, args.downscale,
                           args.max_frames, args.gpu)
        rows.append(row)
        print(f"  AUC={row['auc']:.4f} SR={row['sr']:.4f} FPS={row['fps']:.2f}",
              flush=True)
        del tracker
        torch.cuda.empty_cache()

    keys = ("sr", "sr_dual", "auc", "auc_dual", "fps")
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sequence", "n_frames", *keys))
        for row in rows:
            writer.writerow([row["sequence"], row["n_frames"]] +
                            [f"{row[key]:.6f}" for key in keys])
        writer.writerow(["MEAN", len(rows)] +
                        [f"{np.mean([r[key] for r in rows]):.6f}" for key in keys])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

