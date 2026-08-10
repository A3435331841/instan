#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ODTrack 的 360VOT ERP 三平铺适配器。

适配器不修改 ODTrack 主干，只负责：加载上游 tracker、将 ERP 帧水平三平铺、
把首帧框放到中间副本、运行逐帧跟踪、把预测框横坐标折回原始 ERP 宽度，并写出
results.txt/metrics.json。上游源码和权重通过命令行传入，不进入本仓库。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _install_compat():
    import types
    try:
        import torch._six  # noqa: F401
    except ModuleNotFoundError:
        six = types.ModuleType("torch._six")
        six.string_classes = (str,)
        six.int_classes = (int,)
        sys.modules["torch._six"] = six
    if "visdom" not in sys.modules:
        visdom = types.ModuleType("visdom")
        visdom.__path__ = []
        visdom.Visdom = object
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = types.ModuleType("visdom.server")
    if "lib.vis.visdom_cus" not in sys.modules:
        mod = types.ModuleType("lib.vis.visdom_cus")
        mod.Visdom = type("Visdom", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["lib.vis.visdom_cus"] = mod


def _tile_box(box, width):
    x, y, w, h = (float(v) for v in box)
    return [x % width + width, y, w, h]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--odtrack-root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default="baseline")
    p.add_argument("--seqs", default="all")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-frames", type=int, default=None)
    args = p.parse_args(argv)
    _install_compat()
    import torch
    torch.cuda.set_device(args.gpu)
    root = Path(args.odtrack_root).resolve()
    sys.path.insert(0, str(root))
    from panotrack.data.vot360 import find_sequences, iter_vot360_sequence
    from panotrack.evaluation.metrics import ope_evaluate
    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack
    from lib.test.utils.params import TrackerParams

    update_config_from_file(root / "experiments" / "odtrack" / f"{args.config}.yaml")
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(Path(args.checkpoint).resolve())
    params.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0
    wanted = None if args.seqs.lower() == "all" else {
        x.strip().zfill(4) for x in args.seqs.split(",") if x.strip()}
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for seq_dir in find_sequences(args.data):
        seq = seq_dir.name.zfill(4)
        if wanted is not None and seq not in wanted:
            continue
        frames = iter_vot360_sequence(seq_dir, max_frames=args.max_frames)
        _, frame, first_gt = next(frames)
        height, width = frame.shape[:2]
        tracker = ODTrack(params)
        tracker.initialize(np.concatenate((frame, frame, frame), axis=1),
                           {"init_bbox": _tile_box(first_gt, width)})
        predictions = [np.asarray(first_gt, dtype=float)]
        targets = [np.asarray(first_gt, dtype=float)]
        torch.cuda.synchronize(args.gpu)
        start = time.perf_counter()
        for _, image, gt in frames:
            output = tracker.track(np.concatenate((image, image, image), axis=1))
            pred = np.asarray(output["target_bbox"], dtype=float).reshape(4)
            pred[0] %= width
            predictions.append(pred)
            targets.append(np.asarray(gt, dtype=float))
        torch.cuda.synchronize(args.gpu)
        pred = np.asarray(predictions)
        gt = np.asarray(targets)
        metrics = ope_evaluate(pred, gt, width)
        elapsed = max(time.perf_counter() - start, 1e-9)
        dst = out_root / seq
        dst.mkdir(parents=True, exist_ok=True)
        np.savetxt(dst / "results.txt", pred, delimiter=",", fmt="%.9f")
        with open(dst / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"sequence": seq, "n_frames": len(pred),
                       "auc": float(metrics["auc"]), "sr": float(metrics["sr"]),
                       "auc_dual": float(metrics["auc_dual"]),
                       "sr_dual": float(metrics["sr_dual"]),
                       "fps": float((len(pred) - 1) / elapsed)}, f,
                      ensure_ascii=False, indent=2)
        print(seq, f"AUC={metrics['auc']:.4f} SR={metrics['sr']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
