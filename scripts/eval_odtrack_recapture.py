#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ODTrack + 重捕获 wrapper 的全量 360VOT 评测入口（服务器运行）。

与 scripts/odtrack_360vot.py 同协议（ERP 三平铺、严格 OPE、双口径评分），
tracker 换为 OdtrackRecaptureTracker（可靠性门控 + 球面重捕获）。

用法（服务器，GPU）：
  python scripts/eval_odtrack_recapture.py \
    --odtrack-root /path/to/odtrack \
    --data /path/to/data360 \
    --checkpoint /path/to/ODTrack_ep0300.pth.tar \
    --config baseline --seqs all --gpu 0 --out runs/odtrack_recapture_120

输出：<out>/<seq>/results.txt + metrics.json + summary.csv（与评分器一致），
另附每序列 recapture_stats.json（lost/recovered/false_recovery 统计）。
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
    """ODTrack 上游兼容补丁（与 odtrack_360vot.py 一致）。"""
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
        server = types.ModuleType("visdom.server")
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = server
    if "lib.vis.visdom_cus" not in sys.modules:
        visdom_cus = types.ModuleType("lib.vis.visdom_cus")
        visdom_cus.Visdom = type("Visdom", (),
                                 {"__init__": lambda self, *a, **k: None})
        sys.modules["lib.vis.visdom_cus"] = visdom_cus


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--odtrack-root", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default="baseline")
    p.add_argument("--seqs", default="all")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", required=True)
    # recapture 参数（默认值见 recapture.py）
    p.add_argument("--run-len", type=int, default=5)
    p.add_argument("--search-interval", type=int, default=5)
    p.add_argument("--observe-frames", type=int, default=3)
    p.add_argument("--anchor-min-sim", type=float, default=0.5)
    p.add_argument("--recapture-min-score", type=float, default=0.45)
    p.add_argument("--motion-max-deg", type=float, default=90.0)
    args = p.parse_args(argv)

    _patch_torch_six()
    import torch
    torch.cuda.set_device(args.gpu)
    root = Path(args.odtrack_root).resolve()
    sys.path.insert(0, str(root))

    from panotrack.data.vot360 import find_sequences, iter_vot360_sequence
    from panotrack.evaluation.metrics import ope_evaluate
    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack
    from integrations.odtrack.recapture import OdtrackRecaptureTracker

    update_config_from_file(root / "experiments" / "odtrack"
                            / f"{args.config}.yaml")
    params = _build_params(cfg, args.checkpoint)

    seq_dirs = find_sequences(args.data)
    if args.seqs.lower() != "all":
        wanted = {s.strip().zfill(4) for s in args.seqs.split(",") if s.strip()}
        seq_dirs = [d for d in seq_dirs if d.name.zfill(4) in wanted]
    if not seq_dirs:
        raise SystemExit("no matching 360VOT sequences")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, seq_dir in enumerate(seq_dirs, 1):
        seq = seq_dir.name
        print(f"[{index}/{len(seq_dirs)}] {seq}", flush=True)
        it = iter_vot360_sequence(seq_dir, downscale=args.downscale,
                                  max_frames=args.max_frames)
        _, frame, first_gt = next(it)
        height, width = frame.shape[:2]
        tracker = OdtrackRecaptureTracker(
            ODTrack(params), run_len=args.run_len,
            search_interval=args.search_interval,
            observe_frames=args.observe_frames,
            anchor_min_sim=args.anchor_min_sim,
            recapture_min_score=args.recapture_min_score,
            motion_max_deg=args.motion_max_deg)
        tracker.init(frame, first_gt)
        preds = [[float(first_gt[0]) % width, float(first_gt[1]),
                  float(first_gt[2]), float(first_gt[3])]]
        gts = [np.asarray(first_gt, dtype=float)]
        statuses = ["ok"]
        torch.cuda.synchronize(args.gpu)
        t0 = time.perf_counter()
        for _, image, gt in it:
            out = tracker.update(image)
            box = [float(v) for v in out["bbox"]]
            box[0] %= width
            preds.append(box)
            gts.append(np.asarray(gt, dtype=float))
            statuses.append(str(out["status"]))
        torch.cuda.synchronize(args.gpu)
        elapsed = max(time.perf_counter() - t0, 1e-9)
        pred_arr = np.asarray(preds, dtype=float)
        gt_arr = np.asarray(gts, dtype=float)
        pred_arr[:, 0] %= width
        gt_arr[:, 0] %= width
        metrics = ope_evaluate(pred_arr, gt_arr, width)
        fps = (len(preds) - 1) / elapsed
        n_lost = sum(1 for s in statuses if s == "lost")
        n_recovered = sum(1 for s in statuses if s == "recovered")
        dst = out_dir / seq
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "results.txt", "w", encoding="utf-8") as handle:
            for row in pred_arr:
                handle.write(",".join(f"{v:.6f}" for v in row) + "\n")
        payload = {
            "sequence": seq, "n_frames": len(preds),
            "width": int(width), "height": int(height),
            "downscale": float(args.downscale),
            "sr": float(metrics["sr"]), "sr_dual": float(metrics["sr_dual"]),
            "auc": float(metrics["auc"]), "auc_dual": float(metrics["auc_dual"]),
            "fps": float(fps),
            "n_lost": n_lost, "n_recovered": n_recovered,
        }
        with open(dst / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        with open(dst / "recapture_stats.json", "w", encoding="utf-8") as handle:
            json.dump({"n_lost_frames": n_lost, "n_recovered_events": n_recovered},
                      handle, ensure_ascii=False, indent=2)
        rows.append(payload)
        print(f"  AUC={metrics['auc']:.4f} SR={metrics['sr']:.4f} "
              f"FPS={fps:.2f} lost={n_lost} recovered={n_recovered}", flush=True)
        del tracker
        torch.cuda.empty_cache()

    keys = ("sr", "sr_dual", "auc", "auc_dual", "fps")
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sequence", "n_frames", *keys, "n_lost", "n_recovered"))
        for row in rows:
            writer.writerow([row["sequence"], row["n_frames"]] +
                            [f"{row[k]:.6f}" for k in keys] +
                            [row["n_lost"], row["n_recovered"]])
        writer.writerow(["MEAN", len(rows)] +
                        [f"{np.mean([r[k] for r in rows]):.6f}" for k in keys] +
                        [int(np.mean([r["n_lost"] for r in rows])),
                         int(np.mean([r["n_recovered"] for r in rows]))])
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
