#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方训练集评测 runner（P0-A，2026-08-24）。

对官方 130 条序列（train_real 47 + train_sim 83，1440×720，BFoV GT）做 OPE 评测：
  video.mp4 + init.txt -> tracker 逐帧推理 -> ERP 框 -> 双口径 IoU 评分 -> summary.csv。

官方数据特性（实测）：
  - GT 含 0,0,0,0 丢失帧（全库 11.06%，40/130 条序列）：**评分时跳过**这些帧
    （GT 无效帧不参与 SR/AUC；首帧初始化帧同样不计），但统计进 metrics 供分析；
  - 行号 = 帧号，与 Arena 协议一致。

tracker 后端（--tracker）：
  - mock     : 冻结初始框（管道联通用，不是真跟踪器）
  - gt_echo  : 诊断后端，回显 GT+微扰（验证评分链路，输出 AUC 应≈1；禁止用于上报成绩）
  - odtrack  : 真实 ODTrack 三平铺（arena_protocol 同款加载方式，需 GPU/torch）

用法示例：
  python scripts/eval_official.py --tracker mock --seqs train_real/seq_0001 --max-frames 80
  python scripts/eval_official.py --tracker gt_echo --split valid
  python scripts/eval_official.py --tracker odtrack --split all \
      --odtrack-workspace /opt/odtrack --odtrack-ckpt /opt/models/ODTrack_ep0300.pth.tar
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov  # noqa: E402
from panotrack.evaluation.metrics import auc, dual_iou, iou_xywh, success_rate  # noqa: E402

try:
    import cv2 as cv
except ImportError:
    cv = None

SPLIT_DIR = PROJECT_ROOT / "data360" / "official_split"


# ---------------------------------------------------------------------------
# GT 解析（含丢失帧掩码）
# ---------------------------------------------------------------------------

def load_gt(path: Path):
    """读 GT：返回 (bfov 数组 (N,4), valid 掩码 (N,))。0,0,0,0 -> invalid。"""
    rows, valid = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        f = [float(v) for v in ln.replace(",", " ").split()]
        if len(f) != 4:
            raise ValueError(f"{path}: 非法行 {ln!r}")
        rows.append(f)
        valid.append(f[2] > 0.0 and f[3] > 0.0)
    return np.asarray(rows, dtype=float), np.asarray(valid, dtype=bool)


def load_init(seq_dir: Path):
    """init.txt -> (clon, clat, fov_h, fov_v)。"""
    line = next(ln for ln in (seq_dir / "init.txt").read_text(encoding="utf-8").splitlines() if ln.strip())
    f = [float(v) for v in line.replace(",", " ").split()]
    if len(f) != 4 or f[2] <= 0 or f[3] <= 0:
        raise ValueError(f"{seq_dir}/init.txt 非法: {line!r}")
    return tuple(f)


# ---------------------------------------------------------------------------
# tracker 后端
# ---------------------------------------------------------------------------

class MockTracker:
    """冻结初始框。"""

    def __init__(self, **_):
        pass

    def init(self, frame, erp_box, **_):
        self.box = [float(v) for v in erp_box]

    def track(self, frame, **_):
        return {"target_bbox": list(self.box), "quality": 1.0}


class GtEchoTracker:
    """诊断后端：回显当前帧 GT（+1px 扰动）。仅用于验证评分链路。"""

    def __init__(self, gt_erp=None, **_):
        self.gt_erp = gt_erp  # (N,4) ERP 框, 无效帧为全 0

    def init(self, frame, erp_box, frame_idx=0, **_):
        pass

    def track(self, frame, frame_idx=0, **_):
        b = self.gt_erp[frame_idx]
        if b[2] <= 0:
            return {"target_bbox": [0.0, 0.0, 0.0, 0.0], "quality": 0.0}
        return {"target_bbox": [b[0] + 1.0, b[1] + 1.0, b[2], b[3]], "quality": 1.0}


def build_odtrack_tracker(args):
    """懒加载 ODTrack（与 arena_protocol.py 相同的补丁与参数装配）。"""
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    workspace = Path(args.odtrack_workspace).resolve()
    if not workspace.is_dir():
        raise SystemExit(f"[error] ODTrack workspace 不存在: {workspace}")
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

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
        m = types.ModuleType("lib.vis.visdom_cus")
        m.Visdom = type("Visdom", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["lib.vis.visdom_cus"] = m

    if args.force_cpu:
        import torch
        torch.nn.Module.cuda = lambda self, device=None: self
        torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self

    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack
    from lib.test.utils.params import TrackerParams

    update_config_from_file(workspace / "experiments" / "odtrack" / f"{args.odtrack_config}.yaml")
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(Path(args.odtrack_ckpt))
    params.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0

    class OdtrackAdapter:
        def __init__(self):
            self.tracker = ODTrack(params)
            self.width = None

        def init(self, frame_rgb, erp_box):
            import numpy as _np
            h, w = frame_rgb.shape[:2]
            self.width = w
            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            box = [erp_box[0] % w + w, erp_box[1], erp_box[2], erp_box[3]]
            self.tracker.initialize(tiled, {"init_bbox": box})

        def track(self, frame_rgb):
            import numpy as _np
            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            out = self.tracker.track(tiled)
            box = out.get("target_bbox")
            return {"target_bbox": [float(box[0]) % self.width, float(box[1]),
                                    float(box[2]), float(box[3])],
                    "quality": float(getattr(self.tracker, "last_pred_iou", 1.0))}

    return OdtrackAdapter()


# ---------------------------------------------------------------------------
# 序列选择与主循环
# ---------------------------------------------------------------------------

def select_sequences(args):
    if args.seqs:
        return [s.strip() for s in args.seqs.split(",") if s.strip()]
    if args.split in ("train", "valid"):
        path = SPLIT_DIR / f"seqlist_official_{args.split}.txt"
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # all: 扫描数据目录
    root = Path(args.data)
    seqs = []
    for block in ("train_real", "train_sim"):
        d = root / block
        if d.is_dir():
            seqs.extend(f"{block}/{p.name}" for p in sorted(d.iterdir())
                        if p.is_dir() and (p / "video.mp4").is_file())
    return seqs


def run_sequence(seq_rel, data_root, tracker_factory, max_frames=None):
    seq_dir = Path(data_root) / seq_rel
    if cv is None:
        raise SystemExit("需要 cv2 解码 video.mp4")

    gt_bfov, gt_valid = load_gt(seq_dir / "groundtruth.txt")
    init_bfov = load_init(seq_dir)
    n_total = len(gt_bfov)
    limit = min(n_total, max_frames) if max_frames else n_total

    # GT BFoV -> ERP 框（无效帧置 0）
    cap = cv.VideoCapture(str(seq_dir / "video.mp4"))
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError(f"解码失败: {seq_dir}")
    H, W = first.shape[:2]
    gt_erp = np.zeros((n_total, 4), dtype=float)
    for i in range(n_total):
        if gt_valid[i]:
            x, y, w, h = erp_bbox_from_bfov(
                BFoV(lon=gt_bfov[i][0], lat=gt_bfov[i][1],
                     fov_h=gt_bfov[i][2], fov_v=gt_bfov[i][3]), W, H)
            gt_erp[i] = [x, y, w, h]

    tracker = tracker_factory(gt_erp=gt_erp)
    init_erp = erp_bbox_from_bfov(BFoV(*init_bfov), W, H)
    tracker.init(cv.cvtColor(first, cv.COLOR_BGR2RGB), init_erp)

    pred_erp = np.zeros((limit, 4), dtype=float)
    pred_erp[0] = init_erp
    times = []
    for i in range(1, limit):
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"{seq_rel}: 第 {i} 帧解码失败")
        t0 = time.time()
        out = tracker.track(cv.cvtColor(frame, cv.COLOR_BGR2RGB), frame_idx=i)
        times.append(time.time() - t0)
        b = out["target_bbox"]
        pred_erp[i] = [float(v) for v in b[:4]]
    cap.release()

    # 掩码 OPE 评分：跳过首帧与 GT 无效帧
    ious, ious_dual = [], []
    for i in range(1, limit):
        if not gt_valid[i]:
            continue
        p, g = pred_erp[i], gt_erp[i]
        if p[2] <= 0 or p[3] <= 0:   # 预测宣告丢失 -> IoU 0
            ious.append(0.0)
            ious_dual.append(0.0)
            continue
        ious.append(iou_xywh(p, g))
        ious_dual.append(dual_iou(p, g, W))

    n_scored = len(ious)
    elapsed = float(np.sum(times))
    metrics = {
        "sequence": seq_rel,
        "n_frames": int(limit),
        "n_scored": n_scored,
        "n_gt_absent": int((~gt_valid[:limit]).sum()),
        "sr": success_rate(ious) if n_scored else float("nan"),
        "auc": auc(ious) if n_scored else float("nan"),
        "sr_dual": success_rate(ious_dual) if n_scored else float("nan"),
        "auc_dual": auc(ious_dual) if n_scored else float("nan"),
        "fps": (limit - 1) / elapsed if elapsed > 0 else float("nan"),
        "resolution": f"{W}x{H}",
    }
    return metrics, pred_erp, gt_valid[:limit], W, H


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=r"D:\instan\初赛数据\train",
                    help="官方训练集根目录（含 train_real/ train_sim/）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "official_eval"))
    ap.add_argument("--tracker", default="mock", choices=["mock", "gt_echo", "odtrack"])
    ap.add_argument("--split", default="all", choices=["all", "train", "valid"])
    ap.add_argument("--seqs", default=None, help="逗号分隔 block/seq 覆盖 split")
    ap.add_argument("--max-frames", type=int, default=None)
    # odtrack 专用
    ap.add_argument("--odtrack-workspace", default="/opt/odtrack")
    ap.add_argument("--odtrack-ckpt", default="/opt/models/ODTrack_ep0300.pth.tar")
    ap.add_argument("--odtrack-config", default="baseline")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force-cpu", action="store_true")
    args = ap.parse_args(argv)

    seqs = select_sequences(args)
    if not seqs:
        raise SystemExit("[error] 没有序列被选中")
    out_dir = Path(args.out) / f"{args.tracker}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tracker == "mock":
        factory = lambda gt_erp: MockTracker()
    elif args.tracker == "gt_echo":
        factory = lambda gt_erp: GtEchoTracker(gt_erp=gt_erp)
    else:
        proto = build_odtrack_tracker(args)
        factory = lambda gt_erp: proto  # 单实例顺序复用（ODTrack 逐序列重建较慢，先共用）

    print(f"[eval_official] tracker={args.tracker} seqs={len(seqs)} out={out_dir}")
    rows = []
    for idx, seq_rel in enumerate(seqs, 1):
        t0 = time.time()
        try:
            metrics, pred_erp, valid, W, H = run_sequence(seq_rel, args.data, factory, args.max_frames)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"  [{idx}/{len(seqs)}] {seq_rel}: FAILED ({exc})", file=sys.stderr)
            continue
        seq_out = out_dir / seq_rel
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(seq_out / "results_erp.txt", pred_erp, fmt="%.2f", delimiter=",")
        with open(seq_out / "bfov.txt", "w", encoding="utf-8") as f:
            for b in pred_erp:
                if b[2] <= 0 or b[3] <= 0:
                    f.write("0.000,0.000,0.000,0.000\n")
                else:
                    bf = bfov_from_erp_bbox(b[0], b[1], b[2], b[3], W, H)
                    f.write(f"{bf.lon:.3f},{bf.lat:.3f},{bf.fov_h:.3f},{bf.fov_v:.3f}\n")
        (seq_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(metrics)
        print(f"  [{idx}/{len(seqs)}] {seq_rel}: AUC={metrics['auc']:.4f} "
              f"SR={metrics['sr']:.4f} FPS={metrics['fps']:.1f} "
              f"({time.time()-t0:.1f}s, absent={metrics['n_gt_absent']})")

    if rows:
        def mean(k):
            vals = [r[k] for r in rows if np.isfinite(r[k])]
            return float(np.mean(vals)) if vals else float("nan")
        summary = {
            "tracker": args.tracker, "n_sequences": len(rows),
            "auc": mean("auc"), "sr": mean("sr"),
            "auc_dual": mean("auc_dual"), "sr_dual": mean("sr_dual"),
            "fps": mean("fps"), "n_gt_absent_total": sum(r["n_gt_absent"] for r in rows),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[summary] {args.tracker}: AUC={summary['auc']:.4f} SR={summary['sr']:.4f} "
              f"dual AUC={summary['auc_dual']:.4f} FPS={summary['fps']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
