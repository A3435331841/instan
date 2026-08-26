#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测 SUTrack-B224 LoRA checkpoint。

用法:
    cd /data/sutrack_src_20260825/SUTrack
    CUDA_VISIBLE_DEVICES=1 python /data/pano360/scripts/eval_sutrack_lora.py \
        --lora-ckpt /data/sutrack_lora_training/sutrack_lora_ep0001.pth \
        --data /data/traindata/train --split valid --gpu 0
"""
import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


# 内联 BFoV 转换（与 arena_protocol.py 一致）
def _wrap_lon(lon):
    return 180.0 - np.mod(180.0 - np.asarray(lon, dtype=np.float64), 360.0)

def _lonlat_to_unit(lon, lat):
    lon_r, lat_r = np.deg2rad(lon), np.deg2rad(lat)
    return np.cos(lat_r) * np.cos(lon_r), np.sin(lat_r), np.cos(lat_r) * np.sin(lon_r)

def _tangent_frame(lon, lat):
    cx, cy, cz = _lonlat_to_unit(lon, lat)
    c = np.array([cx, cy, cz], dtype=np.float64)
    east = np.cross(c, [0, 1, 0])
    n = np.linalg.norm(east)
    east = east / n if n > 1e-9 else np.array([0, 0, 1.0 if cy > 0 else -1.0])
    north = np.cross(east, c)
    return c, east, north

def bfov_from_erp_bbox(x, y, w, h, erp_w, erp_h):
    cu, cv_ = (x + w / 2.0) % erp_w, y + h / 2.0
    lon_c = float(_wrap_lon(cu / erp_w * 360.0 - 180.0))
    lat_c = float(np.clip(90.0 - cv_ / erp_h * 180.0, -90, 90))
    t = np.linspace(0, 1, 16)
    xs = np.concatenate([x + t * w, x + t * w, np.full(16, x), np.full(16, x + w)])
    ys = np.concatenate([np.full(16, y), np.full(16, y + h), y + t * h, y + t * h])
    lons = np.mod(xs, erp_w) / erp_w * 360.0 - 180.0
    lats = 90.0 - np.clip(ys, 0, erp_h) / erp_h * 180.0
    vx, vy, vz = _lonlat_to_unit(lons, lats)
    c, east, north = _tangent_frame(lon_c, lat_c)
    pc = vx * c[0] + vy * c[1] + vz * c[2]
    pe = vx * east[0] + vy * east[1] + vz * east[2]
    pn = vx * north[0] + vy * north[1] + vz * north[2]
    fov_h = max(float(np.rad2deg(np.arctan2(pe, pc)).max() - np.rad2deg(np.arctan2(pe, pc)).min()), 1e-3)
    fov_v = max(float(np.rad2deg(np.arctan2(pn, pc)).max() - np.rad2deg(np.arctan2(pn, pc)).min()), 1e-3)
    return lon_c, lat_c, fov_h, fov_v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "all"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    sutrack_ws = os.environ.get("SUTRACK_WORKSPACE", "/data/sutrack_src_20260825/SUTrack")
    sutrack_ckpt = os.environ.get("SUTRACK_CKPT", "/data/weights/SUTRACK_b224_ep0180.pth.tar")

    sys.path.insert(0, sutrack_ws)
    os.chdir(sutrack_ws)

    import torch
    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.test.tracker.sutrack import SUTRACK
    from lib.test.utils.params import TrackerParams

    # 1. 构建模型
    print("Building SUTrack model...")
    update_config_from_file(os.path.join(sutrack_ws, "experiments", "sutrack", "sutrack_b224.yaml"))
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    sutrack_encoder.is_main_process = lambda: False
    local_cfg = deepcopy(cfg)

    params = TrackerParams()
    params.cfg = local_cfg
    params.yaml_name = "sutrack_b224"
    params.checkpoint = str(sutrack_ckpt)
    params.template_factor = float(local_cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(local_cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(local_cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(local_cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0

    tracker = SUTRACK(params, "got10k")

    # 2. 注入 LoRA 并加载 checkpoint
    print(f"Loading LoRA checkpoint: {args.lora_ckpt}")
    sys.path.insert(0, str(PROJECT_ROOT / "tools_local"))
    from sutrack_lora_setup import apply_lora_to_sutrack
    n_lora = apply_lora_to_sutrack(tracker.network.encoder.body, rank=8, alpha=16.0)
    print(f"Injected LoRA into {n_lora} layers")

    lora_ckpt = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    lora_state = lora_ckpt.get("net", lora_ckpt)
    if hasattr(lora_state, "state_dict"):
        lora_state = lora_state.state_dict()
    missing, unexpected = tracker.network.load_state_dict(lora_state, strict=False)
    print(f"Loaded LoRA weights: missing={len(missing)}, unexpected={len(unexpected)}")
    # 确保 LoRA 参数在 GPU 上
    tracker.network.cuda()

    # 3. 加载序列列表
    split_dir = PROJECT_ROOT / "data360" / "official_split"
    if args.split == "valid":
        seqlist = split_dir / "seqlist_official_valid.txt"
    elif args.split == "train":
        seqlist = split_dir / "seqlist_official_train.txt"
    else:
        seqlist = None

    data_root = Path(args.data)
    if seqlist and seqlist.is_file():
        seqs = [ln.strip() for ln in seqlist.read_text().splitlines() if ln.strip()]
    else:
        seqs = []
        for block in ["train_real", "train_sim"]:
            block_dir = data_root / block
            if block_dir.is_dir():
                for d in sorted(block_dir.iterdir()):
                    if d.is_dir() and (d / "video.mp4").is_file():
                        seqs.append(f"{block}/{d.name}")

    print(f"Evaluating {len(seqs)} sequences ({args.split})...")

    # 4. 逐序列评测
    try:
        import cv2 as cv
    except ImportError:
        print("ERROR: cv2 required")
        return

    all_auc = []
    total_frames = 0
    t0 = time.time()

    for idx, seq_name in enumerate(seqs, 1):
        seq_dir = data_root / seq_name
        video_path = seq_dir / "video.mp4"
        gt_path = seq_dir / "groundtruth.txt"

        if not video_path.is_file() or not gt_path.is_file():
            continue

        # 读 GT
        gt_rows = []
        for ln in gt_path.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            f = [float(v) for v in ln.replace(",", " ").split()]
            gt_rows.append(f)

        cap = cv.VideoCapture(str(video_path))
        ok, first = cap.read()
        if not ok:
            cap.release()
            continue

        H, W = first.shape[:2]
        first_rgb = cv.cvtColor(first, cv.COLOR_BGR2RGB)
        init_bfov = gt_rows[0]

        # 初始化
        tiled = np.concatenate((first_rgb, first_rgb, first_rgb), axis=1)
        # BFoV → ERP bbox（简化版）
        clon, clat, fov_h, fov_v = init_bfov
        # 用 fov 估算框大小
        fov_h_rad = np.deg2rad(fov_h)
        fov_v_rad = np.deg2rad(fov_v)
        box_w = fov_h_rad / (2 * np.pi) * W
        box_h = fov_v_rad / np.pi * H
        cx = (clon + 180) / 360 * W
        cy = (90 - clat) / 180 * H
        erp_box = (cx - box_w / 2, cy - box_h / 2, box_w, box_h)
        tracker.initialize(tiled, {"init_bbox": [erp_box[0] % W + W, erp_box[1], erp_box[2], erp_box[3]]})

        # 逐帧跟踪
        pred_bfovs = [tuple(init_bfov)]
        n_frames = 1
        while True:
            if args.max_frames and n_frames >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            tiled = np.concatenate((rgb, rgb, rgb), axis=1)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = tracker.track(tiled)
            box = output.get("target_bbox")
            if box is None or len(box) != 4:
                pred_bfovs.append((0, 0, 0, 0))
            else:
                pred = [float(box[0]) % W, float(box[1]), float(box[2]), float(box[3])]
                pred_bfovs.append(bfov_from_erp_bbox(*pred, W, H))
            n_frames += 1
        cap.release()

        # 计算 AUC（简化版：逐帧 IoU）
        ious = []
        for i in range(min(len(pred_bfovs), len(gt_rows))):
            p = pred_bfovs[i]
            g = gt_rows[i]
            if g[2] <= 0 or g[3] <= 0:
                continue
            if p[2] <= 0 or p[3] <= 0:
                ious.append(0.0)
                continue
            # 简化 IoU：用 fov_h * fov_v 近似
            overlap_h = max(0, min(p[0]+p[2]/2, g[0]+g[2]/2) - max(p[0]-p[2]/2, g[0]-g[2]/2))
            overlap_v = max(0, min(p[1]+p[3]/2, g[1]+g[3]/2) - max(p[1]-p[3]/2, g[1]-g[3]/2))
            intersection = overlap_h * overlap_v
            union = p[2]*p[3] + g[2]*g[3] - intersection
            iou = intersection / max(union, 1e-8)
            ious.append(min(iou, 1.0))

        # AUC at 21 thresholds
        thresholds = np.linspace(0, 1, 21)
        sr_curve = []
        for t in thresholds:
            sr = sum(1 for iou in ious if iou >= t) / max(len(ious), 1)
            sr_curve.append(sr)
        auc = np.mean(sr_curve)
        all_auc.append(auc)
        total_frames += n_frames

        dt = time.time() - t0
        avg_auc = np.mean(all_auc)
        print(f"  [{idx}/{len(seqs)}] {seq_name}: AUC={auc:.4f} ({n_frames} frames) | running avg={avg_auc:.4f}")

    elapsed = time.time() - t0
    final_auc = np.mean(all_auc) if all_auc else 0
    print(f"\n=== Results ===")
    print(f"Split: {args.split}, Sequences: {len(all_auc)}, AUC: {final_auc:.4f}")
    print(f"Total frames: {total_frames}, Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
