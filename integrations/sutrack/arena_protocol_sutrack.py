#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arena 平台提交协议入口（BFoV 输出 · SUTrack-B224 ERP 三平铺）。

基于 ODTrack 版 arena_protocol.py，替换推理内核为 SUTrack-B224：
  - Fast-iTPN backbone (B224: search=224, template=112)
  - ERP 帧水平三平铺 + FP16 AMP 推理
  - BFoV <-> ERP 转换内联纯 numpy（与 panotrack/geometry/bfov.py 同算法）

输入/输出协议与 arena_protocol.py 完全一致：
  输入：/mnt/dataset/<seq>/video.mp4 + init.txt
  输出：/mnt/result/<seq>.txt (每行 clon,clat,fov_h,fov_v)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import types
from copy import deepcopy
from pathlib import Path

import numpy as np

try:
    import cv2 as cv
except ImportError:
    cv = None

# ---------------------------------------------------------------------------
# BFoV <-> ERP 转换（内联纯 numpy）
# ---------------------------------------------------------------------------

D2R = np.pi / 180.0


def _wrap_lon(lon):
    out = 180.0 - np.mod(180.0 - np.asarray(lon, dtype=np.float64), 360.0)
    return float(out) if np.isscalar(lon) else out


def _lonlat_to_unit(lon, lat):
    lon_r = np.deg2rad(np.asarray(lon, dtype=np.float64))
    lat_r = np.deg2rad(np.asarray(lat, dtype=np.float64))
    c = np.cos(lat_r)
    x = c * np.cos(lon_r)
    y = np.sin(lat_r)
    z = c * np.sin(lon_r)
    if x.ndim == 0:
        return float(x), float(y), float(z)
    return x, y, z


def _unit_to_lonlat(x, y, z):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    n = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-12)
    lat = np.rad2deg(np.arcsin(np.clip(y / n, -1.0, 1.0)))
    lon = 180.0 - np.mod(180.0 - np.rad2deg(np.arctan2(z, x)), 360.0)
    if lon.ndim == 0:
        return float(lon), float(lat)
    return lon, lat


def _tangent_frame(lon, lat):
    cx, cy, cz = _lonlat_to_unit(lon, lat)
    c = np.array([cx, cy, cz], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    east = np.cross(c, up)
    n = np.linalg.norm(east)
    if n < 1e-9:
        east = np.array([0.0, 0.0, 1.0 if cy > 0 else -1.0])
    else:
        east = east / n
    north = np.cross(east, c)
    return c, east, north


def _offset_dirs(du, dv, frame, gnomonic=True):
    c, east, north = frame
    du = np.asarray(du, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    if gnomonic:
        kx = np.tan(du)
        ky = np.tan(dv)
        vx = c[0] + kx * east[0] + ky * north[0]
        vy = c[1] + kx * east[1] + ky * north[1]
        vz = c[2] + kx * east[2] + ky * north[2]
    else:
        cd, sd = np.cos(du), np.sin(du)
        cv_, sv = np.cos(dv), np.sin(dv)
        ax = c[0] * cd + east[0] * sd
        ay = c[1] * cd + east[1] * sd
        az = c[2] * cd + east[2] * sd
        vx = cv_ * ax + sv * north[0]
        vy = cv_ * ay + sv * north[1]
        vz = cv_ * az + sv * north[2]
    n = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-12)
    return vx / n, vy / n, vz / n


def _px_to_lonlat(u, v, erp_w, erp_h):
    lon = np.asarray(u, dtype=np.float64) / erp_w * 360.0 - 180.0
    lat = 90.0 - np.asarray(v, dtype=np.float64) / erp_h * 180.0
    return lon, lat


def _bbox_boundary_points(x, y, w, h, n):
    t = np.linspace(0.0, 1.0, n)
    xs = np.concatenate([x + t * w, x + t * w, np.full(n, x), np.full(n, x + w)])
    ys = np.concatenate([np.full(n, y), np.full(n, y + h), y + t * h, y + t * h])
    return xs, ys


def bfov_from_erp_bbox(x, y, w, h, erp_w, erp_h):
    cu = (x + w / 2.0) % erp_w
    cv_ = y + h / 2.0
    lon_c, lat_c = _px_to_lonlat(cu, cv_, erp_w, erp_h)
    lon_c = _wrap_lon(float(lon_c))
    lat_c = float(np.clip(lat_c, -90.0, 90.0))
    xs, ys = _bbox_boundary_points(x, y, w, h, 16)
    lons, lats = _px_to_lonlat(np.mod(xs, erp_w), np.clip(ys, 0.0, erp_h), erp_w, erp_h)
    vx, vy, vz = _lonlat_to_unit(lons, lats)
    c, east, north = _tangent_frame(lon_c, lat_c)
    pc = vx * c[0] + vy * c[1] + vz * c[2]
    pe = vx * east[0] + vy * east[1] + vz * east[2]
    pn = vx * north[0] + vy * north[1] + vz * north[2]
    du = np.rad2deg(np.arctan2(pe, pc))
    dv = np.rad2deg(np.arctan2(pn, pc))
    fov_h = max(float(du.max() - du.min()), 1e-3)
    fov_v = max(float(dv.max() - dv.min()), 1e-3)
    return lon_c, lat_c, fov_h, fov_v


def erp_bbox_from_bfov(clon, clat, fov_h, fov_v, erp_w, erp_h, samples=48):
    frame = _tangent_frame(clon, clat)
    gnomonic = max(fov_h, fov_v) <= 90.0
    t = np.linspace(-0.5, 0.5, samples)
    dh = np.deg2rad(fov_h)
    dvv = np.deg2rad(fov_v)
    du = np.concatenate([t * dh, t * dh, np.full(samples, -0.5 * dh), np.full(samples, 0.5 * dh)])
    dv = np.concatenate([np.full(samples, -0.5 * dvv), np.full(samples, 0.5 * dvv), t * dvv, t * dvv])
    vx, vy, vz = _offset_dirs(du, dv, frame, gnomonic)
    lon, lat = _unit_to_lonlat(vx, vy, vz)
    px = (np.asarray(lon) + 180.0) / 360.0 * erp_w
    py = (90.0 - np.asarray(lat)) / 180.0 * erp_h
    ref = float(np.median(px))
    px = px + erp_w * np.round((ref - px) / erp_w)
    x0, x1 = float(px.min()), float(px.max())
    y0 = float(np.clip(py.min(), 0.0, erp_h))
    y1 = float(np.clip(py.max(), 0.0, erp_h))
    return x0 % erp_w, y0, x1 - x0, y1 - y0


# ---------------------------------------------------------------------------
# 数据集遍历
# ---------------------------------------------------------------------------

def natural_key(name):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', name)]


def list_sequences(dataset_dir):
    dataset_dir = Path(dataset_dir)
    seqlist = dataset_dir / 'seqlist.txt'
    if seqlist.is_file():
        with open(seqlist, 'r', encoding='utf-8-sig') as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    names = []
    for child in sorted(dataset_dir.iterdir(), key=lambda p: natural_key(p.name)):
        if child.is_dir() and (child / 'video.mp4').is_file():
            names.append(child.name)
    return names


def load_init_bfov(seq_dir):
    path = Path(seq_dir) / 'init.txt'
    with open(path, 'r', encoding='utf-8') as f:
        line = next((ln for ln in f if ln.strip()), '')
    fields = re.split(r'[\s,;]+', line.strip())
    if len(fields) != 4:
        raise ValueError(f'init.txt must contain exactly four values ({path})')
    values = [float(v) for v in fields]
    if not all(np.isfinite(values)):
        raise ValueError(f'init.txt contains non-finite value ({path})')
    clon, clat, fov_h, fov_v = values
    if fov_h <= 0.0 or fov_v <= 0.0:
        raise ValueError(f'init.txt fov must be positive ({path})')
    return clon, clat, fov_h, fov_v


def write_bfov_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
        for (clon, clat, fh, fv) in rows:
            handle.write(f'{float(clon):.3f},{float(clat):.3f},'
                         f'{float(fh):.3f},{float(fv):.3f}\n')
    temporary.replace(path)


# ---------------------------------------------------------------------------
# SUTrack 推理内核
# ---------------------------------------------------------------------------

def _tile_box(box, width):
    x, y, w, h = (float(v) for v in box)
    return [x % width + width, y, w, h]


def build_sutrack_tracker(workspace, checkpoint, config_name="sutrack_b224"):
    """构建 SUTrack tracker 实例。"""
    import torch
    workspace = Path(workspace)
    sys.path.insert(0, str(workspace))
    os.chdir(workspace)

    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.test.tracker.sutrack import SUTRACK
    from lib.test.utils.params import TrackerParams

    update_config_from_file(workspace / 'experiments' / 'sutrack' / f'{config_name}.yaml')
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    sutrack_encoder.is_main_process = lambda: False

    local_cfg = deepcopy(cfg)
    params = TrackerParams()
    params.cfg = local_cfg
    params.yaml_name = config_name
    params.checkpoint = str(checkpoint)
    params.template_factor = float(local_cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(local_cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(local_cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(local_cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0

    tracker = SUTRACK(params, "got10k")
    return tracker


def _track_one_sequence(seq_dir, name, tracker, lost_iou_threshold=0.0, max_frames=None):
    """跑单条序列，返回 (rows, n_frames)。"""
    video = Path(seq_dir) / 'video.mp4'
    if not video.is_file():
        raise FileNotFoundError(f'video.mp4 not found in {seq_dir}')
    init_bfov = load_init_bfov(seq_dir)
    if cv is None:
        raise RuntimeError('OpenCV (cv2) is required')

    import torch

    cap = cv.VideoCapture(str(video))
    try:
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError(f'failed to decode first frame of {video}')
        height, width = first.shape[:2]
        first_rgb = cv.cvtColor(first, cv.COLOR_BGR2RGB)

        # 三平铺初始化
        tiled = np.concatenate((first_rgb, first_rgb, first_rgb), axis=1)
        erp_box = erp_bbox_from_bfov(*init_bfov, width, height)
        tracker.initialize(tiled, {'init_bbox': _tile_box(erp_box, width)})

        rows = [tuple(init_bfov)]
        n_frames = 1
        while True:
            if max_frames is not None and n_frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            tiled = np.concatenate((rgb, rgb, rgb), axis=1)

            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = tracker.track(tiled)

            box = output.get('target_bbox')
            if box is None or len(box) != 4:
                raise RuntimeError(f'tracker returned invalid box at frame {n_frames + 1}')
            pred = [float(box[0]) % width, float(box[1]),
                    float(box[2]), float(box[3])]
            if lost_iou_threshold > 0.0:
                quality = float(getattr(tracker, 'last_pred_iou', 1.0))
                if quality < lost_iou_threshold:
                    rows.append((0.0, 0.0, 0.0, 0.0))
                    n_frames += 1
                    continue
            rows.append(bfov_from_erp_bbox(*pred, width, height))
            n_frames += 1
    finally:
        cap.release()
    return rows, n_frames


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--result', default=None)
    parser.add_argument('--workspace', default='/opt/sutrack',
                        help='SUTrack repository root')
    parser.add_argument('--checkpoint', default='/opt/models/SUTRACK_b224_ep0180.pth.tar')
    parser.add_argument('--config', default='sutrack_b224')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--force-cpu', action='store_true')
    parser.add_argument('--lost-iou-threshold', type=float, default=0.0)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--seqs', default=None)
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset or os.environ.get('DATASET_DIR', '/mnt/dataset'))
    result_dir = Path(args.result or os.environ.get('RESULT_DIR', '/mnt/result'))

    if not dataset_dir.is_dir():
        print(f'[error] dataset dir does not exist: {dataset_dir}', file=sys.stderr)
        return 2

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    workspace = Path(args.workspace).resolve()
    checkpoint = Path(args.checkpoint)

    print(f'[arena_protocol_sutrack] Loading SUTrack from {workspace} ...')
    tracker = build_sutrack_tracker(workspace, checkpoint, args.config)
    print(f'[arena_protocol_sutrack] Model loaded. checkpoint={checkpoint}')

    seqs = list_sequences(dataset_dir)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(',') if s.strip()}
        seqs = [s for s in seqs if s in wanted]
    if not seqs:
        print(f'[error] no sequences found under {dataset_dir}', file=sys.stderr)
        return 2

    print(f'[arena_protocol_sutrack] sequences={len(seqs)} dataset={dataset_dir} '
          f'result={result_dir}')
    result_dir.mkdir(parents=True, exist_ok=True)
    total_frames, t0, failures = 0, time.time(), []

    for idx, name in enumerate(seqs, 1):
        seq_dir = dataset_dir / name
        ts = time.time()
        try:
            rows, n_frames = _track_one_sequence(
                seq_dir, name, tracker,
                lost_iou_threshold=args.lost_iou_threshold,
                max_frames=args.max_frames)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            failures.append(name)
            print(f'  [{idx}/{len(seqs)}] {name}: FAILED ({exc})', file=sys.stderr)
            continue
        write_bfov_rows(result_dir / f'{name}.txt', rows)
        total_frames += n_frames
        dt = time.time() - ts
        fps = (n_frames - 1) / dt if dt > 0 and n_frames > 1 else 0.0
        print(f'  [{idx}/{len(seqs)}] {name}: {n_frames} 帧, {dt:.2f}s ({fps:.1f} FPS)')

    elapsed = time.time() - t0
    print(f'[arena_protocol_sutrack] 完成: {total_frames} 帧 / {elapsed:.2f}s, '
          f'平均 {total_frames / elapsed if elapsed > 0 else 0:.1f} FPS, '
          f'结果写入 {result_dir}')
    if failures:
        print(f'[arena_protocol_sutrack] 失败序列: {", ".join(failures)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
