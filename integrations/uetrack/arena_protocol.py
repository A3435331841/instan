#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UETrack Arena 平台官方提交协议入口（BFoV 输出 · 高速版）。

与 ODTrack 版 arena_protocol.py 同一协议，但跟踪内核用 UETrack（ERP-wrap 接缝处理）：
  - 输入 /mnt/dataset（只读）：<seq>/video.mp4 + <seq>/init.txt（BFoV: clon,clat,fov_h,fov_v）
  - 输出 /mnt/result（可写）：<seq>.txt，每行 BFoV，行号=帧号；丢失帧 0,0,0,0
  - 启动：无参自启动，跑完全部序列后退出码 0
  - 环境变量：DATASET_DIR / RESULT_DIR（默认 /mnt/dataset、/mnt/result）

速度：UETrack ERP-wrap 全量 120 序列 57.16 FPS（AUC 0.5143）。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# 内联 BFoV<->ERP 转换与数据读取工具（纯 numpy，与 ODTrack 版 arena_protocol.py 同源）
import numpy as np

def _wrap_lon(lon):
    """经度归一化到 (-180, 180]。"""
    out = 180.0 - np.mod(180.0 - np.asarray(lon, dtype=np.float64), 360.0)
    return float(out) if np.isscalar(lon) else out


def _lonlat_to_unit(lon, lat):
    """经纬度转单位向量（y 轴指北；lon=0,lat=0 对应 +x；lon 增大约 +z）。"""
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
    """单位向量转经纬度（经度自动 wrap 到 (-180, 180]）。"""
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
    """中心方向的切平面右手坐标系（私有）。"""
    cx, cy, cz = _lonlat_to_unit(lon, lat)
    c = np.array([cx, cy, cz], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0])
    east = np.cross(c, up)
    n = np.linalg.norm(east)
    if n < 1e-9:  # 中心恰在极点，任取固定东西方向
        east = np.array([0.0, 0.0, 1.0 if cy > 0 else -1.0])
    else:
        east = east / n
    north = np.cross(east, c)
    return c, east, north


def _offset_dirs(du, dv, frame, gnomonic=True):
    """由切平面角偏移生成单位方向向量（私有）。"""
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
        cv, sv = np.cos(dv), np.sin(dv)
        ax = c[0] * cd + east[0] * sd
        ay = c[1] * cd + east[1] * sd
        az = c[2] * cd + east[2] * sd
        vx = cv * ax + sv * north[0]
        vy = cv * ay + sv * north[1]
        vz = cv * az + sv * north[2]
    n = np.maximum(np.sqrt(vx * vx + vy * vy + vz * vz), 1e-12)
    return vx / n, vy / n, vz / n


def _px_to_lonlat(u, v, erp_w, erp_h):
    """连续像素坐标转经纬度（u 可超界，v 需在 [0,H]）。"""
    lon = np.asarray(u, dtype=np.float64) / erp_w * 360.0 - 180.0
    lat = 90.0 - np.asarray(v, dtype=np.float64) / erp_h * 180.0
    return lon, lat


def _bbox_boundary_points(x, y, w, h, n):
    """ERP 框边界采样点（每边 n 点含角点），返回连续像素坐标。"""
    t = np.linspace(0.0, 1.0, n)
    xs = np.concatenate([x + t * w, x + t * w, np.full(n, x), np.full(n, x + w)])
    ys = np.concatenate([np.full(n, y), np.full(n, y + h), y + t * h, y + t * h])
    return xs, ys


def bfov_from_erp_bbox(x, y, w, h, erp_w, erp_h):
    """由 ERP 框估算 BFoV：采样边界点转球面，取中心与角跨度。

    返回 (clon, clat, fov_h, fov_v)：中心经纬度 + 水平/垂直视场角（度）。
    跨界框先对 x 做模 W 展开；中心取框中心像素对应经纬度。
    """
    cu = (x + w / 2.0) % erp_w
    cv = y + h / 2.0
    lon_c, lat_c = _px_to_lonlat(cu, cv, erp_w, erp_h)
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
    """BFoV 边界采样投影回 ERP，取最小面积轴对齐框；跨界时 x+w 可超 W。

    返回 (x, y, w, h) 浮点 ERP 框，x∈[0,W)。
    """
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
# ODTrack 兼容补丁（与 file_protocol.py 一致）
# ---------------------------------------------------------------------------


def _load_tracker(workspace, parameter, no_erp_wrap):
    """加载 UETrack tracker（包含 ERP-wrap 接缝处理）。

    torch>=2.6 默认 weights_only=True，旧 UETrack 权重含自定义类（如
    AverageMeter）会加载失败；这里打补丁强制 weights_only=False（权重
    来自可信来源，已在镜像内固定 SHA-256）。
    """
    import cv2  # noqa: F401 - 保证 cv2 导入顺序
    import torch
    _orig_torch_load = torch.load
    def _compat_load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_torch_load(*args, **kwargs)
    torch.load = _compat_load
    from lib.test.evaluation import Tracker
    if not no_erp_wrap:
        from erp_wrap import clip_box_erp, sample_target_erp
        import lib.test.tracker.uetrack as tracker_module
        tracker_module.sample_target = sample_target_erp
        tracker_module.clip_box = clip_box_erp
    wrapper = Tracker('uetrack', parameter, 'erp', None)
    params = wrapper.get_parameters()
    params.debug = 0
    return wrapper.create_tracker(params)


def natural_key(name):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', name)]


def list_sequences(dataset_dir):
    """返回待处理序列名列表：优先 seqlist.txt，否则扫描含 video.mp4 的子目录。"""
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
    """读取第 1 帧初始 BFoV：clon,clat,fov_h,fov_v（度）。"""
    path = Path(seq_dir) / 'init.txt'
    with open(path, 'r', encoding='utf-8') as f:
        line = next((ln for ln in f if ln.strip()), '')
    fields = re.split(r'[\s,;]+', line.strip())
    if len(fields) != 4:
        raise ValueError(f'init.txt must contain exactly four values: clon,clat,fov_h,fov_v ({path})')
    values = [float(v) for v in fields]
    if not all(np.isfinite(values)):
        raise ValueError(f'init.txt contains non-finite value ({path})')
    clon, clat, fov_h, fov_v = values
    if fov_h <= 0.0 or fov_v <= 0.0:
        raise ValueError(f'init.txt fov must be positive ({path})')
    return clon, clat, fov_h, fov_v


def write_bfov_rows(path, rows):
    """原子写入 BFoV 行（3 位小数，与官方 demo 一致）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
        for (clon, clat, fh, fv) in rows:
            handle.write(f'{float(clon):.3f},{float(clat):.3f},'
                         f'{float(fh):.3f},{float(fv):.3f}\n')
    temporary.replace(path)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _track_one_sequence(seq_dir, name, parameter, no_erp_wrap, lost_iou_threshold=0.0, max_frames=None):
    """跑单条序列：video.mp4 逐帧 + init.txt BFoV 初始化 -> BFoV 行列表。"""
    import cv2 as cv
    video = Path(seq_dir) / 'video.mp4'
    if not video.is_file():
        raise FileNotFoundError(f'video.mp4 not found in {seq_dir}')
    init_bfov = load_init_bfov(seq_dir)

    cap = cv.VideoCapture(str(video))
    try:
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError(f'failed to decode first frame of {video}')
        height, width = first.shape[:2]
        first_rgb = cv.cvtColor(first, cv.COLOR_BGR2RGB)

        # init BFoV -> ERP 框，UETrack 直接用 xywh（无三平铺，靠 ERP-wrap 处理接缝）
        erp_box = erp_bbox_from_bfov(*init_bfov, width, height)
        tracker = _load_tracker(Path(seq_dir).name + '_tracker', parameter, no_erp_wrap)
        initialized = tracker.initialize(
            first_rgb, {'init_bbox': list(erp_box), 'seq_name': name})
        previous = initialized or {}

        rows = [tuple(init_bfov)]
        n_frames = 1
        while True:
            if max_frames is not None and n_frames >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            output = tracker.track(rgb, {'previous_output': previous})
            box = output.get('target_bbox')
            if box is None or len(box) != 4:
                raise RuntimeError(f'tracker returned invalid box at frame {n_frames + 1}')
            pred = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
            if lost_iou_threshold > 0.0:
                quality = float(getattr(tracker, 'last_pred_iou', 1.0))
                if quality < lost_iou_threshold:
                    rows.append((0.0, 0.0, 0.0, 0.0))
                    n_frames += 1
                    continue
            rows.append(bfov_from_erp_bbox(*pred, width, height))
            n_frames += 1
            previous = output
    finally:
        cap.release()
    return rows, n_frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default=None, help='dataset dir (default: $DATASET_DIR or /mnt/dataset)')
    parser.add_argument('--result', default=None, help='result dir (default: $RESULT_DIR or /mnt/result)')
    parser.add_argument('--workspace', default='/opt/uetrack', help='installed UETrack repo root')
    parser.add_argument('--parameter', default='uetrack_base', help='UETrack parameter name')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--no-erp-wrap', action='store_true', help='disable ERP seam wrapping')
    parser.add_argument('--lost-iou-threshold', type=float, default=0.0, help='lost-frame marking threshold (0=off)')
    parser.add_argument('--max-frames', type=int, default=None, help='limit frames per sequence (debug)')
    parser.add_argument('--seqs', default=None, help='comma-separated sequences to process (default: all)')
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset or os.environ.get('DATASET_DIR', '/mnt/dataset'))
    result_dir = Path(args.result or os.environ.get('RESULT_DIR', '/mnt/result'))
    if not dataset_dir.is_dir():
        print(f'[error] dataset dir does not exist: {dataset_dir}', file=sys.stderr)
        return 2

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        print(f'[error] UETrack workspace does not exist: {workspace}', file=sys.stderr)
        return 2
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

    seqs = list_sequences(dataset_dir)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(',') if s.strip()}
        seqs = [s for s in seqs if s in wanted]
    if not seqs:
        print(f'[error] no sequences found under {dataset_dir}', file=sys.stderr)
        return 2

    print(f'[arena_uetrack] sequences={len(seqs)} dataset={dataset_dir} result={result_dir}')
    result_dir.mkdir(parents=True, exist_ok=True)
    total_frames, t0, failures = 0, time.time(), []

    for idx, name in enumerate(seqs, 1):
        seq_dir = dataset_dir / name
        ts = time.time()
        try:
            rows, n_frames = _track_one_sequence(
                seq_dir, name, args.parameter, args.no_erp_wrap,
                lost_iou_threshold=args.lost_iou_threshold, max_frames=args.max_frames)
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
    print(f'[arena_uetrack] 完成: {total_frames} 帧 / {elapsed:.2f}s, 平均 {total_frames / elapsed if elapsed > 0 else 0:.1f} FPS, 结果写入 {result_dir}')
    if failures:
        print(f'[arena_uetrack] 失败序列: {", ".join(failures)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
