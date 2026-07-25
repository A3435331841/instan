"""panotrack.io.file_protocol —— 图像序列文件协议适配层。

官方评测常见形态：给定 frames 目录与首帧标注文件，逐帧输出跟踪框。
本模块是 8 月官方评测 I/O 适配层之一（另一路见 trax_protocol）。
约定见 CONTRACTS.md：图像 (H,W,3) uint8 RGB；ERP 框 (x,y,w,h) 跨界约定。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
from PIL import Image

_IMG_EXTS = ('.png', '.jpg', '.jpeg')


def _list_frames(frames_dir):
    """列出目录下按文件名排序的图像路径（png/jpg）。"""
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f'frames 目录不存在: {frames_dir}')
    names = [n for n in os.listdir(frames_dir) if n.lower().endswith(_IMG_EXTS)]
    names.sort()
    if not names:
        raise FileNotFoundError(f'frames 目录下没有 png/jpg 图像: {frames_dir}')
    return [os.path.join(frames_dir, n) for n in names]


def _load_image(path):
    """读取图像为 (H,W,3) uint8 RGB 数组（契约通用约定）。"""
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8)


def _read_init(init_file):
    """解析 init 文件首行 'x,y,w,h'（兼容逗号/空白分隔），返回 float 四元组。"""
    with open(init_file, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
    parts = line.replace(',', ' ').split()
    if len(parts) != 4:
        raise ValueError(f'init 文件首行应为 x,y,w,h: {init_file!r} -> {line!r}')
    return tuple(float(v) for v in parts)


def _create_pano_tracker(config):
    """延迟导入并创建 PanoTracker（契约模块 E，集成阶段实现）。

    延迟导入保证模块 E 尚未落地时，本模块仍可被正常导入与单元测试。
    """
    try:
        from panotrack.pipeline.pipeline import PanoTracker
    except Exception as exc:  # ImportError 或模块 E 内部错误
        raise RuntimeError(
            '无法导入 panotrack.pipeline.pipeline.PanoTracker'
            '（契约模块 E 尚未实现或其实现有误）'
        ) from exc
    return PanoTracker(config)


def run_file_protocol(frames_dir, init_file, out_file, config=None):
    """按文件协议在图像序列上运行 PanoTracker 并逐帧写出跟踪框。

    参数:
        frames_dir: 图像目录，按文件名排序读取 png/jpg。
        init_file: 首帧标注文件，首行 'x,y,w,h'（跨界约定同契约）。
        out_file: 输出文件，逐帧追加写入 'x,y,w,h'（保留 2 位小数）；
                  首行为初始化框，之后每帧一行，与输入帧逐一对齐（共 N 行）。
        config: PanoTracker 配置 dict；None 表示使用默认配置。
    返回:
        dict: 耗时与帧率统计
        {'n_frames', 'elapsed_sec', 'fps', 'avg_ms_per_frame', 'out_file'}。
    备注:
        调试日志一律输出到 stderr，stdout 保持干净。
    """
    paths = _list_frames(frames_dir)
    bbox0 = _read_init(init_file)
    tracker = _create_pano_tracker(config)

    print(f'[file_protocol] 共 {len(paths)} 帧, init={bbox0}', file=sys.stderr)
    t0 = time.perf_counter()
    with open(out_file, 'a', encoding='utf-8', newline='') as fout:
        for i, path in enumerate(paths):
            frame = _load_image(path)
            if i == 0:
                tracker.init(frame, bbox0)
                bbox = bbox0
            else:
                res = tracker.update(frame)
                bbox = res['bbox']
                print(f'[file_protocol] 帧 {i}: status={res.get("status")} '
                      f'score={res.get("score")}', file=sys.stderr)
            x, y, w, h = (float(v) for v in bbox)
            fout.write(f'{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n')
            fout.flush()
    elapsed = time.perf_counter() - t0

    n = len(paths)
    stats = {
        'n_frames': n,
        'elapsed_sec': elapsed,
        'fps': n / elapsed if elapsed > 0 else float('inf'),
        'avg_ms_per_frame': elapsed * 1000.0 / n if n else 0.0,
        'out_file': os.path.abspath(out_file),
    }
    print(f'[file_protocol] 完成 {n} 帧, 用时 {elapsed:.3f}s, '
          f'FPS={stats["fps"]:.2f}', file=sys.stderr)
    return stats
