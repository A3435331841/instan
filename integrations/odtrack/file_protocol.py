#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ODTrack ERP 三平铺的文件协议提交入口（对齐 UETrack 镜像入口接口）。

用法（镜像默认 ENTRYPOINT）：
  python file_protocol.py --frames DIR --init init.txt --out results.txt \
      [--timing timing.txt] [--workspace /opt/odtrack] \
      [--checkpoint /opt/models/ODTrack_ep0300.pth.tar] [--config baseline]

逐帧流程：加载 ODTrack 上游 tracker（ERP 帧水平三平铺后喂入，初始框移到
中间副本），输出框横坐标折回 [0, W)，首行输出初始框（与官方 OPE 一致）。
输出 results.txt 每行 x,y,w,h（12 位小数，保留首帧框精度），日志走 stdout。

--force-cpu 仅供无 GPU 环境的结构性冒烟：把上游硬编码的 .cuda() 变为
no-op，让整个前向在 CPU 上执行。正式评测请使用 GPU 路径。
"""
import argparse
import os
import re
import sys
import time
import types
from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {'.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}


def parse_box(value):
    """解析 x,y,w,h：可从文件读取首行，也可直接内联。"""
    candidate = Path(value)
    text = candidate.read_text(encoding='utf-8') if candidate.is_file() else value
    first_line = next((line for line in text.splitlines() if line.strip()), '')
    fields = re.split(r'[\s,;]+', first_line.strip())
    if len(fields) != 4:
        raise ValueError('initial box must contain exactly four values: x,y,w,h')
    try:
        box = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError('initial box contains a non-numeric value') from exc
    if not all(value == value and abs(value) != float('inf') for value in box):
        raise ValueError('initial box contains a non-finite value')
    if box[2] <= 0.0 or box[3] <= 0.0:
        raise ValueError('initial box width and height must be positive')
    return box


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def frame_paths(root):
    """按文件名自然排序返回单层目录中的图像帧列表。"""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'frame directory does not exist: {root}')
    frames = sorted(
        (path for path in root.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if not frames:
        raise FileNotFoundError(f'no supported images found in {root}')
    return frames


def write_rows(path, rows):
    """原子写入数值行（12 位小数，与 UETrack 镜像入口一致）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with open(temporary, 'w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(','.join(f'{float(value):.12f}' for value in row) + '\n')
    temporary.replace(path)


def _patch_torch_six():
    """ODTrack 上游是 PyTorch 2.0 之前的代码：torch._six / visdom 兼容补丁。"""
    try:
        import torch._six  # noqa: F401
    except ModuleNotFoundError:
        six = types.ModuleType('torch._six')
        six.string_classes = (str,)
        six.int_classes = (int,)
        sys.modules['torch._six'] = six
    if 'visdom' not in sys.modules:
        visdom = types.ModuleType('visdom')
        visdom.__path__ = []
        visdom.Visdom = object
        server = types.ModuleType('visdom.server')
        sys.modules['visdom'] = visdom
        sys.modules['visdom.server'] = server
    if 'lib.vis.visdom_cus' not in sys.modules:
        visdom_cus = types.ModuleType('lib.vis.visdom_cus')
        visdom_cus.Visdom = type('Visdom', (),
                                 {'__init__': lambda self, *a, **k: None})
        sys.modules['lib.vis.visdom_cus'] = visdom_cus


def _tile_box(box, width):
    """初始框移到三平铺帧的中间副本。"""
    x, y, w, h = (float(v) for v in box)
    return [x % width + width, y, w, h]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', required=True,
                        help='directory of ordered frames')
    parser.add_argument('--init', required=True,
                        help='text file or inline x,y,w,h initial box')
    parser.add_argument('--out', required=True, help='output x,y,w,h result file')
    parser.add_argument('--timing', default=None,
                        help='optional per-frame timing output file')
    parser.add_argument('--workspace', default='/opt/odtrack',
                        help='installed ODTrack repository root')
    parser.add_argument('--checkpoint', default='/opt/models/ODTrack_ep0300.pth.tar',
                        help='ODTrack checkpoint')
    parser.add_argument('--config', default='baseline',
                        help='experiment yaml name under experiments/odtrack/')
    parser.add_argument('--gpu', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--force-cpu', action='store_true',
                        help='structural CPU smoke: no-op the upstream .cuda()')
    args = parser.parse_args(argv)

    # CUDA visibility must be set before importing torch through ODTrack.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f'ODTrack workspace does not exist: {workspace}')
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f'ODTrack checkpoint does not exist: {checkpoint}')
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

    _patch_torch_six()
    if args.force_cpu:
        import torch
        torch.nn.Module.cuda = lambda self, device=None: self
        torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self

    import cv2 as cv
    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack
    from lib.test.utils.params import TrackerParams

    update_config_from_file(
        workspace / 'experiments' / 'odtrack' / f'{args.config}.yaml')
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(checkpoint)
    params.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0

    def _read(path):
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'failed to decode frame: {path}')
        return cv.cvtColor(image, cv.COLOR_BGR2RGB)

    frames = frame_paths(args.frames)
    initial_box = parse_box(args.init)
    tracker = ODTrack(params)

    first = _read(frames[0])
    height, width = first.shape[:2]
    tiled = np.concatenate((first, first, first), axis=1)
    boxes = [[float(initial_box[0]) % width, float(initial_box[1]),
              float(initial_box[2]), float(initial_box[3])]]
    timings = []
    start = time.perf_counter()
    tracker.initialize(tiled, {'init_bbox': _tile_box(initial_box, width)})
    timings.append(time.perf_counter() - start)

    for frame_path in frames[1:]:
        image = _read(frame_path)
        tiled = np.concatenate((image, image, image), axis=1)
        start = time.perf_counter()
        output = tracker.track(tiled)
        timings.append(time.perf_counter() - start)
        box = output.get('target_bbox')
        if box is None or len(box) != 4:
            raise RuntimeError(f'tracker returned an invalid box for {frame_path}')
        pred = [float(box[0]) % width, float(box[1]),
                float(box[2]), float(box[3])]
        boxes.append(pred)

    write_rows(args.out, boxes)
    if args.timing:
        write_rows(args.timing, ([value] for value in timings))
    elapsed = sum(timings[1:])
    fps = (len(timings) - 1) / elapsed if elapsed > 0.0 else 0.0
    print(f'COMPLETE frames={len(frames)} fps={fps:.4f} output={Path(args.out)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
