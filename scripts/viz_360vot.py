# -*- coding: utf-8 -*-
"""360VOT 跟踪可视化：重跑 PanoTracker 并导出抽帧 GIF（人工检查用）。

用法（工作目录 D:\\instan\\pano360）：
  python scripts/viz_360vot.py --seq 0001 --downscale 0.25 --stride 3 \
      --out runs/360vot --config configs/eval_360vot.json

输出 <out>/<seq>/demo.gif：逐帧重跑跟踪器（OPE 连续），每 stride 帧抽 1 帧；
跟踪框按状态着色（绿=ok 红=lost 黄=recovered），GT 框蓝色；帧缩至 480 宽。
说明：JPEG 用 libjpeg draft 模式按目标尺寸近似抽取解码（比全解码+缩放快
2~3 倍），仅用于本可视化脚本；评测脚本仍走精确解码。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data import vot360
from panotrack.data.viz import draw_bbox, save_gif
from panotrack.pipeline.pipeline import PanoTracker

_COLOR = {'ok': (0, 200, 0), 'lost': (220, 0, 0), 'recovered': (230, 200, 0)}
_GT_COLOR = (40, 80, 255)
_GIF_W = 480


def _iter_frames_draft(paths, gt, downscale):
    """draft 模式快速解码并同步缩放 GT（私有）。

    参数: paths 帧路径列表；gt (N,4) 原分辨率 GT；downscale 缩放比例。
    产出: (i, frame_uint8, gt_row_scaled)。
    """
    for i, p in enumerate(paths):
        img = Image.open(p)
        fw, fh = img.size
        tw, th = max(1, round(fw * downscale)), max(1, round(fh * downscale))
        img.draft('RGB', (tw * 2, th * 2))     # libjpeg 按 1/2/4/8 抽取到 >= 目标
        img = img.convert('RGB').resize((tw, th), Image.BILINEAR)
        yield i, np.asarray(img, dtype=np.uint8), gt[i] * downscale


def main(argv=None):
    """生成单序列跟踪可视化 GIF。

    参数: argv 命令行参数（None 取 sys.argv）。
    返回: 退出码（0 正常）。
    """
    p = argparse.ArgumentParser(description='360VOT 跟踪可视化 GIF 导出')
    p.add_argument('--seq', required=True, help='序列名（如 0001）')
    p.add_argument('--data', default=str(PROJECT_ROOT / 'data360'))
    p.add_argument('--downscale', type=float, default=0.25)
    p.add_argument('--stride', type=int, default=3, help='GIF 抽帧间隔（默认 3）')
    p.add_argument('--out', default=str(PROJECT_ROOT / 'runs' / '360vot'))
    p.add_argument('--config', default=None)
    args = p.parse_args(argv)

    config = None
    if args.config:
        import json
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    paths, gt, _ = vot360._resolve_sequence(Path(args.data) / args.seq)
    frames_it = _iter_frames_draft(paths, gt, float(args.downscale))
    _, f0, g0 = next(frames_it)
    H, W = f0.shape[:2]
    norm = lambda b: (b[0] % W, b[1], b[2], b[3])
    tr = PanoTracker(config)
    tr.init(f0, norm(g0))

    gif_scale = _GIF_W / W

    def _shrink(im):
        """绘制帧立即缩到 GIF 尺寸，避免全尺寸累积占内存（私有）。"""
        return np.asarray(Image.fromarray(im).resize(
            (_GIF_W, max(1, round(H * gif_scale))), Image.BILINEAR))

    vis = [_shrink(draw_bbox(draw_bbox(f0, norm(g0), color=_GT_COLOR, thickness=2),
                             norm(g0), color=_COLOR['ok'], thickness=2))]
    for i, frame, row in frames_it:
        r = tr.update(frame)
        if i % args.stride != 0:
            continue
        img = draw_bbox(frame, norm(row), color=_GT_COLOR, thickness=2)
        img = draw_bbox(img, r['bbox'], color=_COLOR[r['status']], thickness=2)
        vis.append(_shrink(img))
        del frame, img

    out_path = Path(args.out) / args.seq / 'demo.gif'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_gif(vis, out_path, fps=10)
    print(f'{args.seq}: {len(vis)} 帧 -> {out_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
