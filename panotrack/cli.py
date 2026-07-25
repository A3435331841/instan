"""panotrack.cli —— 命令行入口（文件协议）。

用法:
    python -m panotrack.cli --frames DIR --init init.txt --out results.txt
                            [--config configs/default.json] [--visualize OUT_DIR]

说明:
    - --frames: 图像序列目录（png/jpg，按文件名排序）。
    - --init:   首帧标注文件，首行 'x,y,w,h'。
    - --out:    输出结果文件，逐帧追加 'x,y,w,h'（2 位小数）。
    - --config: PanoTracker 配置 JSON（缺省用内置默认）。
    - --visualize: 可选，把跟踪框画到各帧并保存到 OUT_DIR（依赖模块 C 的 viz）。
    - 统计与调试信息输出到 stderr，stdout 保持干净。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog='panotrack',
        description='360° ERP 全景视频实时单目标跟踪（文件协议入口）',
    )
    parser.add_argument('--frames', required=True, metavar='DIR',
                        help='图像序列目录（png/jpg，按文件名排序）')
    parser.add_argument('--init', required=True, dest='init_file', metavar='FILE',
                        help='首帧标注文件，首行 x,y,w,h')
    parser.add_argument('--out', required=True, dest='out_file', metavar='FILE',
                        help='输出结果文件（逐帧追加 x,y,w,h）')
    parser.add_argument('--config', default=None, metavar='FILE',
                        help='PanoTracker 配置 JSON，如 configs/default.json')
    parser.add_argument('--visualize', default=None, metavar='OUT_DIR',
                        help='可选：保存画框后的帧到 OUT_DIR')
    return parser


def _load_config(path):
    """读取 JSON 配置文件；path 为 None 时返回 None（用 PanoTracker 默认）。"""
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _visualize(frames_dir, out_file, vis_dir):
    """把 results 中的框画到对应帧上并保存（延迟导入模块 C 的 viz/io）。"""
    try:
        from panotrack.data.viz import draw_bbox
        from panotrack.io.file_protocol import _list_frames, _load_image
    except Exception as exc:
        print(f'[cli] 可视化不可用（模块 C 未就绪？）: {exc}', file=sys.stderr)
        return
    os.makedirs(vis_dir, exist_ok=True)
    paths = _list_frames(frames_dir)
    with open(out_file, 'r', encoding='utf-8') as f:
        boxes = [tuple(float(v) for v in line.replace(',', ' ').split())
                 for line in f if line.strip()]
    n = min(len(paths), len(boxes))
    for i in range(n):
        img = draw_bbox(_load_image(paths[i]), boxes[i])
        out_path = os.path.join(vis_dir, os.path.basename(paths[i]))
        from PIL import Image
        Image.fromarray(img).save(out_path)
    print(f'[cli] 可视化输出 {n} 帧到 {vis_dir}', file=sys.stderr)


def main(argv=None):
    """CLI 主入口；返回进程退出码。"""
    args = _build_parser().parse_args(argv)
    config = _load_config(args.config)

    # 延迟导入：保证 --help 与参数解析在模块 E 未实现时也能工作
    from panotrack.io.file_protocol import run_file_protocol

    stats = run_file_protocol(args.frames, args.init_file, args.out_file, config)
    print(f'[cli] 结果写入 {stats["out_file"]} | '
          f'{stats["n_frames"]} 帧, {stats["elapsed_sec"]:.3f}s, '
          f'FPS={stats["fps"]:.2f}', file=sys.stderr)

    if args.visualize:
        _visualize(args.frames, args.out_file, args.visualize)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
