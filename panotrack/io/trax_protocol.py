"""panotrack.io.trax_protocol —— trax 风格 stdin/stdout 行协议适配（占位实现）。

【占位声明】8 月官方评测协议公布后，本模块将被替换为正式实现。
当前为最小可用的行协议，用于本地联调与容器内协议冒烟测试。

约定：
  - stdout 只输出协议行；一切调试信息走 stderr。
  - 文本行协议（空格分隔）：
      启动后服务端输出:  `panotrack-trax 0.1 ready`
      C->S: `init <image_path> <x> <y> <w> <h>`   初始化（首帧图像 + 初始框）
            S->C: `ok init`
      C->S: `frame <image_path>`                  跟踪下一帧
            S->C: `bbox <x> <y> <w> <h>`          （保留 2 位小数，跨界约定同契约）
      C->S: `quit`                                结束会话
            S->C: `ok bye`
      未知/非法命令: S->C: `error <message>`
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

_PROTOCOL_VERSION = 'panotrack-trax 0.1'


def _load_image(path):
    """读取图像为 (H,W,3) uint8 RGB 数组（契约通用约定）。"""
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.uint8)


def _create_pano_tracker(config):
    """延迟导入并创建 PanoTracker（契约模块 E，集成阶段实现）。"""
    try:
        from panotrack.pipeline.pipeline import PanoTracker
    except Exception as exc:
        raise RuntimeError(
            '无法导入 panotrack.pipeline.pipeline.PanoTracker'
            '（契约模块 E 尚未实现或其实现有误）'
        ) from exc
    return PanoTracker(config)


def run_trax_protocol(input_stream=None, output_stream=None, config=None):
    """运行 trax 风格行协议会话（占位实现）。

    参数:
        input_stream: 命令输入流，默认 sys.stdin。
        output_stream: 协议输出流（只写协议行），默认 sys.stdout。
        config: PanoTracker 配置 dict；None 表示使用默认配置。
    返回:
        int: 进程退出码（正常 quit 返回 0，未初始化即 frame 等错误返回 1）。
    """
    instream = input_stream if input_stream is not None else sys.stdin
    outstream = output_stream if output_stream is not None else sys.stdout
    tracker = None
    exit_code = 0

    def emit(line):
        outstream.write(line + '\n')
        outstream.flush()

    emit(f'{_PROTOCOL_VERSION} ready')
    for raw in instream:
        parts = raw.strip().split()
        if not parts:
            continue
        cmd, args = parts[0].lower(), parts[1:]
        try:
            if cmd == 'init':
                if len(args) != 5:
                    emit('error init 需要 5 个参数: <image_path> <x> <y> <w> <h>')
                    continue
                if tracker is None:
                    tracker = _create_pano_tracker(config)
                frame = _load_image(args[0])
                bbox = tuple(float(v) for v in args[1:5])
                tracker.init(frame, bbox)
                emit('ok init')
            elif cmd == 'frame':
                if len(args) != 1:
                    emit('error frame 需要 1 个参数: <image_path>')
                    continue
                if tracker is None:
                    emit('error 尚未 init')
                    exit_code = 1
                    continue
                frame = _load_image(args[0])
                res = tracker.update(frame)
                x, y, w, h = (float(v) for v in res['bbox'])
                print(f'[trax] status={res.get("status")} '
                      f'score={res.get("score")}', file=sys.stderr)
                emit(f'bbox {x:.2f} {y:.2f} {w:.2f} {h:.2f}')
            elif cmd == 'quit':
                emit('ok bye')
                break
            else:
                emit(f'error 未知命令: {cmd}')
        except Exception as exc:  # 协议层不崩溃，错误以协议行返回
            emit(f'error {type(exc).__name__}: {exc}')
            exit_code = 1
    return exit_code


if __name__ == '__main__':
    raise SystemExit(run_trax_protocol())
