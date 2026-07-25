"""序列读取：加载合成/真实序列的帧与真值。"""
import os
import numpy as np
from PIL import Image


def load_sequence(seq_dir):
    """读取序列目录下的 frames/*.png 与 gt.txt。

    参数：seq_dir —— 序列目录（含 frames 子目录与 gt.txt）。
    返回：(frames, gt)；frames 为 list[np.ndarray (H,W,3) uint8]，
         gt 为 (N,4) float 数组（x,y,w,h，跨界约定同契约）。
    """
    frames_dir = os.path.join(seq_dir, 'frames')
    names = sorted(n for n in os.listdir(frames_dir) if n.lower().endswith('.png'))
    if not names:
        raise FileNotFoundError(f'未找到帧图像: {frames_dir}')
    frames = [
        np.asarray(Image.open(os.path.join(frames_dir, n)).convert('RGB'))
        for n in names
    ]
    gt = np.loadtxt(os.path.join(seq_dir, 'gt.txt'), delimiter=',', ndmin=2)
    if len(frames) != len(gt):
        raise ValueError(f'帧数({len(frames)})与 GT 行数({len(gt)})不一致')
    return frames, gt
