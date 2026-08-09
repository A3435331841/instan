# -*- coding: utf-8 -*-
"""Siam 风格双分支跟踪器的共享 numpy 工具（裁剪 / 解码 / 归一化）。

供 VitTrack ONNX 与 LightFC ONNX 等 twin-branch 跟踪器复用，避免重复实现。
仅依赖 numpy + Pillow（与生产依赖白名单一致）。

裁剪约定: 水平方向按 ERP 全景宽度回绕（mod W），垂直方向 clip + 零填充，
与 lightfc_onnx 的历史实现保持一致，天然支持 360° 子午线穿越。
"""
import numpy as np
from PIL import Image

# ImageNet 归一化（与 OpenCV TrackerVit / LightFC 官方一致）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def hann1d(n, centered=True):
    """1D 余弦窗（numpy 实现，私有）。"""
    if centered:
        step = 0.5 / (n - 1) if n > 1 else 0.5
        axis = np.arange(n)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * (axis * step - 0.5))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def hann2d(sz, centered=True):
    """2D 余弦窗（numpy 实现，私有），形状 (1,1,H,W)。"""
    h, w = int(sz[0]), int(sz[1])
    return (hann1d(h, centered).reshape(-1, 1)
            * hann1d(w, centered).reshape(1, -1))[None, None].astype(np.float32)


def sample_target(im, target_bb, factor, output_sz):
    """按官方 sample_target 逻辑裁剪方形搜索区（水平回绕）。

    返回: (crop, resize_factor) —— crop 为 (output_sz, output_sz, 3) uint8。
    """
    H, W = im.shape[:2]
    x, y, w, h = (float(v) for v in target_bb)
    crop_sz = int(np.ceil(np.sqrt(w * h) * factor))
    crop_sz = max(crop_sz, 2)
    cx, cy = x + 0.5 * w, y + 0.5 * h
    half = crop_sz // 2
    # 用整数起点构造 arange，杜绝浮点舍入导致长度多 1 的 shape mismatch
    x0 = int(np.floor(cx - half))
    y0 = int(np.floor(cy - half))
    cols = np.mod(np.arange(x0, x0 + crop_sz), W).astype(np.int64)
    rows = np.arange(y0, y0 + crop_sz)
    rows_c = np.clip(rows, 0, H - 1).astype(np.int64)
    out = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
    valid = (rows >= 0) & (rows < H)
    out[valid] = im[rows_c[valid]][:, cols]
    if output_sz is not None and output_sz != crop_sz:
        resize_factor = output_sz / crop_sz
        out = np.asarray(Image.fromarray(out).resize(
            (int(output_sz), int(output_sz)), Image.BILINEAR))
    else:
        resize_factor = 1.0
    return out, resize_factor


def preprocess(patch):
    """uint8 RGB 裁剪图 -> (1,3,H,W) float32 归一化数组。"""
    arr = patch.astype(np.float32).transpose(2, 0, 1)[None]
    return (arr / 255.0 - _MEAN) / _STD


def cal_bbox(score, size, offset, feat_sz):
    """由响应图解码局部框（numpy 版，对应官方 head.cal_bbox）。

    score: (1,1,fs,fs) 加窗后响应; size: (1,2,fs,fs); offset: (1,2,fs,fs)。
    返回: np.array([cx, cy, w, h]) 归一化坐标（cx/cy 在 [0,1] 尺度）。
    """
    fs = int(feat_sz)
    s = score[0, 0]
    idx = int(np.argmax(s))
    iy, ix = divmod(idx, fs)
    sz = size[0, :, iy, ix]        # (2,) w,h
    off = offset[0, :, iy, ix]     # (2,) dx,dy
    cx = (ix + off[0]) / fs
    cy = (iy + off[1]) / fs
    return np.array([cx, cy, sz[0], sz[1]], dtype=np.float64)


def map_box_back(pred_box, state, search_size, resize_factor):
    """局部预测框映射回原图坐标（官方 map_box_back 逻辑）。

    state 为上一帧 bbox (x,y,w,h)；pred_box 为像素尺度 [cx,cy,w,h]。
    """
    cx_prev = state[0] + 0.5 * state[2]
    cy_prev = state[1] + 0.5 * state[3]
    cx, cy, w, h = (float(v) for v in pred_box)
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def clip_box(box, H, W, margin=2):
    """边界钳制: x 回绕 [0,W), y clamp, 尺寸不小于 margin。"""
    x, y, w, h = (float(v) for v in box)
    w = max(w, margin)
    h = max(h, margin)
    x = x % W
    y = float(np.clip(y, 0, max(0.0, H - h)))
    return [x, y, w, h]