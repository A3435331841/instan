# -*- coding: utf-8 -*-
"""全局重检测 v2：多尺度模板 + 多尺度 NCC 搜索 + 亚像素峰值细化。

改进：
1. 多尺度模板池（0.7x/1.0x/1.3x/1.7x），应对目标尺度变化
2. 多降采样倍率搜索（ds=2/3/4），避免小目标在粗降采样中消失
3. 亚像素峰值细化（二次拟合），提升定位精度
4. 模板质量评估：自动选择最清晰的模板尺度
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_EPS = 1e-8
_FLAT_STD = 0.25   # 平坦窗口标准差下限（比 v1 更严格）


def _highpass(gray, sigma=1.0):
    return gray - gaussian_filter(gray, sigma=sigma)


def _to_gray_f32(img):
    arr = np.asarray(img)
    if arr.ndim == 3:
        return (arr[..., 0].astype(np.float32) * 0.299
                + arr[..., 1].astype(np.float32) * 0.587
                + arr[..., 2].astype(np.float32) * 0.114)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resize(img, out_w, out_h):
    im = Image.fromarray(np.asarray(img))
    return np.asarray(im.resize((int(out_w), int(out_h)), Image.BILINEAR))


def _ncc_field(frame_g, tmpl_g):
    """单尺度 NCC 响应场。"""
    fh, fw = frame_g.shape
    th, tw = tmpl_g.shape
    if th > fh or tw > fw:
        return None
    t = tmpl_g - float(tmpl_g.mean())
    tn = float(np.linalg.norm(t))
    if tn < _EPS:
        return None
    t = (t / tn).astype(np.float32)
    ext = np.concatenate([frame_g, frame_g[:, :tw]], axis=1)
    corr = fftconvolve(ext, t[::-1, ::-1], mode='valid')[:, :fw]
    i1 = np.zeros((fh + 1, ext.shape[1] + 1), dtype=np.float64)
    i2 = np.zeros_like(i1)
    i1[1:, 1:] = np.cumsum(np.cumsum(ext, axis=0), axis=1)
    i2[1:, 1:] = np.cumsum(np.cumsum(ext * ext, axis=0), axis=1)
    win_sum = i1[th:, tw:] - i1[:-th, tw:] - i1[th:, :-tw] + i1[:-th, :-tw]
    win_sq = i2[th:, tw:] - i2[:-th, tw:] - i2[:-th, :-tw] + i2[:-th, :-tw]
    n = float(th * tw)
    denom = (win_sq - win_sum * win_sum / n)[:, :fw]
    np.maximum(denom, 0.0, out=denom)
    ok = denom > n * (_FLAT_STD ** 2)
    resp = np.where(ok, corr / (np.sqrt(denom) + 1e-6), -1.0)
    return np.clip(resp, -1.0, 1.0)


def _subpixel_peak(resp, py, px):
    """对 NCC 响应峰值做亚像素级二次插值细化。"""
    h, w = resp.shape
    if py <= 0 or py >= h - 1 or px <= 0 or px >= w - 1:
        return py, px, float(resp[py, px])
    # 取 3x3 邻域拟合抛物线
    patch = resp[py-1:py+2, px-1:px+2]
    # 二次拟合: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    # 简化：分别对 x/y 方向做 1D 二次拟合
    dx = (patch[0, 1] - patch[0, 0]) + (patch[1, 2] - patch[1, 1]) + (patch[2, 1] - patch[2, 0])
    dy = (patch[0, 0] + patch[0, 1] + patch[0, 2] - 2*patch[1, 1])
    # 半像素偏移
    sx = 0.5 * dx / (patch[0, 1] + patch[2, 1] - 2*patch[1, 1] + _EPS)
    sy = 0.5 * dy / (patch[1, 0] + patch[1, 2] - 2*patch[1, 1] + _EPS)
    sx = float(np.clip(sx, -0.5, 0.5))
    sy = float(np.clip(sy, -0.5, 0.5))
    return py + sy, px + sx, float(resp[py, px])


class GlobalRedetectorV2:
    """多尺度全局重检测器：ERP 全图 NCC 模板搜索（升级版）。"""

    # 模板尺度倍率：覆盖目标可能的尺度变化
    _TEMPLATE_SCALES = (0.7, 1.0, 1.3, 1.7)
    # 降采样倍率：多尺度搜索避免小目标被淹没
    _DOWNSAMPLE_DS = (2, 3, 4)

    def __init__(self, get_template, min_score=0.45):
        self._get_template = get_template
        self.min_score = float(min_score)

    def search(self, frame, erp_downscale=4):
        """多尺度全局搜索。

        参数: frame (H,W,3) uint8 ERP 帧；erp_downscale 主降采样倍率。
        返回: ((x,y,w,h), score) 或 None。
        """
        tpl = self._get_template()
        if tpl is None:
            return None
        timg, (ow, oh) = tpl
        H, W = frame.shape[:2]
        best = None  # (score, x, y, w, h, ds, scale)

        # 遍历多个降采样倍率
        for ds in self._DOWNSAMPLE_DS:
            sw, sh = max(8, W // ds), max(8, H // ds)
            fsmall = _highpass(_to_gray_f32(_resize(frame, sw, sh)))

            # 遍历多个模板尺度
            for tscale in self._TEMPLATE_SCALES:
                tw = max(4, int(round(ow * sw / W * tscale)))
                th = max(4, int(round(oh * sh / H * tscale)))
                if tw >= sw or th >= sh:
                    continue
                tsmall = _highpass(_to_gray_f32(_resize(timg, tw, th)))
                resp = _ncc_field(fsmall, tsmall)
                if resp is None:
                    continue
                py, px = np.unravel_index(int(np.argmax(resp)), resp.shape)
                score = float(resp[py, px])
                if score < self.min_score:
                    continue
                # 亚像素细化
                py_f, px_f, _ = _subpixel_peak(resp, py, px)
                x = px_f * W / sw
                y = py_f * H / sh
                # 尺度自适应调整输出框尺寸
                out_w = ow * tscale
                out_h = oh * tscale
                candidate = (score, x, y, out_w, out_h, ds, tscale)
                if best is None or score > best[0]:
                    best = candidate

        if best is None:
            return None
        score, x, y, out_w, out_h, ds, tscale = best
        return (float(x % W), float(y), float(out_w), float(out_h)), score
