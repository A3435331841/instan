# -*- coding: utf-8 -*-
"""全局重检测：ERP 降采样整幅 NCC 模板搜索（模块 E）。

丢失后用于在整幅全景图上找回目标：帧与模板同比例降采样，
FFT 计算零均值归一化互相关（NCC），水平方向按 ERP 回绕扩展，
使跨界目标也能命中。仅依赖 numpy/Pillow/scipy。
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

_EPS = 1e-8
_FLAT_STD = 0.3   # 窗口灰度标准差下限：平坦窗口的 NCC 为噪声相除，会虚报满分


def _highpass(gray, sigma=1.0):
    """灰度高斯高通（私有）：抑制平滑背景斜坡，保留纹理与边缘。"""
    return gray - gaussian_filter(gray, sigma=sigma)


def _to_gray_f32(img):
    """(H,W,3) uint8 RGB 转 (H,W) float32 灰度（私有）。"""
    arr = np.asarray(img)
    if arr.ndim == 3:
        return (arr[..., 0].astype(np.float32) * 0.299
                + arr[..., 1].astype(np.float32) * 0.587
                + arr[..., 2].astype(np.float32) * 0.114)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resize(img, out_w, out_h):
    """PIL 双线性缩放（私有），返回同通道数 uint8 数组。"""
    im = Image.fromarray(np.asarray(img))
    return np.asarray(im.resize((int(out_w), int(out_h)), Image.BILINEAR))


def _ncc_field(frame_g, tmpl_g):
    """降采样灰度域 NCC 响应场（私有）。

    参数: frame_g (H,W) float32；tmpl_g (th,tw) float32。
    返回: (H-th+1, W) float32 响应；列 x ∈ [0, W) 表示模板左上角经度位置，
          水平方向已按 ERP 回绕扩展（帧右侧拼接前 tw 列）。
    """
    fh, fw = frame_g.shape
    th, tw = tmpl_g.shape
    t = tmpl_g - float(tmpl_g.mean())
    tn = float(np.linalg.norm(t))
    if tn < _EPS:
        return None
    t = (t / tn).astype(np.float32)
    ext = np.concatenate([frame_g, frame_g[:, :tw]], axis=1)  # 水平回绕扩展
    # 零均值模板 => 分子即线性相关
    corr = fftconvolve(ext, t[::-1, ::-1], mode='valid')[:, :fw]
    # 分母：各模板窗口去均值能量（积分图）
    i1 = np.zeros((fh + 1, ext.shape[1] + 1), dtype=np.float64)
    i2 = np.zeros_like(i1)
    i1[1:, 1:] = np.cumsum(np.cumsum(ext, axis=0), axis=1)
    i2[1:, 1:] = np.cumsum(np.cumsum(ext * ext, axis=0), axis=1)
    win_sum = i1[th:, tw:] - i1[:-th, tw:] - i1[th:, :-tw] + i1[:-th, :-tw]
    win_sq = i2[th:, tw:] - i2[:-th, tw:] - i2[th:, :-tw] + i2[:-th, :-tw]
    n = float(th * tw)
    denom = (win_sq - win_sum * win_sum / n)[:, :fw]
    np.maximum(denom, 0.0, out=denom)
    # 平坦窗口屏蔽：denom≈0 时 corr 同为浮点噪声，相除会被放大并 clip 成
    # 满分 1.0（天空/糊区虚报满分的根因）；此类窗口直接置 -1 不参与峰值竞争
    ok = denom > n * (_FLAT_STD ** 2)
    resp = np.where(ok, corr / (np.sqrt(denom) + 1e-6), -1.0)
    return np.clip(resp, -1.0, 1.0)


class GlobalRedetector:
    """ERP 全图降采样模板搜索器：用于丢失后的全局目标找回。

    get_template 由上层（PanoTracker）提供，返回 (模板图, (w_erp, h_erp))：
    模板图为目标在 ERP 域的 RGB 裁剪，(w_erp, h_erp) 为其原分辨率像素尺寸。
    """

    def __init__(self, get_template, min_score=0.5):
        """创建全局重检测器。

        参数: get_template 可调用，返回 (模板图 (th,tw,3) uint8, (w,h)) 或 None；
              min_score 命中所需的最低 NCC 分数。
        返回: None
        """
        self._get_template = get_template
        self.min_score = float(min_score)

    def search(self, frame, erp_downscale=4):
        """在整幅 ERP 帧上搜索目标模板（含水平回绕）。

        参数: frame (H,W,3) uint8 ERP 帧；erp_downscale 降采样倍率。
        返回: ((x,y,w,h), score) —— x∈[0,W)、x+w 可超 W（跨界约定）；
              无模板或峰值低于 min_score 时返回 None。
        """
        tpl = self._get_template()
        if tpl is None:
            return None
        timg, (ow, oh) = tpl
        H, W = frame.shape[:2]
        ds = max(1, int(erp_downscale))
        sw, sh = max(8, W // ds), max(8, H // ds)
        tw = max(4, int(round(ow * sw / W)))
        th = max(4, int(round(oh * sh / H)))
        if tw >= sw or th >= sh:
            return None  # 模板大于搜索图，全局搜索无意义
        fsmall = _highpass(_to_gray_f32(_resize(frame, sw, sh)))
        tsmall = _highpass(_to_gray_f32(_resize(timg, tw, th)))
        resp = _ncc_field(fsmall, tsmall)
        if resp is None:
            return None
        py, px = np.unravel_index(int(np.argmax(resp)), resp.shape)
        score = float(resp[py, px])
        if score < self.min_score:
            return None
        x = px * W / sw
        y = py * H / sh
        return (float(x % W), float(y), float(ow), float(oh)), score
