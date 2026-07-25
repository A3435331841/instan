# -*- coding: utf-8 -*-
"""纯 numpy FFT 归一化互相关（NCC）单目标跟踪器（SiamFC 思想经典版）。

核心流程：余弦窗预处理模板 -> 上一帧位置周围按 search_scale 倍切取搜索图
-> 多尺度 FFT-NCC 响应 -> 峰值定位 + 抛物线亚像素细化 -> 门控模板更新。
全部状态保存在实例内，可并发实例化。
"""
from __future__ import annotations

import numpy as np

from .base import BaseTracker

_UPDATE_THR = 0.5   # 模板门控更新阈值：score 低于该值判定不可靠，冻结模板
_PSR_RADIUS = 5     # PSR 统计时排除的峰值邻域半径（响应图像素）
_EPS = 1e-8
_FLAT_STD = 0.3     # 窗口灰度标准差下限（0-255 量纲）：低于则 NCC 无判别力


def _fast_fft_len(n):
    """不小于 n 的最小 5-smooth 数（2/3/5 因子），保证 pocketfft 走快速分解。

    线性相关所需长度常为质数乘积（如 381=3×127），radix 过大极慢，需上取整。
    """
    m = n
    while True:
        x = m
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return m
        m += 1


def _to_gray(image):
    """(H,W,3) uint8 RGB 转 (H,W) float32 灰度（兼容单通道输入），全程 float32。"""
    arr = np.asarray(image)
    if arr.ndim == 3:
        r = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32)
        b = arr[..., 2].astype(np.float32)
        r *= 0.299
        g *= 0.587
        b *= 0.114
        r += g
        r += b
        return r
    return np.ascontiguousarray(arr, dtype=np.float32)


def _sample_patch(gray, cx, cy, pw, ph, out, base=None):
    """以 (cx,cy) 为中心、宽高 (pw,ph) 的图像区域双线性采样为 out×out 方阵。

    参数: gray (H,W) float32 灰度图；cx,cy 中心坐标；pw,ph 采样区域宽高（像素）；
          out 输出边长；base 可选的缓存基坐标 (arange+0.5-out/2)/out。
          越界处复制边缘像素。
    返回: (out,out) float32 采样结果。
    """
    H, W = gray.shape
    if base is None:
        base = (np.arange(out, dtype=np.float32) + 0.5 - out * 0.5) / out
    ys = cy + base * ph
    xs = cx + base * pw
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    Ia = gray[np.ix_(y0, x0)]
    Ib = gray[np.ix_(y0, x1)]
    Ic = gray[np.ix_(y1, x0)]
    Id = gray[np.ix_(y1, x1)]
    top = Ia + (Ib - Ia) * wx[None, :]
    bot = Ic + (Id - Ic) * wx[None, :]
    return top + (bot - top) * wy[:, None]


def _psr(resp, py, px, radius=_PSR_RADIUS):
    """峰值旁瓣比：(峰值 - 邻域外均值) / 邻域外标准差，平坦响应返回 0。"""
    R, C = resp.shape
    y0 = max(0, py - radius)
    y1 = min(R, py + radius + 1)
    x0 = max(0, px - radius)
    x1 = min(C, px + radius + 1)
    mask = np.ones(resp.shape, dtype=bool)
    mask[y0:y1, x0:x1] = False
    side = resp[mask]
    if side.size < 8:
        return 0.0
    mu = float(side.mean())
    sd = float(side.std())
    if sd < _EPS:
        return 0.0
    return float((resp[py, px] - mu) / sd)


def _apce(resp):
    """平均峰值相关能量：(max-min)^2 / mean((r-min)^2)，平坦响应返回 0。"""
    fmax = float(resp.max())
    fmin = float(resp.min())
    denom = float(np.mean((resp - fmin) ** 2))
    if denom < 1e-12:
        return 0.0
    return float((fmax - fmin) ** 2 / denom)


class NCCTracker(BaseTracker):
    """FFT 加速的多尺度归一化互相关模板匹配跟踪器。

    score 为峰值 NCC 相关度（裁剪到 [0,1]）；psr/apce 描述响应峰锐利程度，
    可供上层做丢失判定。模板按 lr 指数滑动更新，score 低于门限时冻结，
    避免遮挡期间污染模板。
    """

    def __init__(self, context=1.0, scales=(0.98, 1.0, 1.02), lr=0.02,
                 search_scale=2.0, template_size=127):
        """创建 NCC 跟踪器实例（无全局状态，可并发实例化）。

        参数:
            context: 模板上下文比例，模板裁剪边长 = 目标边长 × (1+context)。
            scales: 每帧相对尺度搜索因子序列。
            lr: 模板指数滑动更新率（仅 score 不低于门限时生效）。
            search_scale: 搜索区域相对模板裁剪区域的放大倍数。
            template_size: 模板边长（像素）。
        返回: None
        """
        self.context = float(context)
        self.scales = tuple(float(s) for s in scales)
        self.lr = float(lr)
        self.search_scale = float(search_scale)
        self.template_size = int(template_size)
        # 逐轴尺度搜索点集（十字形）：(f,1) 与 (1,f)，支持目标长宽比变化
        # （全景极区目标在切图域会被拉成扁弧，各向同性尺度无法贴合）
        self._scale_pairs = tuple(sorted(
            {(f, 1.0) for f in self.scales} | {(1.0, f) for f in self.scales}))
        # 搜索图边长取奇数，使响应图零偏移恰好落在整数像素（默认 255）
        ss = int(round(self.template_size * self.search_scale))
        self.search_size = ss if ss % 2 == 1 else ss + 1
        # FFT 线性相关尺寸上取整到快速长度（默认 381 -> 384，避免大质数 radix）
        self._fft_h = _fast_fft_len(self.search_size + self.template_size - 1)
        self._fft_w = self._fft_h
        # 余弦窗（每实例独立，避免全局状态）
        w1 = np.hanning(self.template_size).astype(np.float32)
        self._win = np.outer(w1, w1)
        # 预分配复用缓冲（实例私有，线程安全）
        self._base_search = (np.arange(self.search_size, dtype=np.float32)
                             + 0.5 - self.search_size * 0.5) / self.search_size
        self._base_tmpl = (np.arange(self.template_size, dtype=np.float32)
                           + 0.5 - self.template_size * 0.5) / self.template_size
        self._int1 = np.empty((self.search_size + 1, self.search_size + 1), dtype=np.float64)
        self._int2 = np.empty((self.search_size + 1, self.search_size + 1), dtype=np.float64)
        self._prod = np.empty((self._fft_h, self._fft_w // 2 + 1), dtype=np.complex64)
        # 目标状态
        self._cx = 0.0
        self._cy = 0.0
        self._w = 1.0
        self._h = 1.0
        self._tmpl = None       # 零均值单位范数模板 (template_size, template_size)
        self._tmpl_hat = None   # 模板 FFT 共轭缓存
        self._ready = False

    def init(self, image, bbox):
        """用首帧局部透视图与目标框初始化跟踪器。

        参数: image (H,W,3) uint8 RGB 局部透视图；bbox (x,y,w,h) 局部像素框。
        返回: None
        """
        gray = _to_gray(image)
        H, W = gray.shape
        x, y, w, h = (float(v) for v in bbox)
        self._w = max(2.0, w)
        self._h = max(2.0, h)
        self._cx = float(np.clip(x + w / 2.0, 0.0, W - 1.0))
        self._cy = float(np.clip(y + h / 2.0, 0.0, H - 1.0))
        patch = _sample_patch(gray, self._cx, self._cy,
                              self._w * (1.0 + self.context),
                              self._h * (1.0 + self.context),
                              self.template_size, base=self._base_tmpl)
        p = (patch - float(patch.mean())) * self._win
        n = float(np.linalg.norm(p))
        if n < _EPS:
            p = np.zeros_like(p)  # 完全平坦的退化模板
        else:
            p = p / n
        self._tmpl = p.astype(np.float32)
        self._tmpl_hat = np.conj(np.fft.rfft2(self._tmpl, (self._fft_h, self._fft_w)))
        self._ready = True

    def update(self, image):
        """在新一帧局部透视图上定位目标。

        参数: image (H,W,3) uint8 RGB 局部透视图。
        返回: dict {'bbox': (x,y,w,h), 'score': float∈[0,1], 'psr': float, 'apce': float}
        """
        if not self._ready:
            raise RuntimeError("NCCTracker 未初始化，请先调用 init(image, bbox)")
        gray = _to_gray(image)
        H, W = gray.shape
        base_w = self._w * (1.0 + self.context) * self.search_scale
        base_h = self._h * (1.0 + self.context) * self.search_scale
        # 逐轴多尺度搜索：x/y 方向独立缩放裁剪窗口，使目标长宽比可独立演化
        best = None
        for sx, sy in self._scale_pairs:
            pw = base_w * sx
            ph = base_h * sy
            search = _sample_patch(gray, self._cx, self._cy, pw, ph,
                                   self.search_size, base=self._base_search)
            resp = self._ncc_response(search)
            r, c = np.unravel_index(int(np.argmax(resp)), resp.shape)
            val = float(resp[r, c])
            if best is None or val > best[0]:
                best = (val, sx, sy, resp, r, c, pw, ph)
        val, sx, sy, resp, r, c, pw, ph = best
        if val <= 0.0:
            # 整幅搜索图无有效 NCC 窗口（全平坦）：argmax 落在噪声上有害，
            # 保持位置与尺度，返回零置信由上层判丢/重检测
            return {'bbox': (self._cx - self._w / 2.0, self._cy - self._h / 2.0,
                             self._w, self._h),
                    'score': 0.0, 'psr': 0.0, 'apce': 0.0}
        fr, fc = self._subpixel(resp, r, c)
        # 响应图位置 -> 搜索图中心偏移 -> 图像位移（各轴向独立换算）
        off_y = (fr + (self.template_size - 1) * 0.5) - (self.search_size - 1) * 0.5
        off_x = (fc + (self.template_size - 1) * 0.5) - (self.search_size - 1) * 0.5
        self._cx = float(np.clip(self._cx + off_x * (pw / self.search_size), 0.0, W - 1.0))
        self._cy = float(np.clip(self._cy + off_y * (ph / self.search_size), 0.0, H - 1.0))
        # 尺度更新并做范围保护（x/y 轴独立）
        self._w = float(np.clip(self._w * sx, 4.0, 2.0 * W))
        self._h = float(np.clip(self._h * sy, 4.0, 2.0 * H))
        score = float(min(max(val, 0.0), 1.0))
        psr = _psr(resp, r, c)
        apce = _apce(resp)
        # 门控模板更新：低置信帧冻结模板
        if score >= _UPDATE_THR:
            self._update_template(gray)
        return {'bbox': (self._cx - self._w / 2.0, self._cy - self._h / 2.0,
                         self._w, self._h),
                'score': score, 'psr': psr, 'apce': apce}

    def _ncc_response(self, search):
        """FFT 计算搜索图与模板的 NCC 响应图（valid 区域，值域 [-1,1]）。"""
        ts = self.template_size
        M, N = search.shape
        R = M - ts + 1
        C = N - ts + 1
        s_hat = np.fft.rfft2(search, (self._fft_h, self._fft_w))
        # ifft(fft(S)·conj(fft(T))) 即线性互相关（已零 padding 无回绕混叠）
        np.multiply(s_hat, self._tmpl_hat, out=self._prod)
        corr = np.fft.irfft2(self._prod, (self._fft_h, self._fft_w))[:R, :C]
        # 模板零均值单位范数：分子即 corr，分母为各窗口去均值能量
        win_sum, win_sq = self._window_sums(search)
        denom = win_sq - win_sum * win_sum / (ts * ts)
        np.maximum(denom, 0.0, out=denom)
        # 平坦窗口屏蔽：denom≈0 时 corr 同为浮点噪声，相除会被放大并 clip 成
        # 满分 1.0（真实全景中天空/严重模糊区导致 score=1.000 误锁的根因）
        ok = denom > (ts * ts) * (_FLAT_STD ** 2)
        resp = np.where(ok, corr / (np.sqrt(denom) + 1e-6), -1.0)
        return np.clip(resp, -1.0, 1.0)

    def _window_sums(self, search):
        """积分图计算每个 template_size² 窗口的和与平方和（复用实例缓冲）。

        float64 累加保证方差精度；返回 (win_sum, win_sq)，形状为响应图大小。
        """
        k = self.template_size
        i1 = self._int1
        i2 = self._int2
        i1[0, :] = 0.0
        i1[:, 0] = 0.0
        i2[0, :] = 0.0
        i2[:, 0] = 0.0
        np.cumsum(search, axis=0, dtype=np.float64, out=i1[1:, 1:])
        np.cumsum(i1[1:, 1:], axis=1, out=i1[1:, 1:])
        np.cumsum(search * search, axis=0, dtype=np.float64, out=i2[1:, 1:])
        np.cumsum(i2[1:, 1:], axis=1, out=i2[1:, 1:])
        win_sum = i1[k:, k:] - i1[:-k, k:] - i1[k:, :-k] + i1[:-k, :-k]
        win_sq = i2[k:, k:] - i2[:-k, k:] - i2[k:, :-k] + i2[:-k, :-k]
        return win_sum, win_sq

    def _update_template(self, gray):
        """按当前状态重采模板并指数滑动融合（门控由调用方保证）。"""
        patch = _sample_patch(gray, self._cx, self._cy,
                              self._w * (1.0 + self.context),
                              self._h * (1.0 + self.context),
                              self.template_size, base=self._base_tmpl)
        p = (patch - float(patch.mean())) * self._win
        n = float(np.linalg.norm(p))
        if n < _EPS:
            return  # 退化补丁，跳过更新
        p /= n
        t = (1.0 - self.lr) * self._tmpl + self.lr * p
        tn = float(np.linalg.norm(t))
        if tn < _EPS:
            return
        self._tmpl = (t / tn).astype(np.float32)
        self._tmpl_hat = np.conj(np.fft.rfft2(self._tmpl, (self._fft_h, self._fft_w)))

    @staticmethod
    def _subpixel(resp, py, px):
        """响应峰值抛物线亚像素细化，边界或退化时退回整数位置。"""
        fy, fx = float(py), float(px)
        R, C = resp.shape
        if 0 < py < R - 1:
            d = 2.0 * float(resp[py, px]) - float(resp[py - 1, px]) - float(resp[py + 1, px])
            if d > _EPS:
                o = 0.5 * (float(resp[py - 1, px]) - float(resp[py + 1, px])) / d
                fy += float(np.clip(o, -1.0, 1.0))
        if 0 < px < C - 1:
            d = 2.0 * float(resp[py, px]) - float(resp[py, px - 1]) - float(resp[py, px + 1])
            if d > _EPS:
                o = 0.5 * (float(resp[py, px - 1]) - float(resp[py, px + 1])) / d
                fx += float(np.clip(o, -1.0, 1.0))
        return fy, fx
