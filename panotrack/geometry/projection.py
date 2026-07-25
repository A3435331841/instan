# -*- coding: utf-8 -*-
"""tangent 局部切图正投影、双线性重采样、局部框逆投影与 remap LRU 缓存。"""
from collections import OrderedDict

import numpy as np

from .sphere import _tangent_frame, _offset_dirs
from .bfov import BFoV


def tangent_remap(bfov, out_w, out_h, erp_w, erp_h):
    """生成 tangent 切图的 ERP 源坐标映射（fov≤90° 用 gnomonic，否则 eBFoV 球面均匀角采样）。

    参数: bfov BFoV（rotation 保留参数，暂未实现）；out_w, out_h 切图尺寸；
          erp_w, erp_h 全景图尺寸。
    返回: (map_x, map_y)，float32 数组 (out_h, out_w)；map_x 已模 W 回绕，
          map_y clamp 到 [0, erp_h-1]；坐标为源图像素索引（像素中心为整数）。
    """
    out_w, out_h = int(out_w), int(out_h)
    gnomonic = max(bfov.fov_h, bfov.fov_v) <= 90.0
    frame = _tangent_frame(bfov.lon, bfov.lat)
    # 输出像素中心的切平面带符号角偏移（行向下 = 向南，故 dv 取负）
    du = ((np.arange(out_w) + 0.5) / out_w - 0.5) * np.deg2rad(bfov.fov_h)
    dv = (0.5 - (np.arange(out_h) + 0.5) / out_h) * np.deg2rad(bfov.fov_v)
    DU, DV = np.meshgrid(du, dv)
    vx, vy, vz = _offset_dirs(DU, DV, frame, gnomonic)
    lon = np.rad2deg(np.arctan2(vz, vx))
    lat = np.rad2deg(np.arcsin(np.clip(vy, -1.0, 1.0)))
    # 连续源坐标减 0.5 转为像素索引坐标（与 remap_image 双线性约定一致）
    map_x = np.mod((lon + 180.0) / 360.0 * erp_w - 0.5, erp_w)
    map_y = np.clip((90.0 - lat) / 180.0 * erp_h - 0.5, 0.0, erp_h - 1.0)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def remap_image(img, map_x, map_y):
    """纯 numpy 双线性重采样：水平回绕、垂直 clamp。

    参数: img (H,W) 或 (H,W,3) 数组；map_x, map_y 同形 float 数组，源像素索引坐标。
    返回: 重采样结果，形状与 map 相同（彩色带通道维），dtype 与输入一致。
    """
    h, w = img.shape[:2]
    mx = np.asarray(map_x, dtype=np.float64)
    my = np.asarray(map_y, dtype=np.float64)
    x0 = np.floor(mx).astype(np.int64)
    y0 = np.floor(my).astype(np.int64)
    wx = mx - x0
    wy = my - y0
    x0w = np.mod(x0, w)
    x1w = np.mod(x0 + 1, w)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    Ia = img[y0c, x0w].astype(np.float64)
    Ib = img[y0c, x1w].astype(np.float64)
    Ic = img[y1c, x0w].astype(np.float64)
    Id = img[y1c, x1w].astype(np.float64)
    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy
    if img.ndim == 3:
        wa, wb, wc, wd = wa[..., None], wb[..., None], wc[..., None], wd[..., None]
    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(img.dtype)


def _unwrap_rows(mx, width):
    """map_x 按行解缠（私有）：消除 ±W 跳变，得到可安全插值的连续坐标。"""
    d = np.diff(mx, axis=1)
    shift = np.where(d > width / 2.0, -width, np.where(d < -width / 2.0, width, 0.0))
    corr = np.concatenate([np.zeros((mx.shape[0], 1)), np.cumsum(shift, axis=1)], axis=1)
    return mx + corr


def _bilinear_sample(m, ix, iy):
    """在二维网格上按索引坐标双线性取值（私有），越界 clamp。"""
    h, w = m.shape
    ix = np.clip(ix, 0.0, w - 1.0)
    iy = np.clip(iy, 0.0, h - 1.0)
    x0 = np.floor(ix).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = ix - x0
    wy = iy - y0
    return (m[y0, x0] * (1 - wx) * (1 - wy) + m[y0, x1] * wx * (1 - wy)
            + m[y1, x0] * (1 - wx) * wy + m[y1, x1] * wx * wy)


def local_bbox_to_erp(lx, ly, lw, lh, map_x, map_y, erp_w, erp_h):
    """把局部（切图）坐标框经 map 逆投影回 ERP 最小面积框（边界采样 ≥16 点/边）。

    参数: lx, ly, lw, lh 局部框（像素，浮点）；map_x, map_y 为 tangent_remap 输出；
          erp_w, erp_h 全景图尺寸。
    返回: (x, y, w, h) 浮点 ERP 框，x∈[0,W)，跨界时 x+w 可超 W。
    """
    mx = np.asarray(map_x, dtype=np.float64)
    my = np.asarray(map_y, dtype=np.float64)
    mxu = _unwrap_rows(mx, erp_w)  # 先解缠再插值，避免跨界接缝处插值错误
    n = max(16, int(np.ceil(max(lw, lh))) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs = np.concatenate([lx + t * lw, lx + t * lw, np.full(n, lx), np.full(n, lx + lw)])
    ys = np.concatenate([np.full(n, ly), np.full(n, ly + lh), ly + t * lh, ly + t * lh])
    # map 网格节点对应连续坐标 (i+0.5, j+0.5)，故采样索引减 0.5；map 值加 0.5 还原连续源坐标
    sx = _bilinear_sample(mxu, xs - 0.5, ys - 0.5) + 0.5
    sy = _bilinear_sample(my, xs - 0.5, ys - 0.5) + 0.5
    ref = float(np.median(sx))  # 再次相对中位数回绕，保证跨界框连续
    sx = sx + erp_w * np.round((ref - sx) / erp_w)
    x0, x1 = float(sx.min()), float(sx.max())
    y0 = float(np.clip(sy.min(), 0.0, erp_h))
    y1 = float(np.clip(sy.max(), 0.0, erp_h))
    return x0 % erp_w, y0, x1 - x0, y1 - y0


class RemapCache:
    """tangent_remap 的 LRU 缓存：按量化键 (lon/2°, lat/2°, fov/2°) + 尺寸缓存。"""

    def __init__(self, capacity=64):
        """参数: capacity 缓存容量（默认 64）。"""
        self.capacity = int(capacity)
        self._cache = OrderedDict()

    @staticmethod
    def _key(bfov, out_w, out_h, erp_w, erp_h):
        """量化键（私有）：角度按 2° 量化，附输出与全景尺寸防串扰。"""
        return (int(np.round(bfov.lon / 2.0)), int(np.round(bfov.lat / 2.0)),
                int(np.round(bfov.fov_h / 2.0)), int(np.round(bfov.fov_v / 2.0)),
                int(out_w), int(out_h), int(erp_w), int(erp_h))

    def get_remap(self, bfov, out_w, out_h, erp_w, erp_h):
        """按量化键取 remap，未命中则计算并写入 LRU；返回 (map_x, map_y) 副本。

        参数: bfov BFoV；out_w, out_h 切图尺寸；erp_w, erp_h 全景图尺寸。
        返回: (map_x, map_y) float32 数组 (out_h, out_w)，与 tangent_remap 输出一致。
        """
        key = self._key(bfov, out_w, out_h, erp_w, erp_h)
        if key in self._cache:
            self._cache.move_to_end(key)
            mx, my = self._cache[key]
        else:
            mx, my = tangent_remap(bfov, out_w, out_h, erp_w, erp_h)
            self._cache[key] = (mx, my)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return mx.copy(), my.copy()
