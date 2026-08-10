# -*- coding: utf-8 -*-
"""GRT-360 Spherical Multi-view Reacquisition（全局重检测 v3）。

核心思想：不再只对 ERP 全图做一次模板搜索，而是先在球面（S²）上生成
一组**重叠的多视角窗口**（经度环绕 + 纬度分带），对每个视角窗口内的
ERP 子区域与**多模板池**（来自 TemplateMemory 的 anchor/short/long）
做多尺度 NCC 搜索，最后取全局最高分。这带来三方面收益：

1. 视角退避：目标大幅移动/重出现在任意球面位置时，经纬网格覆盖整个球面，
   不会因单窗口受限而漏检；off-center 视角对目标几何畸变（极区拉伸、
   seam 回绕）更鲁棒。
2. 多模板池：外观在遮挡/旋转后变化时，用 anchor 一致性 + 短期/长期模板
   共同比选，比单一模板更可能命中。
3. 局部聚焦：每个视角限制在对应 ERP 子区域搜索，抑制全图平坦区伪峰，
   同时因为按视角分块计算量相当（每视角窗口 ≈ W/3 宽），可接受。

视角网格：纬度分带 lat_bands × 每带经度数 lon_per_band，视角半宽
view_half_lon/lat（度），相邻视角重叠以覆盖 seam/极区。默认生成 12 个
视角（4×3），符合 6-12 规格。
"""
import numpy as np

from .redetect_v2 import (
    _highpass, _to_gray_f32, _resize, _ncc_field, _subpixel_peak,
)


class SphericalMultiViewRedetector:
    """球面多视角全局重检测器：多视角窗口 × 多模板池 × 多尺度 NCC。"""

    def __init__(self, get_templates, min_score=0.45,
                 lat_bands=(-45.0, 0.0, 45.0), lon_per_band=(4, 4, 4),
                 view_half_lon=70.0, view_half_lat=70.0,
                 template_scales=(0.9, 1.0, 1.15)):
        """创建球面多视角重检测器。

        参数: get_templates 回调，返回模板列表（TemplateMemory.get_bank 输出）；
              min_score 最低命中分；lat_bands 纬度分带（度）；lon_per_band 每带
              经度窗口数；view_half_lon/lat 视角半宽（度，> 经度间隔即重叠）；
              template_scales 模板尺度倍率。
        返回: None
        """
        self._get_templates = get_templates
        self.min_score = float(min_score)
        self.lat_bands = tuple(float(b) for b in lat_bands)
        self.lon_per_band = tuple(max(2, int(n)) for n in lon_per_band)
        self.view_half_lon = float(view_half_lon)
        self.view_half_lat = float(view_half_lat)
        self.template_scales = tuple(float(s) for s in template_scales)

    @property
    def n_views(self):
        return sum(self.lon_per_band)

    def view_centers(self):
        """生成视角中心 (lon, lat) 列表（经度环绕 + 纬度分带）。"""
        centers = []
        for lat, nlon in zip(self.lat_bands, self.lon_per_band):
            for k in range(nlon):
                lon = -180.0 + 360.0 * k / nlon
                centers.append((lon, lat))
        return centers

    def _erp_window(self, lon_c, lat_c, W, H):
        """视角中心 -> ERP 像素窗口 (x0, x1, y0, y1)，x 可跨界回绕。"""
        cx = (lon_c + 180.0) / 360.0 * W
        half_x = self.view_half_lon / 360.0 * W
        y0 = (90.0 - (lat_c + self.view_half_lat)) / 180.0 * H
        y1 = (90.0 - (lat_c - self.view_half_lat)) / 180.0 * H
        y0 = int(np.clip(round(y0), 0, H - 1))
        y1 = int(np.clip(round(y1), 0, H))
        x0 = int(round(cx - half_x))
        x1 = int(round(cx + half_x))
        return x0, x1, y0, y1

    @staticmethod
    def _crop_window(frame, x0, x1, y0, y1):
        """裁剪 ERP 窗口（x 回绕），返回 (win, x0有效) 与恢复函数所需偏移。"""
        H, W = frame.shape[:2]
        y0 = int(np.clip(y0, 0, H))
        y1 = int(np.clip(y1, 0, H))
        if y1 <= y0 or x1 <= x0:
            return None, 0
        cols = np.mod(np.arange(x0, x1), W)
        win = frame[y0:y1][:, cols]
        return win, x0

    def search(self, frame, erp_downscale=2):
        """多视角 × 多模板 × 多尺度全局搜索。

        参数: frame (H,W,3) uint8 ERP 帧；erp_downscale 主降采样倍率。
        返回: ((x,y,w,h), score) 或 None。
        """
        templates = self._get_templates()
        if not templates:
            return None
        H, W = frame.shape[:2]
        best = None  # (score, x, y, w, h)

        for (lon_c, lat_c) in self.view_centers():
            x0, x1, y0, y1 = self._erp_window(lon_c, lat_c, W, H)
            win, x0 = self._crop_window(frame, x0, x1, y0, y1)
            if win is None:
                continue
            wh, ww = win.shape[:2]
            sw, sh = max(8, ww // erp_downscale), max(8, wh // erp_downscale)
            fsmall = _highpass(_to_gray_f32(_resize(win, sw, sh)))

            for timg, (ow, oh) in templates:
                for tscale in self.template_scales:
                    tw = max(4, int(round(ow * sw / ww * tscale)))
                    th = max(4, int(round(oh * sh / wh * tscale)))
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
                    py_f, px_f, _ = _subpixel_peak(resp, py, px)
                    # 局部像素 -> ERP 像素（x 回绕）
                    x = (x0 + px_f * ww / sw) % W
                    y = (y0 or 0) + py_f * H / sh
                    out_w = ow * tscale
                    out_h = oh * tscale
                    if best is None or score > best[0]:
                        best = (score, x, y, out_w, out_h)

        if best is None:
            return None
        score, x, y, out_w, out_h = best
        return (float(x), float(y), float(out_w), float(out_h)), score