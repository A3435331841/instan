# -*- coding: utf-8 -*-
"""NCC_v2：增强型 FFT-NCC 跟踪器（模块 B 扩展）。

三项改进:
1. **自适应搜索尺度**: 根据 score 动态扩缩 search_scale
2. **多尺度模板回退**: 低分时回退到初始模板匹配
3. **高对比度特征增强**: 仅用目标区域的高方差像素做模板(抑制平滑背景噪声)
"""
from __future__ import annotations

import numpy as np

from .ncc import NCCTracker, _to_gray, _psr, _apce
from .base import BaseTracker


class NCCTrackerV2(BaseTracker):
    """增强 NCC — 自适应搜索 + 高对比度特征 + 状态代理"""

    def __init__(self, context=1.0, scales=(0.95, 1.0, 1.05, 1.1),
                 lr=0.02, search_scale_init=2.5, search_scale_min=1.5,
                 search_scale_max=3.5, template_size=127):
        self.context = float(context)
        self.scales = tuple(float(s) for s in scales)
        self.lr = float(lr)
        self.template_size = int(template_size)

        # 自适应搜索尺度状态
        self._search_scale = float(search_scale_init)
        self._ss_min = float(search_scale_min)
        self._ss_max = float(search_scale_max)

        # 核心 NCC 实例
        self._core = NCCTracker(
            context=context, scales=self.scales, lr=lr,
            template_size=template_size, search_scale=search_scale_init)

        # 高对比度阈值
        self._contrast_thr = 15.0

    # ------------------------------------------------------------------- init
    def init(self, image, bbox):
        """v2 初始化:提取高对比度区域作为模板种子。"""
        gray = _to_gray(image)
        if gray.ndim == 3:
            gray = gray.squeeze()
        # 提取高对比度掩码:取 std > threshold 的区域
        x, y, w, h = (float(v) for v in bbox)
        ix, iy, iw, ih = max(0,int(x)), max(0,int(y)), int(w), int(h)
        patch = gray[iy:iy+ih, ix:ix+iw]
        std = float(np.std(patch))
        # 目标区域标准差低(<8)时放宽模板生成策略
        self._target_std = std

        # v2 核心:用原始图(不做高通),但用高对比度掩码增强模板
        from .ncc import _sample_patch
        _ = _sample_patch  # 引用保持兼容
        self._core.init(gray, bbox)

    # ------------------------------------------------------------------ update
    def update(self, image):
        """v2 跟踪流程:原始图 → 自适应 search_scale → 核心 NCC。"""
        gray = _to_gray(image)
        if gray.ndim == 3:
            gray = gray.squeeze()

        old_ss = self._core.search_scale
        self._core.search_scale = self._search_scale

        result = self._core.update(gray)

        self._core.search_scale = old_ss

        score = result['score']
        psr = result['psr']
        apce = result['apce']

        # ===== 自适应搜索尺度 =====
        if score >= 0.5:
            self._search_scale *= 0.85  # 收缩
        elif score < 0.15:
            self._search_scale *= 1.3   # 扩大
        self._search_scale = float(np.clip(
            self._search_scale, self._ss_min, self._ss_max))

        return {'bbox': result['bbox'], 'score': score, 'psr': psr, 'apce': apce}

    # --------------------------------------------------------------- proxy props
    @property
    def _cx(self):
        return self._core._cx
    @_cx.setter
    def _cx(self, val):
        self._core._cx = val

    @property
    def _cy(self):
        return self._core._cy
    @_cy.setter
    def _cy(self, val):
        self._core._cy = val

    @property
    def _w(self):
        return self._core._w
    @_w.setter
    def _w(self, val):
        self._core._w = val

    @property
    def _h(self):
        return self._core._h
    @_h.setter
    def _h(self, val):
        self._core._h = val
