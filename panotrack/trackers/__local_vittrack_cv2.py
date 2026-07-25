# -*- coding: utf-8 -*-
"""vittrack_cv2：OpenCV 5 cv2.TrackerVit_create 包装为 panotrack BaseTracker。

仅用于**本地 Windows 实证验证**(用 cv2 推理 ONNX),生产部署请用
panotrack/trackers/vittrack_onnx.py(纯 numpy + onnxruntime)。

关键修复: pipeline 对切图做 _highpass() → float32,均值≈0。
VitTrack DNN 需要 uint8 [0,255] RGB。所以必须在 init/update 前将
float32 highpass 图像还原为 uint8 RGB。
"""
from __future__ import annotations

import numpy as np
import cv2

from .base import BaseTracker


class VitTrackCV2Tracker(BaseTracker):
    """cv2.TrackerVit 包装器,兼容 PanoTracker 的高通输入。"""

    def __init__(self, model_path, **kwargs):
        self.model_path = str(model_path)
        try:
            params = cv2.TrackerVit_Params()
            params.net = self.model_path
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            self._model = cv2.TrackerVit_create(params)
        except Exception as e:
            raise RuntimeError(f"无法创建 cv2.TrackerVit: {e}") from e
        # 状态代理供 pipeline._migrate_tracker 使用
        self._cx = 0.0; self._cy = 0.0; self._w = 1.0; self._h = 1.0
        self._last_bbox = (0, 0, 1, 1)

    @staticmethod
    def _ensure_uint8_rgb(img):
        """将可能是 float32 highpass 的图像还原为 uint8 RGB。"""
        if img.dtype == np.uint8:
            return img
        # float32, 值域可能是 [-inf, inf]; 简单做法: min-max 归一化到 [0,255]
        img_f = img.astype(np.float64)
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx - mn < 1e-6:
            return np.zeros_like(img_f, dtype=np.uint8)
        out = ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return out

    def init(self, image, bbox):
        x, y, w, h = (float(v) for v in bbox)
        rgb = self._ensure_uint8_rgb(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        int_x, int_y, int_w, int_h = (int(round(max(2.0, v))) for v in [x, y, w, h])
        self._init_box = (int_x, int_y, int_w, int_h)
        self._model.init(bgr, self._init_box)
        self._last_score = 1.0
        self._cx = x + w / 2.0; self._cy = y + h / 2.0
        self._w = w; self._h = h
        self._last_bbox = (x, y, w, h)

    def update(self, image):
        rgb = self._ensure_uint8_rgb(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, bbox = self._model.update(bgr)
        score = float(self._model.getTrackingScore())
        self._last_score = score
        x, y, w, h = (float(v) for v in bbox)
        if not ok or w <= 1.0 or h <= 1.0:
            return {'bbox': (x, y, max(2.0, w), max(2.0, h)),
                    'score': 0.0, 'psr': 0.0, 'apce': 0.0}
        self._last_bbox = (x, y, w, h)
        self._cx = x + w / 2.0; self._cy = y + h / 2.0
        self._w = w; self._h = h
        psr = max(0.0, (score - 0.3) * 20.0)
        apce = score * score
        return {'bbox': (x, y, w, h),
                'score': float(np.clip(score, 0.0, 1.0)),
                'psr': float(psr), 'apce': float(apce)}
