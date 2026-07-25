#!/usr/bin/env python3
"""Direct ERP Tracker：直接在全帧 ERP 上运行 VitTrack，绕过 BFoV 框架。

为避开 BFoV 切图带来的状态漂移问题而设计。
自动处理 360° 边界穿越：对输出 bbox 的 x 坐标做回绕。
"""
import numpy as np
import cv2
from .base import BaseTracker


class DirectERPTracker(BaseTracker):
    """直接 ERP 跟踪器：在全帧 ERP 全景图上运行 cv2.TrackerVit。

    不使用 BFoV 切图，避免状态预测漂移。
    自动处理 360° 子午线穿越：x 坐标自动回绕到 [0, erp_w)。
    """

    def __init__(self, model_path, **kwargs):
        """创建直接 ERP 跟踪器。

        参数:
            model_path: ONNX 模型文件路径（cv2.TrackerVit 可直接加载）。
            **kwargs: 预留扩展参数，当前未使用。
        """
        self.model_path = str(model_path)
        try:
            params = cv2.TrackerVit_Params()
            params.net = self.model_path
            params.backend = cv2.dnn.DNN_BACKEND_OPENCV
            params.target = cv2.dnn.DNN_TARGET_CPU
            self._model = cv2.TrackerVit_create(params)
        except Exception as e:
            raise RuntimeError(f"无法创建 cv2.TrackerVit: {e}") from e

        # 状态代理，供 pipeline 或外部调试使用
        self._cx = 0.0
        self._cy = 0.0
        self._w = 1.0
        self._h = 1.0
        self._erp_w = 0
        self._erp_h = 0

    @staticmethod
    def _ensure_uint8_rgb(img):
        """将 float32 highpass 图像还原为 uint8 RGB。"""
        if img.dtype == np.uint8:
            return img
        # float32 值域可能是 [-inf, inf]；做 min-max 归一化到 [0, 255]
        img_f = img.astype(np.float64)
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx - mn < 1e-6:
            return np.zeros_like(img_f, dtype=np.uint8)
        return ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)

    def init(self, image, bbox):
        """在全帧 ERP 上初始化跟踪器。

        参数:
            image: (H,W,3) uint8 ERP 全景帧。
            bbox: (x,y,w,h) ERP 坐标，允许跨界（x+w 可超 erp_w）。
        """
        x, y, w, h = (float(v) for v in bbox)
        rgb = self._ensure_uint8_rgb(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # cv2 要求 int32 且不能越界
        int_x = int(round(np.clip(x, 0, image.shape[1] - 1)))
        int_y = int(round(np.clip(y, 0, image.shape[0] - 1)))
        int_w = int(round(max(2.0, w)))
        int_h = int(round(max(2.0, h)))

        self._model.init(bgr, (int_x, int_y, int_w, int_h))
        self._last_score = 1.0
        self._cx = x + w / 2.0
        self._cy = y + h / 2.0
        self._w = w
        self._h = h
        self._erp_w = image.shape[1]
        self._erp_h = image.shape[0]

    def update(self, image):
        """在全帧 ERP 上更新跟踪器。

        返回:
            dict: {'bbox': (x,y,w,h), 'score': float, 'psr': float, 'apce': float}
            bbox 会自动处理 360° 边界穿越（x 回绕到 [0, erp_w)）。
        """
        rgb = self._ensure_uint8_rgb(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, bbox = self._model.update(bgr)
        score = float(self._model.getTrackingScore())

        x, y, w, h = (float(v) for v in bbox)

        # 处理 360° 边界穿越：x 回绕到 [0, erp_w)
        if x < 0:
            x += self._erp_w
        elif x >= self._erp_w:
            x -= self._erp_w

        # 尺寸与位置钳制，防止非法值
        w = max(2.0, w)
        h = max(2.0, h)
        x = float(np.clip(x, 0, self._erp_w - w))
        y = float(np.clip(y, 0, self._erp_h - h))

        self._last_score = score
        self._cx = x + w / 2.0
        self._cy = y + h / 2.0
        self._w = w
        self._h = h

        # PSR / APCE 代理指标（与 pipeline 契约一致）
        psr = max(0.0, (score - 0.3) * 20.0)
        apce = score * score

        return {
            'bbox': (x, y, w, h),
            'score': float(np.clip(score, 0.0, 1.0)),
            'psr': float(psr),
            'apce': float(apce)
        }
