#!/usr/bin/env python3
"""Direct ERP Tracker：直接在全帧 ERP 上运行 VitTrack，绕过 BFoV 框架。

为避开 BFoV 切图带来的状态漂移问题而设计。
自动处理 360° 边界穿越：对输出 bbox 的 x 坐标做回绕。

生产路径复用 VitTrackONNX 的真实 onnxruntime 推理（仅 numpy/Pillow +
onnxruntime，无 cv2 依赖）；cv2.TrackerVit 仅作为本地验证 fallback。
"""
import numpy as np

from .base import BaseTracker
from .vittrack_onnx import VitTrackONNX


class DirectERPTracker(BaseTracker):
    """直接 ERP 跟踪器：在全帧 ERP 全景图上运行 VitTrack ONNX。

    不使用 BFoV 切图，避免状态预测漂移。
    自动处理 360° 子午线穿越：x 坐标由 VitTrackONNX 的 clip_box 回绕到
    [0, erp_w)。内部委托 VitTrackONNX，生产路径无 cv2 依赖。
    """

    input_space = 'erp_full'

    def __init__(self, model_path, **kwargs):
        """创建直接 ERP 跟踪器。

        参数:
            model_path: ONNX 模型文件路径（object_tracking_vittrack_2023sep.onnx）。
            **kwargs: 透传给 VitTrackONNX 的扩展参数（score_thr/backend 等）。
        """
        self.model_path = str(model_path)
        # 仅透传 VitTrackONNX 认识的参数，忽略 BFoV/pipeline 相关与无关键
        vit_kwargs = {k: v for k, v in kwargs.items()
                      if k in ('context', 'search_pad', 'score_thr', 'backend')}
        self._tracker = VitTrackONNX(model_path, **vit_kwargs)

        # 状态代理，供 pipeline 或外部调试使用
        self._cx = 0.0
        self._cy = 0.0
        self._w = 1.0
        self._h = 1.0
        self._erp_w = 0
        self._erp_h = 0

    @staticmethod
    def _ensure_uint8_rgb(img):
        """将 float32 highpass/归一化图像还原为 uint8 RGB。"""
        if img.dtype == np.uint8:
            return img
        # float32 值域可能是 [-inf, inf]；做 min-max 归一化到 [0, 255]
        img_f = img.astype(np.float64)
        mn, mx = float(img_f.min()), float(img_f.max())
        if mx - mn < 1e-6:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img_f - mn) / (mx - mn) * 255.0).astype(np.uint8)

    def init(self, image, bbox):
        """在全帧 ERP 上初始化跟踪器。

        参数:
            image: (H,W,3) uint8 ERP 全景帧。
            bbox: (x,y,w,h) ERP 坐标，允许跨界（x+w 可超 erp_w）。
        返回: None
        """
        x, y, w, h = (float(v) for v in bbox)
        rgb = self._ensure_uint8_rgb(image)
        self._tracker.init(rgb, (x, y, w, h))
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
        self._erp_h, self._erp_w = rgb.shape[:2]

        res = self._tracker.update(rgb)
        x, y, w, h = res['bbox']
        self._last_score = res['score']
        self._cx = x + w / 2.0
        self._cy = y + h / 2.0
        self._w = w
        self._h = h
        return res