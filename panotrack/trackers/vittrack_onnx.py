# -*- coding: utf-8 -*-
"""vittrack_onnx：ONNXRuntime/CPU 包装为 panotrack BaseTracker（模块 B 契约）。

生产部署用这个实现（只依赖 numpy/Pillow/scipy + onnxruntime）。
本地 Windows 开发时如果 onnxruntime DLL 加载失败，fallback 到 cv2.TrackerVit 验证。

接口对齐 BaseTracker：init(image, bbox) / update(image) -> dict{bbox,score,psr,apce}
切图坐标系下运行，不感知 ERP 全景。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from .base import BaseTracker


class VitTrackONNX(BaseTracker):
    """ONNXRuntime 实现的 VitTrack 跟踪器。

    使用方式:
        tracker = VitTrackONNX(model_path)
        tracker.init(patch_img, bbox)   # patch_img: (H,W,3) uint8 RGB
        result = tracker.update(patch_img)
    """

    # --------------------------------------------------------------------- init
    def __init__(self, model_path, context=0.5, search_pad=1.5,
                 score_thr=0.1, backend=None):
        """创建 vittrack(onnxruntime) 跟踪器实例。

        参数:
            model_path: ONNX 模型文件路径(字符串或 Path)。
            context: 模板上下文比例(预留,内部未直接使用)。
            search_pad: 搜索图扩展比例(预留,内部未直接使用)。
            score_thr: 丢失判定分数阈值(低于此值标记为 lost)。
            backend: 'cpu'/'cuda'(None 时自动选 CPU)。
        """
        self.model_path = str(model_path)
        self.context = float(context)
        self.search_pad = float(search_pad)
        self.score_thr = float(score_thr)
        self.backend = backend or 'cpu'

        self._sess = None
        self._last_result = None
        self._init_ok = False
        self._try_load()

    # ------------------------------------------------------------------- load
    def _try_load(self):
        """尝试加载 ONNX 模型(优先 onnxruntime,失败则 fallback 到 cv2)。"""
        loaded = False

        # 路径: ONNX Runtime
        if not loaded:
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if self.backend == 'cuda':
                    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                sess = ort.InferenceSession(
                    self.model_path, providers=providers
                )
                self._sess = sess
                self._onnx = True
                loaded = True
            except Exception as e:
                print(f'[VitTrackONNX] onnxruntime load failed: {e}', file=sys.stderr)

        # Fallback: OpenCV TrackerVit(cv2, 仅本地验证用)
        if not loaded:
            try:
                import cv2
                params = cv2.TrackerVit_Params()
                params.net = self.model_path
                params.backend = cv2.dnn.DNN_BACKEND_OPENCV
                params.target = cv2.dnn.DNN_TARGET_CPU
                self._cv_model = cv2.TrackerVit_create(params)
                self._onnx = False
                loaded = True
            except Exception as e:
                print(f'[VitTrackONNX] cv2 fallback failed: {e}', file=sys.stderr)

        if not loaded:
            raise RuntimeError(
                "VitTrackONNX: 无法加载模型(onnxruntime 和 cv2 均失败)"
            )

    # --------------------------------------------------------------------- init
    def init(self, image, bbox):
        """用首帧局部透视图与目标框初始化跟踪器。

        参数: image (H,W,3) uint8 RGB 局部透视图; bbox (x,y,w,h) 局部像素框。
        返回: None
        """
        x, y, w, h = (float(v) for v in bbox)
        if self._onnx:
            # ONNX: 预分配状态,init 时不做推理
            self._init_box = (x, y, max(2.0, w), max(2.0, h))
            self._curr_bbox = list(self._init_box)
        else:
            # cv2: 需要 int32 ndarray,且要求 BGR
            bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
            init_box = np.array([x, y, max(2.0, w), max(2.0, h)], dtype=np.int32)
            self._cv_model.init(bgr, init_box)
            self._init_box = (x, y, max(2.0, w), max(2.0, h))
            self._curr_bbox = [x, y, max(2.0, w), max(2.0, h)]
        self._init_ok = True

    # ------------------------------------------------------------------ update
    def update(self, image):
        """在新一帧局部透视图上更新目标状态。

        参数: image (H,W,3) uint8 RGB 局部透视图。
        返回: dict {'bbox': (x,y,w,h), 'score': float∈[0,1],
                    'psr': float, 'apce': float}
        """
        if not self._init_ok:
            raise RuntimeError("VitTrackONNX 未初始化")

        if self._onnx:
            return self._onnx_update(image)
        return self._cv_update(image)

    # ---------------------------------------------------------------- onnx path
    def _onnx_update(self, image):
        """ONNXRuntime 推理路径(骨架代码)。实际推理需根据模型输入/输出形状调整。"""
        # TODO: 实现 ONNX 前向传播
        # 模型输入: (1,3,H,W) 或 (1,C,H,W) 灰度
        # 模型输出: (1,4) bbox + (1,1) score
        # 此处暂返回初始跟踪状态(占位)
        bx, by, bw, bh = [float(v) for v in self._curr_bbox]
        score = 0.95  # 占位
        psr = max(0.0, (score - 0.3) * 20.0)
        apce = score * score
        return {'bbox': (bx, by, bw, bh),
                'score': float(np.clip(score, 0.0, 1.0)),
                'psr': float(psr), 'apce': float(apce)}

    # ------------------------------------------------------------------- cv2 path
    def _cv_update(self, image):
        """OpenCV cv2.TrackerVit 推理路径(本地验证用)。"""
        global cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        ok, bbox = self._cv_model.update(bgr)
        score = float(self._cv_model.getTrackingScore())
        x, y, w, h = (float(v) for v in bbox)
        if not ok or w < 1.0 or h < 1.0:
            # 丢失:保持上一帧
            x, y, w, h = self._curr_bbox[:4]
        self._curr_bbox = [x, y, max(2.0, w), max(2.0, h)]
        psr = max(0.0, (score - 0.3) * 20.0)
        apce = score * score
        return {'bbox': (x, y, w, h),
                'score': float(np.clip(score, 0.0, 1.0)),
                'psr': float(psr), 'apce': float(apce)}
