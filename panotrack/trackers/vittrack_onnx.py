# -*- coding: utf-8 -*-
"""vittrack_onnx：ONNXRuntime/CPU 包装为 panotrack BaseTracker（模块 B 契约）。

生产部署用这个实现（只依赖 numpy/Pillow/scipy + onnxruntime）。
本地 Windows 开发时如果 onnxruntime DLL 加载失败，fallback 到 cv2.TrackerVit 验证。

接口对齐 BaseTracker：init(image, bbox) / update(image) -> dict{bbox,score,psr,apce}
切图坐标系下运行，不感知 ERP 全景。

模型结构（object_tracking_vittrack_2023sep.onnx）：
  输入 template [1,3,128,128] + search [1,3,256,256]
  输出 output1 [1,1,16,16]（score）、output2 [1,2,16,16]（size）、
       output3 [1,2,16,16]（offset）
  Onnx 路径在 init 时裁剪并缓存 template 裁剪图，update 时对 template + search
  一起前向推理（与 OpenCV TrackerVit 的 constant-pad 裁剪不同，这里按 ERP
  水平回绕裁剪，与 lightfc_onnx 一致，天然支持 360° 子午线穿越）。
"""
from __future__ import annotations

import sys

import numpy as np

from .base import BaseTracker
from ._siam_common import (sample_target, preprocess, cal_bbox,
                           map_box_back, clip_box, hann2d)

# VitTrack 模型固有尺寸（与 OpenCV TrackerVit 默认一致）
_TEMPLATE_SIZE = 128   # template 输入分辨率
_SEARCH_SIZE = 256     # search 输入分辨率
_TEMPLATE_FACTOR = 2.0
_SEARCH_FACTOR = 4.0
_FEAT_SZ = 16          # 特征图分辨率（search / 16）


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
        self._last_score = 1.0
        self._init_ok = False
        self.state = None
        self.feat_sz = _FEAT_SZ
        self.output_window = hann2d(np.array([_FEAT_SZ, _FEAT_SZ]), centered=True)
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
                self._sess = ort.InferenceSession(
                    self.model_path, providers=providers
                )
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
        self.state = [x, y, max(2.0, w), max(2.0, h)]
        if self._onnx:
            # 缓存模板裁剪图(模型每次推理都需要 template 输入)
            t_patch, _ = sample_target(image, self.state,
                                       _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
            self._template_blob = preprocess(t_patch)
        else:
            # cv2: 需要 int32 ndarray,且要求 BGR
            import cv2
            bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
            init_box = np.array(self.state, dtype=np.int32)
            self._cv_model.init(bgr, init_box)
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
        """ONNXRuntime 推理路径: template + search 一起前向,deck 响应图。"""
        image = np.asarray(image)
        H, W = image.shape[:2]
        x_patch, resize_factor = sample_target(
            image, self.state, _SEARCH_FACTOR, _SEARCH_SIZE)
        x_blob = preprocess(x_patch)

        outs = self._sess.run(None, {'template': self._template_blob,
                                     'search': x_blob})
        score_map, size_map, offset_map = outs[0], outs[1], outs[2]

        response = self.output_window * score_map
        pred_box = cal_bbox(response, size_map, offset_map, self.feat_sz)
        # 与官方 compute_box 一致:解码坐标乘 search_size/resize_factor 还原像素
        pred_box = pred_box * _SEARCH_SIZE / resize_factor
        pred_box = map_box_back(pred_box, self.state, _SEARCH_SIZE, resize_factor)
        self.state = clip_box(pred_box, H, W, margin=2)

        score = float(score_map.max())
        self._last_score = max(0.0, min(1.0, score))
        psr = max(0.0, (self._last_score - 0.3) * 20.0)
        apce = self._last_score * self._last_score

        return {'bbox': tuple(float(v) for v in self.state),
                'score': self._last_score,
                'psr': float(psr), 'apce': float(apce)}

    # ------------------------------------------------------------------- cv2 path
    def _cv_update(self, image):
        """OpenCV cv2.TrackerVit 推理路径(本地验证用)。"""
        import cv2
        bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        ok, bbox = self._cv_model.update(bgr)
        score = float(self._cv_model.getTrackingScore())
        x, y, w, h = (float(v) for v in bbox)
        if not ok or w < 1.0 or h < 1.0:
            # 丢失:保持上一帧
            x, y, w, h = self.state[:4]
        self.state = [x, y, max(2.0, w), max(2.0, h)]
        psr = max(0.0, (score - 0.3) * 20.0)
        apce = score * score
        return {'bbox': tuple(self.state),
                'score': float(np.clip(score, 0.0, 1.0)),
                'psr': float(psr), 'apce': float(apce)}