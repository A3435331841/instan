# -*- coding: utf-8 -*-
"""LightFC ONNXRuntime 推理封装(对齐 panotrack BaseTracker 接口)。

用双子图 ONNX(backbone + tracking)做真实推理,不依赖 torch:
  - init: backbone(z) -> z_feat(模板特征缓存,只算一次)
  - update: tracking(z_feat, x) -> score/size/offset,纯 numpy 后处理

满足 Docker 断网自包含 + 仅 numpy/Pillow/scipy+onnxruntime 约束。
输入为全帧 ERP(等距柱状),自动处理 360° 跨界(搜索区裁剪水平回绕)。
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .base import BaseTracker

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def _hann1d(n, centered=True):
    """1D 余弦窗(numpy 实现,私有)。"""
    if centered:
        step = 0.5 / (n - 1) if n > 1 else 0.5
        axis = np.arange(n)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * (axis * step - 0.5))
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))


def _hann2d(sz, centered=True):
    """2D 余弦窗(numpy 实现,私有),形状 (1,1,H,W)。"""
    h, w = int(sz[0]), int(sz[1])
    return (_hann1d(h, centered).reshape(-1, 1)
            * _hann1d(w, centered).reshape(1, -1))[None, None].astype(np.float32)


class LightFCONNX(BaseTracker):
    """LightFC ONNXRuntime 跟踪器(ERP 全帧输入,跨界回绕)。

    init(image, bbox): image (H,W,3) uint8 RGB ERP;bbox=(x,y,w,h) 可跨界。
    update(image) -> {'bbox','score','psr','apce'}。
    """

    input_space = 'erp_full'

    def __init__(self, backbone_path, tracking_path, search_size=256,
                 search_factor=4.0, template_size=128, template_factor=2.0,
                 backend='cpu', **kwargs):
        """创建 LightFC ONNX 跟踪器。

        参数: backbone_path —— lightfc_backbone.onnx;tracking_path ——
              lightfc_tracking.onnx;search_size/search_factor —— 搜索区尺寸与
              裁剪倍率;template_size/template_factor —— 模板尺寸与裁剪倍率;
              backend —— 'cpu' 或 'cuda'(onnxruntime provider)。
        返回: None
        """
        del kwargs
        self.backbone_path = str(backbone_path)
        self.tracking_path = str(tracking_path)
        self.search_size = int(search_size)
        self.search_factor = float(search_factor)
        self.template_size = int(template_size)
        self.template_factor = float(template_factor)
        self.feat_sz = self.search_size // 16
        self.output_window = _hann2d(np.array([self.feat_sz, self.feat_sz]),
                                     centered=True)
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if str(backend).lower() == 'cuda'
                     else ['CPUExecutionProvider'])
        self._sess_b = ort.InferenceSession(self.backbone_path,
                                            providers=providers)
        self._sess_t = ort.InferenceSession(self.tracking_path,
                                            providers=providers)
        self.state = None
        self.z_feat = None
        self._last_score = 1.0
        # 状态代理,供 pipeline 或外部调试使用(与 DirectERPTracker 一致)
        self._cx = 0.0
        self._cy = 0.0
        self._w = 1.0
        self._h = 1.0
        self._erp_w = 0
        self._erp_h = 0

    # ------------------------------------------------------------ 内部工具

    def _sample_target(self, im, target_bb, factor, output_sz):
        """按官方 sample_target 逻辑裁剪方形搜索区(私有,水平回绕)。

        返回: (crop, resize_factor) —— crop 为 (output_sz, output_sz, 3) uint8。
        """
        H, W = im.shape[:2]
        x, y, w, h = (float(v) for v in target_bb)
        crop_sz = int(np.ceil(np.sqrt(w * h) * factor))
        crop_sz = max(crop_sz, 2)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        half = crop_sz // 2
        # 用整数起点构造 arange,避免浮点误差导致长度多1(cx/cy 为浮点时)
        start_col = int(np.round(cx - half))
        start_row = int(np.round(cy - half))
        cols = np.mod(np.arange(start_col, start_col + crop_sz), W).astype(np.int64)
        rows = np.arange(start_row, start_row + crop_sz)
        rows_c = np.clip(rows, 0, H - 1).astype(np.int64)
        out = np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
        valid = (rows >= 0) & (rows < H)
        out[valid] = im[rows_c[valid]][:, cols]
        if output_sz is not None and output_sz != crop_sz:
            resize_factor = output_sz / crop_sz
            from PIL import Image
            out = np.asarray(Image.fromarray(out).resize(
                (int(output_sz), int(output_sz)), Image.BILINEAR))
        else:
            resize_factor = 1.0
        return out, resize_factor

    def _preprocess(self, patch):
        """uint8 RGB 裁剪图 -> (1,3,H,W) float32 归一化数组(私有)。"""
        arr = patch.astype(np.float32).transpose(2, 0, 1)[None]
        return (arr / 255.0 - _MEAN) / _STD

    def _cal_bbox(self, score, size, offset):
        """由响应图解码局部框(numpy 版,私有,对应官方 head.cal_bbox)。

        score: (1,1,fs,fs) 加窗后响应;size: (1,2,fs,fs);offset: (1,2,fs,fs)。
        返回: np.array([cx, cy, w, h]) 归一化坐标(cx/cy 在 [0,1] 尺度)。
        """
        fs = self.feat_sz
        s = score[0, 0]
        idx = int(np.argmax(s))
        iy, ix = divmod(idx, fs)
        sz = size[0, :, iy, ix]        # (2,) w,h
        off = offset[0, :, iy, ix]     # (2,) dx,dy
        cx = (ix + off[0]) / fs
        cy = (iy + off[1]) / fs
        return np.array([cx, cy, sz[0], sz[1]], dtype=np.float64)

    def _map_box_back(self, pred_box, resize_factor):
        """局部预测框映射回 ERP 坐标(私有,官方 map_box_back 逻辑)。"""
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def _clip_box(self, box, H, W, margin=2):
        """边界钳制(私有):x 回绕 [0,W),y clamp,尺寸不小于 margin。"""
        x, y, w, h = (float(v) for v in box)
        w = max(w, margin)
        h = max(h, margin)
        x = x % W
        y = float(np.clip(y, 0, max(0.0, H - h)))
        return [x, y, w, h]

    # ------------------------------------------------------------ 契约接口

    def init(self, image, bbox):
        """用首帧 ERP 与目标框初始化。

        参数: image (H,W,3) uint8 RGB ERP 全帧;bbox (x,y,w,h) 可跨界。
        返回: None
        """
        image = np.asarray(image)
        self._erp_h, self._erp_w = image.shape[:2]
        self.state = [float(v) for v in bbox]
        self._cx = self.state[0] + 0.5 * self.state[2]
        self._cy = self.state[1] + 0.5 * self.state[3]
        self._w = self.state[2]
        self._h = self.state[3]
        z_patch, _ = self._sample_target(image, self.state, self.template_factor,
                                         self.template_size)
        z_t = self._preprocess(z_patch)
        # 模板特征缓存:backbone 只跑一次
        self.z_feat = self._sess_b.run(None, {'z': z_t})[0]

    def update(self, image):
        """在新帧 ERP 上更新目标状态。

        参数: image (H,W,3) uint8 RGB ERP 全帧。
        返回: dict {'bbox': (x,y,w,h) ERP 坐标(跨界约定), 'score': float,
                    'psr': float, 'apce': float}
        """
        image = np.asarray(image)
        H, W = image.shape[:2]
        x_patch, resize_factor = self._sample_target(
            image, self.state, self.search_factor, self.search_size)
        x_t = self._preprocess(x_patch)

        score_map, size_map, offset_map = self._sess_t.run(
            None, {'z_feat': self.z_feat, 'x': x_t})

        response = self.output_window * score_map
        pred_box = self._cal_bbox(response, size_map, offset_map)
        # 与官方 compute_box 一致:解码坐标乘 search_size/resize_factor 还原像素
        pred_box = pred_box * self.search_size / resize_factor
        pred_box = self._map_box_back(pred_box, resize_factor)
        self.state = self._clip_box(pred_box, H, W, margin=2)
        self._cx = self.state[0] + 0.5 * self.state[2]
        self._cy = self.state[1] + 0.5 * self.state[3]
        self._w = self.state[2]
        self._h = self.state[3]

        score = float(score_map.max())
        self._last_score = max(0.0, min(1.0, score))
        psr = max(0.0, (self._last_score - 0.3) * 20.0)
        apce = self._last_score * self._last_score

        return {'bbox': tuple(self.state), 'score': self._last_score,
                'psr': float(psr), 'apce': float(apce)}
