# -*- coding: utf-8 -*-
"""LightFC CPU 推理封装(对齐 panotrack BaseTracker 接口)。

直接在全帧 ERP 上运行 LightFC(类似 Direct ERP 思路,绕过 BFoV 框架),
自动处理 360° 跨界:搜索区裁剪时水平回绕。

依赖:torch(CPU) + LightFC 仓库代码(通过 sys.path 注入 tools_local/lightfc)。
生产部署可后续导出 ONNX 用 onnxruntime 替换 torch。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 注入 LightFC 仓库路径(模型定义与权重加载依赖 lib 包)
_LIGHTFC_ROOT = Path(__file__).resolve().parents[2] / 'tools_local' / 'lightfc'
if str(_LIGHTFC_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIGHTFC_ROOT))

from lib.models.tracker_model import LightFC  # noqa: E402
from lib.test.utils.hann import hann2d  # noqa: E402

from .base import BaseTracker  # noqa: E402

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class LightFCTracker(BaseTracker):
    """LightFC 全帧 CPU 跟踪器(ERP 输入,跨界回绕)。

    init(image, bbox): image 为 (H,W,3) uint8 RGB ERP 全帧;bbox=(x,y,w,h)。
    update(image) -> {'bbox','score','psr','apce'}。
    """

    def __init__(self, model_path, search_size=256, search_factor=4.0,
                 template_size=128, template_factor=2.0, **kwargs):
        """创建 LightFC 跟踪器(CPU)。

        参数: model_path —— lightfc_ep0400.pth.tar 权重路径;
              search_size/search_factor —— 搜索区尺寸与裁剪倍率;
              template_size/template_factor —— 模板尺寸与裁剪倍率。
        返回: None
        """
        del kwargs
        self.model_path = str(model_path)
        self.search_size = int(search_size)
        self.search_factor = float(search_factor)
        self.template_size = int(template_size)
        self.template_factor = float(template_factor)

        # 构造最小化配置(仅模型所需字段)
        from easydict import EasyDict as edict
        cfg = edict()
        cfg.MODEL = edict()
        cfg.MODEL.BACKBONE = edict(TYPE='MobileNetV2', STRIDE=16, CHANNEL=96,
                                   USE_PRETRAINED=False, PRETRAIN_FILE='',
                                   LOAD_MODE=0)
        cfg.MODEL.FUSION = edict(TYPE='pwcorr_se_scf_sc_iab_sc_concat',
                                 CHANNEL=96,
                                 PARAMS=edict(num_kernel=64, adj_channel=96))
        cfg.MODEL.HEAD = edict(TYPE='repn33_se_center_concat', CHANNEL=96,
                               PARAMS=edict(inplanes=192, channel=256,
                                            feat_sz=self.search_size // 16,
                                            stride=16, freeze_bn=False))
        cfg.MODEL.NECK = edict(USE_NECK=False)
        cfg.TEST = edict(SEARCH_SIZE=self.search_size,
                         SEARCH_FACTOR=self.search_factor,
                         TEMPLATE_SIZE=self.template_size,
                         TEMPLATE_FACTOR=self.template_factor)
        self.cfg = cfg

        # 加载网络
        self.network = LightFC(cfg=cfg, env_num=None, training=False)
        state = torch.load(self.model_path, map_location='cpu', weights_only=False)
        self.network.load_state_dict(state['net'], strict=True)
        for module in self.network.backbone.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        for module in self.network.head.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        self.network.eval()

        self.feat_sz = cfg.MODEL.HEAD.PARAMS.feat_sz
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(),
                                    centered=True)
        self.state = None
        self.z_feat = None
        self._last_score = 1.0

    # ------------------------------------------------------------ 内部工具

    def _sample_target(self, im, target_bb, factor, output_sz):
        """按官方 sample_target 逻辑裁剪方形搜索区(私有,纯 numpy/PIL)。

        水平方向按 ERP 回绕拼接处理跨界;垂直方向 clamp + 常量填充。
        返回: (crop, resize_factor) —— crop 为 (output_sz, output_sz, 3) uint8。
        """
        H, W = im.shape[:2]
        x, y, w, h = (float(v) for v in target_bb)
        crop_sz = int(np.ceil(np.sqrt(w * h) * factor))
        crop_sz = max(crop_sz, 2)
        cx, cy = x + 0.5 * w, y + 0.5 * h

        # 水平:以 cx 为中心,从 ERP 环形取宽 crop_sz 的条带(跨界回绕)
        half = crop_sz // 2
        # 源列索引(连续,可跨界)
        cols = np.mod(np.arange(cx - half, cx - half + crop_sz), W).astype(np.int64)
        rows = np.arange(cy - half, cy - half + crop_sz)
        # 垂直 clamp
        rows_c = np.clip(rows, 0, H - 1).astype(np.int64)
        # 边界填充标志:垂直越界部分填 0(与官方 copyMakeBorder 常量填充一致)
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
        """uint8 RGB 裁剪图 -> (1,3,H,W) float32 归一化张量(私有)。"""
        arr = patch.astype(np.float32).transpose(2, 0, 1)[None]
        arr = (arr / 255.0 - _MEAN) / _STD
        return torch.from_numpy(arr.astype(np.float32))

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
        """边界钳制(私有):x 回绕到 [0,W),y clamp;尺寸不小于 margin。"""
        x, y, w, h = (float(v) for v in box)
        w = max(w, margin)
        h = max(h, margin)
        x = x % W
        y = float(np.clip(y, 0, max(0.0, H - h)))
        return [x, y, w, h]

    # ------------------------------------------------------------ 契约接口

    def init(self, image, bbox):
        """用首帧 ERP 与目标框初始化。

        参数: image (H,W,3) uint8 RGB ERP 全帧;bbox (x,y,w,h) ERP 坐标(可跨界)。
        返回: None
        """
        image = np.asarray(image)
        self._erp_h, self._erp_w = image.shape[:2]
        self.state = [float(v) for v in bbox]
        z_patch, _ = self._sample_target(image, self.state, self.template_factor,
                                         self.template_size)
        z_t = self._preprocess(z_patch)
        with torch.no_grad():
            self.z_feat = self.network.forward_backbone(z_t)

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

        with torch.no_grad():
            out = self.network.forward_tracking(z_feat=self.z_feat, x=x_t)

        response = self.output_window * out['score_map']
        pred_box = self._compute_box(response, out, resize_factor)
        pred_box = self._map_box_back(pred_box, resize_factor)
        self.state = self._clip_box(pred_box, H, W, margin=2)

        # 置信度代理:score_map 峰值 -> score, PSR/APCE 代理
        score = float(out['score_map'].max())
        self._last_score = max(0.0, min(1.0, score))
        psr = max(0.0, (self._last_score - 0.3) * 20.0)
        apce = self._last_score * self._last_score

        return {'bbox': tuple(self.state), 'score': self._last_score,
                'psr': float(psr), 'apce': float(apce)}

    def _compute_box(self, response, out_dict, resize_factor):
        """由响应图解码局部框(私有,官方 compute_box 逻辑)。"""
        pred_boxes = self.network.head.cal_bbox(
            response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        pred_boxes = (pred_boxes.mean(dim=0) * self.search_size / resize_factor)
        return pred_boxes.tolist()
