#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方训练集评测 runner（P0-A，2026-08-24）。

对官方 130 条序列（train_real 47 + train_sim 83，1440×720，BFoV GT）做 OPE 评测：
  video.mp4 + init.txt -> tracker 逐帧推理 -> ERP 框 -> 双口径 IoU 评分 -> summary.csv。

官方数据特性（实测）：
  - GT 含 0,0,0,0 丢失帧（全库 11.06%，40/130 条序列）：**评分时跳过**这些帧
    （GT 无效帧不参与 SR/AUC；首帧初始化帧同样不计），但统计进 metrics 供分析；
  - 行号 = 帧号，与 Arena 协议一致。

tracker 后端（--tracker）：
  - mock     : 冻结初始框（管道联通用，不是真跟踪器）
  - gt_echo  : 诊断后端，回显 GT+微扰（验证评分链路，输出 AUC 应≈1；禁止用于上报成绩）
  - odtrack  : 真实 ODTrack 三平铺（arena_protocol 同款加载方式，需 GPU/torch）
  - odtrack_recapture: ODTrack + 可靠性门控 + 球面重捕获（实验冲分后端）
  - lightfc_onnx: LightFC ONNX full-frame ERP tracker（高速单方案验证）
  - direct_erp: VitTrack ONNX full-frame ERP tracker（直接全景图单方案验证）
  - uetrack  : UETrack ERP-wrap full-frame tracker（单方案验证）
  - lorat    : LoRAT DINOv2-B 三平铺 ERP tracker（单方案验证）

用法示例：
  python scripts/eval_official.py --tracker mock --seqs train_real/seq_0001 --max-frames 80
  python scripts/eval_official.py --tracker gt_echo --split valid
  python scripts/eval_official.py --tracker odtrack --split all \
      --odtrack-workspace /opt/odtrack --odtrack-ckpt /opt/models/ODTrack_ep0300.pth.tar
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov  # noqa: E402
from panotrack.evaluation.metrics import auc, dual_iou, iou_xywh, success_rate  # noqa: E402

try:
    import cv2 as cv
except ImportError:
    cv = None

SPLIT_DIR = PROJECT_ROOT / "data360" / "official_split"


# ---------------------------------------------------------------------------
# GT 解析（含丢失帧掩码）
# ---------------------------------------------------------------------------

def load_gt(path: Path):
    """读 GT：返回 (bfov 数组 (N,4), valid 掩码 (N,))。0,0,0,0 -> invalid。"""
    rows, valid = [], []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        f = [float(v) for v in ln.replace(",", " ").split()]
        if len(f) != 4:
            raise ValueError(f"{path}: 非法行 {ln!r}")
        rows.append(f)
        valid.append(f[2] > 0.0 and f[3] > 0.0)
    return np.asarray(rows, dtype=float), np.asarray(valid, dtype=bool)


def load_init(seq_dir: Path):
    """init.txt -> (clon, clat, fov_h, fov_v)。"""
    line = next(ln for ln in (seq_dir / "init.txt").read_text(encoding="utf-8").splitlines() if ln.strip())
    f = [float(v) for v in line.replace(",", " ").split()]
    if len(f) != 4 or f[2] <= 0 or f[3] <= 0:
        raise ValueError(f"{seq_dir}/init.txt 非法: {line!r}")
    return tuple(f)


# ---------------------------------------------------------------------------
# tracker 后端
# ---------------------------------------------------------------------------

class MockTracker:
    """冻结初始框。"""

    def __init__(self, **_):
        pass

    def init(self, frame, erp_box, **_):
        self.box = [float(v) for v in erp_box]

    def track(self, frame, **_):
        return {"target_bbox": list(self.box), "quality": 1.0}


class GtEchoTracker:
    """诊断后端：回显当前帧 GT（+1px 扰动）。仅用于验证评分链路。"""

    def __init__(self, gt_erp=None, **_):
        self.gt_erp = gt_erp  # (N,4) ERP 框, 无效帧为全 0

    def init(self, frame, erp_box, frame_idx=0, **_):
        pass

    def track(self, frame, frame_idx=0, **_):
        b = self.gt_erp[frame_idx]
        if b[2] <= 0:
            return {"target_bbox": [0.0, 0.0, 0.0, 0.0], "quality": 0.0}
        return {"target_bbox": [b[0] + 1.0, b[1] + 1.0, b[2], b[3]], "quality": 1.0}


class FullFrameAdapter:
    """把 panotrack BaseTracker 的 update/bbox 契约适配到 eval_official。"""

    def __init__(self, tracker):
        self.tracker = tracker
        self.width = None

    def init(self, frame_rgb, erp_box, **_):
        self.width = frame_rgb.shape[1]
        self.tracker.init(frame_rgb, erp_box)

    def track(self, frame_rgb, **_):
        out = self.tracker.update(frame_rgb)
        box = out.get("bbox")
        return {"target_bbox": [float(box[0]) % self.width, float(box[1]),
                                float(box[2]), float(box[3])],
                "quality": float(out.get("score", 1.0)),
                "status": out.get("status", "ok")}


def build_fullframe_tracker(args):
    """构建 panotrack full-frame ERP tracker（LightFC/DirectERP 等单方案）。"""
    from panotrack.trackers.factory import create_tracker

    if args.tracker == "lightfc_onnx":
        return FullFrameAdapter(create_tracker(
            "lightfc_onnx",
            backbone_path=args.lightfc_backbone,
            tracking_path=args.lightfc_tracking,
            backend=args.lightfc_backend,
            search_size=args.lightfc_search_size,
            search_factor=args.lightfc_search_factor,
            template_size=args.lightfc_template_size,
            template_factor=args.lightfc_template_factor,
            max_crop_size=args.lightfc_max_crop_size,
        ))
    if args.tracker == "direct_erp":
        return FullFrameAdapter(create_tracker(
            "direct_erp",
            model_path=args.direct_erp_model,
            backend=args.direct_erp_backend,
            score_thr=args.direct_erp_score_thr,
        ))
    raise ValueError(f"unsupported full-frame tracker: {args.tracker}")


def build_uetrack_tracker(args):
    """构建 UETrack full-frame ERP tracker adapter。"""
    import os
    import sys as _sys

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    workspace = Path(args.uetrack_workspace).resolve()
    if not workspace.is_dir():
        raise SystemExit(f"[error] UETrack workspace 不存在: {workspace}")
    os.chdir(workspace)
    _sys.path.insert(0, str(workspace))

    from integrations.uetrack.arena_protocol import _load_tracker

    class UETrackAdapter:
        def __init__(self):
            self.tracker = None
            self.previous = {}
            self.width = None

        def init(self, frame_rgb, erp_box, **_):
            self.width = frame_rgb.shape[1]
            self.tracker = _load_tracker(
                workspace, args.uetrack_parameter, args.uetrack_no_erp_wrap)
            initialized = self.tracker.initialize(
                frame_rgb,
                {"init_bbox": [float(v) for v in erp_box], "seq_name": "official_eval"},
            )
            self.previous = initialized or {}

        def track(self, frame_rgb, **_):
            out = self.tracker.track(frame_rgb, {"previous_output": self.previous})
            self.previous = out
            box = out.get("target_bbox")
            if box is None or len(box) != 4:
                raise RuntimeError("UETrack returned invalid target_bbox")
            return {
                "target_bbox": [float(box[0]) % self.width, float(box[1]),
                                float(box[2]), float(box[3])],
                "quality": float(getattr(self.tracker, "last_pred_iou", 1.0)),
            }

    return UETrackAdapter()


def build_sutrack_tracker(args):
    """构建 SUTrack 三平铺 ERP tracker adapter。"""
    import copy as _copy
    import os
    import sys as _sys

    # 在 chdir 前保存 tools_local 绝对路径（LoRA 注入需要）
    _TOOLS_DIR = str(Path(os.path.abspath(__file__)).parent.parent / "tools_local")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    workspace = Path(args.sutrack_workspace).resolve()
    if not workspace.is_dir():
        raise SystemExit(f"[error] SUTrack workspace 不存在: {workspace}")
    ckpt = Path(args.sutrack_ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"[error] SUTrack checkpoint 不存在: {ckpt}")
    os.chdir(workspace)
    _sys.path.insert(0, str(workspace))

    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.test.tracker.sutrack import SUTRACK
    from lib.test.utils.params import TrackerParams

    update_config_from_file(workspace / "experiments" / "sutrack" / f"{args.sutrack_config}.yaml")
    # We load a full SUTrack checkpoint immediately after model construction.
    # Avoid requiring the separate iTPN pretrain file during adapter smoke/eval.
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    sutrack_encoder.is_main_process = lambda: False
    # Each adaptive route owns an independent config snapshot.  Keeping the
    # upstream module-global cfg here makes T224/B224 experts overwrite one
    # another when both are constructed in the same process.
    local_cfg = _copy.deepcopy(cfg)
    params = TrackerParams()
    params.cfg = local_cfg
    params.yaml_name = args.sutrack_config
    params.checkpoint = str(ckpt)
    params.template_factor = float(local_cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(local_cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(local_cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(local_cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0
    params.visualization = False

    class SUTrackAdapter:
        def __init__(self):
            self.tracker = SUTRACK(params, "got10k")
            # 如果指定了 LoRA checkpoint，注入 LoRA 并加载权重
            if getattr(args, 'sutrack_lora_ckpt', None):
                import torch as _torch
                lora_path = Path(args.sutrack_lora_ckpt).resolve()
                if lora_path.is_file():
                    # 注入 LoRA
                    if _TOOLS_DIR not in _sys.path:
                        _sys.path.insert(0, _TOOLS_DIR)
                    import sutrack_lora_setup as _lora_mod
                    apply_lora_to_sutrack = _lora_mod.apply_lora_to_sutrack
                    n = apply_lora_to_sutrack(self.tracker.network.encoder.body, rank=8, alpha=16.0)
                    print(f'[sutrack] injected LoRA into {n} layers')
                    # 加载 LoRA checkpoint
                    lora_ckpt = _torch.load(str(lora_path), map_location="cpu", weights_only=False)
                    lora_state = lora_ckpt.get("net", lora_ckpt)
                    if hasattr(lora_state, "state_dict"):
                        lora_state = lora_state.state_dict()
                    missing, unexpected = self.tracker.network.load_state_dict(lora_state, strict=False)
                    print(f'[sutrack] loaded LoRA ckpt: missing={len(missing)}, unexpected={len(unexpected)}')
                    self.tracker.network.cuda()
            self.width = None

        def init(self, frame_rgb, erp_box, **_):
            import numpy as _np

            h, w = frame_rgb.shape[:2]
            self.width = w
            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            box = [erp_box[0] % w + w, erp_box[1], erp_box[2], erp_box[3]]
            self.tracker.initialize(tiled, {"init_bbox": [float(v) for v in box]})

        def track(self, frame_rgb, **_):
            import numpy as _np
            import torch

            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            with torch.no_grad():
                if args.sutrack_amp and torch.cuda.is_available() and not args.force_cpu:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.tracker.track(tiled)
                else:
                    out = self.tracker.track(tiled)
            box = out.get("target_bbox")
            if box is None or len(box) != 4:
                raise RuntimeError("SUTrack returned invalid target_bbox")
            return {
                "target_bbox": [float(box[0]) % self.width, float(box[1]),
                                float(box[2]), float(box[3])],
                "quality": float(out.get("best_score", 1.0)),
            }

    return SUTrackAdapter()


def build_lorat_tracker(args):
    """构建 LoRAT DINOv2-B 三平铺 ERP tracker adapter。"""
    import os
    import sys as _sys

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    workspace = Path(args.lorat_workspace).resolve()
    if not workspace.is_dir():
        raise SystemExit(f"[error] LoRAT workspace 不存在: {workspace}")
    ckpt = Path(args.lorat_ckpt).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"[error] LoRAT checkpoint 不存在: {ckpt}")
    os.chdir(workspace)
    _sys.path.insert(0, str(workspace))

    # LoRAT upstream assumes a consts.yaml generated by its launcher.  The
    # standalone eval path only needs these inference defaults.
    consts = workspace / "consts.yaml"
    if not consts.exists():
        consts.write_text(
            "on_shared_file_system: false\n"
            "use_safetensors: true\n"
            "TIMM_USE_OLD_CACHE: true\n",
            encoding="utf-8",
        )

    import cv2 as _cv
    import safetensors.torch
    import torch

    from trackit.models.backbone.builder import build_backbone
    from trackit.models.methods.LoRAT.funcs.vit_lora_utils import attach_lora_state_dict_hooks_
    from trackit.models.methods.LoRAT.lorat import LoRAT_DINOv2
    from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
        PostProcessing_BoxWithScoreMap,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    backbone_cfg = {"type": "DINOv2", "parameters": {"name": "ViT-B/14", "acc": "default"}}
    backbone = build_backbone(backbone_cfg, load_pretrained=True, device=device, dtype=torch.float32)
    model = LoRAT_DINOv2(backbone, (8, 8), (16, 16)).to(device).eval()
    attach_lora_state_dict_hooks_(model)
    state = safetensors.torch.load_file(str(ckpt), device=str(device))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"LoRAT unexpected checkpoint keys: {unexpected[:20]}")
    print(f"[LoRAT] loaded {len(state)} tensors, missing base/head keys={len(missing)}")
    post = PostProcessing_BoxWithScoreMap(device, (16, 16), (224, 224), window_penalty_ratio=0.45)
    post.start()

    imagenet_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def xywh_to_xyxy(box):
        x, y, w, h = [float(v) for v in box]
        return np.asarray([x, y, x + w, y + h], dtype=np.float32)

    def xyxy_to_xywh(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def crop_params(box_xyxy, area_factor, out_size):
        out = np.asarray(out_size, dtype=np.float32)
        wh = np.maximum(box_xyxy[2:4] - box_xyxy[0:2], 1.0)
        area = wh + (float(area_factor) - 1.0) * (wh.sum() * 0.5)
        scale = np.sqrt(float(out.prod()) / float(area.prod()))
        scale = np.asarray([scale, scale], dtype=np.float32)
        src_center = (box_xyxy[0:2] + box_xyxy[2:4]) * 0.5
        dst_center = out * 0.5
        translation = dst_center - src_center * scale
        return np.stack([scale, translation], axis=0)

    def apply_crop(frame_rgb, params, out_size):
        scale, translation = params
        mat = np.asarray([[scale[0], 0.0, translation[0]],
                          [0.0, scale[1], translation[1]]], dtype=np.float32)
        mean_color = tuple(float(v) for v in frame_rgb.reshape(-1, 3).mean(axis=0))
        return _cv.warpAffine(
            frame_rgb,
            mat,
            (int(out_size[0]), int(out_size[1])),
            flags=_cv.INTER_LINEAR,
            borderMode=_cv.BORDER_CONSTANT,
            borderValue=mean_color,
        )

    def box_apply_params(box_xyxy, params):
        scale, translation = params
        out = box_xyxy.copy()
        out[0::2] = out[0::2] * scale[0] + translation[0]
        out[1::2] = out[1::2] * scale[1] + translation[1]
        return out

    def box_reverse_params(box_xyxy, params):
        scale, translation = params
        out = box_xyxy.copy()
        out[0::2] = (out[0::2] - translation[0]) / scale[0]
        out[1::2] = (out[1::2] - translation[1]) / scale[1]
        return out

    def to_tensor(crop):
        arr = crop.astype(np.float32) / 255.0
        arr = (arr - imagenet_mean) / imagenet_std
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def make_template_mask(init_xyxy, z_params):
        stride = np.asarray([14.0, 14.0], dtype=np.float32)
        z_box = box_apply_params(init_xyxy, z_params)
        z_box[0::2] /= stride[0]
        z_box[1::2] /= stride[1]
        x1 = int(np.floor(np.clip(z_box[0], 0, 8)))
        y1 = int(np.floor(np.clip(z_box[1], 0, 8)))
        x2 = int(np.ceil(np.clip(z_box[2], 0, 8)))
        y2 = int(np.ceil(np.clip(z_box[3], 0, 8)))
        if x2 <= x1:
            x2 = min(8, x1 + 1)
        if y2 <= y1:
            y2 = min(8, y1 + 1)
        mask = torch.zeros((1, 8, 8), dtype=torch.long, device=device)
        mask[:, y1:y2, x1:x2] = 1
        return mask

    class LoRATAdapter:
        def __init__(self):
            self.width = None
            self.height = None
            self.template = None
            self.z_mask = None
            self.box_xywh_tiled = None

        def init(self, frame_rgb, erp_box, **_):
            h, w = frame_rgb.shape[:2]
            self.width, self.height = w, h
            tiled = np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            box = np.asarray([erp_box[0] % w + w, erp_box[1], erp_box[2], erp_box[3]], dtype=np.float32)
            box_xyxy = xywh_to_xyxy(box)
            z_params = crop_params(box_xyxy, 2.0, (112, 112))
            z_crop = apply_crop(tiled, z_params, (112, 112))
            self.template = to_tensor(z_crop)
            self.z_mask = make_template_mask(box_xyxy, z_params)
            self.box_xywh_tiled = box

        def track(self, frame_rgb, **_):
            tiled = np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            prev_xyxy = xywh_to_xyxy(self.box_xywh_tiled)
            x_params = crop_params(prev_xyxy, 4.0, (224, 224))
            x_crop = apply_crop(tiled, x_params, (224, 224))
            x_tensor = to_tensor(x_crop)
            with torch.no_grad():
                if args.lorat_no_amp or device.type != "cuda":
                    out = model(self.template, x_tensor, self.z_mask)
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(self.template, x_tensor, self.z_mask)
                pp = post(out)
            pred_xyxy = pp["box"][0].detach().cpu().numpy().astype(np.float32)
            conf = float(pp["confidence"][0].detach().cpu())
            pred_xyxy = box_reverse_params(pred_xyxy, x_params)
            pred_xyxy[1::2] = np.clip(pred_xyxy[1::2], 0, self.height)
            pred_xyxy[2] = max(pred_xyxy[2], pred_xyxy[0] + 1.0)
            pred_xyxy[3] = max(pred_xyxy[3], pred_xyxy[1] + 1.0)

            # Keep internal state on the middle panorama tile to preserve wrap context.
            cx = float((pred_xyxy[0] + pred_xyxy[2]) * 0.5)
            while cx < self.width:
                pred_xyxy[0::2] += self.width
                cx += self.width
            while cx >= 2 * self.width:
                pred_xyxy[0::2] -= self.width
                cx -= self.width

            self.box_xywh_tiled = xyxy_to_xywh(pred_xyxy)
            return {
                "target_bbox": [float(self.box_xywh_tiled[0]) % self.width,
                                float(self.box_xywh_tiled[1]),
                                float(self.box_xywh_tiled[2]),
                                float(self.box_xywh_tiled[3])],
                "quality": conf,
            }

    return LoRATAdapter()


def build_odtrack_tracker(args):
    """懒加载 ODTrack（与 arena_protocol.py 相同的补丁与参数装配）。"""
    import os
    import types

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    workspace = Path(args.odtrack_workspace).resolve()
    if not workspace.is_dir():
        raise SystemExit(f"[error] ODTrack workspace 不存在: {workspace}")
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))

    try:
        import torch._six  # noqa: F401
    except ModuleNotFoundError:
        six = types.ModuleType("torch._six")
        six.string_classes = (str,)
        six.int_classes = (int,)
        sys.modules["torch._six"] = six
    if "visdom" not in sys.modules:
        visdom = types.ModuleType("visdom")
        visdom.__path__ = []
        visdom.Visdom = object
        sys.modules["visdom"] = visdom
        sys.modules["visdom.server"] = types.ModuleType("visdom.server")
    if "lib.vis.visdom_cus" not in sys.modules:
        m = types.ModuleType("lib.vis.visdom_cus")
        m.Visdom = type("Visdom", (), {"__init__": lambda self, *a, **k: None})
        sys.modules["lib.vis.visdom_cus"] = m

    if args.force_cpu:
        import torch
        torch.nn.Module.cuda = lambda self, device=None: self
        torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self

    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.test.tracker.odtrack import ODTrack
    from lib.test.utils.params import TrackerParams

    update_config_from_file(workspace / "experiments" / "odtrack" / f"{args.odtrack_config}.yaml")
    params = TrackerParams()
    params.cfg = cfg
    params.checkpoint = str(Path(args.odtrack_ckpt))
    params.template_factor = float(cfg.TEST.TEMPLATE_FACTOR)
    params.template_size = int(cfg.TEST.TEMPLATE_SIZE)
    params.search_factor = float(cfg.TEST.SEARCH_FACTOR)
    params.search_size = int(cfg.TEST.SEARCH_SIZE)
    params.save_all_boxes = False
    params.debug = 0

    class OdtrackAdapter:
        def __init__(self):
            self.tracker = ODTrack(params)
            self.width = None

        def init(self, frame_rgb, erp_box, **_):
            import numpy as _np
            h, w = frame_rgb.shape[:2]
            self.width = w
            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            box = [erp_box[0] % w + w, erp_box[1], erp_box[2], erp_box[3]]
            self.tracker.initialize(tiled, {"init_bbox": box})

        def track(self, frame_rgb, **_):
            import numpy as _np
            tiled = _np.concatenate((frame_rgb, frame_rgb, frame_rgb), axis=1)
            out = self.tracker.track(tiled)
            box = out.get("target_bbox")
            return {"target_bbox": [float(box[0]) % self.width, float(box[1]),
                                    float(box[2]), float(box[3])],
                    "quality": float(getattr(self.tracker, "last_pred_iou", 1.0))}

    class OdtrackTangentAdapter:
        """Fixed tangent-plane ODTrack expert for polar/spherical distortion.

        The local view is fixed at initialization, so the upstream ODTrack
        state remains coherent across frames.  It is selected only by an
        inference-visible initialization geometry gate; no sequence lookup or
        ground truth is used.
        """
        def __init__(self):
            self.tracker = ODTrack(params)
            self.width = self.height = None
            self.patch_size = int(args.tangent_patch_size)
            self.view = None
            self.map_x = self.map_y = None
            self.grid = None  # GPU 重采样 grid（(1,P,P,2) 归一化坐标），None 则走 numpy 回退

        def _patch(self, frame_rgb):
            if self.grid is not None:
                import numpy as _np
                import torch
                from torch.nn import functional as _F
                dev = self.grid.device
                t = torch.from_numpy(_np.ascontiguousarray(frame_rgb)).permute(2, 0, 1).float()
                tiled = torch.cat([t, t, t], dim=2).unsqueeze(0).to(dev, non_blocking=True)
                out = _F.grid_sample(tiled, self.grid, mode="bilinear",
                                     padding_mode="border", align_corners=True)
                patch = out[0].permute(1, 2, 0).detach().cpu().numpy()
                return _np.clip(_np.rint(patch), 0, 255).astype(_np.uint8)
            from panotrack.geometry.projection import remap_image
            return remap_image(frame_rgb, self.map_x, self.map_y)

        def init(self, frame_rgb, erp_box, **_):
            import numpy as _np
            from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox
            from panotrack.geometry.projection import tangent_remap
            from panotrack.pipeline.pipeline import _erp_bbox_to_local

            self.height, self.width = frame_rgb.shape[:2]
            target = bfov_from_erp_bbox(*erp_box, self.width, self.height)
            fov_h = min(160.0, max(float(args.tangent_fov_deg),
                                   target.fov_h * float(args.tangent_context)))
            fov_v = min(160.0, max(float(args.tangent_fov_deg),
                                   target.fov_v * float(args.tangent_context)))
            self.view = BFoV(target.lon, target.lat, fov_h, fov_v)
            self.map_x, self.map_y = tangent_remap(
                self.view, self.patch_size, self.patch_size, self.width, self.height)
            self.grid = None
            if not args.force_cpu:
                import torch
                if torch.cuda.is_available():
                    # 与 numpy remap_image 语义对齐：三平铺输入(3W) + 源坐标平移 +W，
                    # align_corners=True（-1↔像素0中心，+1↔像素W-1中心），
                    # border 垂直 clamp 对应 numpy 的 y clamp；x 邻域始终落在 3W 张量内无需回绕。
                    gx = torch.from_numpy(self.map_x.astype(_np.float32))
                    gy = torch.from_numpy(self.map_y.astype(_np.float32))
                    tw = 3 * self.width
                    gx = (gx + self.width) / (tw - 1.0) * 2.0 - 1.0
                    gy = gy / (self.height - 1.0) * 2.0 - 1.0
                    self.grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).cuda()
            patch = self._patch(frame_rgb)
            local = _erp_bbox_to_local(
                erp_box, self.view, self.patch_size, self.patch_size,
                self.width, self.height)
            tiled = _np.concatenate((patch, patch, patch), axis=1)
            box = [local[0] + self.patch_size, local[1], local[2], local[3]]
            self.tracker.initialize(tiled, {"init_bbox": [float(value) for value in box]})

        def track(self, frame_rgb, **_):
            import numpy as _np
            from panotrack.geometry.projection import local_bbox_to_erp

            patch = self._patch(frame_rgb)
            tiled = _np.concatenate((patch, patch, patch), axis=1)
            out = self.tracker.track(tiled)
            box = [float(value) for value in out.get("target_bbox")]
            # Keep the tracker on the middle tile, then expose a local box.
            while box[0] < self.patch_size:
                box[0] += self.patch_size
            while box[0] >= 2 * self.patch_size:
                box[0] -= self.patch_size
            local_x = float(_np.clip(box[0] - self.patch_size, -self.patch_size * 0.25,
                                      self.patch_size * 1.25))
            local_y = float(_np.clip(box[1], -self.patch_size * 0.25,
                                      self.patch_size * 1.25))
            local_w = float(_np.clip(box[2], 2.0, self.patch_size))
            local_h = float(_np.clip(box[3], 2.0, self.patch_size))
            erp = local_bbox_to_erp(
                local_x, local_y, local_w, local_h, self.map_x, self.map_y,
                self.width, self.height)
            return {
                "target_bbox": [float(erp[0]) % self.width, float(erp[1]),
                                float(erp[2]), float(erp[3])],
                "quality": float(getattr(self.tracker, "last_pred_iou", 1.0)),
                "projection": "tangent_fixed",
            }

    class OdtrackRecaptureAdapter:
        def __init__(self):
            from integrations.odtrack.recapture import OdtrackRecaptureTracker

            self.tracker = OdtrackRecaptureTracker(
                ODTrack(params),
                run_len=args.recapture_run_len,
                search_interval=args.recapture_search_interval,
                observe_frames=args.recapture_observe_frames,
                anchor_min_sim=args.recapture_anchor_min_sim,
                recapture_min_score=args.recapture_min_score,
                motion_max_deg=args.recapture_motion_max_deg,
                threshold=args.recapture_threshold,
            )
            self.width = None

        def init(self, frame_rgb, erp_box, **_):
            h, w = frame_rgb.shape[:2]
            self.width = w
            self.tracker.init(frame_rgb, erp_box)

        def track(self, frame_rgb, **_):
            out = self.tracker.update(frame_rgb)
            box = out.get("bbox")
            return {"target_bbox": [float(box[0]) % self.width, float(box[1]),
                                    float(box[2]), float(box[3])],
                    "quality": float(out.get("reliability", out.get("score", 1.0))),
                "status": out.get("status", "ok")}

    class OdtrackGeometryAdapter:
        """One submission-time tracker that selects ERP or tangent geometry at init.

        The choice uses only the initial BFoV latitude.  ERP remains the
        default for ordinary scenes; a fixed tangent view is reserved for the
        polar regime where ERP distortion is a known failure source.
        """
        def __init__(self):
            self.erp = OdtrackAdapter()
            self.tangent = OdtrackTangentAdapter()
            self.active = None
            self.projection = None

        def init(self, frame_rgb, erp_box, **kwargs):
            from panotrack.geometry.bfov import bfov_from_erp_bbox

            target = bfov_from_erp_bbox(*erp_box, frame_rgb.shape[1], frame_rgb.shape[0])
            if abs(target.lat) >= float(args.geo_tangent_lat_deg):
                self.active, self.projection = self.tangent, "tangent"
            else:
                self.active, self.projection = self.erp, "erp"
            self.active.init(frame_rgb, erp_box, **kwargs)

        def track(self, frame_rgb, **kwargs):
            out = dict(self.active.track(frame_rgb, **kwargs))
            out["projection"] = self.projection
            return out

    if args.tracker == "odtrack_recapture":
        return OdtrackRecaptureAdapter()
    if args.tracker == "odtrack_tangent":
        return OdtrackTangentAdapter()
    if args.tracker == "odtrack_geo":
        return OdtrackGeometryAdapter()
    return OdtrackAdapter()


def _adaptive_backend(args, name, checkpoint=None):
    """Build one same-family adaptive backend without sequence-name routing."""
    import argparse as _argparse

    child = _argparse.Namespace(**vars(args))
    if name == "odtrack":
        child.tracker = "odtrack"
        if checkpoint:
            child.odtrack_ckpt = checkpoint
        return build_odtrack_tracker(child), "odtrack"
    if name in ("sutrack_t224", "sutrack_b224"):
        child.tracker = "sutrack"
        child.sutrack_config = name
        if checkpoint:
            child.sutrack_ckpt = checkpoint
        return build_sutrack_tracker(child), "sutrack"
    raise SystemExit(f"[error] adaptive backend 暂不支持: {name}")


def build_adaptive_tracker(args):
    """Build route P (ODTrack family) or route S (SUTrack family)."""
    from panotrack.pipeline.adaptive_spherical import (
        AdaptiveRouterConfig,
        AdaptiveSphericalTracker,
    )
    from panotrack.pipeline.risk_policy import LinearRiskPolicy

    main, family = _adaptive_backend(
        args, args.adaptive_main, args.adaptive_main_ckpt)
    expert = None
    expert_family = family
    if args.adaptive_expert != "none":
        expert, expert_family = _adaptive_backend(
            args, args.adaptive_expert, args.adaptive_expert_ckpt)
        if expert_family != family:
            raise SystemExit(
                "[error] 同进程 adaptive tracker 只允许同代码族组合；"
                "跨 ODTrack/SUTrack 的 lib 命名空间冲突需走隔离 worker")
    config = AdaptiveRouterConfig(
        suspect_quality=args.adaptive_suspect_quality,
        lost_quality=args.adaptive_lost_quality,
        recover_quality=args.adaptive_recover_quality,
        suspect_run=args.adaptive_suspect_run,
        lost_run=args.adaptive_lost_run,
        verify_frames=args.adaptive_verify_frames,
        geometry_risk=args.adaptive_geometry_risk,
        expert_interval=args.adaptive_expert_interval,
        expert_max_fraction=args.adaptive_expert_max_fraction,
        target_frame_ms=args.adaptive_target_frame_ms,
        expert_quality=args.adaptive_expert_quality,
        expert_episode_frames=args.adaptive_expert_episode_frames,
        redetect_interval=args.adaptive_redetect_interval,
        enable_global_redetect=not args.adaptive_no_global_redetect,
        redetect_min_score=args.adaptive_redetect_min_score,
        anchor_min_similarity=args.adaptive_anchor_min_similarity,
        motion_max_deg=args.adaptive_motion_max_deg,
    )
    risk_policy = LinearRiskPolicy.load(args.adaptive_risk_policy) \
        if args.adaptive_risk_policy else None
    return AdaptiveSphericalTracker(
        main, expert, config=config,
        risk_policy=risk_policy,
        main_name=args.adaptive_main,
        expert_name=args.adaptive_expert,
    )


# ---------------------------------------------------------------------------
# 序列选择与主循环
# ---------------------------------------------------------------------------

def select_sequences(args):
    if args.seqs:
        return [s.strip() for s in args.seqs.split(",") if s.strip()]
    if args.split in ("train", "valid"):
        path = SPLIT_DIR / f"seqlist_official_{args.split}.txt"
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # all: 扫描数据目录
    root = Path(args.data)
    seqs = []
    for block in ("train_real", "train_sim"):
        d = root / block
        if d.is_dir():
            seqs.extend(f"{block}/{p.name}" for p in sorted(d.iterdir())
                        if p.is_dir() and (p / "video.mp4").is_file())
    return seqs


def run_sequence(seq_rel, data_root, tracker_factory, max_frames=None):
    seq_dir = Path(data_root) / seq_rel
    if cv is None:
        raise SystemExit("需要 cv2 解码 video.mp4")

    gt_bfov, gt_valid = load_gt(seq_dir / "groundtruth.txt")
    init_bfov = load_init(seq_dir)
    n_total = len(gt_bfov)
    limit = min(n_total, max_frames) if max_frames else n_total

    # GT BFoV -> ERP 框（无效帧置 0）
    cap = cv.VideoCapture(str(seq_dir / "video.mp4"))
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError(f"解码失败: {seq_dir}")
    H, W = first.shape[:2]
    gt_erp = np.zeros((n_total, 4), dtype=float)
    for i in range(n_total):
        if gt_valid[i]:
            x, y, w, h = erp_bbox_from_bfov(
                BFoV(lon=gt_bfov[i][0], lat=gt_bfov[i][1],
                     fov_h=gt_bfov[i][2], fov_v=gt_bfov[i][3]), W, H)
            gt_erp[i] = [x, y, w, h]

    tracker = tracker_factory(gt_erp=gt_erp)
    init_erp = erp_bbox_from_bfov(BFoV(*init_bfov), W, H)
    tracker.init(cv.cvtColor(first, cv.COLOR_BGR2RGB), init_erp)

    pred_erp = np.zeros((limit, 4), dtype=float)
    pred_erp[0] = init_erp
    times = []
    frame_times = []
    decode_times = []
    qualities = np.ones((limit,), dtype=float)
    statuses = ["init"] + [""] * max(0, limit - 1)
    traces = [{
        "frame_index": 0,
        "target_bbox": [float(v) for v in init_erp],
        "quality": 1.0,
        "status": "init",
        "latency_ms": 0.0,
        "expert_used": None,
        "route_reasons": [],
    }]
    for i in range(1, limit):
        frame_t0 = time.perf_counter()
        decode_t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"{seq_rel}: 第 {i} 帧解码失败")
        decode_times.append(time.perf_counter() - decode_t0)
        t0 = time.perf_counter()
        out = tracker.track(cv.cvtColor(frame, cv.COLOR_BGR2RGB), frame_idx=i)
        tracker_elapsed = time.perf_counter() - t0
        times.append(tracker_elapsed)
        frame_times.append(time.perf_counter() - frame_t0)
        b = out["target_bbox"]
        pred_erp[i] = [float(v) for v in b[:4]]
        qualities[i] = float(out.get("quality", 1.0))
        statuses[i] = str(out.get("status", "ok"))
        traces.append({
            "frame_index": i,
            "target_bbox": [float(v) for v in b[:4]],
            "quality": qualities[i],
            "status": statuses[i],
            "latency_ms": float(out.get("latency_ms", tracker_elapsed * 1000.0)),
            "main_latency_ms": float(out.get("main_latency_ms", tracker_elapsed * 1000.0)),
            "expert_latency_ms": float(out.get("expert_latency_ms", 0.0)),
            "redetect_latency_ms": float(out.get("redetect_latency_ms", 0.0)),
            "expert_used": out.get("expert_used"),
            "expert_probed": bool(out.get("expert_probed", False)),
            "expert_call_fraction": float(out.get("expert_call_fraction", 0.0)),
            "expert_budget_fraction": float(out.get("expert_budget_fraction", 0.0)),
            "expert_episode_remaining": int(out.get("expert_episode_remaining", 0)),
            "anchor_similarity": out.get("anchor_similarity"),
            "response_entropy": out.get("response_entropy"),
            "route_reasons": list(out.get("route_reasons", [])),
            "decode_ms": decode_times[-1] * 1000.0,
            "frame_e2e_ms": frame_times[-1] * 1000.0,
        })
    cap.release()

    # 掩码 OPE 评分：跳过首帧与 GT 无效帧
    ious, ious_dual = [], []
    for i in range(1, limit):
        if not gt_valid[i]:
            continue
        p, g = pred_erp[i], gt_erp[i]
        if p[2] <= 0 or p[3] <= 0:   # 预测宣告丢失 -> IoU 0
            ious.append(0.0)
            ious_dual.append(0.0)
            continue
        ious.append(iou_xywh(p, g))
        ious_dual.append(dual_iou(p, g, W))

    n_scored = len(ious)
    elapsed = float(np.sum(times))
    e2e_elapsed = float(np.sum(frame_times))
    tracker_ms = np.asarray(times, dtype=float) * 1000.0
    e2e_ms = np.asarray(frame_times, dtype=float) * 1000.0
    expert_calls = sum(bool(trace.get("expert_probed")) for trace in traces[1:])
    metrics = {
        "sequence": seq_rel,
        "n_frames": int(limit),
        "n_scored": n_scored,
        "n_gt_absent": int((~gt_valid[:limit]).sum()),
        "sr": success_rate(ious) if n_scored else float("nan"),
        "auc": auc(ious) if n_scored else float("nan"),
        "sr_dual": success_rate(ious_dual) if n_scored else float("nan"),
        "auc_dual": auc(ious_dual) if n_scored else float("nan"),
        "fps": (limit - 1) / elapsed if elapsed > 0 else float("nan"),
        "tracker_fps": (limit - 1) / elapsed if elapsed > 0 else float("nan"),
        "e2e_fps": (limit - 1) / e2e_elapsed if e2e_elapsed > 0 else float("nan"),
        "tracker_latency_p50_ms": float(np.percentile(tracker_ms, 50)) if len(tracker_ms) else float("nan"),
        "tracker_latency_p95_ms": float(np.percentile(tracker_ms, 95)) if len(tracker_ms) else float("nan"),
        "e2e_latency_p50_ms": float(np.percentile(e2e_ms, 50)) if len(e2e_ms) else float("nan"),
        "e2e_latency_p95_ms": float(np.percentile(e2e_ms, 95)) if len(e2e_ms) else float("nan"),
        "decode_latency_mean_ms": float(np.mean(decode_times) * 1000.0) if decode_times else float("nan"),
        "expert_calls": expert_calls,
        "expert_call_fraction": expert_calls / max(1, limit - 1),
        "resolution": f"{W}x{H}",
    }
    latency = {key: metrics[key] for key in (
        "tracker_fps", "e2e_fps", "tracker_latency_p50_ms",
        "tracker_latency_p95_ms", "e2e_latency_p50_ms",
        "e2e_latency_p95_ms", "decode_latency_mean_ms",
        "expert_calls", "expert_call_fraction")}
    return metrics, pred_erp, gt_valid[:limit], W, H, qualities, statuses, traces, latency


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=r"D:\instan\初赛数据\train",
                    help="官方训练集根目录（含 train_real/ train_sim/）")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "official_eval"))
    ap.add_argument("--tracker", default="mock",
                    choices=["mock", "gt_echo", "odtrack", "odtrack_recapture",
                             "odtrack_tangent", "odtrack_geo",
                             "lightfc_onnx", "direct_erp", "uetrack", "sutrack", "lorat",
                             "adaptive_spherical"])
    ap.add_argument("--split", default="all", choices=["all", "train", "valid"])
    ap.add_argument("--seqs", default=None, help="逗号分隔 block/seq 覆盖 split")
    ap.add_argument("--max-frames", type=int, default=None)
    # odtrack 专用
    ap.add_argument("--odtrack-workspace", default="/opt/odtrack")
    ap.add_argument("--odtrack-ckpt", default="/opt/models/ODTrack_ep0300.pth.tar")
    ap.add_argument("--odtrack-config", default="baseline")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--force-cpu", action="store_true")
    # odtrack_recapture 专用
    ap.add_argument("--recapture-run-len", type=int, default=5)
    ap.add_argument("--recapture-search-interval", type=int, default=5)
    ap.add_argument("--recapture-observe-frames", type=int, default=3)
    ap.add_argument("--recapture-anchor-min-sim", type=float, default=0.5)
    ap.add_argument("--recapture-min-score", type=float, default=0.45)
    ap.add_argument("--recapture-motion-max-deg", type=float, default=90.0)
    ap.add_argument("--recapture-threshold", type=float, default=0.55)
    # odtrack_tangent 专用
    ap.add_argument("--tangent-patch-size", type=int, default=720)
    ap.add_argument("--tangent-fov-deg", type=float, default=110.0)
    ap.add_argument("--tangent-context", type=float, default=3.5)
    ap.add_argument("--geo-tangent-lat-deg", type=float, default=60.0)
    # full-frame tracker / LightFC 专用
    ap.add_argument("--lightfc-backbone", default=str(PROJECT_ROOT / "models" / "lightfc_backbone.onnx"))
    ap.add_argument("--lightfc-tracking", default=str(PROJECT_ROOT / "models" / "lightfc_tracking.onnx"))
    ap.add_argument("--lightfc-backend", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--lightfc-search-size", type=int, default=256)
    ap.add_argument("--lightfc-search-factor", type=float, default=4.0)
    ap.add_argument("--lightfc-template-size", type=int, default=128)
    ap.add_argument("--lightfc-template-factor", type=float, default=2.0)
    ap.add_argument("--lightfc-max-crop-size", type=int, default=2048)
    # full-frame tracker / DirectERP 专用
    ap.add_argument("--direct-erp-model", default=str(PROJECT_ROOT / "models" / "object_tracking_vittrack_2023sep.onnx"))
    ap.add_argument("--direct-erp-backend", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--direct-erp-score-thr", type=float, default=0.1)
    # UETrack 专用
    ap.add_argument("--uetrack-workspace", default="/opt/uetrack")
    ap.add_argument("--uetrack-parameter", default="uetrack_base")
    ap.add_argument("--uetrack-no-erp-wrap", action="store_true")
    # SUTrack 专用
    ap.add_argument("--sutrack-workspace", default="/opt/sutrack")
    ap.add_argument("--sutrack-ckpt", default="/opt/models/SUTRACK_ep0300.pth.tar")
    ap.add_argument("--sutrack-config", default="sutrack_b224")
    ap.add_argument("--sutrack-amp", action="store_true", default=True,
                    help="use CUDA FP16 autocast for SUTrack inference (default: on)")
    ap.add_argument("--no-sutrack-amp", dest="sutrack_amp", action="store_false",
                    help="disable CUDA FP16 autocast for SUTrack")
    ap.add_argument("--sutrack-lora-ckpt", default=None,
                    help="LoRA checkpoint to inject into SUTrack (merged with base ckpt)")
    # LoRAT 专用
    ap.add_argument("--lorat-workspace", default="/opt/lorat")
    ap.add_argument("--lorat-ckpt", default="/opt/models/lorat_base.bin")
    ap.add_argument("--lorat-no-amp", action="store_true")
    # adaptive_spherical: route P=ODTrack family, route S=SUTrack T/B family
    ap.add_argument("--adaptive-main", default="odtrack",
                    choices=["odtrack", "sutrack_t224", "sutrack_b224"])
    ap.add_argument("--adaptive-expert", default="none",
                    choices=["none", "odtrack", "sutrack_t224", "sutrack_b224"])
    ap.add_argument("--adaptive-main-ckpt", default=None)
    ap.add_argument("--adaptive-expert-ckpt", default=None)
    ap.add_argument("--adaptive-suspect-quality", type=float, default=0.45)
    ap.add_argument("--adaptive-lost-quality", type=float, default=0.25)
    ap.add_argument("--adaptive-recover-quality", type=float, default=0.55)
    ap.add_argument("--adaptive-suspect-run", type=int, default=2)
    ap.add_argument("--adaptive-lost-run", type=int, default=5)
    ap.add_argument("--adaptive-verify-frames", type=int, default=3)
    ap.add_argument("--adaptive-geometry-risk", type=float, default=0.55)
    ap.add_argument("--adaptive-expert-interval", type=int, default=1)
    ap.add_argument("--adaptive-expert-max-fraction", type=float, default=0.20)
    ap.add_argument("--adaptive-target-frame-ms", type=float, default=33.333)
    ap.add_argument("--adaptive-expert-quality", type=float, default=0.50)
    ap.add_argument("--adaptive-expert-episode-frames", type=int, default=10)
    ap.add_argument("--adaptive-redetect-interval", type=int, default=5)
    ap.add_argument("--adaptive-redetect-min-score", type=float, default=0.45)
    ap.add_argument("--adaptive-no-global-redetect", action="store_true")
    ap.add_argument("--adaptive-anchor-min-similarity", type=float, default=0.50)
    ap.add_argument("--adaptive-motion-max-deg", type=float, default=120.0)
    ap.add_argument("--adaptive-risk-policy", default=None,
                    help="validated LinearRiskPolicy JSON; diagnostic policies are rejected by the launch queue")
    args = ap.parse_args(argv)

    seqs = select_sequences(args)
    if not seqs:
        raise SystemExit("[error] 没有序列被选中")
    out_dir = Path(args.out) / f"{args.tracker}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tracker == "mock":
        factory = lambda gt_erp: MockTracker()
    elif args.tracker == "gt_echo":
        factory = lambda gt_erp: GtEchoTracker(gt_erp=gt_erp)
    elif args.tracker in ("lightfc_onnx", "direct_erp"):
        factory = lambda gt_erp: build_fullframe_tracker(args)
    elif args.tracker == "uetrack":
        factory = lambda gt_erp: build_uetrack_tracker(args)
    elif args.tracker == "sutrack":
        proto = build_sutrack_tracker(args)
        factory = lambda gt_erp: proto
    elif args.tracker == "lorat":
        proto = build_lorat_tracker(args)
        factory = lambda gt_erp: proto
    elif args.tracker == "adaptive_spherical":
        proto = build_adaptive_tracker(args)
        factory = lambda gt_erp: proto
    else:
        proto = build_odtrack_tracker(args)
        factory = lambda gt_erp: proto  # 单实例顺序复用（ODTrack 逐序列重建较慢，先共用）

    print(f"[eval_official] tracker={args.tracker} seqs={len(seqs)} out={out_dir}")
    rows = []
    for idx, seq_rel in enumerate(seqs, 1):
        t0 = time.time()
        try:
            (metrics, pred_erp, valid, W, H, qualities, statuses,
             traces, latency) = run_sequence(
                seq_rel, args.data, factory, args.max_frames)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(f"  [{idx}/{len(seqs)}] {seq_rel}: FAILED ({exc})", file=sys.stderr)
            continue
        seq_out = out_dir / seq_rel
        seq_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(seq_out / "results_erp.txt", pred_erp, fmt="%.2f", delimiter=",")
        np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
        (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
        with (seq_out / "trace.jsonl").open("w", encoding="utf-8") as f:
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False, allow_nan=True) + "\n")
        (seq_out / "latency.json").write_text(
            json.dumps(latency, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8")
        with open(seq_out / "bfov.txt", "w", encoding="utf-8") as f:
            for b in pred_erp:
                if b[2] <= 0 or b[3] <= 0:
                    f.write("0.000,0.000,0.000,0.000\n")
                else:
                    bf = bfov_from_erp_bbox(b[0], b[1], b[2], b[3], W, H)
                    f.write(f"{bf.lon:.3f},{bf.lat:.3f},{bf.fov_h:.3f},{bf.fov_v:.3f}\n")
        (seq_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(metrics)
        print(f"  [{idx}/{len(seqs)}] {seq_rel}: AUC={metrics['auc']:.4f} "
              f"SR={metrics['sr']:.4f} trackerFPS={metrics['fps']:.1f} "
              f"e2eFPS={metrics['e2e_fps']:.1f} "
              f"({time.time()-t0:.1f}s, absent={metrics['n_gt_absent']})")

    if rows:
        def mean(k):
            vals = [r[k] for r in rows if np.isfinite(r[k])]
            return float(np.mean(vals)) if vals else float("nan")
        summary = {
            "tracker": args.tracker, "n_sequences": len(rows),
            "auc": mean("auc"), "sr": mean("sr"),
            "auc_dual": mean("auc_dual"), "sr_dual": mean("sr_dual"),
            "fps": mean("fps"), "tracker_fps": mean("tracker_fps"),
            "e2e_fps": mean("e2e_fps"),
            "tracker_latency_p50_ms": mean("tracker_latency_p50_ms"),
            "tracker_latency_p95_ms": mean("tracker_latency_p95_ms"),
            "e2e_latency_p50_ms": mean("e2e_latency_p50_ms"),
            "e2e_latency_p95_ms": mean("e2e_latency_p95_ms"),
            "decode_latency_mean_ms": mean("decode_latency_mean_ms"),
            "expert_calls_total": sum(r["expert_calls"] for r in rows),
            "expert_call_fraction": mean("expert_call_fraction"),
            "n_gt_absent_total": sum(r["n_gt_absent"] for r in rows),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        latency_summary = {
            key: summary[key] for key in (
                "tracker_fps", "e2e_fps", "tracker_latency_p50_ms",
                "tracker_latency_p95_ms", "e2e_latency_p50_ms",
                "e2e_latency_p95_ms", "decode_latency_mean_ms",
                "expert_calls_total", "expert_call_fraction")
        }
        latency_summary["sequences"] = {
            row["sequence"]: {
                key: row[key] for key in (
                    "tracker_fps", "e2e_fps", "tracker_latency_p50_ms",
                    "tracker_latency_p95_ms", "e2e_latency_p50_ms",
                    "e2e_latency_p95_ms", "expert_calls", "expert_call_fraction")
            } for row in rows
        }
        (out_dir / "latency.json").write_text(
            json.dumps(latency_summary, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8")
        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[summary] {args.tracker}: AUC={summary['auc']:.4f} SR={summary['sr']:.4f} "
              f"dual AUC={summary['auc_dual']:.4f} trackerFPS={summary['fps']:.1f} "
              f"e2eFPS={summary['e2e_fps']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
