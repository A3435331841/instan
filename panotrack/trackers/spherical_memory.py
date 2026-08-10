# -*- coding: utf-8 -*-
"""GRT360-Spherical-Memory 的 BaseTracker 适配器。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.nn import functional as F

from ..models.spherical_memory import SphericalMemoryNet
from .base import BaseTracker


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("SphericalMemoryTracker 需要三通道图像")
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo > 1e-6 and (lo < 0.0 or hi > 1.0):
        arr = (arr - lo) / (hi - lo) * 255.0
    return arr


def _crop_tensor(image: np.ndarray, bbox, out_size: int, device: torch.device) -> torch.Tensor:
    arr = _as_rgb(image)
    h, w = arr.shape[:2]
    x, y, bw, bh = (float(v) for v in bbox)
    yy = (torch.arange(out_size, device=device, dtype=torch.float32) + 0.5) / out_size - 0.5
    xx = (torch.arange(out_size, device=device, dtype=torch.float32) + 0.5) / out_size - 0.5
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    px = x + gx * bw + 0.5 * bw
    py = y + gy * bh + 0.5 * bh
    grid = torch.stack((2.0 * px / max(w - 1, 1) - 1.0,
                        2.0 * py / max(h - 1, 1) - 1.0), dim=-1)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return F.grid_sample(tensor, grid.unsqueeze(0), mode="bilinear",
                         padding_mode="border", align_corners=True)


class SphericalMemoryTracker(BaseTracker):
    """单模型球面记忆跟踪器，输出与 NCCTracker 兼容的字典。"""

    input_space = "local_patch"

    def __init__(self, model_path: str, device: str = "auto", template_size: int = 64,
                 update_threshold: float = 0.45, channels: int = 24, **kwargs):
        del kwargs
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.template_size = int(template_size)
        self.update_threshold = float(update_threshold)
        self.model = SphericalMemoryNet(channels=channels).to(self.device).eval()
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"SphericalMemory checkpoint 不存在: {path}")
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self._memory: Optional[torch.Tensor] = None
        self._template: Optional[torch.Tensor] = None
        self._geometry: Optional[torch.Tensor] = None
        self._cx = self._cy = 0.0
        self._w = self._h = 1.0
        self._ready = False

    def set_geometry(self, bfov) -> None:
        self._geometry = torch.tensor([[float(bfov.lon), float(bfov.lat),
                                        float(bfov.fov_h), float(bfov.fov_v)]],
                                       device=self.device)

    def init(self, image, bbox):
        image = _as_rgb(image)
        x, y, w, h = (float(v) for v in bbox)
        self._cx, self._cy, self._w, self._h = x + w / 2.0, y + h / 2.0, max(w, 2.0), max(h, 2.0)
        self._template = _crop_tensor(image, (x, y, w, h), self.template_size, self.device)
        with torch.inference_mode():
            self._memory = self.model.init_memory(self._template, self._geometry)
        self._ready = True

    def update(self, image):
        if not self._ready or self._template is None or self._memory is None:
            raise RuntimeError("SphericalMemoryTracker 尚未 init")
        image = _as_rgb(image)
        search = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out = self.model(self._template, search, self._geometry, self._memory)
        delta = out["delta"][0].detach().cpu().numpy()
        confidence = float(out["confidence"][0].detach().cpu())
        sh, sw = image.shape[:2]
        self._cx = float(np.clip(sw * 0.5 + delta[0] * sw * 0.5, 0.0, sw - 1.0))
        self._cy = float(np.clip(sh * 0.5 + delta[1] * sh * 0.5, 0.0, sh - 1.0))
        self._w = float(np.clip(self._w * np.exp(delta[2]), 2.0, 0.8 * sw))
        self._h = float(np.clip(self._h * np.exp(delta[3]), 2.0, 0.8 * sh))
        if confidence >= self.update_threshold:
            with torch.inference_mode():
                self._memory = self.model.update_memory(self._memory,
                                                         out["search_feature"],
                                                         out["confidence"])
        return {"bbox": (self._cx - self._w / 2.0, self._cy - self._h / 2.0,
                          self._w, self._h),
                "score": confidence, "psr": 2.0 + 8.0 * confidence,
                "apce": 0.5 + 4.0 * confidence}
