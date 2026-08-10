# -*- coding: utf-8 -*-
"""GRT360-Spherical-Memory：面向 ERP 360°跟踪的底层单模型。

这个模块不接收其它跟踪器的预测框，也不做结果级投票。它在特征层完成：

* 水平方向环形卷积，避免 ERP 子午线处的人工断缝；
* 经纬度正余弦编码，把极区拉伸和 seam 位置显式输入网络；
* 模板--搜索区域相关匹配，输出球面局部位移和尺度变化；
* 因果记忆门控，只在置信度足够高时更新模板，降低遮挡污染。

网络故意保持轻量，方便先在 360VOT 上验证架构，再替换为更大的 backbone。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _coord_channels(
    batch: int,
    height: int,
    width: int,
    geometry: Optional[Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """生成 [sin(lon), cos(lon), sin(lat), cos(lat)] 四个球面通道。

    geometry 的四列是中心经度、中心纬度、水平 FoV、垂直 FoV，单位为度。
    允许传入空值，空值时退化为以 patch 中心为原点的局部球面坐标。
    """
    yy = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height - 0.5
    xx = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width - 0.5
    vy, vx = torch.meshgrid(yy, xx, indexing="ij")
    vx = vx.unsqueeze(0).expand(batch, -1, -1)
    vy = vy.unsqueeze(0).expand(batch, -1, -1)
    if geometry is None:
        lon = vx * 90.0
        lat = -vy * 90.0
    else:
        geometry = geometry.to(device=device, dtype=dtype).reshape(batch, 4)
        lon = geometry[:, 0, None, None] + vx * geometry[:, 2, None, None]
        lat = geometry[:, 1, None, None] - vy * geometry[:, 3, None, None]
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat.clamp(-89.9, 89.9))
    return torch.stack((lon.sin(), lon.cos(), lat.sin(), lat.cos()), dim=1)


class CircularConv2d(nn.Module):
    """水平方向 circular、垂直方向 replicate 的卷积。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.pad = pad
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.pad:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


class SepBlock(nn.Module):
    """轻量 depthwise-separable block，减少实验阶段的算力开销。"""

    def __init__(self, channels: int):
        super().__init__()
        self.dw = CircularConv2d(channels, channels, 3)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = F.gelu(self.norm(self.pw(self.dw(x))))
        return x + residual


class SphericalEncoder(nn.Module):
    """带球面坐标通道的共享模板/搜索编码器。"""

    def __init__(self, channels: int = 24):
        super().__init__()
        self.stem = CircularConv2d(7, channels, 5, stride=2)
        self.norm = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(SepBlock(channels), SepBlock(channels))

    def forward(self, image: Tensor, geometry: Optional[Tensor]) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"需要 [B,3,H,W] 输入，实际为 {tuple(image.shape)}")
        coords = _coord_channels(image.shape[0], image.shape[2], image.shape[3],
                                 geometry, image.device, image.dtype)
        x = torch.cat((image, coords), dim=1)
        return self.blocks(F.gelu(self.norm(self.stem(x))))


def _group_correlation(template: Tensor, search: Tensor) -> Tensor:
    """按 batch 做归一化模板相关，返回 [B,1,H,W] 响应图。"""
    b, c, ht, wt = template.shape
    _, _, hs, ws = search.shape
    if ht > hs or wt > ws:
        raise ValueError("template 必须小于 search")
    template = F.normalize(template.flatten(1), dim=1).view(b, c, ht, wt)
    search = F.normalize(search, dim=1)
    # group convolution: one independent C-channel kernel per batch item.
    kernels = template.reshape(b, c, ht, wt)
    search_batched = search.reshape(1, b * c, hs, ws)
    corr = F.conv2d(search_batched, kernels, groups=b)
    return corr.reshape(b, 1, corr.shape[-2], corr.shape[-1])


class SphericalMemoryNet(nn.Module):
    """GRT360-Spherical-Memory 主网络。

    ``forward`` 返回训练和推理共用的中间量。target_delta 为相对 search
    中心的 (dx, dy, log_w, log_h)，前三者均按 search 尺寸归一化。
    """

    def __init__(self, channels: int = 24, memory_momentum: float = 0.08):
        super().__init__()
        self.encoder = SphericalEncoder(channels)
        self.memory_momentum = float(memory_momentum)
        self.box_head = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 5)
        )

    @staticmethod
    def _soft_argmax(response: Tensor) -> Tuple[Tensor, Tensor]:
        b, _, h, w = response.shape
        flat = response.flatten(1)
        weights = torch.softmax(flat * 12.0, dim=1).reshape(b, h, w)
        yy = torch.linspace(-1.0, 1.0, h, device=response.device, dtype=response.dtype)
        xx = torch.linspace(-1.0, 1.0, w, device=response.device, dtype=response.dtype)
        cy = (weights.sum(2) * yy).sum(1)
        cx = (weights.sum(1) * xx).sum(1)
        conf = flat.max(1).values.sigmoid()
        return torch.stack((cx, cy), dim=1), conf

    def encode(self, image: Tensor, geometry: Optional[Tensor] = None) -> Tensor:
        return self.encoder(image.float() / 255.0 if image.max() > 1.5 else image, geometry)

    def init_memory(self, template: Tensor, geometry: Optional[Tensor] = None) -> Tensor:
        return self.encode(template, geometry)

    def update_memory(self, memory: Tensor, current: Tensor, confidence: Tensor) -> Tensor:
        """因果更新：置信度低于 0.35 时完全冻结记忆。"""
        current = F.adaptive_avg_pool2d(current, memory.shape[-2:])
        gate = ((confidence.detach() - 0.35) / 0.45).clamp(0.0, 1.0)
        momentum = self.memory_momentum * gate.reshape(-1, 1, 1, 1)
        return (1.0 - momentum) * memory + momentum * current

    def forward(
        self,
        template: Tensor,
        search: Tensor,
        geometry: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        search_feat = self.encode(search, geometry)
        if memory is None:
            memory = self.init_memory(template, geometry)
        corr = _group_correlation(memory, search_feat)
        xy, corr_conf = self._soft_argmax(corr)
        pooled = torch.cat((xy, corr_conf[:, None],
                            memory.mean(dim=(1, 2, 3), keepdim=False)[:, None],
                            search_feat.mean(dim=(1, 2, 3), keepdim=False)[:, None]), dim=1)
        head = self.box_head(pooled)
        # 位移由相关图给出，head 只学习尺度、微调和置信度残差。
        delta = torch.cat((xy + 0.15 * torch.tanh(head[:, :2]),
                           0.25 * torch.tanh(head[:, 2:4])), dim=1)
        confidence = torch.sigmoid(head[:, 4] + torch.logit(corr_conf.clamp(1e-4, 1 - 1e-4)))
        return {"delta": delta, "confidence": confidence,
                "correlation": corr, "search_feature": search_feat,
                "memory": memory}

    def loss(self, output: Dict[str, Tensor], target_delta: Tensor,
             target_visible: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """几何一致的训练损失：位移 Smooth-L1 + 可见性 BCE。"""
        target_delta = target_delta.to(output["delta"].device).float()
        loc = F.smooth_l1_loss(output["delta"], target_delta)
        if target_visible is None:
            target_visible = torch.ones_like(output["confidence"])
        vis = F.binary_cross_entropy(output["confidence"].clamp(1e-5, 1 - 1e-5),
                                     target_visible.float())
        return {"total": loc + 0.25 * vis, "localization": loc, "visibility": vis}
