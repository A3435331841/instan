#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练/验证 GRT360-Spherical-Memory 的最小实验入口。

默认先做合成球面局部位移过拟合，验证网络确实学习相关位移；使用
``--data`` 时可在后续接入 360VOT 帧对训练。该脚本不会读取其它跟踪器结果。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.models.spherical_memory import SphericalMemoryNet


def synthetic_batch(batch: int, template_size: int, search_size: int, device: torch.device):
    """生成带纹理目标的可控平移样本。"""
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, search_size, device=device),
        torch.linspace(-1.0, 1.0, search_size, device=device), indexing="ij")
    cx = torch.empty(batch, device=device).uniform_(-0.45, 0.45)
    cy = torch.empty(batch, device=device).uniform_(-0.45, 0.45)
    sigma = torch.empty(batch, device=device).uniform_(0.08, 0.18)
    blob = torch.exp(-((xx[None] - cx[:, None, None]) ** 2
                       + (yy[None] - cy[:, None, None]) ** 2)
                    / (2.0 * sigma[:, None, None] ** 2))
    texture = 0.25 * torch.sin(13.0 * xx)[None] * torch.cos(9.0 * yy)[None]
    search = (blob + texture).unsqueeze(1).repeat(1, 3, 1, 1)
    template = torch.zeros(batch, 3, template_size, template_size, device=device)
    # 模板是以目标中心为原点的局部观察，网络必须从 search 中找回 cx/cy。
    ty, tx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, template_size, device=device),
        torch.linspace(-1.0, 1.0, template_size, device=device), indexing="ij")
    tblob = torch.exp(-(tx[None] ** 2 + ty[None] ** 2) / 0.18)
    template[:] = (tblob + 0.25 * torch.sin(13.0 * tx)[None]
                   * torch.cos(9.0 * ty)[None]).unsqueeze(1)
    target = torch.stack((cx, cy, torch.zeros_like(cx), torch.zeros_like(cx)), dim=1)
    return template, search, target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/spherical_memory_synthetic.pt")
    args = p.parse_args(argv)
    device = torch.device(args.device)
    model = SphericalMemoryNet(channels=12).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    geometry = torch.tensor([[0.0, 0.0, 90.0, 90.0]], device=device).repeat(args.batch, 1)
    for step in range(1, args.steps + 1):
        template, search, target = synthetic_batch(args.batch, 32, 96, device)
        out = model(template, search, geometry)
        losses = model.loss(out, target)
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step == 1 or step % max(1, args.steps // 8) == 0 or step == args.steps:
            err = (out["delta"].detach()[:, :2] - target[:, :2]).norm(dim=1).mean()
            print(f"[step {step:04d}/{args.steps}] loss={losses['total'].item():.4f} "
                  f"loc={losses['localization'].item():.4f} err={err.item():.4f}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out_path)
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
