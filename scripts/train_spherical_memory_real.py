#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 360VOT GT 相邻帧训练 GRT360-Spherical-Memory。

这是架构训练入口，不读取 ODTrack/UETrack/LightFC 预测，因此不会把结果级
融合伪装成模型训练。默认只取少量序列和步数，便于先做端到端管线验证。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from panotrack.data.vot360 import find_sequences, load_vot360_annotations  # noqa: E402
from panotrack.models.spherical_memory import SphericalMemoryNet  # noqa: E402


def erp_crop(image, center_x, center_y, crop_w, crop_h, out_size):
    h, w = image.shape[:2]
    u = (np.arange(out_size, dtype=np.float32) + 0.5) / out_size - 0.5
    v = (np.arange(out_size, dtype=np.float32) + 0.5) / out_size - 0.5
    xs = center_x + u * crop_w
    ys = center_y + v * crop_h
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    wx, wy = xs - x0, ys - y0
    x0 = np.mod(x0, w); x1 = np.mod(x0 + 1, w)
    y0 = np.clip(y0, 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    a = image[np.ix_(y0, x0)].astype(np.float32)
    b = image[np.ix_(y0, x1)].astype(np.float32)
    c = image[np.ix_(y1, x0)].astype(np.float32)
    d = image[np.ix_(y1, x1)].astype(np.float32)
    out = (a * (1 - wx[None, :, None]) * (1 - wy[:, None, None])
           + b * wx[None, :, None] * (1 - wy[:, None, None])
           + c * (1 - wx[None, :, None]) * wy[:, None, None]
           + d * wx[None, :, None] * wy[:, None, None])
    return (out / 255.0).transpose(2, 0, 1).astype(np.float32)


def build_pairs(data, max_seqs, max_pairs, downscale, seed):
    rng = random.Random(seed)
    seqs = find_sequences(data)[:max_seqs]
    pairs = []
    for seq in seqs:
        paths, gt = load_vot360_annotations(seq)
        n = min(len(paths), len(gt))
        for i in range(max(0, n - 1)):
            pairs.append((paths[i], paths[i + 1], gt[i], gt[i + 1], downscale))
    rng.shuffle(pairs)
    return pairs[:max_pairs]


def sample(pair, template_size, search_size):
    p0, p1, b0, b1, scale = pair
    with Image.open(p0) as im0, Image.open(p1) as im1:
        im0 = np.asarray(im0.convert("RGB").resize(
            (round(im0.width * scale), round(im0.height * scale)), Image.Resampling.BILINEAR))
        im1 = np.asarray(im1.convert("RGB").resize(
            (round(im1.width * scale), round(im1.height * scale)), Image.Resampling.BILINEAR))
    b0, b1 = np.asarray(b0, dtype=np.float32) * scale, np.asarray(b1, dtype=np.float32) * scale
    h, w = im0.shape[:2]
    c0 = np.array([b0[0] + b0[2] / 2, b0[1] + b0[3] / 2], dtype=np.float32)
    c1 = np.array([b1[0] + b1[2] / 2, b1[1] + b1[3] / 2], dtype=np.float32)
    search_w = max(4.0 * b0[2], 4.0 * b0[3], 32.0)
    search_h = max(4.0 * b0[2], 4.0 * b0[3], 32.0)
    template = erp_crop(im0, c0[0], c0[1], max(2.0 * b0[2], 8.0),
                        max(2.0 * b0[3], 8.0), template_size)
    search = erp_crop(im1, c0[0], c0[1], search_w, search_h, search_size)
    dx = ((c1[0] - c0[0] + w / 2.0) % w - w / 2.0) / (search_w / 2.0)
    dy = (c1[1] - c0[1]) / (search_h / 2.0)
    target = np.array([dx, dy, np.log(max(b1[2], 2.0) / max(b0[2], 2.0)),
                       np.log(max(b1[3], 2.0) / max(b0[3], 2.0))], dtype=np.float32)
    lon = (c0[0] / w) * 360.0 - 180.0
    lat = 90.0 - (c0[1] / h) * 180.0
    geometry = np.array([lon, lat, search_w / w * 360.0,
                         search_h / h * 180.0], dtype=np.float32)
    return template, search, target, geometry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=512)
    p.add_argument("--downscale", type=float, default=0.25)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="artifacts/grt360_spherical_memory.pt")
    p.add_argument("--seed", type=int, default=20260810)
    args = p.parse_args(argv)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = build_pairs(args.data, args.max_seqs, args.max_pairs, args.downscale, args.seed)
    if not pairs:
        raise SystemExit("没有找到可训练的 360VOT 相邻帧")
    # 只解码一次。反复从 4K 原图取 patch 会让 GPU 长时间空转，浪费两张卡。
    print(f"preparing cached patches: {len(pairs)}", flush=True)
    cached = [sample(x, 64, 128) for x in pairs]
    print("patch cache ready", flush=True)
    model = SphericalMemoryNet(channels=24).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        random.shuffle(cached)
        total = 0.0
        for start in range(0, len(cached), args.batch):
            batch = cached[start:start + args.batch]
            t, s, y, g = zip(*batch)
            t = torch.tensor(np.stack(t), device=device)
            s = torch.tensor(np.stack(s), device=device)
            y = torch.tensor(np.stack(y), device=device)
            g = torch.tensor(np.stack(g), device=device)
            out = model(t, s, g)
            losses = model.loss(out, y)
            opt.zero_grad(set_to_none=True); losses["total"].backward(); opt.step()
            total += float(losses["total"].item()) * len(batch)
        avg = total / len(cached); history.append(avg)
        print(f"[epoch {epoch}/{args.epochs}] pairs={len(cached)} loss={avg:.5f}", flush=True)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out_path)
    out_path.with_suffix(".json").write_text(json.dumps({"args": vars(args), "loss": history},
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
