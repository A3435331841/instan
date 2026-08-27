#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU feasibility smoke test for an XMem-style mask expert on 360 ERP data.

This is deliberately an experiment adapter, not a submission tracker.  It
uses only the protocol init BFoV to seed a rectangular mask, propagates that
mask with XMem memory, and converts the predicted mask back to a seam-aware
ERP box.  The output is useful for deciding whether a low-frequency mask
expert is worth integrating behind B224's reliability gate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XMEM_DEFAULT = PROJECT_ROOT.parent / "grt360_scratch" / "research" / "XMem"
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov  # noqa: E402


def _wrapped_box_mask(box, height: int, width: int) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    mask = np.zeros((height, width), dtype=np.uint8)
    if w <= 0 or h <= 0:
        return mask
    y0 = max(0, int(np.floor(y)))
    y1 = min(height, int(np.ceil(y + h)))
    if y1 <= y0:
        return mask
    x0 = int(np.floor(x)) % width
    x1 = x0 + int(np.ceil(w))
    if x1 <= width:
        mask[y0:y1, x0:x1] = 1
    else:
        mask[y0:y1, x0:width] = 1
        mask[y0:y1, :x1 - width] = 1
    return mask


def _mask_to_wrapped_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    height, width = mask.shape
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    cols = np.unique(xs)
    if cols.size == 1:
        x0, w = int(cols[0]), 1
    else:
        extended = np.concatenate([cols, [cols[0] + width]])
        gaps = np.diff(extended)
        cut = int(np.argmax(gaps))
        start = int(cols[(cut + 1) % cols.size])
        end = int(cols[cut]) + (width if cut < cols.size - 1 else 0)
        # If the largest gap is the wrap gap, ``end`` already lives in the
        # next turn; otherwise shift the endpoint into the same turn.
        if end < start:
            end += width
        x0, w = start % width, max(1, end - start + 1)
    return [float(x0), float(y0), float(w), float(y1 - y0)]


def _frame_tensor(frame_rgb: np.ndarray, height: int) -> torch.Tensor:
    height = int(height)
    width = int(round(height * frame_rgb.shape[1] / frame_rgb.shape[0]))
    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype)[:, None, None]
    return (t - mean) / std


def _load_init(seq_dir: Path) -> tuple[float, float, float, float]:
    vals = [float(x) for x in (seq_dir / "init.txt").read_text(encoding="utf-8").strip().split(",")]
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise ValueError(f"invalid init.txt: {seq_dir / 'init.txt'}")
    return tuple(vals)  # type: ignore[return-value]


def _load_xmem(xmem_root: Path, model_path: Path, size: int, mem_every: int):
    sys.path.insert(0, str(xmem_root))
    from model.network import XMem  # pylint: disable=import-outside-toplevel
    from inference.inference_core import InferenceCore  # pylint: disable=import-outside-toplevel

    config = {
        # The released XMem.pth is the multi-object checkpoint (its value
        # encoder expects RGB + mask + other-mask = 5 channels).  We still
        # propagate one object, but must instantiate the matching architecture.
        "single_object": False,
        "enable_long_term": True,
        "enable_long_term_count_usage": False,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": int(mem_every),
        "deep_update_every": -1,
        "size": int(size),
    }
    network = XMem(config, str(model_path), map_location="cpu").cpu().eval()
    return network, InferenceCore, config


def run(args: argparse.Namespace) -> dict:
    data_root = Path(args.data).resolve()
    seq_dir = data_root / Path(args.seq)
    video_path = seq_dir / "video.mp4"
    init = _load_init(seq_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_w <= 0 or source_h <= 0:
        raise RuntimeError(f"invalid video dimensions: {video_path}")
    target_h = int(args.size)
    target_w = int(round(target_h * source_w / source_h))
    init_box = erp_bbox_from_bfov(BFoV(*init), source_w, source_h)
    seed_mask = _wrapped_box_mask(
        [init_box[0] * target_w / source_w, init_box[1] * target_h / source_h,
         init_box[2] * target_w / source_w, init_box[3] * target_h / source_h],
        target_h, target_w)

    model_path = Path(args.model).resolve()
    xmem_root = Path(args.xmem_root).resolve()
    network, InferenceCore, config = _load_xmem(xmem_root, model_path, target_h, args.mem_every)
    processor = InferenceCore(network, config=config)
    processor.set_all_labels([1])
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    traces = []
    total_s = 0.0
    for frame_idx in range(int(args.max_frames)):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = _frame_tensor(frame_rgb, target_h)
        start = time.perf_counter()
        with torch.inference_mode():
            if frame_idx == 0:
                mask = torch.from_numpy(seed_mask).float().unsqueeze(0)
                prob = processor.step(rgb, mask=mask, valid_labels=[1], end=False)
            else:
                prob = processor.step(rgb, mask=None, valid_labels=None,
                                      end=(frame_idx == int(args.max_frames) - 1))
        elapsed = time.perf_counter() - start
        total_s += elapsed
        prob_obj = prob[1] if prob.ndim == 3 and prob.shape[0] > 1 else prob[0]
        prob_obj = prob_obj.detach().cpu().float()
        binary = (prob_obj.numpy() >= float(args.threshold)).astype(np.uint8)
        small_box = _mask_to_wrapped_box(binary)
        box = [small_box[0] * source_w / target_w, small_box[1] * source_h / target_h,
               small_box[2] * source_w / target_w, small_box[3] * source_h / target_h]
        results.append(box)
        traces.append({
            "frame_index": frame_idx,
            "target_bbox": box,
            "mask_fraction": float(binary.mean()),
            "mask_probability_mean": float(prob_obj.mean()),
            "latency_ms": elapsed * 1000.0,
        })
    cap.release()
    n = len(results)
    payload = {
        "schema": "grt360.xmem_bbox_smoke.v1",
        "sequence": str(args.seq).replace("\\", "/"),
        "model": str(model_path),
        "xmem_root": str(xmem_root),
        "input_resolution": f"{source_w}x{source_h}",
        "inference_resolution": f"{target_w}x{target_h}",
        "frames": n,
        "threshold": float(args.threshold),
        "mem_every": int(args.mem_every),
        "mean_latency_ms": (total_s / n * 1000.0) if n else None,
        "fps": (n / total_s) if total_s > 0 else None,
        "init_bfov": list(init),
        "init_erp_bbox": list(init_box),
        "status": "ok" if n else "empty",
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "results_erp.txt").write_text(
        "\n".join(",".join(f"{v:.6f}" for v in row) for row in results) + "\n",
        encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--xmem-root", default=str(XMEM_DEFAULT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--size", type=int, default=240,
                        help="inference height; ERP width is inferred from the video")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mem-every", type=int, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
