#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal global-motion probe for very large-FoV ERP targets.

This is a deliberately small expert, not a replacement for B224.  For a
near-hemisphere target the dominant signal can be camera/scene translation;
frame-to-frame phase correlation on a low-resolution ERP estimates that
translation without GT, sequence names, or a result table.  The expert keeps
the protocol-initialized angular extent fixed and only transports its centre.
It is intended for a geometry-gated bake-off on very large FoV views.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_official import run_sequence  # noqa: E402


class GlobalPhaseTracker:
    """Transport the initial ERP box with causal global phase motion."""

    def __init__(self, width_px: int = 180, height_px: int = 90,
                 max_shift_px: float = 18.0, smoothing: float = 0.65,
                 use_window: bool = True):
        self.width_px = max(32, int(width_px))
        self.height_px = max(16, int(height_px))
        self.max_shift_px = max(1.0, float(max_shift_px))
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.use_window = bool(use_window)
        self.width = self.height = 0
        self.prev = None
        self.window = None
        self.initial_box = None
        self.center = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self.last_response = 1.0
        self.frame_id = 0

    def _gray(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (self.width_px, self.height_px),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        gray -= float(gray.mean())
        return gray

    def init(self, frame_rgb, erp_box, **_kwargs):
        self.height, self.width = frame_rgb.shape[:2]
        self.initial_box = np.asarray([float(v) for v in erp_box], dtype=np.float64)
        self.center = np.asarray([
            (self.initial_box[0] + 0.5 * self.initial_box[2]) % self.width,
            self.initial_box[1] + 0.5 * self.initial_box[3],
        ], dtype=np.float64)
        self.prev = self._gray(frame_rgb)
        self.window = (cv2.createHanningWindow(
            (self.width_px, self.height_px), cv2.CV_32F)
                       if self.use_window else None)
        self.velocity[:] = 0.0
        self.last_response = 1.0
        self.frame_id = 0

    def track(self, frame_rgb, **_kwargs):
        current = self._gray(frame_rgb)
        shift, response = cv2.phaseCorrelate(self.prev, current, self.window)
        # phaseCorrelate reports low-resolution pixels.  Ignore pathological
        # wrap/blur jumps; valid fast motion is accumulated over subsequent
        # frames instead of teleporting the box across the sphere.
        delta = np.asarray(shift, dtype=np.float64)
        delta[0] *= self.width / float(self.width_px)
        delta[1] *= self.height / float(self.height_px)
        delta = np.clip(delta, -self.max_shift_px *
                         np.asarray([self.width / self.width_px,
                                     self.height / self.height_px]),
                        self.max_shift_px *
                        np.asarray([self.width / self.width_px,
                                    self.height / self.height_px]))
        # Integrate the filtered displacement.  Accumulating every raw phase
        # estimate lets sub-pixel noise drift hundreds of pixels on long
        # videos; smoothing the applied motion is especially important for
        # the 8k-frame real large-FoV sequences.
        self.velocity = self.smoothing * self.velocity + (1.0 - self.smoothing) * delta
        applied = self.velocity.copy()
        self.center[0] = (self.center[0] + applied[0]) % self.width
        self.center[1] = float(np.clip(self.center[1] + applied[1], 0.0, self.height))
        self.prev = current
        self.frame_id += 1
        self.last_response = float(response)
        bw, bh = self.initial_box[2], self.initial_box[3]
        box = [float((self.center[0] - 0.5 * bw) % self.width),
               float(np.clip(self.center[1] - 0.5 * bh, 0.0, self.height)),
               float(bw), float(bh)]
        return {
            "target_bbox": box,
            "quality": float(max(0.0, self.last_response)),
            "status": "normal",
            "expert_used": "global_phase_motion",
            "phase_response": float(self.last_response),
            "phase_shift_x_px": float(applied[0]),
            "phase_shift_y_px": float(applied[1]),
        }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--width", type=int, default=180)
    ap.add_argument("--height", type=int, default=90)
    ap.add_argument("--max-shift", type=float, default=18.0)
    ap.add_argument("--smoothing", type=float, default=0.65)
    ap.add_argument("--no-window", action="store_true",
                    help="disable the Hanning window for the raw phase probe")
    args = ap.parse_args(argv)
    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    for seq in seqs:
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)

        def factory(**_kwargs):
            return GlobalPhaseTracker(args.width, args.height,
                                      args.max_shift, args.smoothing,
                                      not args.no_window)

        metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
            seq, args.data, factory, args.max_frames)
        metrics.update({
            "router_schema": "grt360.global_phase_motion.v1",
            "phase_width": args.width,
            "phase_height": args.height,
            "phase_max_shift": args.max_shift,
            "phase_smoothing": args.smoothing,
        })
        np.savetxt(seq_out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
        np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
        (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
        (seq_out / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        with (seq_out / "trace.jsonl").open("w", encoding="utf-8") as handle:
            for trace in traces:
                handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=True) + "\n")
        rows.append(metrics)
        print(f"{seq}: AUC={metrics['auc']:.4f} SR={metrics['sr']:.4f} "
              f"e2eFPS={metrics['e2e_fps']:.2f}", flush=True)
    if rows:
        keys = ["sequence", "n_frames", "auc", "sr", "e2e_fps", "tracker_fps"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in keys} for row in rows)
        (out_root / "summary.json").write_text(json.dumps({
            "schema": "grt360.global_phase_motion_summary.v1",
            "n_sequences": len(rows),
            "wall_seconds": time.perf_counter() - t0,
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "mean_sr": float(np.mean([row["sr"] for row in rows])),
            "mean_e2e_fps": float(np.mean([row["e2e_fps"] for row in rows])),
            "rows": rows,
        }, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
