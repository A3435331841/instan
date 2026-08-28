#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Interleave fast B224 and ODTrack calls under one causal frame schedule.

This experiment tests whether a slow, accurate ODTrack expert can be sampled
every N frames while B224 handles the intervening frames.  It is intentionally
an A/B probe: no result table or GT signal chooses the model, and each branch
sees only the frames assigned by the fixed stride.
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

from scripts.eval_official import run_sequence  # noqa: E402
from scripts.run_odtrack_openvino_sequence import OpenVinoODTrackTracker  # noqa: E402
from scripts.run_sutrack_b224_openvino_sequence import OpenVinoB224Tracker  # noqa: E402


class InterleavedTracker:
    def __init__(self, b_model, od_model, od_first_model, stride: int,
                 projection_mode: str = "tangent"):
        self.b224 = OpenVinoB224Tracker(
            b_model, search_size=224, template_size=112,
            search_factor=4.0, template_factor=2.0,
            update_interval=25, update_threshold=0.70,
            search_factor_mode="fixed")
        self.od = OpenVinoODTrackTracker(
            od_model, search_size=384, template_size=192,
            search_factor=5.0, template_factor=2.0,
            update_interval=25, update_threshold=0.55,
            seam_recenter=True, first_compiled_model=od_first_model,
            projection_mode=projection_mode)
        self.stride = max(2, int(stride))
        self.frame_id = 0

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        self.b224.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.od.init(frame_rgb, erp_box, init_bfov=init_bfov)
        self.frame_id = 0

    def track(self, frame_rgb, **kwargs):
        self.frame_id += 1
        if self.frame_id % self.stride == 0:
            out = dict(self.od.track(frame_rgb, **kwargs))
            out["expert_used"] = "odtrack_interleaved"
        else:
            out = dict(self.b224.track(frame_rgb, **kwargs))
            out["expert_used"] = "sutrack_b224_interleaved"
        out["interleave_stride"] = self.stride
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-xml", required=True)
    ap.add_argument("--od-xml", required=True)
    ap.add_argument("--od-first-xml", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--projection-mode", choices=["erp", "tangent"], default="tangent")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)
    import openvino as ov

    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    if not seqs:
        raise SystemExit("no sequences supplied")
    t0 = time.perf_counter()
    core = ov.Core()
    b_model = core.compile_model(str(Path(args.b_xml).resolve()), args.device)
    od_model = core.compile_model(str(Path(args.od_xml).resolve()), args.device)
    od_first_model = core.compile_model(str(Path(args.od_first_xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - t0
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in seqs:
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        def factory(**_kwargs):
            tracker = InterleavedTracker(
                b_model, od_model, od_first_model, args.stride,
                args.projection_mode)
            holder["tracker"] = tracker
            return tracker

        metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
            seq, args.data, factory, args.max_frames)
        metrics.update({
            "router_schema": "grt360.interleaved_b224_od.v1",
            "stride": args.stride,
            "projection_mode": args.projection_mode,
            "compile_seconds": compile_seconds,
            "device": args.device,
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
    keys = ["sequence", "n_frames", "auc", "sr", "e2e_fps", "tracker_fps",
            "stride", "projection_mode"]
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    (out_root / "summary.json").write_text(json.dumps({
        "schema": "grt360.interleaved_b224_od_summary.v1",
        "n_sequences": len(rows), "compile_seconds": compile_seconds,
        "mean_auc": float(np.mean([row["auc"] for row in rows])),
        "mean_sr": float(np.mean([row["sr"] for row in rows])),
        "mean_e2e_fps": float(np.mean([row["e2e_fps"] for row in rows])),
        "rows": rows,
    }, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
