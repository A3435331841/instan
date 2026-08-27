#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal geometry router for a B224 main and T224 fast expert.

The router is intentionally small and auditable: the method is selected once
from the protocol init BFoV (angle and latitude), never from sequence names,
GT or an offline result table.  B224 keeps the adaptive geometry/high-template
path; T224 is selected only for compact targets where its dense-token model is
the better inexpensive hypothesis.  This script is an experiment runner and
can be promoted only after locked valid/full evaluation.
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
from scripts.run_sutrack_b224_openvino_sequence import (  # noqa: E402
    MotionAdaptiveTracker,
    OpenVinoB224Tracker,
)


def route_t224(init_bfov) -> tuple[bool, list[str]]:
    """Return a geometry-only T224 decision and an auditable reason."""
    if init_bfov is None:
        return False, ["missing_init_bfov_b224_fallback"]
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]), float(init_bfov[1]))
    # These bands came from the pre-registered 8-sequence factor sweep.  They
    # distinguish compact, non-extreme-pole targets from the high-latitude
    # 8--12-degree regime in which B224's memory branch is more reliable.
    if fh <= 6.0 and abs(lat) < 85.0:
        return True, ["compact_fov_h_le_6", "non_extreme_pole"]
    if 10.0 <= fh < 15.0 and abs(lat) < 65.0:
        return True, ["compact_fov_h_10_15", "non_polar"]
    return False, ["b224_geometry_default"]


class GeometryRoutedTracker:
    """Select B224 or T224 once, then run exactly one model per frame."""

    def __init__(self, b_model, b_high_model, t_model, **kwargs):
        self.b = MotionAdaptiveTracker(b_model, b_high_model, **kwargs)
        tracker_keys = {
            "fallback_search_factor", "fallback_quality_threshold", "fallback_min_gain",
            "fallback_cooldown", "fallback_run", "fallback_start_frame",
            "fallback_motion_lead", "anchor_update_threshold", "auto_freeze_scale_threshold",
            "auto_freeze_scale_window", "auto_freeze_quality_slope", "auto_freeze_scale_step_p95",
            "auto_freeze_quality_floor", "auto_freeze_scale_step_median_max",
            "auto_freeze_scale_step_override", "auto_freeze_max_frame", "scale_clamp_factor",
            "motion_predict_horizon", "motion_velocity_alpha", "large_fov_fallback_search_factor",
            "seam_recenter", "polar_rectify", "polar_latitude_threshold", "polar_aspect_max",
            "polar_small_width", "polar_max_frame", "polar_require_initial", "small_template_factor",
            "small_template_width", "small_template_require_initial", "projection_mode",
        }
        fast_kwargs = {k: v for k, v in kwargs.items() if k in tracker_keys}
        # T224 IR has the same 224 search and 112 template tensor contract;
        # its own adaptive factor logic is retained, but no B high-template
        # model is called on this branch.
        self.t = OpenVinoB224Tracker(
            t_model, search_size=224, template_size=112,
            search_factor=4.0, template_factor=2.0,
            update_interval=25, update_threshold=0.70,
            search_factor_mode="adaptive", **fast_kwargs)
        self.active = self.b
        self.selected_method = "sutrack_b224"
        self.route_reasons = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        use_t, reasons = route_t224(init_bfov)
        self.route_reasons = reasons
        if use_t:
            self.active = self.t
            self.selected_method = "sutrack_t224"
            self.t.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        else:
            self.active = self.b
            self.selected_method = "sutrack_b224"
            self.b.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)

    def track(self, frame_rgb, **kwargs):
        out = dict(self.active.track(frame_rgb, **kwargs))
        out["expert_used"] = self.selected_method
        out["route_reasons"] = list(self.route_reasons)
        return out

    @property
    def base(self):
        return self.b.base

    @property
    def switch_frame(self):
        return self.b.switch_frame if self.selected_method == "sutrack_b224" else None


def build_kwargs(args):
    return dict(
        warmup=5,
        threshold_deg=1.5,
        quality_threshold=args.quality_threshold,
        quality_run=3,
        switch_deadline=30,
        search_factor=4.0,
        search_factor_mode="adaptive",
        fallback_search_factor=3.25,
        fallback_quality_threshold=0.4,
        fallback_min_gain=-1.0,
        fallback_cooldown=10,
        fallback_run=3,
        fallback_start_frame=20,
        fallback_motion_lead=0.0,
        anchor_update_threshold=None,
        auto_freeze_scale_threshold=0.25,
        auto_freeze_scale_window=40,
        auto_freeze_quality_slope=-0.003,
        auto_freeze_scale_step_p95=0.045,
        auto_freeze_quality_floor=0.75,
        auto_freeze_scale_step_median_max=0.018,
        auto_freeze_scale_step_override=None,
        auto_freeze_max_frame=100,
        scale_clamp_factor=None,
        motion_predict_horizon=0.0,
        motion_velocity_alpha=0.4,
        large_fov_fallback_search_factor=5.0,
        seam_recenter=True,
        polar_rectify=True,
        polar_latitude_threshold=55.0,
        polar_aspect_max=2.5,
        polar_small_width=100.0,
        polar_max_frame=20,
        polar_require_initial=True,
        small_template_factor=1.5,
        small_template_width=100.0,
        small_template_require_initial=True,
        projection_mode="erp",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-xml", required=True)
    parser.add_argument("--b-high-xml", required=True)
    parser.add_argument("--t-xml", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seqs", required=True,
                        help="comma-separated paths such as train_sim/seq_0002")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--quality-threshold", type=float, default=0.4)
    args = parser.parse_args(argv)
    import openvino as ov

    compile_t0 = time.perf_counter()
    core = ov.Core()
    b_model = core.compile_model(str(Path(args.b_xml).resolve()), args.device)
    b_high_model = core.compile_model(str(Path(args.b_high_xml).resolve()), args.device)
    t_model = core.compile_model(str(Path(args.t_xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - compile_t0
    seqs = [x.strip().replace("\\", "/") for x in args.seqs.split(",") if x.strip()]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, seq in enumerate(seqs, 1):
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}
        kwargs = build_kwargs(args)

        def factory(**_kwargs):
            tracker = GeometryRoutedTracker(b_model, b_high_model, t_model, **kwargs)
            holder["tracker"] = tracker
            return tracker

        try:
            metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
                seq, args.data, factory, args.max_frames)
            tracker = holder["tracker"]
            metrics.update({
                "selected_method": tracker.selected_method,
                "route_reasons": tracker.route_reasons,
                "compile_seconds": compile_seconds,
                "router_schema": "geometry_b224_t224.v1",
                "search_factor_mode": "adaptive",
                "e2e_fps": metrics.get("e2e_fps"),
            })
            np.savetxt(seq_out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
            np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
            (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
            (seq_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            with (seq_out / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=True) + "\n")
            rows.append(metrics)
            print(f"[{idx}/{len(seqs)}] {seq}: {tracker.selected_method} "
                  f"AUC={metrics['auc']:.4f} SR={metrics['sr']:.4f} "
                  f"e2eFPS={metrics['e2e_fps']:.2f}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(seqs)}] {seq}: FAILED {exc}", file=sys.stderr, flush=True)
    if rows:
        keys = ["sequence", "selected_method", "n_frames", "n_scored", "auc", "sr",
                "e2e_fps", "route_reasons", "switch_frame", "fallback_calls"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in rows)
        (out_root / "summary.json").write_text(json.dumps({
            "schema": "grt360.geometry_router_summary.v1",
            "n_sequences": len(rows),
            "compile_seconds": compile_seconds,
            "mean_auc": float(np.mean([r["auc"] for r in rows])),
            "mean_sr": float(np.mean([r["sr"] for r in rows])),
            "mean_e2e_fps": float(np.mean([r["e2e_fps"] for r in rows])),
            "rows": rows,
        }, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return 0 if len(rows) == len(seqs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
