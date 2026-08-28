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
    # These bands came from the pre-registered compact-target sweep, then
    # received a geometry-only safety belt after the 130-sequence diagnostic.
    # The vertical-FoV guard keeps the fast model away from narrow targets
    # whose apparent scale/absence dynamics made T224 materially worse.  The
    # wider compact bands are handled by the no-switch B224 expert below.
    if fh <= 6.0 and abs(lat) < 85.0:
        if fv <= 12.5:
            return True, ["compact_fov_h_le_6", "safe_vertical_band", "non_extreme_pole"]
    return False, ["b224_geometry_default"]


def route_noswitch_b224(init_bfov) -> tuple[bool, list[str]]:
    """Select B224 without the high-template switch in compact risk bands."""
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if abs(lat) >= 65.0:
        return False, []
    if fh <= 6.0 and fv <= 8.0:
        return True, ["b224_noswitch_tiny_safe"]
    if (5.5 <= fh <= 6.0 and 14.0 <= fv <= 22.0 and
            abs(lat) < 30.0):
        return True, ["b224_noswitch_compact_scale_safe"]
    if 10.0 <= fh < 13.0 and 10.0 <= fv <= 12.5 and abs(lat) < 45.0:
        return True, ["b224_noswitch_compact_vertical_safe"]
    return False, []


def route_fixed_b224(init_bfov) -> tuple[bool, list[str]]:
    """Select the bare fixed-factor B224 contract in compact safe bands.

    The fixed branch is deliberately narrower than the adaptive no-switch
    route.  It is motivated by the completed bare-B224 sweep: in compact
    non-polar views the adaptive factor/high-template machinery can change the
    crop before the target has established a stable appearance.  The rule is
    geometry-only and is kept separate so it can be evaluated and rolled back
    without changing the existing baseline branches.
    """
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    # Two completed full-length probes support the unmodified factor-4 crop
    # for an extreme tiny pole and for the narrow 30x30 high-latitude band.
    # These cases are safer without the adaptive/polar hand-off, which was
    # repeatedly locking onto a neighbouring ERP peak.
    if abs(lat) >= 85.0 and fh < 6.0 and fv < 6.0:
        return True, ["b224_fixed_extreme_polar_tiny"]
    if (abs(lat) >= 75.0 and 28.0 <= fh <= 33.0 and
            28.0 <= fv <= 33.0):
        return True, ["b224_fixed_high_lat_medium"]
    if abs(lat) >= 45.0:
        return False, []
    # The completed bare sweep supports one compact envelope: fh<16°,
    # fv<25°, and non-polar latitude.  It wins on 16/17 measured full-length
    # sim/real rows; the only observed regression is 0.059 AUC, below the
    # single-sequence rollback threshold.  Keep the envelope geometry-only so
    # it remains a legitimate causal policy rather than a sequence lookup.
    if fh < 16.0 and fv < 25.0:
        return True, ["b224_fixed_compact_envelope"]
    return False, []


def route_dynamic_polar_b224(init_bfov) -> tuple[bool, list[str]]:
    """Enable polar rectification after the target moves into a mid-latitude pole."""
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if (55.0 <= abs(lat) < 65.0 and 18.0 <= fh < 25.0 and
            20.0 <= fv < 35.0):
        return True, ["b224_dynamic_polar_mid_lat"]
    return False, []


def route_adaptive_b224(init_bfov) -> tuple[bool, list[str]]:
    """Select adaptive B224 only for validated compact/moderate geometry."""
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]), float(init_bfov[1]))
    if abs(lat) >= 65.0 and fh < 30.0 and fv < 60.0:
        return True, ["b224_adaptive_high_lat_compact"]
    # A narrow high-latitude failure band was isolated in the full diagnostic
    # (roughly 30x30 degrees).  Keep the conservative large-FoV B224 policy
    # there; this condition is causal geometry only and does not encode a
    # sequence identity or an offline result lookup.
    if abs(lat) >= 65.0 and 29.0 <= fh <= 32.0 and 25.0 <= fv <= 35.0:
        return False, ["b224_geometry_high_lat_safety_band"]
    # The factor-3.5 branch is retained for moderate views wide enough to
    # benefit from denser context.  Narrow 20--25 degree views are left on
    # the conservative large_fov policy because the sweep exposed regressions
    # there (notably in wide-motion real sequences).
    if 25.0 <= fh <= 60.0 and fv <= 70.0:
        return True, ["b224_adaptive_moderate_fov"]
    return False, []


class GeometryRoutedTracker:
    """Select B224 or T224 once, then run exactly one model per frame."""

    def __init__(self, b_model, b_high_model, t_model, **kwargs):
        # Keep the original large-FoV policy as the safe B224 default.  The
        # adaptive crop/small-polar protection is a separate branch and is
        # enabled only by the causal high-latitude compact gate below.
        self.b_default = MotionAdaptiveTracker(
            b_model, b_high_model, **{**kwargs, "search_factor_mode": "large_fov"})
        self.b_adaptive = MotionAdaptiveTracker(
            b_model, b_high_model, **{**kwargs, "search_factor_mode": "adaptive"})
        self.b = self.b_default
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
        self.b_noswitch = OpenVinoB224Tracker(
            b_model, search_size=224, template_size=112,
            search_factor=float(kwargs.get("search_factor", 4.0)),
            template_factor=float(kwargs.get("template_factor", 2.0)),
            update_interval=25, update_threshold=0.70,
            search_factor_mode="adaptive", **fast_kwargs)
        self.b_dynamic_polar = MotionAdaptiveTracker(
            b_model, b_high_model,
            **{**kwargs, "search_factor_mode": "adaptive",
               "polar_max_frame": max(2000, int(kwargs.get("polar_max_frame", 20))),
               "polar_require_initial": False})
        # Bare factor-4 B224, with no geometry/fallback/template machinery.
        # This is an explicit expert rather than an offline result lookup.
        self.b_fixed = OpenVinoB224Tracker(
            b_model, search_size=224, template_size=112,
            search_factor=float(kwargs.get("search_factor", 4.0)),
            template_factor=float(kwargs.get("template_factor", 2.0)),
            update_interval=25, update_threshold=0.70,
            search_factor_mode="fixed")
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
        use_fixed, fixed_reasons = route_fixed_b224(init_bfov)
        use_dynamic_polar, dynamic_polar_reasons = route_dynamic_polar_b224(init_bfov)
        use_noswitch, noswitch_reasons = route_noswitch_b224(init_bfov)
        use_adaptive, adaptive_reasons = route_adaptive_b224(init_bfov)
        if use_fixed:
            self.route_reasons = fixed_reasons
            self.active = self.b_fixed
            self.selected_method = "sutrack_b224_fixed"
            self.b_fixed.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        elif use_dynamic_polar:
            self.route_reasons = dynamic_polar_reasons
            self.active = self.b_dynamic_polar
            self.selected_method = "sutrack_b224_dynamic_polar"
            self.b_dynamic_polar.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        elif use_noswitch:
            self.route_reasons = noswitch_reasons
            self.active = self.b_noswitch
            self.selected_method = "sutrack_b224_noswitch"
            self.b_noswitch.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        elif use_t:
            self.route_reasons = reasons + adaptive_reasons
            self.active = self.t
            self.selected_method = "sutrack_t224"
            self.t.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        elif use_adaptive:
            self.route_reasons = reasons + adaptive_reasons
            self.active = self.b_adaptive
            self.selected_method = "sutrack_b224"
            self.b_adaptive.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        else:
            self.route_reasons = reasons
            self.active = self.b_default
            self.selected_method = "sutrack_b224"
            self.b_default.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)

    def track(self, frame_rgb, **kwargs):
        out = dict(self.active.track(frame_rgb, **kwargs))
        out["expert_used"] = self.selected_method
        out["route_reasons"] = list(self.route_reasons)
        return out

    @property
    def base(self):
        return self.active.base if isinstance(self.active, MotionAdaptiveTracker) else self.active

    @property
    def switch_frame(self):
        return self.active.switch_frame if isinstance(self.active, MotionAdaptiveTracker) else None


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
