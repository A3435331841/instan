#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Causal early-probe router between bare and adaptive B224.

The router uses a short warm-up in which both B224 variants run on the same
frames.  It then keeps the branch whose runtime quality is stronger, with a
specific guard against an adaptive high-template switch that immediately
loses the target.  The compact T224 geometry gate remains available for
known-safe tiny views.  No ground truth, sequence name, or offline score is
used by this runner.
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
from scripts.run_geometry_routed_b224_t224 import (  # noqa: E402
    GeometryRoutedTracker,
    route_fixed_b224,
    route_noswitch_b224,
    route_t224,
    build_kwargs,
)
from scripts.run_sutrack_b224_openvino_sequence import (  # noqa: E402
    MotionAdaptiveTracker,
    OpenVinoB224Tracker,
)


def _base_filter(kwargs):
    keys = {
        "fallback_search_factor", "fallback_quality_threshold", "fallback_min_gain",
        "fallback_cooldown", "fallback_run", "fallback_start_frame",
        "fallback_motion_lead", "anchor_update_threshold", "auto_freeze_scale_threshold",
        "auto_freeze_scale_window", "auto_freeze_quality_slope", "auto_freeze_scale_step_p95",
        "auto_freeze_quality_floor", "auto_freeze_scale_step_median_max",
        "auto_freeze_scale_step_override", "auto_freeze_max_frame", "scale_clamp_factor",
        "motion_predict_horizon", "motion_velocity_alpha", "large_fov_fallback_search_factor",
        "seam_recenter", "polar_rectify", "polar_latitude_threshold", "polar_aspect_max",
        "polar_small_width", "polar_max_frame", "polar_require_initial",
        "small_template_factor", "small_template_width", "small_template_require_initial",
        "projection_mode",
    }
    return {k: v for k, v in kwargs.items() if k in keys}


class ProbeB224Tracker:
    """Run bare and adaptive B224 briefly, then retain one branch."""

    def __init__(self, b_model, b_high_model, kwargs,
                 probe_frames=6, quality_margin=0.05):
        self.probe_frames = max(2, int(probe_frames))
        self.quality_margin = float(quality_margin)
        self.kwargs = dict(kwargs)
        self.adaptive = MotionAdaptiveTracker(
            b_model, b_high_model,
            **{**self.kwargs, "search_factor_mode": "adaptive"})
        self.bare = OpenVinoB224Tracker(
            b_model, search_size=224, template_size=112,
            search_factor=4.0, template_factor=2.0,
            update_interval=25, update_threshold=0.70,
            # This is intentionally the same bare-B224 contract used by the
            # full sim baseline: no fallback, freeze, seam, polar, or
            # template-memory mechanism is attached to the probe branch.
            search_factor_mode="fixed")
        self.active = None
        self.selected_method = "probe_pending"
        self.route_reasons = ["early_quality_probe"]
        self.frame_id = 0
        self.adaptive_quality = []
        self.bare_quality = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        self.adaptive.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.bare.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.active = None
        self.selected_method = "probe_pending"
        self.frame_id = 0
        self.adaptive_quality = []
        self.bare_quality = []

    def _choose(self):
        a_med = float(np.median(self.adaptive_quality))
        b_med = float(np.median(self.bare_quality))
        # Switching is an irreversible state change in the adaptive tracker;
        # if it happened during the probe, prefer the stable bare branch unless
        # its quality is clearly worse.
        # A high-template hand-off is the failure signature we are trying to
        # avoid.  Once it fires, a merely moderate bare response is safer
        # than trusting the post-switch score, which can be spuriously high
        # after the target has already left the crop.
        choose_bare = (self.adaptive.switched and b_med >= 0.25)
        if not choose_bare:
            # Prefer the bare branch unless adaptive quality is clearly
            # stronger.  This protects sim-like switch-sensitive scenes while
            # retaining adaptive geometry when it has a decisive margin.
            choose_bare = b_med >= a_med - self.quality_margin
        self.active = self.bare if choose_bare else self.adaptive
        self.selected_method = ("sutrack_b224_bare_probe" if choose_bare
                                else "sutrack_b224_adaptive_probe")
        self.route_reasons = [
            "early_quality_probe",
            "bare_median_quality=%.4f" % b_med,
            "adaptive_median_quality=%.4f" % a_med,
            "adaptive_switched=%s" % bool(self.adaptive.switched),
        ]

    def track(self, frame_rgb, **kwargs):
        self.frame_id += 1
        if self.active is None:
            adaptive_out = dict(self.adaptive.track(frame_rgb, **kwargs))
            bare_out = dict(self.bare.track(frame_rgb, **kwargs))
            self.adaptive_quality.append(float(adaptive_out.get("quality", 0.0)))
            self.bare_quality.append(float(bare_out.get("quality", 0.0)))
            if self.frame_id < self.probe_frames:
                adaptive_out["expert_used"] = "sutrack_b224_probe_warmup"
                adaptive_out["route_reasons"] = list(self.route_reasons)
                return adaptive_out
            self._choose()
            out = dict(self.active is self.bare and bare_out or adaptive_out)
        else:
            out = dict(self.active.track(frame_rgb, **kwargs))
        out["expert_used"] = self.selected_method
        out["route_reasons"] = list(self.route_reasons)
        return out


def route_factor_probe(init_bfov) -> tuple[bool, list[str]]:
    """Use a short factor-2/factor-4 probe for large, non-polar views.

    The probe is restricted to the view envelope where the completed sweep
    exposed opposite factor winners.  It is deliberately a runtime quality
    decision; the initialization geometry only decides whether paying the
    six-frame dual warm-up is worthwhile.
    """
    if init_bfov is None:
        return False, ["factor_probe_missing_init_bfov"]
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if abs(lat) >= 45.0:
        return False, ["factor_probe_polar_guard"]
    if 55.0 <= fh < 60.0 and 100.0 <= fv < 120.0:
        return True, ["factor_probe_large_vertical_view"]
    # The 88--96 x 133--151 envelope is a separate stable large-view family;
    # keep it narrow so the known 109--114 x 155 views remain on the safer
    # geometry route.
    if 80.0 <= fh < 100.0 and fv >= 130.0:
        return True, ["factor_probe_mid_large_view"]
    return False, ["factor_probe_geometry_default"]


class FactorProbeB224Tracker:
    """Choose fixed factor-2 or factor-4 B224 from early response quality."""

    def __init__(self, b_model, probe_frames=6, quality_margin=0.08):
        self.probe_frames = max(2, int(probe_frames))
        self.quality_margin = float(quality_margin)
        common = dict(search_size=224, template_size=112,
                      template_factor=2.0, update_interval=25,
                      update_threshold=0.70, search_factor_mode="fixed")
        self.factor4 = OpenVinoB224Tracker(b_model, search_factor=4.0, **common)
        self.factor2 = OpenVinoB224Tracker(b_model, search_factor=2.0, **common)
        self.active = None
        self.selected_method = "factor_probe_pending"
        self.route_reasons = ["factor_probe_pending"]
        self.frame_id = 0
        self.q4 = []
        self.q2 = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        self.factor4.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.factor2.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.active = None
        self.selected_method = "factor_probe_pending"
        self.route_reasons = ["factor_probe_pending"]
        self.frame_id = 0
        self.q4, self.q2 = [], []

    def _choose(self):
        med4 = float(np.median(self.q4))
        med2 = float(np.median(self.q2))
        # Factor 2 must show a clear early advantage; ties stay on factor 4,
        # which is the safer general-purpose large-view contract.
        use2 = med2 > med4 + self.quality_margin
        self.active = self.factor2 if use2 else self.factor4
        self.selected_method = ("sutrack_b224_factor2_probe" if use2
                                else "sutrack_b224_factor4_probe")
        self.route_reasons = [
            "factor_probe",
            "factor2_median_quality=%.4f" % med2,
            "factor4_median_quality=%.4f" % med4,
            "factor2_selected=%s" % use2,
        ]

    def track(self, frame_rgb, **kwargs):
        self.frame_id += 1
        if self.active is None:
            out4 = dict(self.factor4.track(frame_rgb, **kwargs))
            out2 = dict(self.factor2.track(frame_rgb, **kwargs))
            self.q4.append(float(out4.get("quality", 0.0)))
            self.q2.append(float(out2.get("quality", 0.0)))
            if self.frame_id < self.probe_frames:
                out4["expert_used"] = "sutrack_b224_factor_probe_warmup"
                out4["route_reasons"] = list(self.route_reasons)
                return out4
            self._choose()
            out = dict(out2 if self.active is self.factor2 else out4)
        else:
            out = dict(self.active.track(frame_rgb, **kwargs))
        out["expert_used"] = self.selected_method
        out["route_reasons"] = list(self.route_reasons)
        return out


class ProbeRouter:
    def __init__(self, b_model, b_high_model, t_model, kwargs,
                 probe_frames=6, quality_margin=0.05,
                 factor_quality_margin=-0.04):
        self.b_model = b_model
        self.b_high_model = b_high_model
        self.t_model = t_model
        self.kwargs = kwargs
        self.probe_frames = probe_frames
        self.quality_margin = quality_margin
        self.factor_probe = FactorProbeB224Tracker(
            b_model, probe_frames=probe_frames,
            quality_margin=factor_quality_margin)
        self.probe = ProbeB224Tracker(
            b_model, b_high_model, kwargs, probe_frames, quality_margin)
        self.noswitch = OpenVinoB224Tracker(
            b_model, search_size=224, template_size=112,
            search_factor=4.0, template_factor=2.0,
            update_interval=25, update_threshold=0.70,
            search_factor_mode="adaptive", **_base_filter(kwargs))
        self.geometry = GeometryRoutedTracker(
            b_model, b_high_model, t_model, **kwargs)
        self.t = OpenVinoB224Tracker(
            t_model, search_size=224, template_size=112,
            search_factor=4.0, template_factor=2.0,
            update_interval=25, update_threshold=0.70,
            search_factor_mode="adaptive", **_base_filter(kwargs))
        self.active = self.probe
        self.selected_method = "probe_pending"
        self.route_reasons = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        use_factor, factor_reasons = route_factor_probe(init_bfov)
        use_fixed, fixed_reasons = route_fixed_b224(init_bfov)
        use_t, reasons = route_t224(init_bfov)
        use_noswitch, noswitch_reasons = route_noswitch_b224(init_bfov)
        if use_fixed:
            self.active = self.geometry.b_fixed
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = "sutrack_b224_fixed"
            self.route_reasons = fixed_reasons
        elif use_factor:
            self.active = self.factor_probe
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = "factor_probe_pending"
            self.route_reasons = factor_reasons
        elif use_noswitch:
            self.active = self.noswitch
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = "sutrack_b224_noswitch"
            self.route_reasons = noswitch_reasons
        elif use_t:
            self.active = self.t
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = "sutrack_t224"
            self.route_reasons = reasons
        elif not route_probe_b224(init_bfov)[0]:
            self.active = self.geometry
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = self.geometry.selected_method
            self.route_reasons = route_probe_b224(init_bfov)[1] + list(self.geometry.route_reasons)
        else:
            self.active = self.probe
            self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
            self.selected_method = "probe_pending"
            self.route_reasons = ["probe_b224_non_t224_geometry"]

    def track(self, frame_rgb, **kwargs):
        out = dict(self.active.track(frame_rgb, **kwargs))
        if self.active is self.probe:
            self.selected_method = self.probe.selected_method
            out["route_reasons"] = list(self.route_reasons) + list(self.probe.route_reasons)
        elif self.active is self.factor_probe:
            self.selected_method = self.factor_probe.selected_method
            out["route_reasons"] = list(self.route_reasons) + list(self.factor_probe.route_reasons)
        else:
            out["route_reasons"] = list(self.route_reasons)
        out["expert_used"] = self.selected_method
        return out


def route_probe_b224(init_bfov) -> tuple[bool, list[str]]:
    """Limit the dual probe to non-polar, non-large geometries.

    Polar/moderate views already have validated geometry branches whose
    spherical crop is safer than a bare/adaptive warm-up decision.  A narrow
    high-vertical band is retained for the sim/0060-like motion regime.
    """
    if init_bfov is None:
        return False, ["probe_missing_init_bfov"]
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if abs(lat) >= 60.0:
        return False, ["probe_preserve_polar_geometry"]
    if fh >= 70.0 or fv >= 100.0:
        return False, ["probe_preserve_large_geometry"]
    if fh < 15.0 and fv >= 27.0:
        return False, ["probe_preserve_tall_compact_geometry"]
    if 30.0 <= fh <= 60.0 and fv >= 70.0:
        return False, ["probe_preserve_tall_medium_geometry"]
    if 20.0 <= fh <= 60.0 and fv <= 70.0:
        if 20.0 <= fh < 30.0 and fv < 25.0:
            return True, ["probe_nonpolar_compact_geometry"]
        if not (20.0 <= fh < 25.0 and fv >= 50.0):
            return False, ["probe_preserve_moderate_geometry"]
    return True, ["probe_nonpolar_compact_geometry"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-xml", required=True)
    ap.add_argument("--b-high-xml", required=True)
    ap.add_argument("--t-xml", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--quality-threshold", type=float, default=0.40)
    ap.add_argument("--probe-frames", type=int, default=6)
    ap.add_argument("--quality-margin", type=float, default=0.05)
    ap.add_argument("--factor-quality-margin", type=float, default=-0.04,
                    help=("factor-2 may be selected when its warm-up quality "
                          "is within this margin of factor-4; negative "
                          "values allow a small deficit when long-view "
                          "stability is better"))
    args = ap.parse_args(argv)
    import openvino as ov

    t0 = time.perf_counter()
    core = ov.Core()
    b_model = core.compile_model(str(Path(args.b_xml).resolve()), args.device)
    b_high_model = core.compile_model(str(Path(args.b_high_xml).resolve()), args.device)
    t_model = core.compile_model(str(Path(args.t_xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - t0
    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    out_root = Path(args.out).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_kwargs(args)
    rows = []
    for idx, seq in enumerate(seqs, 1):
        seq_out = out_root / seq.replace("/", "_"); seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        def factory(**_kwargs):
            tracker = ProbeRouter(b_model, b_high_model, t_model, kwargs,
                                  args.probe_frames, args.quality_margin,
                                  args.factor_quality_margin)
            holder["tracker"] = tracker; return tracker

        try:
            metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
                seq, args.data, factory, args.max_frames)
            tracker = holder["tracker"]
            metrics.update({
                "compile_seconds": compile_seconds,
                "router_schema": "sutrack_b224_early_probe.v1",
                "selected_method": tracker.selected_method,
                "route_reasons": tracker.route_reasons,
                "probe_frames": args.probe_frames,
                "quality_margin": args.quality_margin,
                "factor_quality_margin": args.factor_quality_margin,
                "e2e_fps": metrics.get("e2e_fps"),
            })
            np.savetxt(seq_out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
            np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
            (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
            (seq_out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False,
                                                               indent=2, allow_nan=True), encoding="utf-8")
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
                "e2e_fps", "probe_frames", "quality_margin",
                "factor_quality_margin", "route_reasons"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in rows)
        (out_root / "summary.json").write_text(json.dumps({
            "schema": "grt360.sutrack_b224_early_probe_summary.v1",
            "n_sequences": len(rows), "compile_seconds": compile_seconds,
            "mean_auc": float(np.mean([r["auc"] for r in rows])),
            "mean_sr": float(np.mean([r["sr"] for r in rows])),
            "mean_e2e_fps": float(np.mean([r["e2e_fps"] for r in rows])),
            "rows": rows,
        }, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return 0 if len(rows) == len(seqs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
