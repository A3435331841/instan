#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Geometry router with an eBFoV-only sparse ODTrack recovery branch.

Normal geometry uses the v4 causal B224/T224 router.  Initial views with a
large vertical/horizontal FoV can instead use B224 plus the sparse OD tangent
recovery wrapper.  The branch is selected only from init BFoV and all later
expert decisions use runtime quality/state signals.
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
    build_kwargs,
)
from scripts.run_probe_b224 import ProbeRouter  # noqa: E402
from scripts.run_sutrack_b224_redetect import (  # noqa: E402
    SphericalB224RedetectTracker,
)
from scripts.run_odtrack_openvino_sequence import OpenVinoODTrackTracker  # noqa: E402


def route_recovery(init_bfov) -> tuple[bool, list[str]]:
    """Select the recovery branch from initialization geometry only."""
    if init_bfov is None:
        return False, ["missing_init_bfov"]
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    # The long eBFoV probe is useful when the horizontal view is already
    # broad (or the vertical view is near a hemisphere).  A separate
    # high-latitude medium-view band covers the observed small/scale loss
    # regime without sending ordinary 50--60° views through the slow expert.
    if fh >= 70.0 or fv >= 130.0:
        return True, ["large_fov_sparse_od_recovery"]
    if 30.0 <= fh <= 50.0 and 65.0 <= fv <= 100.0 and abs(lat) >= 40.0:
        return True, ["high_lat_medium_sparse_od_recovery"]
    if 60.0 <= fh < 80.0 and 75.0 <= fv < 100.0 and abs(lat) >= 40.0:
        return True, ["high_lat_broad_sparse_od_recovery"]
    return False, ["geometry_b224_t224_router"]


def route_direct_od(init_bfov) -> tuple[bool, list[str]]:
    """Select a direct tangent OD expert for validated geometry bands.

    The narrow compact and mid-large bands are retained as opt-in quality
    experts.  Their local iGPU latency is recorded in the bake-off; a 5090
    deployment must still repeat the end-to-end speed check.
    """
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if 6.0 <= fh < 7.0 and 10.0 <= fv < 15.0 and abs(lat) >= 60.0:
        return True, ["tiny_polar_direct_od_tangent"]
    # A separate 9x9 high-latitude family is the only completed full-length
    # tiny-polar case where the tangent OD expert beats B224 materially.  Keep
    # its geometry envelope narrow; nearby 4--8° targets have validated T/B
    # routes and must not inherit the ~20 FPS OD path blindly.
    if 8.5 <= fh < 10.0 and 8.0 <= fv < 10.5 and 65.0 <= abs(lat) < 69.9:
        return True, ["tiny_polar_9deg_direct_od"]
    if 8.0 <= fh < 10.0 and 25.0 <= fv < 32.0 and 3.0 <= abs(lat) < 20.0:
        return True, ["compact_direct_od_tangent"]
    if 20.5 <= fh < 23.0 and 45.0 <= fv < 50.0 and abs(lat) < 35.0:
        return True, ["mid_large_direct_od_tangent"]
    return False, []


def route_narrow_recovery(init_bfov) -> tuple[bool, list[str]]:
    """Select only geometry families with a completed full-sequence rescue.

    The broad sparse-OD policy was useful for exploratory 450-frame screens,
    but its false accepts are not safe to promote.  These envelopes are
    deliberately expressed only in the initialization BFoV and were chosen
    from paired full-sequence runs: the 110x155 eBFoV drift family, the nearly
    hemispherical near-equator family, and the 69x83 high-latitude family.
    ``--narrow-recovery-only`` makes this conservative policy explicit while
    retaining the older broad mode for historical experiments.
    """
    if init_bfov is None:
        return False, []
    fh, fv, lat = (float(init_bfov[2]), float(init_bfov[3]),
                   float(init_bfov[1]))
    if 100.0 <= fh <= 120.0 and 150.0 <= fv <= 160.0 and abs(lat) <= 35.0:
        return True, ["narrow_long_ebfov_redetect"]
    # Keep the near-equator hemispherical branch narrow: the nearby -4°
    # sequence is a different motion regime and has no validated rescue yet.
    if fh >= 175.0 and fv >= 175.0 and abs(lat) < 2.0:
        return True, ["narrow_hemisphere_equator_redetect"]
    if 65.0 <= fh <= 72.0 and 80.0 <= fv <= 86.0 and 45.0 <= abs(lat) <= 60.0:
        return True, ["narrow_high_lat_medium_redetect"]
    return False, []


class GeometryRecoveryTracker:
    def __init__(self, b_model, b_high_model, t_model, od_model,
                 od_first_model, tracker_kwargs, args, enable_recovery=True,
                 narrow_recovery_only=False):
        # Keep the latest causal B224 probe/fixed/polar policy on the normal
        # path; direct OD remains an explicitly geometry-gated exception.
        self.geometry = ProbeRouter(
            b_model, b_high_model, t_model, tracker_kwargs,
            probe_frames=6, quality_margin=0.05,
            factor_quality_margin=-0.04)
        self.recovery = (SphericalB224RedetectTracker(
            b_model, b_high_model, tracker_kwargs,
            run_len=args.run_len,
            search_interval=args.search_interval,
            min_score=args.min_score,
            anchor_min_similarity=args.anchor_min_similarity,
            max_motion_deg=args.max_motion_deg,
            erp_downscale=args.erp_downscale,
            od_model=od_model,
            od_first_model=od_first_model,
            od_projection=args.od_projection,
            od_cadence=args.od_cadence,
            od_lost_cadence=args.od_lost_cadence,
            od_quality_threshold=args.od_quality_threshold)
                       if od_model is not None and enable_recovery else None)
        self.od_direct = (OpenVinoODTrackTracker(
            od_model, search_size=384, template_size=192,
            search_factor=5.0, template_factor=2.0,
            update_interval=25, update_threshold=0.55,
            seam_recenter=True, first_compiled_model=od_first_model,
            projection_mode="tangent") if od_model is not None else None)
        # The tiny-polar direct band was measured with the native ERP OD
        # graph: it is both faster and marginally more accurate than tangent
        # remapping on this geometry.  Keep both protocol variants compiled
        # once and choose between them from the causal init-BFoV rule.
        self.od_direct_erp = (OpenVinoODTrackTracker(
            od_model, search_size=384, template_size=192,
            search_factor=5.0, template_factor=2.0,
            update_interval=25, update_threshold=0.55,
            seam_recenter=True, first_compiled_model=od_first_model,
            projection_mode="erp") if od_model is not None else None)
        self.active = self.geometry
        self.selected_method = "geometry_b224_t224"
        self.route_reasons = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        use_direct, direct_reasons = route_direct_od(init_bfov)
        if narrow_recovery_only:
            use_recovery, reasons = route_narrow_recovery(init_bfov)
        else:
            use_recovery, reasons = route_recovery(init_bfov)
        if use_direct and self.od_direct is not None:
            use_erp = any("tiny_polar" in reason for reason in direct_reasons)
            self.active = self.od_direct_erp if use_erp else self.od_direct
            self.selected_method = "odtrack_erp_direct" if use_erp else "odtrack_tangent_direct"
            self.route_reasons = direct_reasons
        elif use_recovery and self.recovery is not None:
            self.active = self.recovery
            self.selected_method = "b224_od_recovery"
            self.route_reasons = reasons
        else:
            self.active = self.geometry
            self.selected_method = "geometry_b224_t224"
            self.route_reasons = reasons
        self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)

    def track(self, frame_rgb, **kwargs):
        if self.active is self.geometry:
            self.selected_method = self.geometry.selected_method
        out = dict(self.active.track(frame_rgb, **kwargs))
        out["expert_used"] = out.get("expert_used", self.selected_method)
        out["route_reasons"] = list(self.route_reasons)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-xml", required=True)
    ap.add_argument("--b-high-xml", required=True)
    ap.add_argument("--t-xml", required=True)
    ap.add_argument("--od-xml", default=None)
    ap.add_argument("--od-first-xml", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--quality-threshold", type=float, default=0.40)
    ap.add_argument("--run-len", type=int, default=5)
    ap.add_argument("--search-interval", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--anchor-min-similarity", type=float, default=0.38)
    ap.add_argument("--max-motion-deg", type=float, default=180.0)
    ap.add_argument("--erp-downscale", type=int, default=4)
    ap.add_argument("--od-projection", choices=["erp", "tangent"], default="tangent")
    ap.add_argument("--od-cadence", type=int, default=30)
    ap.add_argument("--od-lost-cadence", type=int, default=5)
    ap.add_argument("--od-quality-threshold", type=float, default=0.45)
    ap.add_argument("--direct-only", action="store_true",
                    help="disable broad sparse-recovery routing; keep only narrow direct-OD geometry gates")
    ap.add_argument("--narrow-recovery-only", action="store_true",
                    help="use only completed full-sequence geometry rescue envelopes; disable broad exploratory recovery")
    args = ap.parse_args(argv)
    import openvino as ov

    t0 = time.perf_counter()
    core = ov.Core()
    b_model = core.compile_model(str(Path(args.b_xml).resolve()), args.device)
    b_high_model = core.compile_model(str(Path(args.b_high_xml).resolve()), args.device)
    t_model = core.compile_model(str(Path(args.t_xml).resolve()), args.device)
    od_model = (core.compile_model(str(Path(args.od_xml).resolve()), args.device)
                if args.od_xml else None)
    od_first_model = (core.compile_model(str(Path(args.od_first_xml).resolve()), args.device)
                      if args.od_first_xml else None)
    compile_seconds = time.perf_counter() - t0
    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tracker_kwargs = build_kwargs(args)
    rows = []
    for idx, seq in enumerate(seqs, 1):
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        def factory(**_kwargs):
            tracker = GeometryRecoveryTracker(
                b_model, b_high_model, t_model, od_model, od_first_model,
                tracker_kwargs, args, enable_recovery=not args.direct_only,
                narrow_recovery_only=args.narrow_recovery_only)
            holder["tracker"] = tracker
            return tracker

        try:
            metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
                seq, args.data, factory, args.max_frames)
            tracker = holder["tracker"]
            metrics.update({
                "compile_seconds": compile_seconds,
                "router_schema": "geometry_b224_t224_od_recovery.v1",
                "selected_method": tracker.selected_method,
                "route_reasons": tracker.route_reasons,
                "e2e_fps": metrics.get("e2e_fps"),
                "od_calls": (tracker.recovery.od_calls
                              if tracker.recovery is not None else 0),
                "od_selected": (tracker.recovery.od_selected
                                 if tracker.recovery is not None else 0),
                "narrow_recovery_only": bool(args.narrow_recovery_only),
            })
            np.savetxt(seq_out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
            np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
            (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
            (seq_out / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf-8")
            with (seq_out / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace, ensure_ascii=False,
                                             allow_nan=True) + "\n")
            rows.append(metrics)
            print(f"[{idx}/{len(seqs)}] {seq}: {tracker.selected_method} "
                  f"AUC={metrics['auc']:.4f} SR={metrics['sr']:.4f} "
                  f"e2eFPS={metrics['e2e_fps']:.2f} OD={metrics['od_selected']}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(seqs)}] {seq}: FAILED {exc}", file=sys.stderr, flush=True)
    if rows:
        keys = ["sequence", "selected_method", "n_frames", "n_scored", "auc", "sr",
                "e2e_fps", "od_calls", "od_selected", "route_reasons"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in rows)
        (out_root / "summary.json").write_text(json.dumps({
            "schema": "grt360.geometry_b224_t224_od_recovery_summary.v1",
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
