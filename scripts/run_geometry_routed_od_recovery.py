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
    GeometryRoutedTracker,
    build_kwargs,
)
from scripts.run_sutrack_b224_redetect import (  # noqa: E402
    SphericalB224RedetectTracker,
)


def route_recovery(init_bfov) -> tuple[bool, list[str]]:
    """Select the recovery branch from initialization geometry only."""
    if init_bfov is None:
        return False, ["missing_init_bfov"]
    fh, fv = float(init_bfov[2]), float(init_bfov[3])
    if fv >= 100.0 or fh >= 90.0:
        return True, ["large_fov_sparse_od_recovery"]
    return False, ["geometry_b224_t224_router"]


class GeometryRecoveryTracker:
    def __init__(self, b_model, b_high_model, t_model, od_model,
                 od_first_model, tracker_kwargs, args):
        self.geometry = GeometryRoutedTracker(
            b_model, b_high_model, t_model, **tracker_kwargs)
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
                       if od_model is not None else None)
        self.active = self.geometry
        self.selected_method = "geometry_b224_t224"
        self.route_reasons = []

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        use_recovery, reasons = route_recovery(init_bfov)
        if use_recovery and self.recovery is not None:
            self.active = self.recovery
            self.selected_method = "b224_od_recovery"
            self.route_reasons = reasons
        else:
            self.active = self.geometry
            self.selected_method = "geometry_b224_t224"
            self.route_reasons = reasons
        self.active.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)

    def track(self, frame_rgb, **kwargs):
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
    ap.add_argument("--anchor-min-similarity", type=float, default=0.15)
    ap.add_argument("--max-motion-deg", type=float, default=180.0)
    ap.add_argument("--erp-downscale", type=int, default=4)
    ap.add_argument("--od-projection", choices=["erp", "tangent"], default="tangent")
    ap.add_argument("--od-cadence", type=int, default=30)
    ap.add_argument("--od-lost-cadence", type=int, default=5)
    ap.add_argument("--od-quality-threshold", type=float, default=0.45)
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
                tracker_kwargs, args)
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
