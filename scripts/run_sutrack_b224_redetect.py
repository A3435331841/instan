#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B224 main tracker with a causal spherical re-detection state machine.

The main B224 path is unchanged during NORMAL operation.  After a sustained
low-quality run the wrapper freezes the main track, searches the ERP sphere
with an immutable first-frame anchor, verifies the candidate, and only then
re-initializes B224.  No ground truth, sequence name, or offline result table
is used at runtime.  This is an experiment runner; it is promoted only after
locked valid/full evaluation and end-to-end speed measurement.
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

from panotrack.geometry.bfov import bfov_from_erp_bbox  # noqa: E402
from panotrack.pipeline.memory import (  # noqa: E402
    ReliabilityGate,
    TemplateMemory,
    _template_similarity,
)
from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector  # noqa: E402
from scripts.eval_official import run_sequence  # noqa: E402
from scripts.run_geometry_routed_b224_t224 import build_kwargs  # noqa: E402
from scripts.run_odtrack_openvino_sequence import (  # noqa: E402
    OpenVinoODTrackTracker,
)
from scripts.run_sutrack_b224_openvino_sequence import (  # noqa: E402
    MotionAdaptiveTracker,
)


def crop_wrap(frame: np.ndarray, box) -> np.ndarray:
    """Crop an ERP box with circular longitude indexing."""
    h, w = frame.shape[:2]
    x, y, bw, bh = (float(v) for v in box)
    iw, ih = max(2, int(round(bw))), max(2, int(round(bh)))
    iy = int(np.clip(round(y), 0, max(0, h - ih)))
    ix = int(round(x))
    cols = np.mod(ix + np.arange(iw), w)
    return np.ascontiguousarray(frame[iy:iy + ih][:, cols])


def box_center(box, width):
    return (float(box[0]) + 0.5 * float(box[2])) % float(width)


def circular_delta(a, b, width):
    return ((float(a) - float(b) + width / 2.0) % width) - width / 2.0


def bfov_tuple(box, width, height):
    """Return the protocol's indexable BFoV tuple for tracker re-init."""
    bf = bfov_from_erp_bbox(*[float(v) for v in box], width, height)
    return (float(bf.lon), float(bf.lat), float(bf.fov_h), float(bf.fov_v))


class SphericalB224RedetectTracker:
    """B224 plus anchor-verified low-frequency spherical re-detection."""

    def __init__(self, b_model, b_high_model, tracker_kwargs,
                 run_len=5, search_interval=10, min_score=0.45,
                 anchor_min_similarity=0.50, max_motion_deg=120.0,
                 erp_downscale=3, od_model=None, od_first_model=None,
                 od_projection="tangent", od_cadence=15,
                 od_quality_threshold=0.55):
        self.b_model = b_model
        self.b_high_model = b_high_model
        self.tracker_kwargs = dict(tracker_kwargs)
        # The redetection experiment deliberately uses the safe B224 large-FoV
        # policy; geometry routing is a separate candidate.
        self.tracker_kwargs["search_factor_mode"] = "large_fov"
        self.primary = MotionAdaptiveTracker(
            b_model, b_high_model, **self.tracker_kwargs)
        self.run_len = max(2, int(run_len))
        self.search_interval = max(1, int(search_interval))
        self.min_score = float(min_score)
        self.anchor_min_similarity = float(anchor_min_similarity)
        self.max_motion_deg = float(max_motion_deg)
        self.erp_downscale = max(1, int(erp_downscale))
        self.od_cadence = max(1, int(od_cadence))
        self.od_quality_threshold = float(od_quality_threshold)
        self.od = (OpenVinoODTrackTracker(
            od_model, search_size=384, template_size=192,
            search_factor=5.0, template_factor=2.0,
            update_interval=25, update_threshold=0.55,
            seam_recenter=True, first_compiled_model=od_first_model,
            projection_mode=od_projection)
                   if od_model is not None else None)
        self.memory = None
        self.redetector = None
        self.width = self.height = 0
        self.frame_id = 0
        self.low_run = 0
        self.search_counter = 0
        self.verify_good = 0
        self.status = "normal"
        self.last_box = None
        self.last_quality = 1.0
        self.anchor_model_template = None
        self.redetect_calls = 0
        self.recovered_events = 0
        self.last_redetect_score = 0.0
        self.last_candidate_similarity = 0.0
        self.redetect_hits = 0
        self.od_calls = 0
        self.od_selected = 0

    def _new_memory(self, frame, box):
        # A fresh memory per sequence prevents cross-sequence state leakage.
        self.memory = TemplateMemory(
            gate=ReliabilityGate(accept_thr=0.55),
            short_cap=3, long_cap=6, dedup_thr=0.75, min_quality=0.55)
        crop = crop_wrap(frame, box)
        self.memory.set_anchor((crop, (float(box[2]), float(box[3]))), reliability=1.0)
        self.redetector = SphericalMultiViewRedetector(
            self.memory.get_bank,
            min_score=self.min_score,
            lat_bands=(-60.0, 0.0, 60.0),
            lon_per_band=(2, 2, 2),
            view_half_lon=95.0,
            view_half_lat=70.0,
            template_scales=(0.85, 1.0, 1.2),
        )

    def _anchor_similarity(self, frame, box):
        if self.memory is None or self.memory.anchor is None:
            return 0.0
        crop = crop_wrap(frame, box)
        sim = _template_similarity(
            (crop, (float(box[2]), float(box[3]))), self.memory.anchor[0])
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    def _verify_candidate(self, frame, box, score):
        if float(score) < self.min_score:
            return False
        if self._anchor_similarity(frame, box) < self.anchor_min_similarity:
            return False
        if self.last_box is not None:
            old_cx = box_center(self.last_box, self.width)
            new_cx = box_center(box, self.width)
            dx = circular_delta(new_cx, old_cx, self.width)
            old_cy = float(self.last_box[1]) + 0.5 * float(self.last_box[3])
            new_cy = float(box[1]) + 0.5 * float(box[3])
            dy = new_cy - old_cy
            angle = np.hypot(dx * 360.0 / self.width,
                             dy * 180.0 / self.height)
            if float(angle) > self.max_motion_deg:
                return False
        return True

    def _restore_anchor(self):
        if self.anchor_model_template is None:
            return
        self.primary.base.anchor_template = self.anchor_model_template.copy()
        if self.primary.high is not None:
            self.primary.high.anchor_template = self.anchor_model_template.copy()

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        self.height, self.width = frame_rgb.shape[:2]
        self.primary.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.anchor_model_template = self.primary.base.anchor_template.copy()
        self._new_memory(frame_rgb, erp_box)
        self.frame_id = 0
        self.low_run = 0
        self.search_counter = 0
        self.verify_good = 0
        self.status = "normal"
        self.last_box = [float(v) for v in erp_box]
        self.last_quality = 1.0
        self.redetect_calls = 0
        self.recovered_events = 0
        self.last_redetect_score = 0.0
        self.last_candidate_similarity = 0.0
        self.redetect_hits = 0
        self.od_calls = 0
        self.od_selected = 0
        if self.od is not None:
            self.od.init(frame_rgb, erp_box, init_bfov=init_bfov)

    def _out(self, box, quality, status, expert=None, **extra):
        payload = {
            "target_bbox": [float(box[0]) % self.width, float(box[1]),
                            float(box[2]), float(box[3])],
            "quality": float(quality),
            "status": str(status),
            "expert_used": expert or "sutrack_b224",
            "redetect_calls": int(self.redetect_calls),
            "recovered_events": int(self.recovered_events),
            "redetect_score": float(self.last_redetect_score),
            "candidate_similarity": float(self.last_candidate_similarity),
            "redetect_hits": int(self.redetect_hits),
            "od_calls": int(self.od_calls),
            "od_selected": int(self.od_selected),
        }
        payload.update(extra)
        return payload

    def track(self, frame_rgb, **kwargs):
        self.frame_id += 1
        od_out = None
        if self.od is not None and self.frame_id % self.od_cadence == 0:
            od_out = dict(self.od.track(frame_rgb))
            self.od_calls += 1
        if self.status == "lost":
            # A sparse ODTrack tangent prediction gets first refusal in LOST.
            # It is accepted only after the same anchor/motion verification as
            # the spherical NCC detector, then B224 is re-initialized.
            if od_out is not None:
                od_box = [float(v) for v in od_out["target_bbox"]]
                od_quality = float(od_out.get("quality", 0.0))
                if (od_quality >= self.od_quality_threshold and
                        self._verify_candidate(frame_rgb, od_box, od_quality)):
                    self.primary.init(
                        frame_rgb, od_box,
                        init_bfov=bfov_tuple(od_box, self.width, self.height),
                        **kwargs)
                    if self.od is not None:
                        self.od.init(frame_rgb, od_box,
                                     init_bfov=bfov_tuple(od_box, self.width, self.height))
                    self._restore_anchor()
                    self.last_box = od_box
                    self.status = "verify"
                    self.verify_good = 0
                    self.low_run = 0
                    self.od_selected += 1
                    return self._out(od_box, od_quality, "verify",
                                     expert="odtrack_tangent_recovery")
            self.search_counter += 1
            if self.search_counter % self.search_interval == 0:
                self.redetect_calls += 1
                found = self.redetector.search(
                    frame_rgb, erp_downscale=self.erp_downscale)
                if found is not None:
                    candidate, score = found
                    self.redetect_hits += 1
                    self.last_redetect_score = float(score)
                    self.last_candidate_similarity = self._anchor_similarity(
                        frame_rgb, candidate)
                    if self._verify_candidate(frame_rgb, candidate, score):
                        self.primary.init(
                            frame_rgb, candidate,
                            init_bfov=bfov_tuple(candidate, self.width, self.height),
                            **kwargs)
                        if self.od is not None:
                            self.od.init(frame_rgb, candidate,
                                         init_bfov=bfov_tuple(candidate, self.width, self.height))
                        self._restore_anchor()
                        self.last_box = [float(v) for v in candidate]
                        self.status = "verify"
                        self.verify_good = 0
                        self.low_run = 0
                        self.recovered_events += 1
                        return self._out(candidate, score, "verify",
                                         expert="spherical_redetect")
            return self._out(self.last_box, 0.0, "lost",
                             expert="spherical_redetect")

        out = dict(self.primary.track(frame_rgb, **kwargs))
        box = [float(v) for v in out["target_bbox"]]
        quality = float(out.get("quality", 0.0))
        self.last_quality = quality
        self.last_box = box
        if quality <= float(self.tracker_kwargs.get("fallback_quality_threshold", 0.4)):
            self.low_run += 1
        else:
            self.low_run = 0

        if self.status == "verify":
            anchor_sim = self._anchor_similarity(frame_rgb, box)
            if quality >= 0.4 and anchor_sim >= self.anchor_min_similarity:
                self.verify_good += 1
                if self.verify_good >= 3:
                    self.status = "normal"
            else:
                self.status = "lost"
                self.search_counter = 0
                return self._out(box, quality, "lost", expert="spherical_redetect",
                                 anchor_similarity=anchor_sim)
        elif self.low_run >= self.run_len:
            self.status = "lost"
            self.search_counter = 0
            if od_out is not None:
                od_box = [float(v) for v in od_out["target_bbox"]]
                od_quality = float(od_out.get("quality", 0.0))
                if (od_quality >= self.od_quality_threshold and
                        self._verify_candidate(frame_rgb, od_box, od_quality)):
                    self.primary.init(
                        frame_rgb, od_box,
                        init_bfov=bfov_tuple(od_box, self.width, self.height),
                        **kwargs)
                    if self.od is not None:
                        self.od.init(frame_rgb, od_box,
                                     init_bfov=bfov_tuple(od_box, self.width, self.height))
                    self._restore_anchor()
                    self.last_box = od_box
                    self.status = "verify"
                    self.verify_good = 0
                    self.low_run = 0
                    self.od_selected += 1
                    return self._out(od_box, od_quality, "verify",
                                     expert="odtrack_tangent_recovery")
            return self._out(box, quality, "lost", expert="sutrack_b224",
                             anchor_similarity=self._anchor_similarity(frame_rgb, box))

        return self._out(box, quality, self.status,
                         expert="sutrack_b224",
                         anchor_similarity=(self._anchor_similarity(frame_rgb, box)
                                            if self.status != "normal" else None))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b-xml", required=True)
    ap.add_argument("--b-high-xml", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True,
                    help="comma-separated paths such as train_real/seq_0041")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--run-len", type=int, default=5)
    ap.add_argument("--search-interval", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=0.45)
    ap.add_argument("--anchor-min-similarity", type=float, default=0.50)
    ap.add_argument("--max-motion-deg", type=float, default=120.0)
    ap.add_argument("--erp-downscale", type=int, default=3)
    ap.add_argument("--quality-threshold", type=float, default=0.40)
    ap.add_argument("--od-xml", default=None,
                    help="optional ODTrack state graph for sparse LOST recovery")
    ap.add_argument("--od-first-xml", default=None,
                    help="optional ODTrack first-step graph")
    ap.add_argument("--od-projection", choices=["erp", "tangent"], default="tangent")
    ap.add_argument("--od-cadence", type=int, default=15)
    ap.add_argument("--od-quality-threshold", type=float, default=0.55)
    args = ap.parse_args(argv)
    import openvino as ov

    t0 = time.perf_counter()
    core = ov.Core()
    b_model = core.compile_model(str(Path(args.b_xml).resolve()), args.device)
    b_high_model = core.compile_model(str(Path(args.b_high_xml).resolve()), args.device)
    od_model = (core.compile_model(str(Path(args.od_xml).resolve()), args.device)
                if args.od_xml else None)
    od_first_model = (core.compile_model(str(Path(args.od_first_xml).resolve()), args.device)
                      if args.od_first_xml else None)
    compile_seconds = time.perf_counter() - t0
    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    kwargs = build_kwargs(args)
    rows = []
    for idx, seq in enumerate(seqs, 1):
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        def factory(**_kwargs):
            tracker = SphericalB224RedetectTracker(
                b_model, b_high_model, kwargs,
                run_len=args.run_len,
                search_interval=args.search_interval,
                min_score=args.min_score,
                anchor_min_similarity=args.anchor_min_similarity,
                max_motion_deg=args.max_motion_deg,
                erp_downscale=args.erp_downscale,
                od_model=od_model, od_first_model=od_first_model,
                od_projection=args.od_projection,
                od_cadence=args.od_cadence,
                od_quality_threshold=args.od_quality_threshold)
            holder["tracker"] = tracker
            return tracker

        try:
            metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
                seq, args.data, factory, args.max_frames)
            tracker = holder["tracker"]
            metrics.update({
                "compile_seconds": compile_seconds,
                "router_schema": "sutrack_b224_spherical_redetect.v1",
                "run_len": args.run_len,
                "search_interval": args.search_interval,
                "redetect_min_score": args.min_score,
                "anchor_min_similarity": args.anchor_min_similarity,
                "erp_downscale": args.erp_downscale,
                "redetect_calls": tracker.redetect_calls,
                "redetect_hits": tracker.redetect_hits,
                "last_candidate_similarity": tracker.last_candidate_similarity,
                "od_calls": tracker.od_calls,
                "od_selected": tracker.od_selected,
                "recovered_events": tracker.recovered_events,
                "e2e_fps": metrics.get("e2e_fps"),
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
            print(f"[{idx}/{len(seqs)}] {seq}: AUC={metrics['auc']:.4f} "
                  f"SR={metrics['sr']:.4f} e2eFPS={metrics['e2e_fps']:.2f} "
                  f"redetect={tracker.redetect_calls} recovered={tracker.recovered_events}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(seqs)}] {seq}: FAILED {exc}",
                  file=sys.stderr, flush=True)
    if rows:
        keys = ["sequence", "n_frames", "n_scored", "auc", "sr", "e2e_fps",
                "redetect_calls", "redetect_hits", "od_calls", "od_selected",
                "recovered_events"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({k: row.get(k) for k in keys} for row in rows)
        (out_root / "summary.json").write_text(json.dumps({
            "schema": "grt360.spherical_redetect_summary.v1",
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
