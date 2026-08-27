#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-evaluate the adaptive B224 OpenVINO candidate on local sequences.

This deliberately keeps the batch runner thin: all crop, state, gating and
metric logic lives in ``run_sutrack_b224_openvino_sequence.py`` and
``eval_official.run_sequence``.  A single pair of compiled graphs is reused
for the whole batch, while each sequence receives a fresh tracker instance.
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


def discover_sequences(data_root: Path):
    seqs = []
    for block in ("train_real", "train_sim"):
        root = data_root / block
        if not root.is_dir():
            continue
        seqs.extend(
            f"{block}/{p.name}"
            for p in sorted(root.iterdir())
            if p.is_dir() and (p / "video.mp4").is_file()
        )
    return seqs


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--high-xml", default=None)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--seqs", default=None,
                    help="comma-separated sequence paths; default scans train_real/train_sim")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--motion-adaptive", action="store_true")
    ap.add_argument("--quality-threshold", type=float, default=0.40)
    ap.add_argument("--quality-run", type=int, default=3)
    ap.add_argument("--switch-deadline", type=int, default=30)
    ap.add_argument("--search-factor", type=float, default=4.0)
    ap.add_argument("--search-factor-mode",
                    choices=["fixed", "moderate_fov", "large_fov"], default="fixed")
    ap.add_argument("--template-factor", type=float, default=2.0)
    ap.add_argument("--search-size", type=int, default=224)
    ap.add_argument("--template-size", type=int, default=112)
    ap.add_argument("--update-interval", type=int, default=25)
    ap.add_argument("--update-threshold", type=float, default=0.70)
    ap.add_argument("--fallback-search-factor", type=float, default=None)
    ap.add_argument("--fallback-quality-threshold", type=float, default=0.45)
    ap.add_argument("--fallback-min-gain", type=float, default=0.0)
    ap.add_argument("--fallback-cooldown", type=int, default=1)
    ap.add_argument("--fallback-run", type=int, default=1)
    ap.add_argument("--fallback-start-frame", type=int, default=0)
    ap.add_argument("--anchor-update-threshold", type=float, default=None)
    ap.add_argument("--auto-freeze-scale-threshold", type=float, default=None)
    ap.add_argument("--auto-freeze-scale-window", type=int, default=40)
    ap.add_argument("--auto-freeze-quality-slope", type=float, default=None)
    ap.add_argument("--auto-freeze-scale-step-p95", type=float, default=None)
    ap.add_argument("--auto-freeze-quality-floor", type=float, default=0.75)
    ap.add_argument("--auto-freeze-scale-step-median-max", type=float, default=0.018)
    ap.add_argument("--auto-freeze-scale-step-override", type=float, default=None)
    ap.add_argument("--auto-freeze-max-frame", type=int, default=None)
    ap.add_argument("--seam-recenter", action="store_true")
    ap.add_argument("--polar-rectify", action="store_true")
    ap.add_argument("--polar-latitude-threshold", type=float, default=55.0)
    ap.add_argument("--polar-aspect-max", type=float, default=2.5)
    ap.add_argument("--polar-small-width", type=float, default=100.0)
    ap.add_argument("--polar-max-frame", type=int, default=None)
    ap.add_argument("--no-polar-require-initial", dest="polar_require_initial",
                    action="store_false")
    ap.set_defaults(polar_require_initial=True)
    ap.add_argument("--small-template-factor", type=float, default=None)
    ap.add_argument("--small-template-width", type=float, default=100.0)
    ap.add_argument("--no-small-template-require-initial",
                    dest="small_template_require_initial", action="store_false")
    ap.set_defaults(small_template_require_initial=True)
    return ap


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.motion_adaptive and not args.high_xml:
        raise SystemExit("--motion-adaptive requires --high-xml")
    data_root = Path(args.data).resolve()
    seqs = ([s.strip() for s in args.seqs.split(",") if s.strip()]
            if args.seqs else discover_sequences(data_root))
    if not seqs:
        raise SystemExit("no sequences found")

    import openvino as ov
    compile_t0 = time.perf_counter()
    core = ov.Core()
    compiled = core.compile_model(str(Path(args.xml).resolve()), args.device)
    compiled_high = (core.compile_model(str(Path(args.high_xml).resolve()), args.device)
                     if args.motion_adaptive else None)
    compile_seconds = time.perf_counter() - compile_t0

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, seq in enumerate(seqs, 1):
        seq_tag = seq.replace("/", "_")
        seq_out = out_root / seq_tag
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        if args.motion_adaptive:
            def factory(**_kwargs):
                tracker = MotionAdaptiveTracker(
                    compiled, compiled_high,
                    warmup=5,
                    threshold_deg=1.5,
                    quality_threshold=args.quality_threshold,
                    quality_run=args.quality_run,
                    switch_deadline=args.switch_deadline,
                    search_factor=args.search_factor,
                    search_factor_mode=args.search_factor_mode,
                    fallback_search_factor=args.fallback_search_factor,
                    fallback_quality_threshold=args.fallback_quality_threshold,
                    fallback_min_gain=args.fallback_min_gain,
                    fallback_cooldown=args.fallback_cooldown,
                    fallback_run=args.fallback_run,
                    fallback_start_frame=args.fallback_start_frame,
                    anchor_update_threshold=args.anchor_update_threshold,
                    auto_freeze_scale_threshold=args.auto_freeze_scale_threshold,
                    auto_freeze_scale_window=args.auto_freeze_scale_window,
                    auto_freeze_quality_slope=args.auto_freeze_quality_slope,
                    auto_freeze_scale_step_p95=args.auto_freeze_scale_step_p95,
                    auto_freeze_quality_floor=args.auto_freeze_quality_floor,
                    auto_freeze_scale_step_median_max=args.auto_freeze_scale_step_median_max,
                    auto_freeze_scale_step_override=args.auto_freeze_scale_step_override,
                    auto_freeze_max_frame=args.auto_freeze_max_frame,
                    seam_recenter=args.seam_recenter,
                    polar_rectify=args.polar_rectify,
                    polar_latitude_threshold=args.polar_latitude_threshold,
                    polar_aspect_max=args.polar_aspect_max,
                    polar_small_width=args.polar_small_width,
                    polar_max_frame=args.polar_max_frame,
                    polar_require_initial=args.polar_require_initial,
                    small_template_factor=args.small_template_factor,
                    small_template_width=args.small_template_width,
                    small_template_require_initial=args.small_template_require_initial,
                )
                holder["tracker"] = tracker
                return tracker
        else:
            def factory(**_kwargs):
                tracker = OpenVinoB224Tracker(
                    compiled,
                    search_size=args.search_size,
                    template_size=args.template_size,
                    search_factor=args.search_factor,
                    template_factor=args.template_factor,
                    update_interval=args.update_interval,
                    update_threshold=args.update_threshold,
                    search_factor_mode=args.search_factor_mode,
                    fallback_search_factor=args.fallback_search_factor,
                    fallback_quality_threshold=args.fallback_quality_threshold,
                    fallback_min_gain=args.fallback_min_gain,
                    fallback_cooldown=args.fallback_cooldown,
                    fallback_run=args.fallback_run,
                    fallback_start_frame=args.fallback_start_frame,
                    anchor_update_threshold=args.anchor_update_threshold,
                    auto_freeze_scale_threshold=args.auto_freeze_scale_threshold,
                    auto_freeze_scale_window=args.auto_freeze_scale_window,
                    auto_freeze_quality_slope=args.auto_freeze_quality_slope,
                    auto_freeze_scale_step_p95=args.auto_freeze_scale_step_p95,
                    auto_freeze_quality_floor=args.auto_freeze_quality_floor,
                    auto_freeze_scale_step_median_max=args.auto_freeze_scale_step_median_max,
                    auto_freeze_scale_step_override=args.auto_freeze_scale_step_override,
                    auto_freeze_max_frame=args.auto_freeze_max_frame,
                    seam_recenter=args.seam_recenter,
                    polar_rectify=args.polar_rectify,
                    polar_latitude_threshold=args.polar_latitude_threshold,
                    polar_aspect_max=args.polar_aspect_max,
                    polar_small_width=args.polar_small_width,
                    polar_max_frame=args.polar_max_frame,
                    polar_require_initial=args.polar_require_initial,
                    small_template_factor=args.small_template_factor,
                    small_template_width=args.small_template_width,
                    small_template_require_initial=args.small_template_require_initial,
                )
                holder["tracker"] = tracker
                return tracker

        t0 = time.perf_counter()
        try:
            metrics, pred, valid, _w, _h, qualities, statuses, traces, latency = run_sequence(
                seq, str(data_root), factory, args.max_frames)
            np.savetxt(seq_out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
            np.savetxt(seq_out / "quality.txt", qualities, fmt="%.6f")
            (seq_out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
            with (seq_out / "trace.jsonl").open("w", encoding="utf-8") as handle:
                for trace in traces:
                    handle.write(json.dumps(trace, ensure_ascii=False, allow_nan=True) + "\n")
            (seq_out / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf-8")
            tracker = holder.get("tracker")
            if args.motion_adaptive:
                metrics["active_search_factor"] = tracker.base.active_search_factor
                metrics["fallback_calls"] = tracker.base.fallback_calls
                metrics["fallback_selected"] = tracker.base.fallback_selected
                metrics["polar_sample_count"] = tracker.base.polar_sample_count
                metrics["updates_frozen"] = tracker.base.updates_frozen
                metrics["updates_frozen_frame"] = tracker.base.updates_frozen_frame
                metrics["switch_frame"] = tracker.switch_frame
            else:
                metrics["active_search_factor"] = tracker.active_search_factor
                metrics["fallback_calls"] = tracker.fallback_calls
                metrics["fallback_selected"] = tracker.fallback_selected
                metrics["polar_sample_count"] = tracker.polar_sample_count
                metrics["updates_frozen"] = tracker.updates_frozen
                metrics["updates_frozen_frame"] = tracker.updates_frozen_frame
            metrics["compile_seconds"] = compile_seconds
            metrics["batch_wall_seconds"] = time.perf_counter() - t0
            (seq_out / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf-8")
            rows.append(metrics)
            print(f"[{idx}/{len(seqs)}] {seq}: AUC={metrics['auc']:.4f} "
                  f"SR={metrics['sr']:.4f} e2eFPS={metrics['e2e_fps']:.2f} "
                  f"switch={metrics.get('switch_frame')} fb={metrics.get('fallback_calls', 0)}",
                  flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{len(seqs)}] {seq}: FAILED {exc}", file=sys.stderr, flush=True)

    if rows:
        keys = ["sequence", "n_frames", "n_scored", "n_gt_absent", "auc", "sr",
                "auc_dual", "sr_dual", "e2e_fps", "tracker_fps", "switch_frame",
                "fallback_calls", "fallback_selected", "active_search_factor",
                "updates_frozen", "updates_frozen_frame"]
        with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows({key: row.get(key) for key in keys} for row in rows)
        summary = {
            "n_sequences": len(rows),
            "compile_seconds": compile_seconds,
            "mean_auc": float(np.mean([row["auc"] for row in rows])),
            "mean_sr": float(np.mean([row["sr"] for row in rows])),
            "mean_e2e_fps": float(np.mean([row["e2e_fps"] for row in rows])),
            "rows": rows,
        }
        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8")
    return 0 if len(rows) == len(seqs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
