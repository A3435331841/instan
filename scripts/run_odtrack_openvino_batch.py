#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch runner for the explicit-state ODTrack OpenVINO expert.

The single-sequence adapter already owns the state/query contract.  This
thin batch wrapper compiles the first/state graphs once, then evaluates an
explicit list of sequences with identical scoring and end-to-end timing.
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--first-xml", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--search-factor", type=float, default=5.0)
    ap.add_argument("--template-factor", type=float, default=2.0)
    ap.add_argument("--update-interval", type=int, default=25)
    ap.add_argument("--update-threshold", type=float, default=0.55)
    ap.add_argument("--projection-mode", choices=["erp", "tangent"], default="tangent")
    args = ap.parse_args(argv)
    import openvino as ov

    seqs = [s.strip().replace("\\", "/") for s in args.seqs.split(",") if s.strip()]
    if not seqs:
        raise SystemExit("no sequences supplied")
    t0 = time.perf_counter()
    core = ov.Core()
    state = core.compile_model(str(Path(args.xml).resolve()), args.device)
    first = core.compile_model(str(Path(args.first_xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - t0
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in seqs:
        seq_out = out_root / seq.replace("/", "_")
        seq_out.mkdir(parents=True, exist_ok=True)
        holder = {}

        def factory(**_kwargs):
            tracker = OpenVinoODTrackTracker(
                state, search_factor=args.search_factor,
                template_factor=args.template_factor,
                update_interval=args.update_interval,
                update_threshold=args.update_threshold,
                seam_recenter=True, first_compiled_model=first,
                projection_mode=args.projection_mode)
            holder["tracker"] = tracker
            return tracker

        metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
            seq, args.data, factory, args.max_frames)
        metrics.update({
            "compile_seconds": compile_seconds,
            "search_factor": args.search_factor,
            "template_factor": args.template_factor,
            "update_threshold": args.update_threshold,
            "graph": str(Path(args.xml).resolve()),
            "device": args.device,
            "projection_mode": args.projection_mode,
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
            "projection_mode", "search_factor"]
    with (out_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in keys} for row in rows)
    (out_root / "summary.json").write_text(json.dumps({
        "schema": "grt360.odtrack_openvino_batch_summary.v1",
        "n_sequences": len(rows), "compile_seconds": compile_seconds,
        "mean_auc": float(np.mean([row["auc"] for row in rows])),
        "mean_sr": float(np.mean([row["sr"] for row in rows])),
        "mean_e2e_fps": float(np.mean([row["e2e_fps"] for row in rows])),
        "rows": rows,
    }, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
