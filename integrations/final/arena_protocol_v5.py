#!/usr/bin/env python3
"""Arena-compliant ORT CUDA entrypoint for the GRT-360 v5 final route.

The runtime reuses the causally validated B224/T224/ODTrack geometry router.
It reads only video frames and `init.txt`, never ground truth, sequence names,
or offline result tables.  The default contract is the Arena BFoV protocol:
`/mnt/dataset/<sequence>/{video.mp4,init.txt}` to `/mnt/result/<sequence>.txt`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    import cv2 as cv
except ImportError:  # pragma: no cover - image dependency
    cv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.final.ort_adapter import OrtCompiledModel, required_model_paths  # noqa: E402
from integrations.sutrack.arena_protocol_sutrack import (  # noqa: E402
    bfov_from_erp_bbox,
    erp_bbox_from_bfov,
    list_sequences,
    load_init_bfov,
    write_bfov_rows,
)
from scripts.run_geometry_routed_b224_t224 import build_kwargs  # noqa: E402
from scripts.run_geometry_routed_od_recovery import GeometryRecoveryTracker  # noqa: E402


def _runtime_args(quality_threshold: float) -> SimpleNamespace:
    """Keep final inference constants aligned with the locked full130 run."""
    return SimpleNamespace(
        quality_threshold=float(quality_threshold),
        run_len=5,
        search_interval=10,
        min_score=0.45,
        anchor_min_similarity=0.15,
        max_motion_deg=180.0,
        erp_downscale=4,
        od_projection="tangent",
        od_cadence=30,
        od_lost_cadence=5,
        od_quality_threshold=0.45,
        direct_only=False,
        narrow_recovery_only=True,
    )


class FinalOrtRuntime:
    """Compiled final-route graphs shared across Arena sequences."""

    def __init__(self, model_root: Path, device: str, device_id: int,
                 quality_threshold: float, profile: str):
        if profile not in {"v5_final", "geometry_v1", "geometry_v4"}:
            raise ValueError(f"unsupported ORT profile: {profile}")
        paths = required_model_paths(model_root)
        compiled = {name: OrtCompiledModel(path, device=device,
                                            device_id=device_id,
                                            strict=(device == "cuda"))
                    for name, path in paths.items()}
        self.args = _runtime_args(quality_threshold)
        self.kwargs = build_kwargs(self.args)
        self.models = compiled
        self.profile = profile

    @property
    def providers(self) -> dict[str, tuple[str, ...]]:
        return {name: model.providers for name, model in self.models.items()}

    def new_tracker(self) -> GeometryRecoveryTracker:
        # v1/v4 deliberately omit the trained v5 direct expert.  The final
        # v5 profile supplies both v5 graphs; routing remains geometry-only.
        use_v5 = self.profile == "v5_final"
        return GeometryRecoveryTracker(
            self.models["b"], self.models["b_high"], self.models["t"],
            self.models["od"], self.models["od_first"], self.kwargs,
            self.args, enable_recovery=True,
            od_v5_model=self.models["od_v5"] if use_v5 else None,
            od_v5_first_model=self.models["od_v5_first"] if use_v5 else None,
            narrow_recovery_only=True,
        )


def _track_sequence(seq_dir: Path, runtime: FinalOrtRuntime,
                    max_frames: int | None, trace_dir: Path | None):
    if cv is None:
        raise RuntimeError("opencv-python-headless is required")
    init_bfov = load_init_bfov(seq_dir)
    cap = cv.VideoCapture(str(seq_dir / "video.mp4"))
    try:
        ok, first = cap.read()
        if not ok or first is None:
            raise RuntimeError(f"failed to decode first frame: {seq_dir}")
        height, width = first.shape[:2]
        tracker = runtime.new_tracker()
        first_rgb = cv.cvtColor(first, cv.COLOR_BGR2RGB)
        init_box = erp_bbox_from_bfov(*init_bfov, width, height)
        tracker.init(first_rgb, init_box, init_bfov=init_bfov)
        rows = [tuple(init_bfov)]
        trace_rows = [{"frame_index": 0, "status": "init",
                       "expert_used": None, "target_bfov": list(init_bfov)}]
        frame_index = 1
        while max_frames is None or frame_index < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out = dict(tracker.track(cv.cvtColor(frame, cv.COLOR_BGR2RGB),
                                     frame_idx=frame_index))
            box = out.get("target_bbox")
            if box is None or len(box) != 4 or not np.all(np.isfinite(box)):
                raise RuntimeError(f"invalid prediction at frame {frame_index}: {box}")
            bfov = bfov_from_erp_bbox(*[float(v) for v in box], width, height)
            rows.append(tuple(float(v) for v in bfov))
            trace_rows.append({
                "frame_index": frame_index,
                "target_bfov": list(rows[-1]),
                "quality": float(out.get("quality", 0.0)),
                "status": str(out.get("status", "normal")),
                "expert_used": out.get("expert_used"),
                "route_reasons": list(out.get("route_reasons", [])),
            })
            frame_index += 1
    finally:
        cap.release()
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{seq_dir.name}.json").write_text(
            json.dumps(trace_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--result", default=None)
    parser.add_argument("--model-root", default=os.environ.get("GRT360_MODEL_ROOT", "/opt/models"))
    parser.add_argument("--profile", default=os.environ.get("GRT360_PROFILE", "v5_final"),
                        choices=("v5_final", "geometry_v1", "geometry_v4"))
    parser.add_argument("--gpu", type=int, default=int(os.environ.get("GRT360_GPU", "0")))
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--quality-threshold", type=float, default=0.40)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seqs", default=None)
    parser.add_argument("--trace-dir", default=None,
                        help="optional diagnostics directory; omitted in Arena runs")
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset or os.environ.get("DATASET_DIR", "/mnt/dataset"))
    result_dir = Path(args.result or os.environ.get("RESULT_DIR", "/mnt/result"))
    if not dataset_dir.is_dir():
        print(f"[error] dataset missing: {dataset_dir}", file=sys.stderr)
        return 2
    device = "cpu" if args.force_cpu else "cuda"
    runtime = FinalOrtRuntime(Path(args.model_root), device, args.gpu,
                              args.quality_threshold, args.profile)
    print(json.dumps({"profile": args.profile, "device": device,
                      "providers": runtime.providers}, ensure_ascii=False), flush=True)
    sequences = list_sequences(dataset_dir)
    if args.seqs:
        wanted = {item.strip() for item in args.seqs.split(",") if item.strip()}
        sequences = [item for item in sequences if item in wanted]
    if not sequences:
        print("[error] no Arena sequences found", file=sys.stderr)
        return 2
    result_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    failures: list[str] = []
    total_frames = 0
    started = time.perf_counter()
    for index, name in enumerate(sequences, 1):
        begin = time.perf_counter()
        try:
            rows = _track_sequence(dataset_dir / name, runtime, args.max_frames, trace_dir)
            write_bfov_rows(result_dir / f"{name}.txt", rows)
            total_frames += len(rows)
            elapsed = time.perf_counter() - begin
            fps = max(0, len(rows) - 1) / elapsed if elapsed > 0 else 0.0
            print(f"[{index}/{len(sequences)}] {name}: {len(rows)} frames {fps:.2f} FPS", flush=True)
        except Exception as exc:  # noqa: BLE001 - Arena must report every failed sequence
            failures.append(name)
            print(f"[{index}/{len(sequences)}] {name}: FAILED {exc}", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    print(f"[final_ort] frames={total_frames} seconds={elapsed:.2f} "
          f"fps={total_frames / elapsed if elapsed > 0 else 0.0:.2f}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
