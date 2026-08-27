#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare one identical B224 inference on two OpenVINO devices.

This is a diagnostic, not a tracker score.  It builds the exact template and
search tensors used by the B224 runner from one sequence, runs the same graph
on two devices, and records output-map differences.  It is useful for finding
device/compiler numerical problems before a device is allowed into the
precision or speed bake-off.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval_official import erp_bbox_from_bfov, load_init, BFoV
from scripts.run_sutrack_b224_openvino_sequence import OpenVinoB224Tracker, preprocess


def _read_frames(video: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    if not ok or first is None:
        raise RuntimeError(f"decode failed: {video}")
    frame = first
    for _ in range(frame_index):
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"frame {frame_index} unavailable: {video}")
    cap.release()
    return cv2.cvtColor(first, cv2.COLOR_BGR2RGB), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _compile(core, model: Path, device: str, cache_dir: Path | None,
             npu_platform: str | None, npu_compiler_type: str | None):
    config = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache_dir)
    if device == "NPU" and npu_platform:
        config["NPU_PLATFORM"] = str(npu_platform)
    if device == "NPU" and npu_compiler_type:
        config["NPU_COMPILER_TYPE"] = str(npu_compiler_type)
    started = time.perf_counter()
    compiled = core.compile_model(str(model.resolve()), device, config)
    return compiled, config, time.perf_counter() - started


def _candidate(tracker: OpenVinoB224Tracker, frame_rgb: np.ndarray, input_dtype=None):
    tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
    factor = tracker.active_search_factor
    patch, rf = tracker._sample(tiled, tracker.state, factor, tracker.search_size)
    inputs = {
        tracker.template_names[0]: tracker.templates[0],
        tracker.template_names[1]: tracker.templates[1],
        tracker.anno_names[0]: tracker.annos[0][None, :],
        tracker.anno_names[1]: tracker.annos[1][None, :],
        tracker.search_name: preprocess(patch),
    }
    if input_dtype is not None:
        inputs = {name: value.astype(input_dtype, copy=False) for name, value in inputs.items()}
    result = tracker.compiled(inputs)
    arrays = [np.asarray(result[port]).astype(np.float32, copy=True) for port in tracker.outputs]
    return inputs, arrays


def _stats(reference: list[np.ndarray], candidate: list[np.ndarray]) -> dict:
    rows = []
    for index, (a, b) in enumerate(zip(reference, candidate)):
        diff = b - a
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(np.sum(a * b) / denom) if denom else None
        rows.append({
            "output_index": index,
            "shape": list(a.shape),
            "reference_min": float(a.min()),
            "reference_max": float(a.max()),
            "candidate_min": float(b.min()),
            "candidate_max": float(b.max()),
            "max_abs_diff": float(np.max(np.abs(diff))),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff * diff))),
            "cosine": cosine,
            "reference_argmax": int(np.argmax(a)),
            "candidate_argmax": int(np.argmax(b)),
        })
    return {"outputs": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seq", required=True)
    parser.add_argument("--frame-index", type=int, default=1)
    parser.add_argument("--device-a", choices=["CPU", "GPU", "NPU"], default="GPU")
    parser.add_argument("--device-b", choices=["CPU", "GPU", "NPU"], default="NPU")
    parser.add_argument("--cache-dir-a", type=Path, default=None)
    parser.add_argument("--cache-dir-b", type=Path, default=None)
    parser.add_argument("--npu-platform", default=None)
    parser.add_argument("--npu-compiler-type", choices=["PLUGIN", "DRIVER", "PREFER_PLUGIN"], default=None)
    parser.add_argument("--device-a-input-fp16", action="store_true")
    parser.add_argument("--device-b-input-fp16", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frame_index < 1:
        raise SystemExit("--frame-index must be >= 1")

    seq_dir = (args.data / args.seq).resolve()
    first_rgb, frame_rgb = _read_frames(seq_dir / "video.mp4", args.frame_index)
    init = load_init(seq_dir)
    h, w = first_rgb.shape[:2]
    init_erp = erp_bbox_from_bfov(BFoV(*init), w, h)

    import openvino as ov
    core = ov.Core()
    report = {
        "schema": "grt360.openvino_device_compare.v1",
        "sequence": args.seq,
        "frame_index": args.frame_index,
        "resolution": [w, h],
        "devices_available": list(core.available_devices),
        "model": str(args.xml.resolve()),
        "initial_erp": [float(v) for v in init_erp],
    }
    compiled = {}
    for label, device, cache in (("a", args.device_a, args.cache_dir_a),
                                  ("b", args.device_b, args.cache_dir_b)):
        compiled[device], config, seconds = _compile(
            core, args.xml, device, cache, args.npu_platform, args.npu_compiler_type)
        report[f"compile_{label}"] = {
            "device": device,
            "config": config,
            "seconds": round(seconds, 3),
        }

    trackers = {}
    tensors = {}
    outputs = {}
    input_fp16 = {"a": args.device_a_input_fp16, "b": args.device_b_input_fp16}
    for label, device in (("a", args.device_a), ("b", args.device_b)):
        tracker = OpenVinoB224Tracker(compiled[device])
        tracker.init(first_rgb, init_erp, init_bfov=init)
        dtype = np.float16 if input_fp16[label] else None
        tensors[label], outputs[label] = _candidate(tracker, frame_rgb, dtype)
        trackers[label] = tracker
    report["input_equal"] = {
        name: bool(np.array_equal(tensors["a"][name], tensors["b"][name]))
        for name in tensors["a"]
        if name in tensors["b"]
    }
    report["output_compare"] = _stats(outputs["a"], outputs["b"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
