#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an exported ODTrack graph as a causal 360 ERP expert.

The ONNX/OpenVINO graph exposes ODTrack's track-query state explicitly.  The
Python adapter owns ERP tiling, crop geometry, the three-template ring and
state mapping; no GT is consulted after protocol initialization.
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

from scripts.eval_official import run_sequence  # noqa: E402
from scripts.run_sutrack_b224_openvino_sequence import (  # noqa: E402
    bfov_from_local_box,
    clip_box,
    sample_target,
    sample_target_ebfov,
)
from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov  # noqa: E402

MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_rgb(patch: np.ndarray) -> np.ndarray:
    arr = patch.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


class OpenVinoODTrackTracker:
    def __init__(self, compiled_model, search_size=384, template_size=192,
                 search_factor=5.0, template_factor=2.0, update_interval=25,
                 update_threshold=0.55, seam_recenter=True,
                 first_compiled_model=None, projection_mode="erp"):
        # The upstream model has a special first call (one template, no
        # track-query) and steady-state calls (one template + query).  Keep a
        # separate first graph when supplied; the legacy three-template graph
        # remains usable for quick smoke tests.
        self.compiled = compiled_model
        self.first_compiled = first_compiled_model
        self.search_size = int(search_size)
        self.template_size = int(template_size)
        self.search_factor = float(search_factor)
        self.template_factor = float(template_factor)
        self.update_interval = int(update_interval)
        self.update_threshold = float(update_threshold)
        self.seam_recenter = bool(seam_recenter)
        self.projection_mode = str(projection_mode).lower()
        if self.projection_mode not in {"erp", "tangent"}:
            raise ValueError(f"unknown projection_mode: {projection_mode}")
        self.bfov_state = None
        self._ebfov_cache = None
        self.width = self.height = None
        self.state = None
        self.templates = None
        self.track_query = np.zeros((1, 1, 768), dtype=np.float32)
        self.frame_id = 0
        self.last_quality = 1.0
        self._bind_graph(compiled_model)
        self.first_io = self._graph_io(first_compiled_model) if first_compiled_model is not None else None
        side = self.search_size // 16
        axis = 0.5 * (1.0 - np.cos((2.0 * np.pi / (side + 1.0)) * np.arange(1, side + 1, dtype=np.float32)))
        self.window = axis[:, None] * axis[None, :]

    def _graph_io(self, model):
        if model is None:
            return None
        inputs = list(model.inputs)
        outputs = list(model.outputs)
        templates = [x.any_name for x in inputs if list(x.shape) == [1, 3, self.template_size, self.template_size]]
        return {
            "templates": templates,
            "template": templates[0],
            "search": next(x.any_name for x in inputs if list(x.shape) == [1, 3, self.search_size, self.search_size]),
            "query": next((x.any_name for x in inputs if list(x.shape) == [1, 1, 768]), None),
            "score": next(x.any_name for x in outputs if list(x.shape) == [1, 1, self.search_size // 16, self.search_size // 16]),
            "size": next(x.any_name for x in outputs if list(x.shape) == [1, 2, self.search_size // 16, self.search_size // 16]),
            "offset": next(x.any_name for x in outputs if list(x.shape) == [1, 2, self.search_size // 16, self.search_size // 16]),
            "next_query": next(x.any_name for x in outputs if list(x.shape) == [1, 1, 768]),
        }

    def _bind_graph(self, model):
        io = self._graph_io(model)
        self.template_names = list(io["templates"])
        self.search_name = io["search"]
        self.query_name = io["query"]
        self.score_name = io["score"]
        self.size_name = io["size"]
        self.offset_name = io["offset"]
        self.next_query_name = io["next_query"]

    def init(self, frame_rgb, erp_box, init_bfov=None, **_kwargs):
        self.height, self.width = frame_rgb.shape[:2]
        if self.projection_mode == "tangent":
            self.bfov_state = (BFoV(*[float(v) for v in init_bfov[:4]])
                               if init_bfov is not None else
                               bfov_from_erp_bbox(*erp_box, self.width, self.height))
            # A local import avoids constructing a cache for the legacy ERP
            # path; the same cached OpenCV remap implementation as B224 is
            # used by the expert.
            from panotrack.geometry.projection import RemapCache
            self._ebfov_cache = RemapCache(capacity=128)
        # ODTrack's planar adapter tracks on the middle tile.  Keep the same
        # convention so a crop can cross the 0/360 seam without truncation.
        self.state = [float(erp_box[0]) + self.width, float(erp_box[1]),
                      float(erp_box[2]), float(erp_box[3])]
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        if self.projection_mode == "tangent":
            patch, _rf = sample_target_ebfov(
                frame_rgb, erp_box, self.template_factor, self.template_size,
                self.width, self.height, self._ebfov_cache, self.bfov_state)
        else:
            patch, _rf = sample_target(tiled, self.state, self.template_factor, self.template_size)
        t = preprocess_rgb(patch)
        self.templates = [t.copy(), t.copy(), t.copy()] if self.first_compiled is None else [t.copy()]
        self.track_query = np.zeros((1, 1, 768), dtype=np.float32)
        self.frame_id = 0
        self.last_quality = 1.0

    def _map_box(self, normalized, resize_factor):
        _, _, fh, fw = (1, 1, self.search_size // 16, self.search_size // 16)
        # The graph predicts coordinates in the resized search crop.  Undo
        # that resize before mapping the center/size back to the tiled ERP;
        # omitting this division collapses wide/polar boxes at the top edge.
        crop_scale = float(resize_factor)
        cx = float(normalized[0]) * self.search_size / crop_scale
        cy = float(normalized[1]) * self.search_size / crop_scale
        w = max(1.0, float(normalized[2]) * self.search_size / crop_scale)
        h = max(1.0, float(normalized[3]) * self.search_size / crop_scale)
        prev_cx = self.state[0] + 0.5 * self.state[2]
        prev_cy = self.state[1] + 0.5 * self.state[3]
        half = 0.5 * self.search_size / float(resize_factor)
        state = [cx + prev_cx - half - 0.5 * w,
                 cy + prev_cy - half - 0.5 * h, w, h]
        state = clip_box(state, self.height, 3 * self.width, margin=10)
        if self.seam_recenter:
            center = state[0] + 0.5 * state[2]
            state[0] += self.width * round((self.width * 1.5 - center) / self.width)
        return state

    def track(self, frame_rgb, **_kwargs):
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        if self.projection_mode == "tangent":
            patch, resize_factor = sample_target_ebfov(
                frame_rgb, self.state, self.search_factor, self.search_size,
                self.width, self.height, self._ebfov_cache, self.bfov_state)
        else:
            patch, resize_factor = sample_target(tiled, self.state, self.search_factor, self.search_size)
        use_first = self.first_compiled is not None and self.frame_id == 0
        compiled = self.first_compiled if use_first else self.compiled
        io = self.first_io if use_first else {
            "templates": self.template_names, "template": self.template_names[0], "search": self.search_name,
            "query": self.query_name, "score": self.score_name,
            "size": self.size_name, "offset": self.offset_name,
            "next_query": self.next_query_name,
        }
        inputs = {name: self.templates[min(i, len(self.templates) - 1)]
                  for i, name in enumerate(io["templates"])}
        inputs[io["search"]] = preprocess_rgb(patch)
        if io["query"] is not None:
            inputs[io["query"]] = self.track_query
        result = compiled(inputs)
        score = np.asarray(result[io["score"]])
        size = np.asarray(result[io["size"]])
        offset = np.asarray(result[io["offset"]])
        response = score[0, 0] * self.window
        flat = int(np.argmax(response))
        iy, ix = divmod(flat, response.shape[1])
        conf = float(response[iy, ix])
        normalized = np.asarray([(ix + float(offset[0, 0, iy, ix])) / response.shape[1],
                                 (iy + float(offset[0, 1, iy, ix])) / response.shape[0],
                                 float(size[0, 0, iy, ix]), float(size[0, 1, iy, ix])], dtype=np.float32)
        if self.projection_mode == "tangent":
            local_w = float(normalized[2]) * self.search_size
            local_h = float(normalized[3]) * self.search_size
            local_box = [float(normalized[0]) * self.search_size - 0.5 * local_w,
                         float(normalized[1]) * self.search_size - 0.5 * local_h,
                         local_w, local_h]
            self.bfov_state = bfov_from_local_box(local_box, resize_factor["bfov"], self.search_size)
            self.state = list(erp_bbox_from_bfov(self.bfov_state, self.width, self.height))
        else:
            self.state = self._map_box(normalized, resize_factor)
        self.track_query = np.asarray(result[io["next_query"]], dtype=np.float32)
        self.last_quality = conf
        self.frame_id += 1
        if (self.update_interval > 0 and self.frame_id % self.update_interval == 0 and
                conf >= self.update_threshold):
            if self.projection_mode == "tangent":
                z_patch, _ = sample_target_ebfov(
                    frame_rgb, self.state, self.template_factor, self.template_size,
                    self.width, self.height, self._ebfov_cache, self.bfov_state)
            else:
                z_patch, _ = sample_target(tiled, self.state, self.template_factor, self.template_size)
            new_template = preprocess_rgb(z_patch)
            if self.first_compiled is None:
                self.templates = [self.templates[1], self.templates[2], new_template]
            else:
                self.templates = [new_template]
        return {"target_bbox": [self.state[0] % self.width, self.state[1], self.state[2], self.state[3]],
                "quality": conf, "expert_used": "odtrack_openvino"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--first-xml", default=None,
                    help="optional first-step graph exported with one template and no track-query")
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq", default="train_sim/seq_0002")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=["CPU", "GPU", "NPU"], default="GPU")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--search-factor", type=float, default=5.0)
    ap.add_argument("--template-factor", type=float, default=2.0)
    ap.add_argument("--update-interval", type=int, default=25)
    ap.add_argument("--projection-mode", choices=["erp", "tangent"], default="erp")
    args = ap.parse_args(argv)
    import openvino as ov

    t0 = time.perf_counter()
    core = ov.Core()
    compiled = core.compile_model(str(Path(args.xml).resolve()), args.device)
    first_compiled = (core.compile_model(str(Path(args.first_xml).resolve()), args.device)
                      if args.first_xml else None)
    compile_seconds = time.perf_counter() - t0
    holder = {}

    def factory(**_kwargs):
        tracker = OpenVinoODTrackTracker(compiled, search_factor=args.search_factor,
                                         template_factor=args.template_factor,
                                         update_interval=args.update_interval,
                                         seam_recenter=True,
                                         first_compiled_model=first_compiled,
                                         projection_mode=args.projection_mode)
        holder["tracker"] = tracker
        return tracker

    metrics, pred, _valid, _w, _h, qualities, statuses, traces, _latency = run_sequence(
        args.seq, args.data, factory, args.max_frames)
    metrics.update({"compile_seconds": compile_seconds, "search_factor": args.search_factor,
                    "template_factor": args.template_factor, "graph": str(Path(args.xml).resolve()),
                    "device": args.device, "projection_mode": args.projection_mode})
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "results_erp.txt", pred, fmt="%.6f", delimiter=",")
    np.savetxt(out / "quality.txt", qualities, fmt="%.6f")
    (out / "status.txt").write_text("\n".join(statuses) + "\n", encoding="utf-8")
    with (out / "trace.jsonl").open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False, allow_nan=True) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
