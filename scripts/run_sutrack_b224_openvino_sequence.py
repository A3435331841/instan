#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one real sequence with the exported B224 graph on OpenVINO GPU/CPU."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval_official import run_sequence  # noqa: E402


MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def sample_target(image: np.ndarray, box, factor: float, output_size: int):
    x, y, w, h = [float(v) for v in box]
    crop_size = max(1, int(math.ceil(math.sqrt(max(1.0, w * h)) * factor)))
    x1 = int(round(x + 0.5 * w - 0.5 * crop_size))
    x2 = x1 + crop_size
    y1 = int(round(y + 0.5 * h - 0.5 * crop_size))
    y2 = y1 + crop_size
    x1_pad, x2_pad = max(0, -x1), max(x2 - image.shape[1] + 1, 0)
    y1_pad, y2_pad = max(0, -y1), max(y2 - image.shape[0] + 1, 0)
    crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    crop = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return crop, float(output_size) / float(crop_size)


def preprocess(patch: np.ndarray) -> np.ndarray:
    arr = patch.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.concatenate([arr, arr], axis=1).astype(np.float32, copy=False)


def clip_box(box, height: int, width: int, margin: int = 10):
    x1, y1, w, h = [float(v) for v in box]
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0.0, x1), width - margin)
    x2 = min(max(float(margin), x2), float(width))
    y1 = min(max(0.0, y1), height - margin)
    y2 = min(max(float(margin), y2), float(height))
    return [x1, y1, max(float(margin), x2 - x1), max(float(margin), y2 - y1)]


class OpenVinoB224Tracker:
    def __init__(self, compiled_model):
        self.compiled = compiled_model
        self.width = self.height = None
        self.state = None
        self.templates = None
        self.annos = None
        self.frame_id = 0
        axis = 0.5 * (1.0 - np.cos((2.0 * math.pi / 15.0) * np.arange(1, 15, dtype=np.float32)))
        self.window = axis[:, None] * axis[None, :]
        self.inputs = list(compiled_model.inputs)
        self.outputs = list(compiled_model.outputs)
        self.search_name = next(item.any_name for item in self.inputs if list(item.shape) == [1, 6, 224, 224])
        self.template_names = [item.any_name for item in self.inputs if list(item.shape) == [1, 6, 112, 112]]
        self.anno_names = [item.any_name for item in self.inputs if list(item.shape) == [1, 4]]

    def _anno(self, box, resize_factor, size=112):
        cx = (size - 1.0) * 0.5
        wh = np.asarray(box[2:4], dtype=np.float32) * float(resize_factor)
        xy = np.asarray([cx, cx], dtype=np.float32) - 0.5 * wh
        return np.concatenate([xy, wh]) / float(size - 1)

    def init(self, frame_rgb, erp_box, **_kwargs):
        self.height, self.width = frame_rgb.shape[:2]
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        self.state = [float(erp_box[0] % self.width + self.width), float(erp_box[1]),
                      float(erp_box[2]), float(erp_box[3])]
        patch, rf = sample_target(tiled, self.state, 2.0, 112)
        template = preprocess(patch)
        self.templates = [template.copy(), template.copy()]
        anno = self._anno(self.state, rf)
        self.annos = [anno.copy(), anno.copy()]
        self.frame_id = 0

    def track(self, frame_rgb, **_kwargs):
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        patch, rf = sample_target(tiled, self.state, 4.0, 224)
        inputs = {self.template_names[0]: self.templates[0], self.template_names[1]: self.templates[1],
                  self.anno_names[0]: self.annos[0][None, :], self.anno_names[1]: self.annos[1][None, :],
                  self.search_name: preprocess(patch)}
        result = self.compiled(inputs)
        score = np.asarray(result[self.outputs[0]])
        size = np.asarray(result[self.outputs[1]])
        offset = np.asarray(result[self.outputs[2]])
        _, _, fh, fw = score.shape
        response = score * self.window[None, None, :, :]
        flat_index = int(np.argmax(response.reshape(-1)))
        iy, ix = divmod(flat_index, fw)
        conf = float(response.reshape(-1)[flat_index])
        wh = size[0, :, iy, ix]
        off = offset[0, :, iy, ix]
        normalized = np.asarray([(ix + off[0]) / fw, (iy + off[1]) / fh, wh[0], wh[1]], dtype=np.float32)
        pred = normalized * (224.0 / float(rf))
        prev_cx = self.state[0] + 0.5 * self.state[2]
        prev_cy = self.state[1] + 0.5 * self.state[3]
        half = 0.5 * 224.0 / float(rf)
        state = [float(pred[0] + prev_cx - half - 0.5 * pred[2]),
                 float(pred[1] + prev_cy - half - 0.5 * pred[3]),
                 float(pred[2]), float(pred[3])]
        self.state = clip_box(state, self.height, 3 * self.width, margin=10)
        self.frame_id += 1
        if self.frame_id % 25 == 0 and conf > 0.70:
            z_patch, z_rf = sample_target(tiled, self.state, 2.0, 112)
            self.templates.append(preprocess(z_patch))
            self.annos.append(self._anno(self.state, z_rf))
            if len(self.templates) > 2:
                self.templates.pop(1)
                self.annos.pop(1)
        return {"target_bbox": [self.state[0] % self.width, self.state[1], self.state[2], self.state[3]],
                "quality": conf}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq", default="train_sim/seq_0011")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    import openvino as ov

    compile_started = time.perf_counter()
    compiled = ov.Core().compile_model(str(Path(args.xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - compile_started
    run_started = time.perf_counter()
    metrics, pred_erp, _valid, _width, _height, qualities, _statuses, _traces, _latency = run_sequence(
        args.seq, args.data, lambda **_kwargs: OpenVinoB224Tracker(compiled))
    wall_seconds = time.perf_counter() - run_started
    metrics["compile_seconds"] = compile_seconds
    metrics["wall_seconds_including_sequence_only"] = wall_seconds
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "results_erp.txt", pred_erp, fmt="%.6f", delimiter=",")
    np.savetxt(out / "quality.txt", qualities, fmt="%.6f")
    (out / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
