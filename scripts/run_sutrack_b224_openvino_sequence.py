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


def sample_target_polar(image: np.ndarray, box, factor: float, output_size: int,
                        latitude_deg: float):
    """Sample a locally rectified ERP crop near a pole.

    ERP longitude pixels are stretched by roughly ``1/cos(latitude)``.  We
    therefore sample a wider horizontal source window and resize it to the
    square model input.  The caller receives independent x/y resize factors so
    model coordinates can be mapped back without changing the tracker state
    convention.
    """
    x, y, w, h = [float(v) for v in box]
    crop_size = max(1, int(math.ceil(math.sqrt(max(1.0, w * h)) * factor)))
    cos_lat = max(0.25, abs(math.cos(math.radians(float(latitude_deg)))))
    crop_w = max(crop_size, int(math.ceil(float(crop_size) / cos_lat)))
    # The caller normally supplies a three-tile canvas.  Keep the requested
    # rectified window finite even for an extreme, highly elongated box.
    crop_w = min(crop_w, int(image.shape[1]))
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    x1 = int(round(cx - 0.5 * crop_w))
    x2 = x1 + crop_w
    y1 = int(round(cy - 0.5 * crop_size))
    y2 = y1 + crop_size
    x1_pad, x2_pad = max(0, -x1), max(x2 - image.shape[1] + 1, 0)
    y1_pad, y2_pad = max(0, -y1), max(y2 - image.shape[0] + 1, 0)
    crop = image[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]
    crop = cv2.copyMakeBorder(crop, y1_pad, y2_pad, x1_pad, x2_pad,
                              cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return crop, (float(output_size) / float(crop_w),
                  float(output_size) / float(crop_size))


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


def recenter_horizontal(box, width: int):
    """Keep an ERP box center in the middle copy of the three-tile canvas."""
    x, y, w, h = [float(v) for v in box]
    center = x + 0.5 * w
    center = ((center - float(width)) % float(width)) + float(width)
    return [center - 0.5 * w, y, w, h]


class OpenVinoB224Tracker:
    def __init__(self, compiled_model, search_size=224, template_size=112,
                 search_factor=4.0, template_factor=2.0,
                 update_interval=25, update_threshold=0.70,
                 search_factor_mode="fixed", fallback_search_factor=None,
                 fallback_quality_threshold=0.45, fallback_min_gain=0.0,
                 fallback_cooldown=1, fallback_run=1, fallback_start_frame=0,
                 seam_recenter=False,
                 polar_rectify=False, polar_latitude_threshold=55.0,
                 polar_aspect_max=2.5, polar_small_width=100.0,
                 polar_max_frame=None, polar_require_initial=True,
                 small_template_factor=None, small_template_width=100.0,
                 small_template_require_initial=True):
        self.compiled = compiled_model
        self.width = self.height = None
        self.state = None
        self.templates = None
        self.annos = None
        self.frame_id = 0
        self.search_factor = float(search_factor)
        self.search_factor_mode = str(search_factor_mode)
        self.active_search_factor = self.search_factor
        self.template_factor = float(template_factor)
        self.search_size = int(search_size)
        self.template_size = int(template_size)
        self.update_interval = int(update_interval)
        self.update_threshold = float(update_threshold)
        self.fallback_search_factor = (None if fallback_search_factor is None
                                       else float(fallback_search_factor))
        self.fallback_quality_threshold = float(fallback_quality_threshold)
        self.fallback_min_gain = float(fallback_min_gain)
        self.fallback_cooldown = max(1, int(fallback_cooldown))
        self.fallback_run = max(1, int(fallback_run))
        self.fallback_start_frame = max(0, int(fallback_start_frame))
        self.fallback_low_run = 0
        self.seam_recenter = bool(seam_recenter)
        self.polar_rectify = bool(polar_rectify)
        self.polar_latitude_threshold = float(polar_latitude_threshold)
        self.polar_aspect_max = (None if polar_aspect_max is None
                                 else float(polar_aspect_max))
        self.polar_small_width = float(polar_small_width)
        self.polar_max_frame = (None if polar_max_frame is None
                                else int(polar_max_frame))
        self.polar_require_initial = bool(polar_require_initial)
        self.initial_latitude = None
        self.small_template_factor = (None if small_template_factor is None
                                      else float(small_template_factor))
        self.small_template_width = float(small_template_width)
        self.small_template_require_initial = bool(small_template_require_initial)
        self.initial_target_width = None
        self.polar_sample_count = 0
        self.fallback_calls = 0
        self.fallback_selected = 0
        self.last_fallback_used = False
        self.last_motion_deg = 0.0
        self.last_quality = 1.0
        side = self.search_size // 16
        axis = 0.5 * (1.0 - np.cos((2.0 * math.pi / (side + 1.0)) *
                                   np.arange(1, side + 1, dtype=np.float32)))
        self.window = axis[:, None] * axis[None, :]
        self.inputs = list(compiled_model.inputs)
        self.outputs = list(compiled_model.outputs)
        self.search_name = next(item.any_name for item in self.inputs if list(item.shape) == [1, 6, self.search_size, self.search_size])
        self.template_names = [item.any_name for item in self.inputs if list(item.shape) == [1, 6, self.template_size, self.template_size]]
        self.anno_names = [item.any_name for item in self.inputs if list(item.shape) == [1, 4]]

    def _latitude(self, box):
        cy = float(box[1]) + 0.5 * float(box[3])
        return 90.0 - cy / float(self.height) * 180.0

    def _sample(self, image, box, factor, output_size, template=False):
        small_width_ok = (float(box[2]) <= self.small_template_width)
        if self.small_template_require_initial:
            small_width_ok = (self.initial_target_width is not None and
                              self.initial_target_width <= self.small_template_width)
        if (template and self.small_template_factor is not None and small_width_ok):
            factor = self.small_template_factor
        latitude = self._latitude(box)
        aspect = float(box[2]) / max(1.0, float(box[3]))
        polar_shape_ok = (self.polar_aspect_max is None or
                          aspect <= self.polar_aspect_max or
                          float(box[2]) <= self.polar_small_width)
        if (self.polar_rectify and
                abs(latitude) >= self.polar_latitude_threshold and polar_shape_ok and
                (not self.polar_require_initial or
                 (self.initial_latitude is not None and
                  abs(self.initial_latitude) >= self.polar_latitude_threshold)) and
                (self.polar_max_frame is None or self.frame_id <= self.polar_max_frame)):
            self.polar_sample_count += 1
            return sample_target_polar(image, box, factor, output_size, latitude)
        patch, rf = sample_target(image, box, factor, output_size)
        return patch, (rf, rf)

    def _anno(self, box, resize_factor, size=None):
        size = self.template_size if size is None else int(size)
        cx = (size - 1.0) * 0.5
        if np.isscalar(resize_factor):
            resize_factor = (float(resize_factor), float(resize_factor))
        wh = np.asarray(box[2:4], dtype=np.float32) * np.asarray(
            resize_factor, dtype=np.float32)
        xy = np.asarray([cx, cx], dtype=np.float32) - 0.5 * wh
        return np.concatenate([xy, wh]) / float(size - 1)

    def init(self, frame_rgb, erp_box, init_bfov=None, **_kwargs):
        self.height, self.width = frame_rgb.shape[:2]
        self.initial_latitude = self._latitude(erp_box)
        self.initial_target_width = float(erp_box[2])
        if self.search_factor_mode == "moderate_fov":
            # Geometry-only conditional crop: the factor-3.5 ablation helped
            # moderate-FOV OD-dominant scenes but harmed very wide/small-FOV
            # B224 strengths.  This rule uses only the initial BFoV, never a
            # sequence name or ground truth.
            if init_bfov is None:
                from scripts.eval_official import bfov_from_erp_bbox
                initial = bfov_from_erp_bbox(*erp_box, self.width, self.height)
                fov_h, fov_v = initial.fov_h, initial.fov_v
            else:
                fov_h, fov_v = float(init_bfov[2]), float(init_bfov[3])
            if 25.0 <= fov_h <= 60.0 and fov_v <= 70.0:
                self.active_search_factor = 3.5
            else:
                self.active_search_factor = 4.0
        elif self.search_factor_mode == "large_fov":
            # Large targets occupy most of the spherical view.  A factor-4
            # crop makes them too small in the fixed 224 search token grid;
            # use a tighter crop only when both axes are genuinely wide.
            if init_bfov is None:
                from scripts.eval_official import bfov_from_erp_bbox
                initial = bfov_from_erp_bbox(*erp_box, self.width, self.height)
                fov_h, fov_v = initial.fov_h, initial.fov_v
            else:
                fov_h, fov_v = float(init_bfov[2]), float(init_bfov[3])
            if fov_h >= 90.0 and fov_v >= 100.0:
                self.active_search_factor = 2.0
            else:
                self.active_search_factor = 4.0
        elif self.search_factor_mode != "fixed":
            raise ValueError(f"unknown search_factor_mode: {self.search_factor_mode}")
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        self.state = [float(erp_box[0] % self.width + self.width), float(erp_box[1]),
                      float(erp_box[2]), float(erp_box[3])]
        patch, rf = self._sample(tiled, self.state, self.template_factor,
                                 self.template_size, template=True)
        template = preprocess(patch)
        self.templates = [template.copy(), template.copy()]
        anno = self._anno(self.state, rf)
        self.annos = [anno.copy(), anno.copy()]
        self.frame_id = 0
        self.polar_sample_count = 0
        self.fallback_low_run = 0

    def _infer_candidate(self, tiled, factor):
        """Run one pure candidate search without mutating tracker state.

        The extra candidate is deliberately evaluated against the same two
        templates and Hann-windowed response as the primary search.  This is
        the lightweight analogue of ODTrack's candidate-elimination idea: only
        frames whose primary response is weak pay for a second crop.
        """
        patch, rf = self._sample(tiled, self.state, factor, self.search_size)
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
        self.last_quality = conf
        wh = size[0, :, iy, ix]
        off = offset[0, :, iy, ix]
        normalized = np.asarray([(ix + off[0]) / fw, (iy + off[1]) / fh, wh[0], wh[1]], dtype=np.float32)
        scale_x = float(self.search_size) / float(rf[0])
        scale_y = float(self.search_size) / float(rf[1])
        pred = np.asarray([normalized[0] * scale_x, normalized[1] * scale_y,
                           normalized[2] * scale_x, normalized[3] * scale_y],
                          dtype=np.float32)
        prev_cx = self.state[0] + 0.5 * self.state[2]
        prev_cy = self.state[1] + 0.5 * self.state[3]
        half_x = 0.5 * float(self.search_size) / float(rf[0])
        half_y = 0.5 * float(self.search_size) / float(rf[1])
        state = [float(pred[0] + prev_cx - half_x - 0.5 * pred[2]),
                 float(pred[1] + prev_cy - half_y - 0.5 * pred[3]),
                 float(pred[2]), float(pred[3])]
        state = clip_box(state, self.height, 3 * self.width, margin=10)
        if self.seam_recenter:
            state = recenter_horizontal(state, self.width)
        return {"state": state, "conf": conf, "factor": float(factor)}

    def track(self, frame_rgb, **_kwargs):
        tiled = np.concatenate([frame_rgb, frame_rgb, frame_rgb], axis=1)
        prev_cx = self.state[0] + 0.5 * self.state[2]
        prev_cy = self.state[1] + 0.5 * self.state[3]
        primary = self._infer_candidate(tiled, self.active_search_factor)
        chosen = primary
        self.last_fallback_used = False
        if primary["conf"] <= self.fallback_quality_threshold:
            self.fallback_low_run += 1
        else:
            self.fallback_low_run = 0
        if (self.fallback_search_factor is not None and
                self.frame_id >= self.fallback_start_frame and
                self.frame_id % self.fallback_cooldown == 0 and
                self.fallback_low_run >= self.fallback_run and
                abs(self.fallback_search_factor - self.active_search_factor) > 1e-6):
            self.fallback_calls += 1
            alternate = self._infer_candidate(tiled, self.fallback_search_factor)
            if alternate["conf"] >= primary["conf"] + self.fallback_min_gain:
                chosen = alternate
                self.fallback_selected += 1
                self.last_fallback_used = True
        self.state = chosen["state"]
        conf = float(chosen["conf"])
        new_cx = self.state[0] + 0.5 * self.state[2]
        new_cy = self.state[1] + 0.5 * self.state[3]
        # Circular longitude delta: the internal state may be represented in
        # any tile, so a seam crossing must not appear as a 360-degree jump.
        dx = abs(((new_cx - prev_cx + 0.5 * self.width) % self.width)
                 - 0.5 * self.width)
        dy = abs(new_cy - prev_cy)
        self.last_motion_deg = float(np.hypot(dx * 360.0 / self.width,
                                               dy * 180.0 / self.height))
        self.last_quality = conf
        self.frame_id += 1
        if self.update_interval > 0 and self.frame_id % self.update_interval == 0 and conf > self.update_threshold:
            z_patch, z_rf = self._sample(tiled, self.state, self.template_factor,
                                         self.template_size, template=True)
            self.templates.append(preprocess(z_patch))
            self.annos.append(self._anno(self.state, z_rf))
            if len(self.templates) > 2:
                self.templates.pop(1)
                self.annos.pop(1)
        return {"target_bbox": [self.state[0] % self.width, self.state[1], self.state[2], self.state[3]],
                "quality": conf,
                "fallback_used": self.last_fallback_used,
                "fallback_factor": chosen["factor"]}


class MotionAdaptiveTracker:
    """Use early quality evidence to choose a larger-template B224 branch."""
    def __init__(self, base_model, high_model, warmup=5, threshold_deg=1.5,
                 quality_threshold=0.40, quality_run=3, switch_deadline=30,
                 fallback_search_factor=None, fallback_quality_threshold=0.45,
                 fallback_min_gain=0.0, fallback_cooldown=1, fallback_run=1,
                 fallback_start_frame=0,
                 search_factor=4.0, search_factor_mode="fixed",
                 seam_recenter=False, polar_rectify=False,
                 polar_latitude_threshold=55.0, polar_aspect_max=2.5,
                 polar_small_width=100.0, polar_max_frame=None,
                 polar_require_initial=True, small_template_factor=None,
                 small_template_width=100.0, small_template_require_initial=True):
        self.base = OpenVinoB224Tracker(base_model, search_size=224, template_size=112,
                                        search_factor=search_factor,
                                        search_factor_mode=search_factor_mode,
                                        fallback_search_factor=fallback_search_factor,
                                        fallback_quality_threshold=fallback_quality_threshold,
                                        fallback_min_gain=fallback_min_gain,
                                        fallback_cooldown=fallback_cooldown,
                                        fallback_run=fallback_run,
                                        fallback_start_frame=fallback_start_frame,
                                        seam_recenter=seam_recenter,
                                        polar_rectify=polar_rectify,
                                        polar_latitude_threshold=polar_latitude_threshold,
                                        polar_aspect_max=polar_aspect_max,
                                        polar_small_width=polar_small_width,
                                        polar_max_frame=polar_max_frame,
                                        polar_require_initial=polar_require_initial,
                                        small_template_factor=small_template_factor,
                                        small_template_width=small_template_width,
                                        small_template_require_initial=small_template_require_initial)
        self.high_model = high_model
        self.search_factor = float(search_factor)
        self.search_factor_mode = str(search_factor_mode)
        self.fallback_search_factor = (None if fallback_search_factor is None
                                       else float(fallback_search_factor))
        self.fallback_quality_threshold = float(fallback_quality_threshold)
        self.fallback_min_gain = float(fallback_min_gain)
        self.fallback_cooldown = int(fallback_cooldown)
        self.fallback_run = int(fallback_run)
        self.fallback_start_frame = int(fallback_start_frame)
        self.seam_recenter = bool(seam_recenter)
        self.polar_rectify = bool(polar_rectify)
        self.polar_latitude_threshold = float(polar_latitude_threshold)
        self.polar_aspect_max = (None if polar_aspect_max is None
                                 else float(polar_aspect_max))
        self.polar_small_width = float(polar_small_width)
        self.polar_max_frame = (None if polar_max_frame is None
                                else int(polar_max_frame))
        self.polar_require_initial = bool(polar_require_initial)
        self.small_template_factor = (None if small_template_factor is None
                                      else float(small_template_factor))
        self.small_template_width = float(small_template_width)
        self.small_template_require_initial = bool(small_template_require_initial)
        self.warmup = int(warmup)
        self.threshold_deg = float(threshold_deg)
        self.quality_threshold = float(quality_threshold)
        self.quality_run = int(quality_run)
        self.switch_deadline = int(switch_deadline)
        self.active = self.base
        self.motions = []
        self.qualities = []
        self.switched = False
        self.high = None
        self.switch_frame = None

    def init(self, frame_rgb, erp_box, init_bfov=None, **kwargs):
        self.base.init(frame_rgb, erp_box, init_bfov=init_bfov, **kwargs)
        self.active = self.base
        self.motions = []
        self.qualities = []
        self.switched = False
        self.high = None
        self.switch_frame = None

    def track(self, frame_rgb, **kwargs):
        out = self.active.track(frame_rgb, **kwargs)
        if self.switched:
            return out
        self.motions.append(self.base.last_motion_deg)
        self.qualities.append(self.base.last_quality)
        low_quality = (len(self.qualities) >= self.quality_run and
                       float(np.median(self.qualities[-self.quality_run:])) <= self.quality_threshold)
        # Motion estimates can be spuriously small when a lost tracker happens
        # to bounce back near its previous center.  Require the calibrated
        # quality signal for the actual switch; motion is retained only as a
        # diagnostic trace.
        if self.base.frame_id <= self.switch_deadline and low_quality:
            current = [self.base.state[0] % self.base.width, self.base.state[1],
                       self.base.state[2], self.base.state[3]]
            self.high = OpenVinoB224Tracker(self.high_model, search_size=224, template_size=128,
                                            search_factor=self.search_factor,
                                            search_factor_mode=self.search_factor_mode,
                                            fallback_search_factor=self.fallback_search_factor,
                                            fallback_quality_threshold=self.fallback_quality_threshold,
                                            fallback_min_gain=self.fallback_min_gain,
                                            fallback_cooldown=self.fallback_cooldown,
                                            fallback_run=self.fallback_run,
                                            fallback_start_frame=self.fallback_start_frame,
                                            seam_recenter=self.seam_recenter,
                                            polar_rectify=self.polar_rectify,
                                            polar_latitude_threshold=self.polar_latitude_threshold,
                                            polar_aspect_max=self.polar_aspect_max,
                                            polar_small_width=self.polar_small_width,
                                            polar_max_frame=self.polar_max_frame,
                                            polar_require_initial=self.polar_require_initial,
                                            small_template_factor=self.small_template_factor,
                                            small_template_width=self.small_template_width,
                                            small_template_require_initial=self.small_template_require_initial)
            self.high.init(frame_rgb, current, init_bfov=None)
            self.active = self.high
            self.switched = True
            self.switch_frame = self.base.frame_id
        return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seq", default="train_sim/seq_0011")
    parser.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--high-xml", default=None,
                        help="second OpenVINO graph (typically template128) for motion-adaptive mode")
    parser.add_argument("--motion-adaptive", action="store_true")
    parser.add_argument("--motion-warmup", type=int, default=5)
    parser.add_argument("--motion-threshold-deg", type=float, default=1.5)
    parser.add_argument("--quality-threshold", type=float, default=0.40)
    parser.add_argument("--quality-run", type=int, default=3)
    parser.add_argument("--switch-deadline", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--search-factor", type=float, default=4.0)
    parser.add_argument("--search-factor-mode", choices=["fixed", "moderate_fov", "large_fov"], default="fixed")
    parser.add_argument("--template-factor", type=float, default=2.0)
    parser.add_argument("--search-size", type=int, default=224)
    parser.add_argument("--template-size", type=int, default=112)
    parser.add_argument("--update-interval", type=int, default=25)
    parser.add_argument("--update-threshold", type=float, default=0.70)
    parser.add_argument("--fallback-search-factor", type=float, default=None,
                        help="optional second crop factor evaluated only on weak responses")
    parser.add_argument("--fallback-quality-threshold", type=float, default=0.45)
    parser.add_argument("--fallback-min-gain", type=float, default=0.0,
                        help="minimum Hann-windowed response gain required to accept fallback")
    parser.add_argument("--fallback-cooldown", type=int, default=1)
    parser.add_argument("--fallback-run", type=int, default=1,
                        help="consecutive weak primary responses before probing fallback")
    parser.add_argument("--fallback-start-frame", type=int, default=0)
    parser.add_argument("--seam-recenter", action="store_true",
                        help="recenter the internal ERP box in the middle tile after each update")
    parser.add_argument("--polar-rectify", action="store_true",
                        help="horizontally rectify high-latitude ERP crops using cos(latitude)")
    parser.add_argument("--polar-latitude-threshold", type=float, default=55.0)
    parser.add_argument("--polar-aspect-max", type=float, default=2.5,
                        help="only rectify polar boxes with w/h below this value")
    parser.add_argument("--polar-small-width", type=float, default=100.0,
                        help="also rectify polar boxes no wider than this many ERP pixels")
    parser.add_argument("--polar-max-frame", type=int, default=None,
                        help="optional early-frame limit for polar rectification")
    parser.add_argument("--no-polar-require-initial", dest="polar_require_initial",
                        action="store_false",
                        help="allow rectification when a normal-latitude target later reaches a pole")
    parser.set_defaults(polar_require_initial=True)
    parser.add_argument("--small-template-factor", type=float, default=None,
                        help="optional tighter template context for small ERP targets")
    parser.add_argument("--small-template-width", type=float, default=100.0)
    parser.add_argument("--no-small-template-require-initial",
                        dest="small_template_require_initial", action="store_false",
                        help="allow tighter templates when a target becomes small later")
    parser.set_defaults(small_template_require_initial=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)
    import openvino as ov

    compile_started = time.perf_counter()
    compiled = ov.Core().compile_model(str(Path(args.xml).resolve()), args.device)
    compiled_high = None
    if args.motion_adaptive:
        if not args.high_xml:
            raise SystemExit("--motion-adaptive requires --high-xml")
        compiled_high = ov.Core().compile_model(str(Path(args.high_xml).resolve()), args.device)
    compile_seconds = time.perf_counter() - compile_started
    run_started = time.perf_counter()
    tracker_holder = {}
    if args.motion_adaptive:
        def tracker_factory(**_kwargs):
            tracker_holder["tracker"] = MotionAdaptiveTracker(
                compiled, compiled_high, warmup=args.motion_warmup,
                threshold_deg=args.motion_threshold_deg,
                quality_threshold=args.quality_threshold, quality_run=args.quality_run,
                switch_deadline=args.switch_deadline,
                search_factor=args.search_factor,
                search_factor_mode=args.search_factor_mode,
                fallback_search_factor=args.fallback_search_factor,
                fallback_quality_threshold=args.fallback_quality_threshold,
                fallback_min_gain=args.fallback_min_gain,
                fallback_cooldown=args.fallback_cooldown,
                fallback_run=args.fallback_run,
                fallback_start_frame=args.fallback_start_frame,
                seam_recenter=args.seam_recenter,
                polar_rectify=args.polar_rectify,
                polar_latitude_threshold=args.polar_latitude_threshold,
                polar_aspect_max=args.polar_aspect_max,
                polar_small_width=args.polar_small_width,
                polar_max_frame=args.polar_max_frame,
                polar_require_initial=args.polar_require_initial,
                small_template_factor=args.small_template_factor,
                small_template_width=args.small_template_width,
                small_template_require_initial=args.small_template_require_initial)
            return tracker_holder["tracker"]
    else:
        def tracker_factory(**_kwargs):
            tracker_holder["tracker"] = OpenVinoB224Tracker(
                compiled, search_factor=args.search_factor, template_factor=args.template_factor,
                search_size=args.search_size, template_size=args.template_size,
                update_interval=args.update_interval, update_threshold=args.update_threshold,
                search_factor_mode=args.search_factor_mode,
                fallback_search_factor=args.fallback_search_factor,
                fallback_quality_threshold=args.fallback_quality_threshold,
                fallback_min_gain=args.fallback_min_gain,
                fallback_cooldown=args.fallback_cooldown,
                fallback_run=args.fallback_run,
                fallback_start_frame=args.fallback_start_frame,
                seam_recenter=args.seam_recenter,
                polar_rectify=args.polar_rectify,
                polar_latitude_threshold=args.polar_latitude_threshold,
                polar_aspect_max=args.polar_aspect_max,
                polar_small_width=args.polar_small_width,
                polar_max_frame=args.polar_max_frame,
                polar_require_initial=args.polar_require_initial,
                small_template_factor=args.small_template_factor,
                small_template_width=args.small_template_width,
                small_template_require_initial=args.small_template_require_initial)
            return tracker_holder["tracker"]
    metrics, pred_erp, _valid, _width, _height, qualities, _statuses, _traces, _latency = run_sequence(
        args.seq, args.data, tracker_factory, args.max_frames)
    wall_seconds = time.perf_counter() - run_started
    metrics["compile_seconds"] = compile_seconds
    metrics["wall_seconds_including_sequence_only"] = wall_seconds
    metrics["search_factor"] = args.search_factor
    metrics["search_factor_mode"] = args.search_factor_mode
    metrics["motion_adaptive"] = args.motion_adaptive
    metrics["motion_warmup"] = args.motion_warmup
    metrics["motion_threshold_deg"] = args.motion_threshold_deg
    metrics["quality_threshold"] = args.quality_threshold
    metrics["quality_run"] = args.quality_run
    metrics["switch_deadline"] = args.switch_deadline
    if args.motion_adaptive and "tracker" in tracker_holder:
        metrics["motion_history_deg"] = tracker_holder["tracker"].motions[:60]
        metrics["motion_history_count"] = len(tracker_holder["tracker"].motions)
        metrics["switch_frame"] = tracker_holder["tracker"].switch_frame
    metrics["template_factor"] = args.template_factor
    metrics["update_interval"] = args.update_interval
    metrics["update_threshold"] = args.update_threshold
    metrics["fallback_search_factor"] = args.fallback_search_factor
    metrics["fallback_quality_threshold"] = args.fallback_quality_threshold
    metrics["fallback_min_gain"] = args.fallback_min_gain
    metrics["fallback_cooldown"] = args.fallback_cooldown
    metrics["fallback_run"] = args.fallback_run
    metrics["fallback_start_frame"] = args.fallback_start_frame
    metrics["seam_recenter"] = args.seam_recenter
    metrics["polar_rectify"] = args.polar_rectify
    metrics["polar_latitude_threshold"] = args.polar_latitude_threshold
    metrics["polar_aspect_max"] = args.polar_aspect_max
    metrics["polar_small_width"] = args.polar_small_width
    metrics["polar_max_frame"] = args.polar_max_frame
    metrics["polar_require_initial"] = args.polar_require_initial
    metrics["small_template_factor"] = args.small_template_factor
    metrics["small_template_width"] = args.small_template_width
    metrics["small_template_require_initial"] = args.small_template_require_initial
    if not args.motion_adaptive and "tracker" in tracker_holder:
        metrics["active_search_factor"] = tracker_holder["tracker"].active_search_factor
        metrics["fallback_calls"] = tracker_holder["tracker"].fallback_calls
        metrics["fallback_selected"] = tracker_holder["tracker"].fallback_selected
        metrics["polar_sample_count"] = tracker_holder["tracker"].polar_sample_count
    elif args.motion_adaptive and "tracker" in tracker_holder:
        metrics["active_search_factor"] = tracker_holder["tracker"].base.active_search_factor
        metrics["fallback_calls"] = tracker_holder["tracker"].base.fallback_calls
        metrics["fallback_selected"] = tracker_holder["tracker"].base.fallback_selected
        metrics["polar_sample_count"] = tracker_holder["tracker"].base.polar_sample_count
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "results_erp.txt", pred_erp, fmt="%.6f", delimiter=",")
    np.savetxt(out / "quality.txt", qualities, fmt="%.6f")
    (out / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
