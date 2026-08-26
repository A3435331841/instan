# -*- coding: utf-8 -*-
"""Online, inference-only adaptive router for spherical tracking experts.

This module contains no dataset or sequence-name routing.  Decisions are made
from tracker evidence, spherical geometry, motion state and a strict expert
latency budget.  It accepts both the panotrack ``init/update`` contract and the
official-evaluation ``init/track`` adapter contract.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from panotrack.evaluation.metrics import dual_iou
from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov
from panotrack.geometry.descriptor import GeometryDescriptor
from panotrack.pipeline.state import SphericalState
from panotrack.pipeline.risk_policy import LinearRiskPolicy


class AdaptiveStatus(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"
    LOST = "lost"
    VERIFY = "verify"


@dataclass(frozen=True)
class AdaptiveRouterConfig:
    """Rule-router defaults; all values are inference-time observable."""

    suspect_quality: float = 0.45
    lost_quality: float = 0.25
    recover_quality: float = 0.55
    suspect_run: int = 2
    lost_run: int = 5
    verify_frames: int = 3
    geometry_risk: float = 0.55
    polar_lat_deg: float = 60.0
    seam_margin_fraction: float = 0.08
    small_fov_deg: float = 10.0
    large_fov_h_deg: float = 70.0
    large_fov_v_deg: float = 100.0
    expert_interval: int = 1
    expert_max_fraction: float = 0.20
    expert_bucket_capacity: float = 1.0
    target_frame_ms: float = 33.333
    latency_ema_alpha: float = 0.10
    expert_quality: float = 0.50
    expert_switch_margin: float = 0.05
    disagreement_iou: float = 0.30
    expert_episode_frames: int = 10
    redetect_interval: int = 5
    enable_global_redetect: bool = True
    redetect_min_score: float = 0.45
    anchor_min_similarity: float = 0.50
    motion_max_deg: float = 120.0


class ExpertBudget:
    """Token bucket that caps the long-run fraction of slow expert probes."""

    def __init__(self, fraction: float, capacity: float = 1.0):
        self.fraction = float(np.clip(fraction, 0.0, 1.0))
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.frames = 0
        self.calls = 0

    def advance(self) -> None:
        self.frames += 1
        self.tokens = min(self.capacity, self.tokens + self.fraction)

    def consume(self, force: bool = False) -> bool:
        if force:
            self.tokens -= 1.0
            self.calls += 1
            return True
        if self.tokens + 1e-12 < 1.0:
            return False
        self.tokens -= 1.0
        self.calls += 1
        return True

    @property
    def call_fraction(self) -> float:
        return self.calls / max(1, self.frames)


def _call_init(tracker: object, frame: np.ndarray, bbox: Sequence[float]) -> None:
    init = getattr(tracker, "init", None)
    if init is not None:
        init(frame, bbox)
        return
    initialize = getattr(tracker, "initialize", None)
    if initialize is None:
        raise TypeError("tracker must expose init(frame,bbox) or initialize(frame,info)")
    initialize(frame, {"init_bbox": [float(value) for value in bbox]})


def _call_track(tracker: object, frame: np.ndarray) -> Mapping[str, object]:
    track = getattr(tracker, "track", None)
    output = track(frame) if track is not None else tracker.update(frame)
    if not isinstance(output, Mapping):
        raise TypeError("tracker output must be a mapping")
    return output


def _normalize_output(output: Mapping[str, object], width: int, height: int) -> dict[str, object]:
    box = output.get("target_bbox", output.get("bbox"))
    if box is None or len(box) < 4:
        raise ValueError("tracker output is missing a four-value bbox")
    x, y, w, h = (float(value) for value in box[:4])
    if not np.isfinite([x, y, w, h]).all() or w <= 0.0 or h <= 0.0:
        raise ValueError(f"tracker returned invalid bbox: {box}")
    w = float(np.clip(w, 1.0, width))
    h = float(np.clip(h, 1.0, height))
    y = float(np.clip(y, 0.0, max(0.0, height - h)))
    quality = output.get("quality", output.get("score", output.get("best_score", 0.5)))
    try:
        quality = float(quality)
    except (TypeError, ValueError):
        quality = 0.5
    return {
        "target_bbox": [x % width, y, w, h],
        "quality": float(np.clip(quality, 0.0, 1.0)),
        "response_entropy": _optional_float(output.get("response_entropy")),
        "peak_margin": _optional_float(output.get("peak_margin")),
    }


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _crop_wrap_raw(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = (float(value) for value in box)
    crop_w = max(2, int(round(w)))
    crop_h = max(2, int(round(h)))
    y0 = int(np.clip(round(y), 0, max(0, height - crop_h)))
    columns = np.mod(int(round(x)) + np.arange(crop_w), width)
    return np.ascontiguousarray(frame[y0:y0 + crop_h][:, columns])


def _crop_wrap(frame: np.ndarray, box: Sequence[float], out_size: int = 48) -> np.ndarray:
    crop = _crop_wrap_raw(frame, box)
    if crop.size == 0:
        return np.zeros((out_size, out_size), dtype=np.float32)
    # Dependency-free nearest-neighbour normalization is sufficient for the
    # lightweight identity gate and avoids adding OpenCV/Pillow to this module.
    ys = np.minimum((np.arange(out_size) * crop.shape[0] / out_size).astype(int),
                    crop.shape[0] - 1)
    xs = np.minimum((np.arange(out_size) * crop.shape[1] / out_size).astype(int),
                    crop.shape[1] - 1)
    normalized = crop[ys[:, None], xs[None, :]]
    if normalized.ndim == 3:
        normalized = (0.299 * normalized[..., 0]
                      + 0.587 * normalized[..., 1]
                      + 0.114 * normalized[..., 2])
    return normalized.astype(np.float32)


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    a = a - mean_a
    b = b - mean_b
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-9:
        # A constant-color synthetic target can still be a perfect identity
        # match; compare its mean when NCC itself is undefined.
        return float(np.exp(-abs(mean_a - mean_b) / 32.0))
    return float(np.clip((a * b).sum() / denominator, -1.0, 1.0))


def _angular_distance(a: BFoV, b: BFoV) -> float:
    lon1, lon2 = np.deg2rad(a.lon), np.deg2rad(b.lon)
    lat1, lat2 = np.deg2rad(a.lat), np.deg2rad(b.lat)
    cosine = (np.sin(lat1) * np.sin(lat2)
              + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    return float(np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0))))


class AdaptiveSphericalTracker:
    """Fast main tracker plus sparse expert and conservative re-detection."""

    def __init__(
        self,
        main_tracker: object,
        expert_tracker: object | None = None,
        redetector: object | None = None,
        config: AdaptiveRouterConfig | None = None,
        risk_policy: LinearRiskPolicy | None = None,
        main_name: str = "main",
        expert_name: str = "expert",
    ):
        self.main = main_tracker
        self.expert = expert_tracker
        self.redetector = redetector
        self.config = config or AdaptiveRouterConfig()
        self.risk_policy = risk_policy
        self.main_name = str(main_name)
        self.expert_name = str(expert_name)
        self.descriptor = GeometryDescriptor()
        self.budget = ExpertBudget(
            self.config.expert_max_fraction, self.config.expert_bucket_capacity)
        self.width = 0
        self.height = 0
        self.status = AdaptiveStatus.NORMAL
        self.state: SphericalState | None = None
        self.anchor: np.ndarray | None = None
        self.anchor_template: tuple[np.ndarray, tuple[float, float]] | None = None
        self.last_frame: np.ndarray | None = None
        self.last_box: list[float] | None = None
        self.low_run = 0
        self.verify_run = 0
        self.status_run = 0
        self.frame_index = 0
        self.redetect_count = 0
        self.last_quality = 1.0
        self._previous_main_quality = 1.0
        self._previous_main_log_area = 0.0
        self.expert_episode_remaining = 0
        self.main_latency_ema_ms: float | None = None
        self.expert_latency_ema_ms: float | None = None

    def init(self, frame: np.ndarray, init_bfov: BFoV | Sequence[float]) -> None:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (H,W,3)")
        self.height, self.width = frame.shape[:2]
        if isinstance(init_bfov, BFoV):
            box = list(erp_bbox_from_bfov(init_bfov, self.width, self.height))
            bfov = init_bfov
        else:
            box = [float(value) for value in init_bfov[:4]]
            bfov = bfov_from_erp_bbox(*box, self.width, self.height)
        _call_init(self.main, frame, box)
        self.state = SphericalState(bfov)
        self.anchor = _crop_wrap(frame, box)
        self.anchor_template = (
            _crop_wrap_raw(frame, box), (float(box[2]), float(box[3])))
        if self.redetector is None and self.config.enable_global_redetect:
            from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector

            self.redetector = SphericalMultiViewRedetector(
                lambda: [self.anchor_template] if self.anchor_template is not None else [],
                min_score=self.config.redetect_min_score,
            )
        self.last_frame = np.ascontiguousarray(frame)
        self.last_box = box
        self.status = AdaptiveStatus.NORMAL
        self.low_run = self.verify_run = self.status_run = self.frame_index = 0
        self.redetect_count = 0
        self.last_quality = 1.0
        self._previous_main_quality = 1.0
        self._previous_main_log_area = math.log(max(float(box[2] * box[3]), 4.0))
        self.expert_episode_remaining = 0
        self.main_latency_ema_ms = None
        self.expert_latency_ema_ms = None

    def _update_latency_budget(self, main_ms: float, expert_ms: float) -> None:
        alpha = float(np.clip(self.config.latency_ema_alpha, 0.01, 1.0))
        if main_ms > 0.0:
            self.main_latency_ema_ms = (main_ms if self.main_latency_ema_ms is None
                                        else (1.0 - alpha) * self.main_latency_ema_ms
                                        + alpha * main_ms)
        if expert_ms > 0.0:
            self.expert_latency_ema_ms = (expert_ms if self.expert_latency_ema_ms is None
                                          else (1.0 - alpha) * self.expert_latency_ema_ms
                                          + alpha * expert_ms)
        if self.main_latency_ema_ms is None or self.expert_latency_ema_ms is None:
            return
        remaining = max(0.0, self.config.target_frame_ms - self.main_latency_ema_ms)
        allowed = remaining / max(self.expert_latency_ema_ms, 1e-6)
        self.budget.fraction = float(np.clip(
            allowed, 0.0, self.config.expert_max_fraction))

    def _anchor_similarity(self, frame: np.ndarray, box: Sequence[float]) -> float:
        if self.anchor is None:
            return 0.5
        return float(np.clip((_similarity(_crop_wrap(frame, box), self.anchor) + 1.0) / 2.0,
                             0.0, 1.0))

    def _evidence(self, frame: np.ndarray, result: Mapping[str, object]) -> dict[str, object]:
        box = result["target_bbox"]
        measured = bfov_from_erp_bbox(*box, self.width, self.height)
        assert self.state is not None
        motion_error = self.state.prediction_error_deg(measured)
        area = float(np.clip(measured.fov_h * measured.fov_v / (360.0 * 180.0), 0.0, 1.0))
        log_aspect = math.log(max(measured.fov_h, 1e-3) / max(measured.fov_v, 1e-3))
        geometry_risk = self.descriptor.risk(
            measured.lon, measured.lat,
            angular_speed_deg=self.state.angular_speed_deg,
            motion_uncertainty_deg=motion_error,
            tracker_confidence=float(result["quality"]),
        )
        seam_distance_px = min(
            (float(box[0]) + 0.5 * float(box[2])) % self.width,
            self.width - ((float(box[0]) + 0.5 * float(box[2])) % self.width),
        )
        reasons: list[str] = []
        if abs(measured.lat) >= self.config.polar_lat_deg:
            reasons.append("polar")
        if seam_distance_px <= max(2.0 * float(box[2]),
                                   self.config.seam_margin_fraction * self.width):
            reasons.append("seam")
        if min(measured.fov_h, measured.fov_v) < self.config.small_fov_deg:
            reasons.append("small")
        if (measured.fov_h > self.config.large_fov_h_deg
                or measured.fov_v > self.config.large_fov_v_deg):
            reasons.append("large")
        if motion_error > 5.0:
            reasons.append("motion")
        if geometry_risk >= self.config.geometry_risk:
            reasons.append("geometry_risk")
        descriptor = self.descriptor.descriptor(
            measured.lon, measured.lat, self.state.angular_speed_deg,
            motion_error, float(result["quality"]), area, log_aspect)
        log_area = math.log(max(float(box[2] * box[3]), 4.0))
        policy_features = {
            "main_quality": float(result["quality"]),
            "quality_drop": max(0.0, self._previous_main_quality - float(result["quality"])),
            "abs_latitude": abs(measured.lat) / 90.0,
            "seam_proximity": 1.0 - seam_distance_px / (self.width / 2.0),
            "log_box_area": float(np.clip(log_area / math.log(self.width * self.height), 0.0, 1.0)),
            "log_scale_change": min(1.0, abs(log_area - self._previous_main_log_area)),
            "angular_motion": min(1.0, self.state.angular_speed_deg / 45.0),
        }
        policy_score = self.risk_policy.score(policy_features) if self.risk_policy else None
        self._previous_main_quality = float(result["quality"])
        self._previous_main_log_area = log_area
        return {
            "bfov": measured,
            "motion_error_deg": motion_error,
            "geometry_risk": geometry_risk,
            "geometry_descriptor": descriptor,
            "anchor_similarity": self._anchor_similarity(frame, box),
            "route_reasons": reasons,
            "policy_features": policy_features,
            "policy_score": policy_score,
        }

    def _probe_expert(self, frame: np.ndarray) -> tuple[dict[str, object] | None, float]:
        if self.expert is None or self.last_frame is None or self.last_box is None:
            return None, 0.0
        if self.frame_index % max(1, self.config.expert_interval) != 0:
            return None, 0.0
        if not self.budget.consume():
            return None, 0.0
        start = time.perf_counter()
        _call_init(self.expert, self.last_frame, self.last_box)
        output = _normalize_output(_call_track(self.expert, frame), self.width, self.height)
        return output, (time.perf_counter() - start) * 1000.0

    def _verify_candidate(self, frame: np.ndarray, box: Sequence[float]) -> tuple[bool, float]:
        similarity = self._anchor_similarity(frame, box)
        if similarity < self.config.anchor_min_similarity:
            return False, similarity
        if self.state is not None:
            candidate = bfov_from_erp_bbox(*box, self.width, self.height)
            if _angular_distance(self.state.bfov, candidate) > self.config.motion_max_deg:
                return False, similarity
        return True, similarity

    def _set_status(self, status: AdaptiveStatus) -> None:
        if status == self.status:
            self.status_run += 1
        else:
            self.status = status
            self.status_run = 1

    def _reinitialize_main(self, frame: np.ndarray, box: Sequence[float]) -> None:
        _call_init(self.main, frame, box)
        measured = bfov_from_erp_bbox(*box, self.width, self.height)
        self.state = SphericalState(measured)
        self.last_box = [float(value) for value in box]
        self.low_run = 0
        self.verify_run = 0

    def track(self, frame: np.ndarray, **_: object) -> dict[str, object]:
        if self.last_frame is None or self.last_box is None or self.state is None:
            raise RuntimeError("AdaptiveSphericalTracker must be initialized first")
        frame = np.asarray(frame)
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError("frame dimensions differ from initialization")
        self.frame_index += 1
        self.budget.advance()
        total_start = time.perf_counter()
        main_ms = expert_ms = redetect_ms = 0.0
        expert_output: dict[str, object] | None = None
        expert_used: str | None = None
        reasons: list[str] = []

        if self.status == AdaptiveStatus.LOST:
            candidate = None
            candidate_quality = 0.0
            if (self.redetector is not None
                    and self.frame_index % max(1, self.config.redetect_interval) == 0):
                start = time.perf_counter()
                found = self.redetector.search(frame, erp_downscale=2)
                redetect_ms = (time.perf_counter() - start) * 1000.0
                self.redetect_count += 1
                if found is not None:
                    candidate, candidate_quality = found
                    reasons.append("global_redetect")
            if candidate is None:
                expert_output, expert_ms = self._probe_expert(frame)
                if expert_output is not None and expert_output["quality"] >= self.config.expert_quality:
                    candidate = expert_output["target_bbox"]
                    candidate_quality = float(expert_output["quality"])
                    expert_used = self.expert_name
                    reasons.append("expert_recovery")
            if candidate is not None:
                accepted, anchor_similarity = self._verify_candidate(frame, candidate)
                if accepted:
                    self._reinitialize_main(frame, candidate)
                    self._set_status(AdaptiveStatus.VERIFY)
                    box = list(candidate)
                    quality = candidate_quality
                else:
                    reasons.append("candidate_rejected")
                    box = list(self.last_box)
                    quality = 0.0
            else:
                anchor_similarity = self._anchor_similarity(frame, self.last_box)
                box = list(self.last_box)
                quality = 0.0
        elif self.expert_episode_remaining > 0 and self.expert is not None:
            # Keep the expert's temporal state for a short episode.  Running
            # only the expert (instead of main+expert together) makes a
            # specialist useful on sustained polar/scale failures without
            # paying both model latencies on every episode frame.
            self.budget.consume(force=True)
            start = time.perf_counter()
            expert_output = _normalize_output(
                _call_track(self.expert, frame), self.width, self.height)
            expert_ms = (time.perf_counter() - start) * 1000.0
            expert_used = self.expert_name
            evidence = self._evidence(frame, expert_output)
            reasons = list(evidence["route_reasons"]) + ["expert_episode"]
            anchor_similarity = float(evidence["anchor_similarity"])
            quality = float(expert_output["quality"])
            box = list(expert_output["target_bbox"])
            self.expert_episode_remaining -= 1
            if quality < self.config.lost_quality:
                self.expert_episode_remaining = 0
                self._set_status(AdaptiveStatus.LOST)
            elif self.expert_episode_remaining == 0:
                self._reinitialize_main(frame, box)
                self._set_status(AdaptiveStatus.VERIFY)
            else:
                self._set_status(AdaptiveStatus.VERIFY)
                self.state.update(bfov_from_erp_bbox(*box, self.width, self.height))
        else:
            start = time.perf_counter()
            main_output = _normalize_output(_call_track(self.main, frame), self.width, self.height)
            main_ms = (time.perf_counter() - start) * 1000.0
            evidence = self._evidence(frame, main_output)
            reasons = list(evidence["route_reasons"])
            anchor_similarity = float(evidence["anchor_similarity"])
            quality = float(main_output["quality"])
            box = list(main_output["target_bbox"])

            policy_trigger = (evidence["policy_score"] is not None
                              and evidence["policy_score"] >= self.risk_policy.threshold)
            needs_expert = (policy_trigger or quality < self.config.suspect_quality
                            or self.status == AdaptiveStatus.SUSPECT
                            or "motion" in reasons)
            if needs_expert:
                expert_output, expert_ms = self._probe_expert(frame)
            if expert_output is not None:
                agreement = float(dual_iou(
                    main_output["target_bbox"], expert_output["target_bbox"], self.width))
                expert_better = float(expert_output["quality"]) >= quality + self.config.expert_switch_margin
                recovery_vote = (quality < self.config.lost_quality
                                 and float(expert_output["quality"]) >= self.config.expert_quality)
                if (expert_better or recovery_vote) and agreement < self.config.disagreement_iou:
                    accepted, expert_anchor = self._verify_candidate(
                        frame, expert_output["target_bbox"])
                    if accepted:
                        box = list(expert_output["target_bbox"])
                        quality = float(expert_output["quality"])
                        anchor_similarity = expert_anchor
                        expert_used = self.expert_name
                        self.expert_episode_remaining = max(
                            0, int(self.config.expert_episode_frames) - 1)
                        self._set_status(AdaptiveStatus.VERIFY)
                        reasons.append("expert_switch")

            if self.status == AdaptiveStatus.VERIFY:
                if quality >= self.config.recover_quality:
                    self.verify_run += 1
                    if self.verify_run >= self.config.verify_frames:
                        self._set_status(AdaptiveStatus.NORMAL)
                        self.verify_run = 0
                else:
                    self._set_status(AdaptiveStatus.LOST)
                    self.verify_run = 0
            elif quality < self.config.lost_quality:
                self.low_run += 1
                if self.low_run >= self.config.lost_run:
                    self._set_status(AdaptiveStatus.LOST)
                else:
                    self._set_status(AdaptiveStatus.SUSPECT)
            elif (quality < self.config.suspect_quality
                  or "motion" in reasons
                  or anchor_similarity < self.config.anchor_min_similarity):
                self.low_run += 1
                if self.low_run >= self.config.suspect_run:
                    self._set_status(AdaptiveStatus.SUSPECT)
            else:
                self.low_run = 0
                self._set_status(AdaptiveStatus.NORMAL)

            if self.status != AdaptiveStatus.LOST:
                measured = bfov_from_erp_bbox(*box, self.width, self.height)
                self.state.update(measured)

        self.last_quality = float(quality)
        self.last_box = [float(value) for value in box]
        self.last_frame = np.ascontiguousarray(frame)
        self._update_latency_budget(main_ms, expert_ms)
        latency_ms = (time.perf_counter() - total_start) * 1000.0
        return {
            "target_bbox": list(self.last_box),
            "bbox": tuple(self.last_box),
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "status": self.status.value,
            "response_entropy": (expert_output or {}).get("response_entropy"),
            "anchor_similarity": float(anchor_similarity),
            "latency_ms": latency_ms,
            "main_latency_ms": main_ms,
            "expert_latency_ms": expert_ms,
            "redetect_latency_ms": redetect_ms,
            "expert_used": expert_used,
            "expert_probed": expert_output is not None,
            "expert_call_fraction": self.budget.call_fraction,
            "expert_budget_fraction": self.budget.fraction,
            "expert_episode_remaining": self.expert_episode_remaining,
            "policy_score": (evidence["policy_score"] if 'evidence' in locals() else None),
            "route_reasons": reasons,
            "frame_index": self.frame_index,
        }

    def update(self, frame: np.ndarray) -> dict[str, object]:
        """Alias for the panotrack BaseTracker-style contract."""

        return self.track(frame)
