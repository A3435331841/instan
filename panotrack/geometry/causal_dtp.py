# -*- coding: utf-8 -*-
"""Causal DTP-style expert router for ERP 360 tracking.

This module is deliberately prediction-only: it consumes tracker boxes and
never uses ground truth.  It implements the first runnable slice of the
GRT360-Causal-DTP-ERP proposal:

* causal constant-velocity prior on the circular ERP longitude;
* temporal reliability calibration from innovation, scale change, and expert
  agreement;
* geometry risk penalty near ERP seams and poles;
* teacher-as-anchor behavior: ODTrack is preferred when reliable, UETrack is
  allowed to recover when the teacher becomes unreliable;
* circular blending so seam crossing does not create a false long jump.

The trainable KV-cache/token-compression parts of the proposal are intentionally
not faked here.  They belong to the model implementation, while this module is
the safe causal routing layer that can be validated on existing predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


def _center(box: Sequence[float], width: float) -> float:
    return (float(box[0]) + 0.5 * float(box[2])) % float(width)


def _circ_delta(a: float, b: float, width: float) -> float:
    return ((float(a) - float(b) + width / 2.0) % width) - width / 2.0


def _circ_distance(a: float, b: float, width: float) -> float:
    return abs(_circ_delta(a, b, width))


def _blend_boxes(a: Sequence[float], b: Sequence[float], alpha: float,
                 width: float) -> np.ndarray:
    """Blend boxes with circular x-center and linear y/size components."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    ca = _center(a, width)
    cb = ca + _circ_delta(_center(b, width), ca, width)
    c = (ca + alpha * (cb - ca)) % float(width)
    ywh = (1.0 - alpha) * a[1:] + alpha * b[1:]
    return np.array([c - 0.5 * ywh[0], ywh[0], ywh[1], ywh[2]], dtype=np.float64)


def _box_from_state(center: float, y: float, w: float, h: float,
                    width: float) -> np.ndarray:
    return np.array([center - 0.5 * w, y, w, h], dtype=np.float64)


@dataclass
class CausalState:
    center: float
    y: float
    width: float
    height: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    velocity_w: float = 0.0
    velocity_h: float = 0.0


class CausalDTPRouter:
    """Causal reliability router for OD/UE/LightFC prediction streams."""

    def __init__(self, width: int, height: int, teacher_index: int = 0,
                 student_index: int = 1, scout_index: int = 2,
                 velocity_alpha: float = 0.35,
                 reliability_decay: float = 18.0,
                 scale_decay: float = 3.0,
                 agreement_decay: float = 12.0,
                 geometry_penalty: float = 0.35,
                 teacher_margin: float = 0.04,
                 recovery_margin: float = 0.02,
                 hold_frames: int = 3,
                 blend_alpha: float = 0.18):
        self.width = float(width)
        self.height = float(height)
        self.teacher_index = int(teacher_index)
        self.student_index = int(student_index)
        self.scout_index = int(scout_index)
        self.velocity_alpha = float(np.clip(velocity_alpha, 0.0, 1.0))
        self.reliability_decay = max(float(reliability_decay), 1e-6)
        self.scale_decay = max(float(scale_decay), 1e-6)
        self.agreement_decay = max(float(agreement_decay), 1e-6)
        self.geometry_penalty = max(float(geometry_penalty), 0.0)
        self.teacher_margin = max(float(teacher_margin), 0.0)
        self.recovery_margin = max(float(recovery_margin), 0.0)
        self.hold_frames = max(int(hold_frames), 0)
        self.blend_alpha = float(np.clip(blend_alpha, 0.0, 1.0))
        self.state: Optional[CausalState] = None
        self.active = self.teacher_index
        self.hold = 0

    def reset(self) -> None:
        self.state = None
        self.active = self.teacher_index
        self.hold = 0

    def _geometry_risk(self, box: Sequence[float]) -> float:
        cy = float(box[1]) + 0.5 * float(box[3])
        lat = abs(90.0 - 180.0 * np.clip(cy, 0.0, self.height) / self.height)
        pole = np.clip((lat - 55.0) / 35.0, 0.0, 1.0)
        cx = _center(box, self.width)
        seam_dist = min(cx, self.width - cx)
        seam = 1.0 - np.clip(seam_dist / max(1.0, 0.12 * self.width), 0.0, 1.0)
        aspect = abs(np.log(max(float(box[2]), 1.0) / max(float(box[3]), 1.0)))
        return float(np.clip(0.50 * pole + 0.35 * seam + 0.15 * np.clip(aspect / 4.0, 0.0, 1.0), 0.0, 1.0))

    def _predict(self) -> np.ndarray:
        assert self.state is not None
        s = self.state
        return _box_from_state((s.center + s.velocity_x) % self.width,
                               s.y + s.velocity_y,
                               max(2.0, s.width + s.velocity_w),
                               max(2.0, s.height + s.velocity_h), self.width)

    def _reliability(self, candidate: Sequence[float], predicted: Sequence[float],
                     other: Sequence[float]) -> float:
        diag = max(2.0, float(np.hypot(candidate[2], candidate[3])))
        innovation = _circ_distance(_center(candidate, self.width),
                                    _center(predicted, self.width), self.width)
        innovation = np.hypot(innovation, float(candidate[1] - predicted[1])) / diag
        old_area = max(4.0, float(predicted[2] * predicted[3]))
        new_area = max(4.0, float(candidate[2] * candidate[3]))
        scale_change = abs(np.log(new_area / old_area))
        disagreement = _circ_distance(_center(candidate, self.width),
                                       _center(other, self.width), self.width) / diag
        geometry = self._geometry_risk(candidate)
        raw = (np.exp(-innovation / self.reliability_decay)
               * np.exp(-scale_change / self.scale_decay)
               * np.exp(-disagreement / self.agreement_decay)
               * np.exp(-self.geometry_penalty * geometry))
        return float(np.clip(raw, 0.0, 1.0))

    def update(self, candidates: Iterable[Sequence[float]]) -> Tuple[np.ndarray, int, np.ndarray]:
        """Return ``(box, selected_expert, reliabilities)`` for one frame."""
        boxes = np.asarray(list(candidates), dtype=np.float64)
        if boxes.shape != (3, 4):
            raise ValueError(f'expected three [x,y,w,h] candidates, got {boxes.shape}')
        if self.state is None:
            chosen = boxes[self.teacher_index].copy()
            self.state = CausalState(_center(chosen, self.width), chosen[1],
                                     chosen[2], chosen[3])
            return chosen, self.teacher_index, np.ones(3, dtype=np.float64)

        predicted = self._predict()
        reliabilities = np.array([
            self._reliability(boxes[i], predicted,
                              boxes[self.student_index if i == self.teacher_index else self.teacher_index])
            for i in range(3)
        ], dtype=np.float64)
        teacher = self.teacher_index
        student = self.student_index
        # The teacher wins when it is not materially less reliable.  Otherwise
        # the student can take over, with hysteresis to avoid oscillation.
        best = int(np.argmax(reliabilities))
        if self.active == teacher:
            if reliabilities[student] > reliabilities[teacher] + self.teacher_margin:
                self.active = student
                self.hold = self.hold_frames
            elif best == self.scout_index and reliabilities[best] > reliabilities[teacher] + 2.0 * self.teacher_margin:
                self.active = best
                self.hold = self.hold_frames
        else:
            self.hold = max(0, self.hold - 1)
            if self.hold == 0 and reliabilities[teacher] >= reliabilities[self.active] + self.recovery_margin:
                self.active = teacher
        selected = boxes[self.active].copy()
        if self.active != teacher and reliabilities[teacher] > 0.45:
            selected = _blend_boxes(selected, boxes[teacher], self.blend_alpha, self.width)

        s = self.state
        center = _center(selected, self.width)
        alpha = self.velocity_alpha
        dx = _circ_delta(center, s.center, self.width)
        dy = float(selected[1] - s.y)
        dw = float(selected[2] - s.width)
        dh = float(selected[3] - s.height)
        s.velocity_x = (1.0 - alpha) * s.velocity_x + alpha * dx
        s.velocity_y = (1.0 - alpha) * s.velocity_y + alpha * dy
        s.velocity_w = (1.0 - alpha) * s.velocity_w + alpha * dw
        s.velocity_h = (1.0 - alpha) * s.velocity_h + alpha * dh
        s.center, s.y = center, float(selected[1])
        s.width, s.height = float(selected[2]), float(selected[3])
        return selected, self.active, reliabilities

