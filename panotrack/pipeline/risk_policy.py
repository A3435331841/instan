# -*- coding: utf-8 -*-
"""Small serializable inference-only risk policy for adaptive tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LinearRiskPolicy:
    """Standardized logistic gate exported by sequence-disjoint OOF training."""

    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LinearRiskPolicy":
        names = tuple(str(value) for value in payload["feature_names"])
        mean = np.asarray(payload["mean"], dtype=float)
        std = np.maximum(np.asarray(payload["std"], dtype=float), 1e-6)
        weights = np.asarray(payload["weights"], dtype=float)
        if not names or len(names) != len(mean) or len(mean) != len(std) or len(std) != len(weights):
            raise ValueError("risk-policy feature dimensions do not match")
        return cls(names, mean, std, weights, float(payload["bias"]),
                   float(payload["threshold"]))

    @classmethod
    def load(cls, path: str | Path) -> "LinearRiskPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, features: Mapping[str, float]) -> float:
        values = np.asarray([float(features[name]) for name in self.feature_names], dtype=float)
        logits = float(np.clip(((values - self.mean) / self.std) @ self.weights + self.bias,
                               -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-logits)))

    def should_probe(self, features: Mapping[str, float]) -> bool:
        return self.score(features) >= self.threshold
