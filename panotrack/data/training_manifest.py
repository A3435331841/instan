# -*- coding: utf-8 -*-
"""Failure-balanced train95 manifest and sequence-level OOF fold builder."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


TAG_WEIGHT = {
    "small": 0.60,
    "large": 0.30,
    "polar": 0.60,
    "seam": 0.40,
    "fast": 0.50,
    "scale": 0.50,
    "absent": 0.40,
    "drift": 0.80,
    "hard": 0.80,
}


def parse_tags(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(sorted({part.strip() for part in value.split(";") if part.strip()}))
    if isinstance(value, Sequence):
        return tuple(sorted({str(part).strip() for part in value if str(part).strip()}))
    return ()


def augmentation_policy(tags: Iterable[str]) -> dict[str, float]:
    tags = set(tags)
    return {
        "longitude_roll": 0.50,
        "pole_rotation": 0.60 if "polar" in tags else 0.15,
        "seam_crossing": 0.60 if "seam" in tags else 0.20,
        "small_target_degrade": 0.60 if "small" in tags else 0.10,
        "log_scale_jitter": 0.60 if tags & {"small", "large", "scale"} else 0.20,
        "motion_blur": 0.50 if "fast" in tags else 0.15,
        "temporal_occlusion": 0.50 if tags & {"absent", "drift"} else 0.10,
        "distractor_insert": 0.40 if tags & {"hard", "drift"} else 0.10,
        "projection_consistency": 0.50,
    }


def _assign_folds(records: list[dict], n_folds: int) -> None:
    """Greedy multi-label stratification by domain and diagnostic scene tags."""

    label_totals = Counter()
    for record in records:
        for label in (record["domain"], *record["scene_tags"]):
            label_totals[label] += 1
    fold_sizes = [0] * n_folds
    fold_labels = [Counter() for _ in range(n_folds)]

    def rarity(record):
        labels = (record["domain"], *record["scene_tags"])
        return (sum(1.0 / max(1, label_totals[label]) for label in labels),
                float(record.get("sample_weight_raw", 1.0)), record["sequence"])

    for record in sorted(records, key=rarity, reverse=True):
        labels = (record["domain"], *record["scene_tags"])
        scores = []
        for fold in range(n_folds):
            imbalance = sum((fold_labels[fold][label] + 1.0)
                            / max(1.0, label_totals[label] / n_folds)
                            for label in labels)
            scores.append((imbalance + 0.05 * fold_sizes[fold], fold_sizes[fold], fold))
        selected = min(scores)[2]
        record["oof_fold"] = selected
        fold_sizes[selected] += 1
        fold_labels[selected].update(labels)


def build_training_manifest(
    failure_rows: Sequence[Mapping[str, object]],
    train_sequences: Iterable[str],
    n_folds: int = 5,
    weight_cap: float = 4.0,
) -> list[dict]:
    """Create a leakage-safe train-only manifest with capped failure weights."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    by_sequence = {str(row["sequence"]).replace("\\", "/"): row for row in failure_rows}
    records: list[dict] = []
    missing: list[str] = []
    for sequence in sorted({str(item).strip().replace("\\", "/")
                            for item in train_sequences if str(item).strip()}):
        row = by_sequence.get(sequence)
        if row is None:
            missing.append(sequence)
            continue
        tags = parse_tags(row.get("scene_tags"))
        raw_weight = min(weight_cap, 1.0 + sum(TAG_WEIGHT.get(tag, 0.0) for tag in tags))
        records.append({
            "sequence": sequence,
            "domain": sequence.split("/", 1)[0],
            "scene_tags": list(tags),
            "baseline_auc": float(row.get("baseline_auc", float("nan"))),
            "sample_weight_raw": raw_weight,
            "augmentation": augmentation_policy(tags),
        })
    if missing:
        raise ValueError(f"failure matrix is missing {len(missing)} train sequences: {missing[:5]}")
    if not records:
        raise ValueError("training manifest would be empty")

    # Balance the total real/sim sampling mass, then normalize mean weight to 1.
    domain_mass = defaultdict(float)
    for record in records:
        domain_mass[record["domain"]] += record["sample_weight_raw"]
    target_mass = float(np.mean(list(domain_mass.values())))
    for record in records:
        record["sample_weight"] = (record["sample_weight_raw"] * target_mass
                                   / max(domain_mass[record["domain"]], 1e-9))
    normalizer = float(np.mean([record["sample_weight"] for record in records]))
    for record in records:
        record["sample_weight"] /= max(normalizer, 1e-9)
    _assign_folds(records, n_folds)
    return records


def write_training_manifest(records: Sequence[Mapping[str, object]], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
    fold_counts = Counter(int(record["oof_fold"]) for record in records)
    tag_counts = Counter(tag for record in records for tag in record["scene_tags"])
    domain_weight = defaultdict(float)
    for record in records:
        domain_weight[str(record["domain"])] += float(record["sample_weight"])
    summary = {
        "n_sequences": len(records),
        "fold_counts": dict(sorted(fold_counts.items())),
        "tag_counts": dict(sorted(tag_counts.items())),
        "domain_sampling_mass": dict(sorted(domain_weight.items())),
        "mean_sample_weight": float(np.mean([record["sample_weight"] for record in records])),
        "max_sample_weight": float(max(record["sample_weight"] for record in records)),
        "valid35_included": False,
    }
    summary_path = output_dir / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "summary": summary_path}
