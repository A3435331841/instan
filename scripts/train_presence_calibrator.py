#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a sequence-disjoint causal presence/failure calibrator on OOF traces.

The tracker traces contain only inference-time signals.  Ground-truth is read
solely to make train95 labels and is never serialized as an inference feature.
The exported JSON follows ``LinearRiskPolicy`` so it can be loaded by the
existing adaptive router after it passes its OOF gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data360" / "official_split" / "seqlist_official_train.txt"

FEATURE_NAMES = (
    "quality",
    "quality_drop",
    "area_log_change",
    "center_motion_norm",
    "seam_proximity",
    "abs_latitude",
    "bbox_area_log",
    "low_quality_run",
    "fallback_used",
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _auc(y, score):
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1, dtype=np.float64)
    n_pos = float(pos.sum())
    n_neg = float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1.0) / 2.0)
                 / (n_pos * n_neg))


def _fit_logistic(x, y, l2=1e-2, epochs=500, learning_rate=0.15):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    if n == 0 or np.unique(y).size < 2:
        raise ValueError("training labels must contain both classes")
    counts = np.bincount(y.astype(np.int64), minlength=2).astype(np.float64)
    class_weight = np.where(y > 0.5, 0.5 / max(counts[1], 1.0), 0.5 / max(counts[0], 1.0))
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for step in range(int(epochs)):
        p = _sigmoid(x @ w + b)
        err = (p - y) * class_weight
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        rate = learning_rate / math.sqrt(1.0 + step / 50.0)
        w -= rate * grad_w
        b -= rate * grad_b
    return w, float(b)


def _read_gt(path: Path):
    valid = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [float(value) for value in line.replace(",", " ").split()]
        if len(fields) != 4:
            raise ValueError(f"invalid GT row in {path}: {line!r}")
        valid.append(fields[2] > 0.0 and fields[3] > 0.0)
    return np.asarray(valid, dtype=bool)


def _runtime_features(traces, width, height):
    rows = []
    previous_quality = 1.0
    previous_area = None
    previous_center = None
    low_run = 0
    for trace in traces:
        box = [float(value) for value in trace.get("target_bbox", [0, 0, 0, 0])[:4]]
        x, y, w, h = box
        quality = float(trace.get("quality", 0.0) or 0.0)
        area = max(1.0, w * h)
        center_x = (x + 0.5 * w) % max(float(width), 1.0)
        center_y = y + 0.5 * h
        if previous_center is None:
            motion = 0.0
        else:
            dx = abs(((center_x - previous_center[0] + width / 2.0) % width) - width / 2.0)
            dy = abs(center_y - previous_center[1])
            motion = math.hypot(dx / max(width, 1), dy / max(height, 1))
        area_change = 0.0 if previous_area is None else abs(math.log(area) - math.log(previous_area))
        seam_distance = min(center_x, width - center_x)
        seam_proximity = float(np.clip(1.0 - seam_distance / max(width / 2.0, 1.0), 0.0, 1.0))
        latitude = 90.0 - center_y / max(height, 1) * 180.0
        if quality <= 0.40:
            low_run += 1
        else:
            low_run = 0
        rows.append({
            "frame_index": int(trace.get("frame_index", len(rows))),
            "features": [
                quality,
                max(0.0, previous_quality - quality),
                area_change,
                motion,
                seam_proximity,
                abs(latitude) / 90.0,
                float(np.clip(math.log(area / max(width * height, 1.0)), -12.0, 0.0)),
                float(min(low_run, 30) / 30.0),
                1.0 if trace.get("fallback_used") else 0.0,
            ],
        })
        previous_quality = quality
        previous_area = area
        previous_center = (center_x, center_y)
    return rows


def _load_sequence(results_root: Path, data_root: Path, sequence: str):
    tag = sequence.replace("/", "_")
    candidates = [results_root / tag, results_root / sequence]
    result_dir = next((path for path in candidates if (path / "trace.jsonl").is_file()), None)
    if result_dir is None:
        return None
    seq_dir = data_root / sequence
    gt_path = seq_dir / "groundtruth.txt"
    video_meta = seq_dir / "video.mp4"
    if not gt_path.is_file() or not video_meta.is_file():
        return None
    import cv2
    cap = cv2.VideoCapture(str(video_meta))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    traces = []
    with (result_dir / "trace.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(item, dict):
                traces.append(item)
    gt = _read_gt(gt_path)
    features = _runtime_features(traces, width, height)
    limit = min(len(features), len(gt))
    return features[:limit], gt[:limit], result_dir


def _threshold(y, p):
    best = (0.5, -1.0)
    for threshold in np.linspace(0.10, 0.90, 81):
        pred = p >= threshold
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        fn = float(np.sum((~pred) & (y == 1)))
        f1 = 2.0 * tp / max(2.0 * tp + fp + fn, 1.0)
        if f1 > best[1]:
            best = (float(threshold), f1)
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--label", choices=["presence"], default="presence")
    args = parser.parse_args()
    sequences = [line.strip().replace("\\", "/") for line in args.split.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    loaded = []
    missing = []
    for sequence in sequences:
        item = _load_sequence(args.results.resolve(), args.data.resolve(), sequence)
        if item is None:
            missing.append(sequence)
        else:
            features, labels, result_dir = item
            loaded.append((sequence, features, labels, result_dir))
    if not loaded:
        raise SystemExit("no train95 trace/GT pairs found")

    folds = max(2, min(int(args.folds), len(loaded)))
    fold_ids = np.arange(len(loaded), dtype=np.int64) % folds
    oof_rows = []
    for fold in range(folds):
        train_items = [item for i, item in enumerate(loaded) if fold_ids[i] != fold]
        test_items = [item for i, item in enumerate(loaded) if fold_ids[i] == fold]
        x_train = np.concatenate([
            np.asarray([row["features"] for row in item[1]], dtype=np.float64)
            for item in train_items
        ], axis=0)
        y_train = np.concatenate([item[2].astype(np.int64) for item in train_items], axis=0)
        mean = x_train.mean(axis=0)
        std = np.maximum(x_train.std(axis=0), 1e-6)
        w, b = _fit_logistic((x_train - mean) / std, y_train)
        for sequence, features, labels, _result_dir in test_items:
            x_test = np.asarray([row["features"] for row in features], dtype=np.float64)
            p = _sigmoid(((x_test - mean) / std) @ w + b)
            for row, label, probability in zip(features, labels, p):
                oof_rows.append({
                    "sequence": sequence,
                    "frame_index": row["frame_index"],
                    "label": int(label),
                    "probability": float(probability),
                    "fold": fold,
                })

    y_oof = np.asarray([row["label"] for row in oof_rows], dtype=np.int64)
    p_oof = np.asarray([row["probability"] for row in oof_rows], dtype=np.float64)
    threshold, f1 = _threshold(y_oof, p_oof)
    # Fit the deployable model on all available train95 sequences only.
    x_all = np.concatenate([
        np.asarray([row["features"] for row in item[1]], dtype=np.float64)
        for item in loaded
    ], axis=0)
    y_all = np.concatenate([item[2].astype(np.int64) for item in loaded], axis=0)
    mean = x_all.mean(axis=0)
    std = np.maximum(x_all.std(axis=0), 1e-6)
    w, b = _fit_logistic((x_all - mean) / std, y_all)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "grt360.linear_presence_policy.v1",
        "label": args.label,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "threshold": float(threshold),
        "train_sequences": [item[0] for item in loaded],
        "missing_sequences": missing,
        "folds": folds,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "presence_policy.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "frame_index", "label", "probability", "fold"])
        writer.writeheader()
        writer.writerows(oof_rows)
    summary = {
        "schema": "grt360.presence_calibrator_summary.v1",
        "label": args.label,
        "sequences_available": len(loaded),
        "sequences_missing": len(missing),
        "frames": int(len(y_oof)),
        "positive_rate": float(y_oof.mean()),
        "oof_auc": _auc(y_oof, p_oof),
        "oof_f1": float(f1),
        "threshold": float(threshold),
        "inference_features_only": True,
        "gt_used_for_training_labels_only": True,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
