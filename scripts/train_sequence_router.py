#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sequence-level OOF router for existing full-result roots.

Unlike the frame router, this diagnostic asks whether the first BFoV and a
short causal warm-up can predict which complete tracker should own a sequence.
Ground-truth labels are used only to score/train the sequence-disjoint OOF
experiment; the exported policy contains no sequence names or GT values.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from panotrack.evaluation.metrics import auc, iou_xywh, success_rate  # noqa: E402
from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov  # noqa: E402


FEATURE_NAMES = (
    "init_lon", "init_lat", "init_fov_h", "init_fov_v", "abs_init_lat",
    "init_seam_proximity", "init_log_fov_area", "init_aspect",
    "warmup_quality_mean", "warmup_quality_p10", "warmup_area_log_std",
    "warmup_step_p95", "warmup_motion_mean", "warmup_motion_p95",
)


def load_records(root: Path) -> dict[str, Path]:
    found = {}
    for metrics in root.rglob("metrics.json"):
        try:
            row = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        sequence = str(row.get("sequence", "")).replace("\\", "/").strip()
        if sequence:
            found[sequence] = metrics.parent
    return found


def load_gt(data_root: Path, sequence: str, width: int, height: int):
    rows, valid = [], []
    for line in (data_root / sequence / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        value = [float(item) for item in line.replace(",", " ").split()]
        rows.append(value[:4])
        valid.append(value[2] > 0.0 and value[3] > 0.0)
    gt = np.zeros((len(rows), 4), dtype=float)
    for index, value in enumerate(rows):
        if valid[index]:
            gt[index] = erp_bbox_from_bfov(BFoV(*value), width, height)
    return gt, np.asarray(valid, dtype=bool)


def read_boxes(path: Path) -> np.ndarray:
    array = np.loadtxt(path, delimiter=",")
    return array.reshape(1, -1) if array.ndim == 1 else array


def read_quality(path: Path, length: int) -> np.ndarray:
    if not path.is_file():
        return np.full(length, 0.5, dtype=float)
    value = np.loadtxt(path).reshape(-1)
    return value if len(value) >= length else np.pad(value, (0, length - len(value)), constant_values=0.5)


def features(data_root: Path, sequence: str, result_dir: Path, warmup: int) -> np.ndarray:
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    width, height = (int(value) for value in str(metrics.get("resolution", "1440x720")).split("x"))
    init = [float(value) for value in (data_root / sequence / "init.txt").read_text(encoding="utf-8").split(",")[:4]]
    boxes = read_boxes(result_dir / "results_erp.txt")
    quality = read_quality(result_dir / "quality.txt", len(boxes))
    n = min(max(2, int(warmup)), len(boxes))
    area = np.log(np.maximum(boxes[:n, 2] * boxes[:n, 3], 4.0))
    cx = (boxes[:n, 0] + boxes[:n, 2] / 2.0) % width
    cy = np.clip(boxes[:n, 1] + boxes[:n, 3] / 2.0, 0.0, height)
    dx = np.abs((np.diff(cx) + width / 2.0) % width - width / 2.0) * 360.0 / width
    dy = np.abs(np.diff(cy)) * 180.0 / height
    motion = np.hypot(dx, dy)
    init_lon, init_lat, fov_h, fov_v = init
    # ERP seam is at ±180°; lon=0 is the safest center, not a seam.
    seam = min(abs(((init_lon + 180.0) % 360.0) - 180.0) / 180.0, 1.0)
    values = [
        init_lon / 180.0, init_lat / 90.0, fov_h / 180.0, fov_v / 180.0,
        abs(init_lat) / 90.0, seam, math.log(max(1.0, fov_h * fov_v)) / math.log(32400.0),
        math.log(max(1e-3, fov_h / max(fov_v, 1e-3))),
        float(np.mean(quality[:n])), float(np.percentile(quality[:n], 10)),
        float(np.std(area)), float(np.percentile(np.abs(np.diff(area)), 95)) if len(area) > 1 else 0.0,
        float(np.mean(motion)) if len(motion) else 0.0,
        float(np.percentile(motion, 95)) if len(motion) else 0.0,
    ]
    return np.asarray(values, dtype=float)


def sequence_metrics(result_dir: Path, data_root: Path, sequence: str):
    row = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    width, height = (int(value) for value in str(row.get("resolution", "1440x720")).split("x"))
    pred = read_boxes(result_dir / "results_erp.txt")
    gt, valid = load_gt(data_root, sequence, width, height)
    n = min(len(pred), len(gt), len(valid))
    values = [iou_xywh(pred[i], gt[i]) if valid[i] and pred[i, 2] > 0 and pred[i, 3] > 0 else 0.0 for i in range(1, n)]
    return np.asarray(values, dtype=float), valid[1:n]


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.08):
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    z = (x - mean) / std
    weight = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    pos_weight = (len(y) - y.sum()) / max(1.0, y.sum())
    for _ in range(steps):
        probability = 1.0 / (1.0 + np.exp(-np.clip(z @ weight + bias, -30.0, 30.0)))
        sample_weight = np.where(y > 0.5, pos_weight, 1.0)
        error = (probability - y) * sample_weight
        weight -= lr * (z.T @ error / len(y) + 1e-4 * weight)
        bias -= lr * float(error.mean())
    return mean, std, weight, float(bias)


def predict(x, model):
    mean, std, weight, bias = model
    return 1.0 / (1.0 + np.exp(-np.clip(((x - mean) / std) @ weight + bias, -30.0, 30.0)))


def metric(values: np.ndarray, valid: np.ndarray):
    values = values[:len(valid)][valid]
    return auc(values) if len(values) else math.nan, success_rate(values) if len(values) else math.nan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--main-root", required=True)
    ap.add_argument("--expert-root", required=True)
    ap.add_argument("--sequence-file", required=True)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--expert-margin", type=float, default=0.02)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data_root = Path(args.data).resolve()
    main = load_records(Path(args.main_root).resolve())
    expert = load_records(Path(args.expert_root).resolve())
    sequences = [line.strip().replace("\\", "/") for line in Path(args.sequence_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for sequence in sequences:
        if sequence not in main or sequence not in expert:
            continue
        try:
            x = features(data_root, sequence, main[sequence], args.warmup)
            miou, mvalid = sequence_metrics(main[sequence], data_root, sequence)
            eiou, evalid = sequence_metrics(expert[sequence], data_root, sequence)
            n = min(len(miou), len(eiou), len(mvalid), len(evalid))
            miou, eiou, valid = miou[:n], eiou[:n], mvalid[:n] & evalid[:n]
            ma, ms = metric(miou, valid)
            ea, es = metric(eiou, valid)
            samples.append({"sequence": sequence, "x": x, "main_auc": ma, "main_sr": ms,
                            "expert_auc": ea, "expert_sr": es,
                            "label": float(ea > ma + args.expert_margin)})
        except (OSError, ValueError, IndexError):
            continue
    if len(samples) < args.folds:
        raise SystemExit(f"insufficient paired sequences: {len(samples)}")
    results = []
    for fold in range(args.folds):
        test = [s for index, s in enumerate(samples) if index % args.folds == fold]
        train = [s for index, s in enumerate(samples) if index % args.folds != fold]
        x_train = np.stack([s["x"] for s in train])
        y_train = np.asarray([s["label"] for s in train])
        model = fit_logistic(x_train, y_train)
        train_scores = predict(x_train, model)
        threshold = float(np.quantile(train_scores, 0.5))
        for sample in test:
            score = float(predict(sample["x"][None, :], model)[0])
            choose_expert = score >= threshold
            chosen_auc = sample["expert_auc"] if choose_expert else sample["main_auc"]
            chosen_sr = sample["expert_sr"] if choose_expert else sample["main_sr"]
            results.append({"sequence": sample["sequence"], "fold": fold,
                            "label": sample["label"], "score": score,
                            "chosen_expert": choose_expert,
                            "main_auc": sample["main_auc"], "expert_auc": sample["expert_auc"],
                            "chosen_auc": chosen_auc, "chosen_sr": chosen_sr,
                            "oracle_auc": max(sample["main_auc"], sample["expert_auc"])})
    mean = lambda key: float(np.mean([row[key] for row in results]))
    report = {
        "n_sequences": len(samples), "folds": args.folds, "warmup": args.warmup,
        "expert_margin": args.expert_margin, "features": FEATURE_NAMES,
        "main_auc": mean("main_auc"), "expert_auc": mean("expert_auc"),
        "chosen_auc": mean("chosen_auc"), "chosen_sr": mean("chosen_sr"),
        "oracle_auc": mean("oracle_auc"),
        "chosen_auc_delta": mean("chosen_auc") - mean("main_auc"),
        "expert_selection_fraction": float(np.mean([row["chosen_expert"] for row in results])),
        "per_sequence": results,
    }
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "sequence_router_oof.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_sequence"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
