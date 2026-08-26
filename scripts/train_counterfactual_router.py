#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OOF counterfactual gate for an online fast-main / precision-expert tracker.

This is an *analysis gate*, not a submission-time fusion tool.  It trains only
on sequence-disjoint folds and asks a strict question before any online router
is enabled: can main-tracker evidence predict when the expert is worth a short
episode, under the configured latency budget?
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

from panotrack.evaluation.metrics import auc, iou_xywh, success_rate
from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov


FEATURE_NAMES = (
    "main_quality",
    "quality_drop",
    "abs_latitude",
    "seam_proximity",
    "log_box_area",
    "log_scale_change",
    "angular_motion",
)


def _load_result_root(root: Path) -> dict[str, Path]:
    records = {}
    for metrics in root.rglob("metrics.json"):
        row = json.loads(metrics.read_text(encoding="utf-8"))
        sequence = row.get("sequence")
        if sequence:
            records[str(sequence)] = metrics.parent
    return records


def _groundtruth(data_root: Path, sequence: str, width: int = 1440, height: int = 720):
    rows = []
    valid = []
    for line in (data_root / sequence / "groundtruth.txt").read_text(encoding="utf-8").splitlines():
        values = [float(value) for value in line.replace(",", " ").split()]
        rows.append(values[:4])
        valid.append(values[2] > 0.0 and values[3] > 0.0)
    gt = np.zeros((len(rows), 4), dtype=float)
    for index, value in enumerate(rows):
        if valid[index]:
            gt[index] = erp_bbox_from_bfov(BFoV(*value), width, height)
    return gt, np.asarray(valid, dtype=bool)


def _read_array(path: Path) -> np.ndarray:
    array = np.loadtxt(path, delimiter="," if path.name.endswith("erp.txt") else None)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _features(boxes: np.ndarray, quality: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float)
    quality = np.asarray(quality, dtype=float).reshape(-1)
    n = min(len(boxes), len(quality))
    boxes, quality = boxes[:n], quality[:n]
    cx = (boxes[:, 0] + boxes[:, 2] / 2.0) % width
    cy = np.clip(boxes[:, 1] + boxes[:, 3] / 2.0, 0.0, height)
    lat = np.abs(90.0 - 180.0 * cy / height) / 90.0
    seam = 1.0 - np.minimum(cx, width - cx) / (width / 2.0)
    log_area = np.log(np.maximum(boxes[:, 2] * boxes[:, 3], 4.0))
    quality_drop = np.maximum(0.0, np.r_[0.0, quality[:-1] - quality[1:]])
    scale = np.abs(np.r_[0.0, np.diff(log_area)])
    # Seam-aware angular motion from ERP centres; latitude is included above.
    dx = np.abs((np.diff(cx) + width / 2.0) % width - width / 2.0) / width * 360.0
    dy = np.abs(np.diff(cy)) / height * 180.0
    motion = np.r_[0.0, np.hypot(dx, dy)] / 45.0
    return np.column_stack((
        np.clip(quality, 0.0, 1.0), np.clip(quality_drop, 0.0, 1.0),
        np.clip(lat, 0.0, 1.0), np.clip(seam, 0.0, 1.0),
        np.clip(log_area / math.log(width * height), 0.0, 1.0),
        np.clip(scale, 0.0, 1.0), np.clip(motion, 0.0, 1.0),
    ))


def _iou_series(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = min(len(pred), len(gt), len(valid))
    out = np.zeros(n, dtype=float)
    for index in range(1, n):
        if valid[index] and pred[index, 2] > 0.0 and pred[index, 3] > 0.0:
            out[index] = iou_xywh(pred[index], gt[index])
    return out


def _rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return math.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Average ranks for ties.
    for value in np.unique(score):
        tied = score == value
        ranks[tied] = ranks[tied].mean()
    return float((ranks[y].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def _fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 400, lr: float = 0.08):
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1e-6)
    z = (x - mean) / std
    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    pos_weight = (len(y) - y.sum()) / max(1, y.sum())
    for _ in range(steps):
        logits = np.clip(z @ weights + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        sample_weight = np.where(y > 0, pos_weight, 1.0)
        error = (probability - y) * sample_weight
        weights -= lr * (z.T @ error / len(y) + 1e-4 * weights)
        bias -= lr * error.mean()
    return mean, std, weights, float(bias)


def _predict(x, model):
    mean, std, weights, bias = model
    logits = np.clip(((x - mean) / std) @ weights + bias, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _episodes(score: np.ndarray, threshold: float, budget: float, episode: int,
              expert_quality: np.ndarray | None = None, expert_quality_min: float = 0.0) -> np.ndarray:
    """Turn frame risk into contiguous expert episodes with a hard budget."""
    selected = np.zeros(len(score), dtype=bool)
    debt = 0.0
    index = 1
    while index < len(score):
        debt = max(0.0, debt - budget)
        can_verify = (expert_quality is None
                      or expert_quality[index] >= expert_quality_min)
        if score[index] >= threshold and debt <= 1e-9 and can_verify:
            end = min(len(score), index + episode)
            selected[index:end] = True
            debt += end - index
            index = end
        else:
            index += 1
    return selected


def _metrics(ious: np.ndarray, valid: np.ndarray):
    values = ious[1:len(valid)][valid[1:len(ious)]]
    return {"auc": auc(values) if len(values) else math.nan,
            "sr": success_rate(values) if len(values) else math.nan}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--main-root", required=True)
    parser.add_argument("--expert-root", required=True)
    parser.add_argument("--sequence-file", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--episode", type=int, default=15)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--expert-quality-min", type=float, default=0.0,
                        help="accept an expert episode only when its first probe clears this quality")
    parser.add_argument("--out", required=True)
    parser.add_argument("--export-policy", action="store_true",
                        help="write a diagnostic full-data policy; deployment still requires OOF promotion")
    args = parser.parse_args(argv)
    data_root = Path(args.data)
    main_records = _load_result_root(Path(args.main_root))
    expert_records = _load_result_root(Path(args.expert_root))
    sequences = [line.strip() for line in Path(args.sequence_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    records = []
    for sequence in sequences:
        if sequence not in main_records or sequence not in expert_records:
            continue
        main_dir, expert_dir = main_records[sequence], expert_records[sequence]
        main_box = _read_array(main_dir / "results_erp.txt")
        expert_box = _read_array(expert_dir / "results_erp.txt")
        quality_path = main_dir / "quality.txt"
        quality = np.loadtxt(quality_path) if quality_path.is_file() else np.full(len(main_box), 0.5)
        expert_quality_path = expert_dir / "quality.txt"
        expert_quality = (np.loadtxt(expert_quality_path) if expert_quality_path.is_file()
                          else np.full(len(expert_box), 1.0))
        gt, valid = _groundtruth(data_root, sequence)
        n = min(len(main_box), len(expert_box), len(quality), len(expert_quality), len(gt))
        main_box, expert_box, quality, expert_quality, gt, valid = (
            main_box[:n], expert_box[:n], quality[:n], expert_quality[:n], gt[:n], valid[:n])
        main_iou = _iou_series(main_box, gt, valid)
        expert_iou = _iou_series(expert_box, gt, valid)
        # Expert must beat by a meaningful margin somewhere in the imminent
        # episode; a one-frame tie is not worth losing the speed budget.
        advantage = np.zeros(n, dtype=bool)
        for index in range(1, n):
            end = min(n, index + args.horizon)
            if valid[index:end].any():
                delta = expert_iou[index:end] - main_iou[index:end]
                advantage[index] = bool(delta.mean() > 0.10 and expert_iou[index:end].mean() > 0.50)
        records.append({"sequence": sequence, "x": _features(main_box, quality, 1440, 720),
                        "y": advantage, "valid": valid, "main_iou": main_iou,
                        "expert_iou": expert_iou, "expert_quality": expert_quality})
    if len(records) < args.folds:
        raise SystemExit("insufficient paired sequences for sequence-disjoint OOF routing")

    all_results = []
    for fold in range(args.folds):
        test = [record for index, record in enumerate(records) if index % args.folds == fold]
        train = [record for index, record in enumerate(records) if index % args.folds != fold]
        x_train = np.concatenate([record["x"][record["valid"]] for record in train])
        y_train = np.concatenate([record["y"][record["valid"]] for record in train]).astype(float)
        model = _fit_logistic(x_train, y_train)
        train_scores = _predict(x_train, model)
        threshold = float(np.quantile(train_scores, max(0.0, 1.0 - args.budget)))
        for record in test:
            scores = _predict(record["x"], model)
            select = _episodes(scores, threshold, args.budget, args.episode,
                               record["expert_quality"], args.expert_quality_min)
            fused = np.where(select, record["expert_iou"], record["main_iou"])
            base = _metrics(record["main_iou"], record["valid"])
            policy = _metrics(fused, record["valid"])
            oracle = _metrics(np.maximum(record["main_iou"], record["expert_iou"]), record["valid"])
            all_results.append({
                "sequence": record["sequence"], "fold": fold,
                "router_auc": _rank_auc(record["y"][record["valid"]], scores[record["valid"]]),
                "base_auc": base["auc"], "base_sr": base["sr"],
                "policy_auc": policy["auc"], "policy_sr": policy["sr"],
                "oracle_auc": oracle["auc"], "oracle_sr": oracle["sr"],
                "expert_frame_fraction": float(select[record["valid"]].mean()),
            })
    def mean(name):
        values = [row[name] for row in all_results if math.isfinite(row[name])]
        return float(np.mean(values)) if values else math.nan
    report = {
        "n_sequences": len(all_results), "features": FEATURE_NAMES,
        "budget": args.budget, "episode": args.episode, "horizon": args.horizon,
        "expert_quality_min": args.expert_quality_min,
        "base_auc": mean("base_auc"), "base_sr": mean("base_sr"),
        "policy_auc": mean("policy_auc"), "policy_sr": mean("policy_sr"),
        "oracle_auc": mean("oracle_auc"), "oracle_sr": mean("oracle_sr"),
        "router_auc": mean("router_auc"),
        "expert_frame_fraction": mean("expert_frame_fraction"),
        "policy_auc_delta": mean("policy_auc") - mean("base_auc"),
        "per_sequence": all_results,
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "router_oof_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    if args.export_policy:
        x_all = np.concatenate([record["x"][record["valid"]] for record in records])
        y_all = np.concatenate([record["y"][record["valid"]] for record in records]).astype(float)
        model = _fit_logistic(x_all, y_all)
        scores = _predict(x_all, model)
        mean, std, weights, bias = model
        policy = {
            "feature_names": FEATURE_NAMES,
            "mean": mean.tolist(), "std": std.tolist(),
            "weights": weights.tolist(), "bias": bias,
            "threshold": float(np.quantile(scores, max(0.0, 1.0 - args.budget))),
            "diagnostic_only": True,
            "oof_policy_auc_delta": report["policy_auc_delta"],
        }
        (output / "router_policy_diagnostic.json").write_text(
            json.dumps(policy, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in report if key != "per_sequence"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
