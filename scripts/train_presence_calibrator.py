#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a causal, CPU-only presence/failure calibrator from tracker traces.

The tracker never sees the labels produced here.  Ground truth is used only by
this offline trainer to define ``failure_next_15_frames`` and to produce
sequence-disjoint OOF scores.  The exported policy is compatible with
``panotrack.pipeline.risk_policy.LinearRiskPolicy`` and consumes only signals
available at inference time (quality, motion, scale, geometry and state).

This intentionally has no scikit-learn dependency: the local recovery machine
can train it with the bundled NumPy runtime.  It is a diagnostic/calibration
artifact until the OOF gates pass; it does not silently alter a submission
router.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov
except ImportError:  # pragma: no cover - the repo root is supplied by the runner
    BFoV = None
    erp_bbox_from_bfov = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_DATA = Path(r"D:\instan\grt360_storage\datasets\official_train\train")
DEFAULT_TRAIN_LIST = PROJECT_ROOT / "data360" / "official_split" / "seqlist_official_train.txt"

FEATURE_NAMES = (
    "quality", "quality_delta", "quality_mean5", "quality_std5",
    "anchor_similarity", "response_entropy", "entropy_missing",
    "center_x", "center_y", "width_norm", "height_norm", "log_area",
    "motion_x", "motion_y", "motion_speed", "log_area_delta",
    "fov_h_norm", "fov_v_norm", "latitude_norm", "seam_distance",
    "fallback_used", "expert_probed", "status_suspect", "status_lost",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -30.0, 30.0)))


def _read_list(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    return {line.strip().replace("\\", "/") for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def _parse_resolution(metrics: dict, seq_dir: Path) -> tuple[float, float]:
    value = str(metrics.get("resolution", ""))
    match = re.search(r"(\d+)\s*x\s*(\d+)", value)
    if match:
        return float(match.group(1)), float(match.group(2))
    # Metadata is normally present in metrics.json.  Keep a cv2-free fallback
    # for trace-only recovery archives; dimensions only scale features.
    try:
        import cv2

        cap = cv2.VideoCapture(str(seq_dir / "video.mp4"))
        width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if width > 0 and height > 0:
            return width, height
    except Exception:  # noqa: BLE001
        pass
    return 1440.0, 720.0


def _load_gt(path: Path) -> list[tuple[float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(x) for x in line.replace(",", " ").split()]
        if len(values) < 4:
            raise ValueError(f"groundtruth row has fewer than four values: {path}")
        rows.append(tuple(values[:4]))
    return rows


def _iou_xywh(a, b) -> float:
    ax, ay, aw, ah = (float(x) for x in a)
    bx, by, bw, bh = (float(x) for x in b)
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-12)


def _dual_iou(a, b, width: float) -> float:
    return max(_iou_xywh(a, b), _iou_xywh((a[0] - width, a[1], a[2], a[3]), b),
               _iou_xywh((a[0] + width, a[1], a[2], a[3]), b))


def _discover(root: Path, allowed: set[str] | None) -> dict[str, tuple[Path, dict]]:
    newest: dict[str, tuple[int, Path, dict]] = {}
    for metrics_path in root.rglob("metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        sequence = str(metrics.get("sequence", "")).replace("\\", "/")
        trace_path = metrics_path.parent / "trace.jsonl"
        if not sequence or not trace_path.is_file() or (allowed is not None and sequence not in allowed):
            continue
        stamp = metrics_path.stat().st_mtime_ns
        previous = newest.get(sequence)
        if previous is None or stamp >= previous[0]:
            newest[sequence] = (stamp, trace_path, metrics)
    return {key: (item[1], item[2]) for key, item in newest.items()}


def _trace(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except (ValueError, TypeError):
            continue
        bbox = item.get("target_bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        item = dict(item)
        item["target_bbox"] = tuple(_finite(v) for v in bbox[:4])
        rows.append(item)
    rows.sort(key=lambda item: int(_finite(item.get("frame_index"), len(rows))))
    return rows


def _features(trace: list[dict], metrics: dict, init: tuple[float, ...], width: float,
              height: float, failure_iou: float, horizon: int, stride: int,
              max_per_sequence: int) -> list[dict]:
    if not trace:
        return []
    predictions = [row["target_bbox"] for row in trace]
    # ``_features`` receives GT-derived labels separately below.  Build causal
    # values first, with no future frame or GT access.
    out = []
    q_history: list[float] = []
    prev_center = None
    prev_area = None
    _lon, lat, fov_h, fov_v = (float(init[0]), float(init[1]), float(init[2]), float(init[3]))
    selected_indices = list(range(1, len(trace), max(1, stride)))
    if max_per_sequence > 0 and len(selected_indices) > max_per_sequence:
        selected_indices = np.linspace(1, len(trace) - 1, max_per_sequence, dtype=int).tolist()
    for index in range(1, len(trace)):
        item = trace[index]
        box = item["target_bbox"]
        cx = (box[0] + box[2] / 2.0) % max(width, 1.0)
        cy = (box[1] + box[3] / 2.0) / max(height, 1.0)
        w_norm = max(box[2], 0.0) / max(width, 1.0)
        h_norm = max(box[3], 0.0) / max(height, 1.0)
        area = max(box[2] * box[3], 1e-6)
        q = np.clip(_finite(item.get("quality"), 0.5), 0.0, 1.0)
        anchor = np.clip(_finite(item.get("anchor_similarity"), 1.0), 0.0, 1.0)
        entropy_raw = item.get("response_entropy")
        entropy_missing = int(not math.isfinite(_finite(entropy_raw, float("nan"))))
        entropy = _finite(entropy_raw, 0.0)
        if prev_center is None:
            motion_x = motion_y = motion_speed = log_area_delta = 0.0
            quality_delta = 0.0
        else:
            dx = cx - prev_center[0]
            if dx > 0.5:
                dx -= 1.0
            elif dx < -0.5:
                dx += 1.0
            motion_x = dx
            motion_y = cy - prev_center[1]
            motion_speed = float(math.hypot(motion_x, motion_y))
            log_area_delta = float(math.log(area / max(prev_area, 1e-6)))
            quality_delta = q - q_history[-1]
        prev_center = (cx, cy)
        prev_area = area
        q_history.append(q)
        recent = np.asarray(q_history[-5:], dtype=float)
        seam_distance = min(cx, 1.0 - cx)
        values = {
            "quality": q, "quality_delta": quality_delta,
            "quality_mean5": float(np.mean(recent)),
            "quality_std5": float(np.std(recent)),
            "anchor_similarity": anchor, "response_entropy": entropy,
            "entropy_missing": entropy_missing, "center_x": cx, "center_y": cy,
            "width_norm": w_norm, "height_norm": h_norm,
            "log_area": float(np.log(area / max(width * height, 1.0))),
            "motion_x": motion_x, "motion_y": motion_y,
            "motion_speed": motion_speed, "log_area_delta": log_area_delta,
            "fov_h_norm": fov_h / 180.0, "fov_v_norm": fov_v / 180.0,
            "latitude_norm": abs(lat) / 90.0, "seam_distance": seam_distance,
            "fallback_used": float(bool(item.get("fallback_used", False))),
            "expert_probed": float(bool(item.get("expert_probed", False))),
            "status_suspect": float(str(item.get("status", "")).lower() == "suspect"),
            "status_lost": float(str(item.get("status", "")).lower() == "lost"),
        }
        if index in selected_indices:
            out.append({"index": index, "bbox": box, "features": values,
                        "horizon": horizon, "failure_iou": failure_iou})
    return out


def _logistic_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-3,
                  iterations: int = 18) -> tuple[np.ndarray, float]:
    if x.ndim != 2 or not len(x):
        raise ValueError("empty calibrator matrix")
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std < 1e-6] = 1.0
    z = (x - mean) / std
    design = np.concatenate([z, np.ones((len(z), 1), dtype=float)], axis=1)
    beta = np.zeros(design.shape[1], dtype=float)
    prevalence = float(np.clip(np.mean(y), 1e-4, 1.0 - 1e-4))
    beta[-1] = math.log(prevalence / (1.0 - prevalence))
    reg = np.eye(design.shape[1], dtype=float) * ridge
    reg[-1, -1] = ridge * 0.1
    for _ in range(max(1, iterations)):
        p = np.asarray(_sigmoid(design @ beta), dtype=float)
        h = np.maximum(p * (1.0 - p), 1e-5)
        hessian = (design.T * h) @ design + reg
        gradient = design.T @ (p - y) + reg @ beta
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta -= step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    return beta[:-1], float(beta[-1]), mean, std


def _predict(x: np.ndarray, mean: np.ndarray, std: np.ndarray,
             weights: np.ndarray, bias: float) -> np.ndarray:
    return np.asarray(_sigmoid(((x - mean) / std) @ weights + bias), dtype=float)


def _auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    positives = int(np.sum(y == 1)); negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1, dtype=float)
    # Ties receive their average rank, preserving the standard Mann-Whitney AUROC.
    unique, first, counts = np.unique(score[order], return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        if count > 1:
            indices = order[start:start + count]
            ranks[indices] = start + 1.0 + (count - 1) / 2.0
    rank_sum = float(np.sum(ranks[y == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _best_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    candidates = np.unique(np.asarray(score, dtype=float))
    if len(candidates) > 128:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 128))
    best = (0.5, -1.0, {})
    for threshold in candidates:
        pred = score >= threshold
        tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[1]:
            best = (float(threshold), f1, {"precision": precision, "recall": recall, "f1": f1,
                                           "probe_rate": float(np.mean(pred))})
    return best[0], best[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="one immutable tracker result root")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--train-list", "--split", dest="train_list", default=str(DEFAULT_TRAIN_LIST),
                        help="train95 sequence list (``--split`` is kept as a compatibility alias)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--label", choices=["presence"], default="presence")
    parser.add_argument("--folds", type=int, default=5, help="sequence-disjoint OOF folds")
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument("--max-per-sequence", type=int, default=20000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--failure-iou", type=float, default=0.30)
    args = parser.parse_args(argv)
    result_root = Path(args.results).resolve()
    data_root = Path(args.data).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    allowed = _read_list(Path(args.train_list).resolve() if args.train_list else None)
    discovered = _discover(result_root, allowed)
    if not discovered:
        raise SystemExit("no trace/metrics pairs found for the requested train split")
    rows: list[dict] = []
    sequence_meta = {}
    for sequence in sorted(discovered):
        trace_path, metrics = discovered[sequence]
        seq_dir = data_root / sequence
        gt_bfov = _load_gt(seq_dir / "groundtruth.txt")
        init_values = tuple(float(x) for x in (seq_dir / "init.txt").read_text(encoding="utf-8").strip().split(","))
        width, height = _parse_resolution(metrics, seq_dir)
        if erp_bbox_from_bfov is None:
            raise SystemExit("panotrack.geometry.bfov is required; set PYTHONPATH to the repository root")
        # The official annotation is BFoV (lon, lat, fov_h, fov_v), while
        # tracker traces are ERP pixel boxes.  Converting here is essential:
        # comparing those two coordinate systems would label every frame as a
        # failure and produce a useless presence model.
        gt = []
        for values in gt_bfov:
            if values[2] <= 0.0 or values[3] <= 0.0:
                gt.append((0.0, 0.0, 0.0, 0.0))
            else:
                gt.append(tuple(float(v) for v in erp_bbox_from_bfov(
                    BFoV(lon=values[0], lat=values[1], fov_h=values[2], fov_v=values[3]),
                    width, height)))
        trace = _trace(trace_path)
        n = min(len(gt), len(trace))
        if n < 2:
            continue
        # Causal features are generated once; labels inspect only future GT.
        feature_rows = _features(trace[:n], metrics, init_values, width, height,
                                 args.failure_iou, args.horizon, args.sample_stride,
                                 args.max_per_sequence)
        dual = np.asarray([_dual_iou(trace[i]["target_bbox"], gt[i], width) for i in range(n)], dtype=float)
        fold_count = max(2, min(int(args.folds), len(discovered)))
        fold = sorted(discovered).index(sequence) % fold_count
        for item in feature_rows:
            i = int(item["index"])
            future = dual[i + 1:min(n, i + 1 + max(1, args.horizon))]
            label = int(len(future) > 0 and float(np.min(future)) < args.failure_iou)
            current_label = int(dual[i] < args.failure_iou)
            rows.append({"sequence": sequence, "frame_index": i, "fold": fold,
                         "label": label, "current_label": current_label,
                         "dual_iou": float(dual[i]), "features": item["features"]})
        sequence_meta[sequence] = {"n_frames": n, "fold": fold,
                                   "metrics_auc": _finite(metrics.get("auc"), float("nan")),
                                   "trace": str(trace_path)}
    if not rows:
        raise SystemExit("no usable causal rows")
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=float)
    current_y = np.asarray([row["current_label"] for row in rows], dtype=float)
    oof = np.full(len(rows), np.nan, dtype=float)
    folds = []
    n_folds = max(2, min(int(args.folds), len(discovered)))
    for fold in range(n_folds):
        test = np.asarray([row["fold"] == fold for row in rows], dtype=bool)
        train = ~test
        if not np.any(test) or len(np.unique(y[train])) < 2:
            continue
        weights, bias, mean, std = _logistic_fit(x[train], y[train])
        oof[test] = _predict(x[test], mean, std, weights, bias)
        folds.append({"fold": fold, "train_rows": int(np.sum(train)), "test_rows": int(np.sum(test)),
                      "test_positive_rate": float(np.mean(y[test])),
                      "auroc": _auroc(y[test], oof[test])})
    valid_oof = np.isfinite(oof)
    threshold, threshold_stats = _best_threshold(y[valid_oof].astype(int), oof[valid_oof]) if np.any(valid_oof) else (0.5, {})
    # Fit the deployable policy only after OOF scores and threshold are fixed.
    weights, bias, mean, std = _logistic_fit(x, y)
    final_score = _predict(x, mean, std, weights, bias)
    policy = {
        "schema": "grt360.linear_presence_policy.v1",
        "label": f"failure_next_{args.horizon}_frames_iou_lt_{args.failure_iou:g}",
        "feature_names": list(FEATURE_NAMES), "mean": mean.tolist(), "std": std.tolist(),
        "weights": weights.tolist(), "bias": bias, "threshold": threshold,
        "inference_contract": "causal tracker signals only; no sequence names, GT or offline results",
        "trained_sequences": sorted(discovered), "train_split": str(Path(args.train_list).resolve()),
        "result_root": str(result_root), "created_at": _utc(),
    }
    (out_root / "presence_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep a compact OOF file useful for routing experiments and audits.
    with (out_root / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "frame_index", "fold", "label", "current_label", "dual_iou", "oof_failure_score"])
        for row, score in zip(rows, oof):
            writer.writerow([row["sequence"], row["frame_index"], row["fold"], row["label"],
                             row["current_label"], f"{row['dual_iou']:.8f}",
                             "" if not np.isfinite(score) else f"{score:.8f}"])
    summary = {
        "schema": "grt360.presence_calibration_summary.v1",
        "created_at": _utc(), "n_sequences": len(discovered), "n_rows": len(rows),
        "positive_rate": float(np.mean(y)), "current_failure_rate": float(np.mean(current_y)),
        "oof_rows": int(np.sum(valid_oof)), "oof_auroc_next_failure": _auroc(y[valid_oof].astype(int), oof[valid_oof]) if np.any(valid_oof) else float("nan"),
        "oof_auroc_current_failure": _auroc(current_y[valid_oof].astype(int), oof[valid_oof]) if np.any(valid_oof) else float("nan"),
        "folds": folds, "n_folds": n_folds, "threshold": threshold, "threshold_stats": threshold_stats,
        # Compatibility fields used by the first calibrator implementation.
        "oof_auc": _auroc(y[valid_oof].astype(int), oof[valid_oof]) if np.any(valid_oof) else float("nan"),
        "oof_f1": threshold_stats.get("f1", float("nan")),
        "expert_selection": {"status": "not_trained", "reason": "only one main trace supplied; expert labels require paired candidate traces"},
        "policy_path": str(out_root / "presence_policy.json"),
        "safety": {"gt_used_only_offline": True, "sequence_name_routing": False, "offline_lookup_at_inference": False},
    }
    (out_root / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    with (out_root / "sequence_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "n_frames", "fold", "metrics_auc", "trace"])
        writer.writeheader(); writer.writerows({"sequence": key, **value} for key, value in sequence_meta.items())
    (out_root / "README.md").write_text(
        "# Presence calibrator\n\n"
        "This is an offline, sequence-disjoint OOF calibration artifact. "
        "The exported policy consumes only causal runtime signals; it is not a deployment route until the OOF gate is reviewed.\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
