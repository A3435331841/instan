# -*- coding: utf-8 -*-
"""Sequence-level failure audit and component bake-off for GRT-360.

The audit is deliberately split from the online router.  Ground truth is used
here to explain failures and to create out-of-fold training labels, but none of
the generated scene tags may be consumed by the submission-time tracker.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from panotrack.evaluation.metrics import dual_iou, iou_xywh
from panotrack.geometry.bfov import BFoV, erp_bbox_from_bfov


@dataclass(frozen=True)
class AuditThresholds:
    """Stable scene-bucketing thresholds used by the diagnostic audit."""

    small_fov_deg: float = 10.0
    large_fov_h_deg: float = 70.0
    large_fov_v_deg: float = 100.0
    polar_lat_deg: float = 70.0
    polar_fraction: float = 0.15
    seam_lon_deg: float = 160.0
    seam_fraction: float = 0.10
    fast_step_p95_deg: float = 5.0
    scale_ratio: float = 4.0
    absent_fraction: float = 0.05
    hard_auc: float = 0.40
    lost_iou: float = 0.50


@dataclass(frozen=True)
class PromotionThresholds:
    """Potential-preserving gates from the agreed GRT-360 bake-off plan."""

    auc_near_precision: float = 0.02
    speedup_near_precision: float = 1.20
    cluster_auc_gain: float = 0.05
    cluster_win_rate: float = 0.60
    cluster_min_unique_rescues: int = 3
    fast_fps: float = 35.0
    auc_near_speed: float = 0.02
    unique_rescue_gain: float = 0.10
    unique_rescue_count: int = 3


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_sequence_records(root: str | Path) -> dict[str, dict]:
    """Find the newest metrics/result pair for every sequence below *root*."""

    root = Path(root)
    found: dict[str, tuple[int, dict]] = {}
    if not root.exists():
        return {}
    for path in root.rglob("metrics.json"):
        try:
            metrics = _load_json(path)
        except (OSError, ValueError, TypeError):
            continue
        sequence = str(metrics.get("sequence", "")).strip().replace("\\", "/")
        if not sequence:
            continue
        stamp = path.stat().st_mtime_ns
        previous = found.get(sequence)
        if previous is not None and previous[0] > stamp:
            continue
        metrics = dict(metrics)
        metrics["metrics_path"] = str(path)
        results = path.parent / "results_erp.txt"
        metrics["results_path"] = str(results) if results.is_file() else None
        found[sequence] = (stamp, metrics)
    return {sequence: record for sequence, (_, record) in found.items()}


def load_groundtruth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load BFoV ground truth and its validity mask."""

    rows: list[list[float]] = []
    valid: list[bool] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        fields = [float(value) for value in line.replace(",", " ").split()]
        if len(fields) < 4:
            raise ValueError(f"invalid ground-truth row in {path}: {line!r}")
        rows.append(fields[:4])
        valid.append(fields[2] > 0.0 and fields[3] > 0.0)
    return np.asarray(rows, dtype=float), np.asarray(valid, dtype=bool)


def _resolution(record: Mapping[str, object], default: tuple[int, int]) -> tuple[int, int]:
    text = str(record.get("resolution", ""))
    match = re.fullmatch(r"\s*(\d+)x(\d+)\s*", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return default


def _angular_steps(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if lon.size < 2:
        return np.zeros((1,), dtype=float)
    lon1, lon2 = np.deg2rad(lon[:-1]), np.deg2rad(lon[1:])
    lat1, lat2 = np.deg2rad(lat[:-1]), np.deg2rad(lat[1:])
    cosine = (np.sin(lat1) * np.sin(lat2)
              + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _prediction_iou_diagnostics(
    gt: np.ndarray,
    valid: np.ndarray,
    results_path: str | None,
    width: int,
    height: int,
    lost_iou: float,
) -> dict[str, object]:
    empty = {
        "baseline_scored_frames": int(max(0, valid[1:].sum())),
        "baseline_mean_frame_iou": math.nan,
        "baseline_mean_frame_dual_iou": math.nan,
        "lost_segment_count": 0,
        "longest_lost_segment": 0,
        "first_lost_frame": -1,
        "lost_frame_fraction": math.nan,
    }
    if not results_path or not Path(results_path).is_file():
        return empty
    predictions = np.loadtxt(results_path, delimiter=",")
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    limit = min(len(gt), len(predictions))
    frame_ious: list[float] = []
    frame_dual: list[float] = []
    frame_indices: list[int] = []
    for index in range(1, limit):
        if not valid[index]:
            continue
        truth = erp_bbox_from_bfov(BFoV(*gt[index, :4]), width, height)
        pred = predictions[index, :4]
        if pred[2] <= 0.0 or pred[3] <= 0.0:
            overlap = overlap_dual = 0.0
        else:
            overlap = float(iou_xywh(pred, truth))
            overlap_dual = float(dual_iou(pred, truth, width))
        frame_indices.append(index)
        frame_ious.append(overlap)
        frame_dual.append(overlap_dual)
    if not frame_ious:
        return empty
    lost = np.asarray(frame_ious) < float(lost_iou)
    runs = _run_lengths(lost)
    return {
        "baseline_scored_frames": len(frame_ious),
        "baseline_mean_frame_iou": float(np.mean(frame_ious)),
        "baseline_mean_frame_dual_iou": float(np.mean(frame_dual)),
        "lost_segment_count": len(runs),
        "longest_lost_segment": max((end - start for start, end in runs), default=0),
        "first_lost_frame": frame_indices[runs[0][0]] if runs else -1,
        "lost_frame_fraction": float(np.mean(lost)),
    }


def audit_sequence(
    sequence: str,
    data_root: str | Path,
    baseline: Mapping[str, object],
    methods: Mapping[str, Mapping[str, object]],
    thresholds: AuditThresholds = AuditThresholds(),
    default_resolution: tuple[int, int] = (1440, 720),
) -> dict[str, object]:
    """Build one row of the sequence failure matrix."""

    gt, valid = load_groundtruth(Path(data_root) / sequence / "groundtruth.txt")
    usable = gt[valid]
    if usable.size == 0:
        raise ValueError(f"{sequence}: no valid ground-truth frames")
    lon, lat, fov_h, fov_v = usable.T
    area = np.maximum(fov_h * fov_v, 1e-6)
    steps = _angular_steps(lon, lat)
    area_p05 = float(np.percentile(area, 5))
    area_p95 = float(np.percentile(area, 95))
    scale_ratio = area_p95 / max(area_p05, 1e-6)
    polar_fraction = float(np.mean(np.abs(lat) > thresholds.polar_lat_deg))
    seam_fraction = float(np.mean(np.abs(lon) > thresholds.seam_lon_deg))
    absent_fraction = float(1.0 - valid.mean())
    width, height = _resolution(baseline, default_resolution)

    tags: list[str] = []
    if np.median(fov_h) < thresholds.small_fov_deg or np.median(fov_v) < thresholds.small_fov_deg:
        tags.append("small")
    if np.median(fov_h) > thresholds.large_fov_h_deg or np.median(fov_v) > thresholds.large_fov_v_deg:
        tags.append("large")
    if polar_fraction > thresholds.polar_fraction:
        tags.append("polar")
    if seam_fraction > thresholds.seam_fraction:
        tags.append("seam")
    if float(np.percentile(steps, 95)) > thresholds.fast_step_p95_deg:
        tags.append("fast")
    if scale_ratio > thresholds.scale_ratio:
        tags.append("scale")
    if absent_fraction > thresholds.absent_fraction:
        tags.append("absent")
    baseline_auc = float(baseline.get("auc", math.nan))
    if np.isfinite(baseline_auc) and baseline_auc < thresholds.hard_auc:
        tags.append("hard")

    row: dict[str, object] = {
        "sequence": sequence,
        "domain": sequence.split("/", 1)[0],
        "n_frames": len(gt),
        "n_valid": int(valid.sum()),
        "absence_rate": absent_fraction,
        "median_fov_h": float(np.median(fov_h)),
        "median_fov_v": float(np.median(fov_v)),
        "fov_area_p05": area_p05,
        "fov_area_p95": area_p95,
        "scale_ratio_p95_p05": scale_ratio,
        "polar_fraction": polar_fraction,
        "seam_fraction": seam_fraction,
        "angular_step_p50_deg": float(np.percentile(steps, 50)),
        "angular_step_p95_deg": float(np.percentile(steps, 95)),
        "scene_tags": ";".join(tags),
        "manual_video_label": "",
        "baseline_auc": baseline_auc,
        "baseline_sr": float(baseline.get("sr", math.nan)),
        "baseline_fps": float(baseline.get("fps", math.nan)),
    }
    row.update(_prediction_iou_diagnostics(
        gt, valid, baseline.get("results_path"), width, height, thresholds.lost_iou))
    if (np.isfinite(float(row["lost_frame_fraction"]))
            and float(row["lost_frame_fraction"]) > 0.50):
        tags.append("drift")
        row["scene_tags"] = ";".join(tags)

    best_name = "baseline"
    best_auc = baseline_auc
    for method_name, record in methods.items():
        prefix = _safe_name(method_name)
        auc_value = float(record.get("auc", math.nan)) if record else math.nan
        row[f"{prefix}_auc"] = auc_value
        row[f"{prefix}_sr"] = float(record.get("sr", math.nan)) if record else math.nan
        row[f"{prefix}_fps"] = float(record.get("fps", math.nan)) if record else math.nan
        row[f"{prefix}_auc_delta"] = auc_value - baseline_auc \
            if np.isfinite(auc_value) and np.isfinite(baseline_auc) else math.nan
        if np.isfinite(auc_value) and (not np.isfinite(best_auc) or auc_value > best_auc):
            best_name, best_auc = method_name, auc_value
    row["best_method"] = best_name
    row["best_auc"] = best_auc
    row["best_auc_delta"] = best_auc - baseline_auc \
        if np.isfinite(best_auc) and np.isfinite(baseline_auc) else math.nan
    return row


def build_failure_matrix(
    data_root: str | Path,
    baseline_records: Mapping[str, Mapping[str, object]],
    method_records: Mapping[str, Mapping[str, Mapping[str, object]]],
    sequences: Iterable[str] | None = None,
    thresholds: AuditThresholds = AuditThresholds(),
) -> list[dict[str, object]]:
    """Audit all requested sequences for which the precision baseline exists."""

    selected = sorted(sequences if sequences is not None else baseline_records)
    rows: list[dict[str, object]] = []
    for sequence in selected:
        baseline = baseline_records.get(sequence)
        if baseline is None:
            continue
        methods = {name: records.get(sequence, {}) for name, records in method_records.items()}
        rows.append(audit_sequence(
            sequence, data_root, baseline, methods, thresholds=thresholds))
    return rows


def _finite(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            result.append(numeric)
    return result


def summarize_bakeoff(
    rows: Sequence[Mapping[str, object]],
    method_names: Sequence[str],
    precision_baseline_name: str,
    speed_baseline_name: str | None = None,
    thresholds: PromotionThresholds = PromotionThresholds(),
) -> dict[str, object]:
    """Summarize global and scene-cluster component potential."""

    baseline_auc = float(np.mean(_finite(row.get("baseline_auc") for row in rows)))
    baseline_fps_values = _finite(row.get("baseline_fps") for row in rows)
    baseline_fps = float(np.mean(baseline_fps_values)) if baseline_fps_values else math.nan
    summary: dict[str, object] = {
        "precision_baseline": precision_baseline_name,
        "speed_baseline": speed_baseline_name,
        "n_sequences": len(rows),
        "baseline_auc": baseline_auc,
        "baseline_fps": baseline_fps,
        "audit_thresholds": asdict(AuditThresholds()),
        "promotion_thresholds": asdict(thresholds),
        "methods": {},
    }

    for method_name in method_names:
        prefix = _safe_name(method_name)
        common = [row for row in rows
                  if np.isfinite(float(row.get(f"{prefix}_auc", math.nan)))
                  and np.isfinite(float(row.get("baseline_auc", math.nan)))]
        auc_values = [float(row[f"{prefix}_auc"]) for row in common]
        common_baseline_auc_values = [float(row["baseline_auc"]) for row in common]
        common_baseline_fps_values = _finite(row.get("baseline_fps") for row in common)
        common_baseline_auc = (float(np.mean(common_baseline_auc_values))
                               if common_baseline_auc_values else math.nan)
        common_baseline_fps = (float(np.mean(common_baseline_fps_values))
                               if common_baseline_fps_values else math.nan)
        sr_values = _finite(row.get(f"{prefix}_sr") for row in common)
        fps_values = _finite(row.get(f"{prefix}_fps") for row in common)
        deltas = [float(row[f"{prefix}_auc_delta"]) for row in common]
        unique_rescues = [row["sequence"] for row in common
                          if float(row[f"{prefix}_auc_delta"]) >= thresholds.unique_rescue_gain]
        clusters: dict[str, object] = {}
        for tag in ("small", "large", "polar", "seam", "fast", "scale", "absent", "drift", "hard"):
            cluster = [row for row in common if tag in str(row.get("scene_tags", "")).split(";")]
            if not cluster:
                continue
            cluster_deltas = [float(row[f"{prefix}_auc_delta"]) for row in cluster]
            wins = sum(delta > 0.01 for delta in cluster_deltas)
            rescues = sum(delta >= thresholds.unique_rescue_gain for delta in cluster_deltas)
            mean_delta = float(np.mean(cluster_deltas))
            win_rate = wins / len(cluster)
            clusters[tag] = {
                "n": len(cluster),
                "mean_auc_delta": mean_delta,
                "win_rate_gt_0_01": win_rate,
                "unique_rescues_gt_0_10": rescues,
                "promote": bool(
                    len(cluster) >= thresholds.cluster_min_unique_rescues
                    and mean_delta >= thresholds.cluster_auc_gain
                    and win_rate >= thresholds.cluster_win_rate
                    and rescues >= thresholds.cluster_min_unique_rescues
                ),
            }
        mean_auc = float(np.mean(auc_values)) if auc_values else math.nan
        mean_fps = float(np.mean(fps_values)) if fps_values else math.nan
        speed_auc = math.nan
        if speed_baseline_name:
            speed_key = f"{_safe_name(speed_baseline_name)}_auc"
            speed_values = _finite(row.get(speed_key) for row in common)
            if speed_values:
                speed_auc = float(np.mean(speed_values))
        reasons: list[str] = []
        if (np.isfinite(mean_auc) and np.isfinite(mean_fps)
                and np.isfinite(common_baseline_fps)
                and mean_auc >= common_baseline_auc - thresholds.auc_near_precision
                and mean_fps >= common_baseline_fps * thresholds.speedup_near_precision):
            reasons.append("near_precision_and_20pct_faster")
        if any(bool(value["promote"]) for value in clusters.values()):
            reasons.append("scene_expert")
        if (np.isfinite(mean_auc) and np.isfinite(mean_fps) and np.isfinite(speed_auc)
                and mean_fps >= thresholds.fast_fps
                and mean_auc >= speed_auc - thresholds.auc_near_speed):
            reasons.append("fast_main")
        if len(unique_rescues) >= thresholds.unique_rescue_count:
            reasons.append("three_unique_rescues")
        summary["methods"][method_name] = {
            "n": len(common),
            "baseline_auc_common": common_baseline_auc,
            "baseline_fps_common": common_baseline_fps,
            "auc": mean_auc,
            "sr": float(np.mean(sr_values)) if sr_values else math.nan,
            "fps": mean_fps,
            "auc_delta": float(np.mean(deltas)) if deltas else math.nan,
            "wins_gt_0_01": sum(delta > 0.01 for delta in deltas),
            "unique_rescues_gt_0_10": unique_rescues,
            "clusters": clusters,
            "retain": bool(reasons),
            "retain_reasons": reasons,
        }
    return summary


def write_failure_artifacts(
    rows: Sequence[Mapping[str, object]],
    bakeoff: Mapping[str, object],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write CSV/JSON artifacts without requiring pandas."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "failure_matrix.csv"
    json_path = output_dir / "failure_matrix.json"
    bakeoff_path = output_dir / "bakeoff.json"
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False, allow_nan=True),
                         encoding="utf-8")
    bakeoff_path.write_text(json.dumps(bakeoff, indent=2, ensure_ascii=False, allow_nan=True),
                            encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "bakeoff": bakeoff_path}
