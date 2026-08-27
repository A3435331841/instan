#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRT-360 local autonomous precision loop.

The controller is intentionally conservative: it never deletes artifacts,
never pushes Docker images, and never starts a second GPU bake-off while a
tracked batch/sequence process is alive.  It can snapshot and diagnose a run
without mutation, or (with ``--apply``) launch one whitelisted next experiment
into a fresh directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(r"D:\instan\grt360_storage\datasets\official_train\train")
DEFAULT_OUT = Path(r"D:\instan\grt360_scratch\autonomous_precision")
DEFAULT_FAILURE_MATRIX = Path(
    r"D:\instan\grt360_storage\experiments\server_exit_20260827\runs\failure_audit_v3_20260826\failure_matrix.csv"
)
DEFAULT_POLICY = PROJECT_ROOT / "configs" / "autonomous_local_b224.json"
TRACKER_MARKERS = ("run_sutrack_b224_openvino_batch", "run_sutrack_b224_openvino_sequence")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def device_probe() -> dict:
    result = {"python": sys.version, "platform": platform.platform(), "devices": []}
    try:
        import openvino as ov

        core = ov.Core()
        result["openvino"] = ov.__version__
        result["devices"] = list(core.available_devices)
        result["device_properties"] = {
            name: {
                "full_name": str(core.get_property(name, "FULL_DEVICE_NAME"))
                if name in core.available_devices else None,
            }
            for name in core.available_devices
        }
    except Exception as exc:  # noqa: BLE001
        result["openvino_error"] = str(exc)
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_devices"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:  # noqa: BLE001
        result["torch_error"] = str(exc)
    return result


def process_snapshot() -> list[dict]:
    rows = []
    try:
        import psutil

        for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "status"]):
            info = process.info
            cmd = " ".join(info.get("cmdline") or [])
            if any(marker in cmd for marker in TRACKER_MARKERS):
                rows.append({
                    "pid": info.get("pid"), "ppid": info.get("ppid"),
                    "name": info.get("name"), "status": info.get("status"),
                    "command": cmd,
                })
    except Exception as exc:  # noqa: BLE001
        rows.append({"error": str(exc)})
    return rows


def discover_metrics(root: Path) -> list[dict]:
    records = []
    if not root.exists():
        return records
    for path in root.rglob("metrics.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        sequence = str(row.get("sequence", "")).replace("\\", "/").strip()
        if not sequence:
            continue
        row = dict(row)
        row["metrics_path"] = str(path)
        row["results_path"] = str(path.parent / "results_erp.txt")
        row["trace_path"] = str(path.parent / "trace.jsonl")
        records.append(row)
    # A root may contain reruns.  Keep newest record per sequence.
    newest = {}
    for row in records:
        path = Path(row["metrics_path"])
        key = row["sequence"]
        stamp = path.stat().st_mtime_ns if path.exists() else 0
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, row)
    return [item[1] for item in sorted(newest.values(), key=lambda item: item[1]["sequence"])]


def discover_metrics_many(roots: list[Path]) -> list[dict]:
    """Merge immutable result roots, preferring the newest run per sequence."""
    merged: dict[str, dict] = {}
    for root in roots:
        for row in discover_metrics(root):
            key = row["sequence"]
            if key not in merged:
                merged[key] = row
                continue
            previous = Path(merged[key]["metrics_path"])
            candidate = Path(row["metrics_path"])
            if candidate.stat().st_mtime_ns >= previous.stat().st_mtime_ns:
                merged[key] = row
    return [merged[key] for key in sorted(merged)]


def read_failure_tags(path: Path | None) -> dict[str, dict]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row.get("sequence", "").replace("\\", "/"): row for row in csv.DictReader(handle)}


def audit_data(data_root: Path) -> list[dict]:
    """Check video/annotation cardinality and basic decode metadata for 130 sequences."""
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        return [{"sequence": "*", "issue": "cv2_unavailable", "detail": str(exc)}]
    rows = []
    for block in ("train_real", "train_sim"):
        root = data_root / block
        if not root.is_dir():
            rows.append({"sequence": block, "issue": "missing_block", "detail": str(root)})
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir() or not (seq_dir / "video.mp4").is_file():
                continue
            sequence = f"{block}/{seq_dir.name}"
            gt_path = seq_dir / "groundtruth.txt"
            try:
                gt_lines = len([line for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip()])
            except OSError as exc:
                rows.append({"sequence": sequence, "issue": "missing_groundtruth", "detail": str(exc)})
                continue
            cap = cv2.VideoCapture(str(seq_dir / "video.mp4"))
            opened = bool(cap.isOpened())
            frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if opened else 0
            width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) if opened else 0
            height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) if opened else 0
            cap.release()
            issue = "ok"
            if not opened:
                issue = "decode_open_failed"
            elif frame_count != gt_lines:
                issue = "frame_gt_mismatch"
            elif width <= 0 or height <= 0:
                issue = "invalid_resolution"
            rows.append({"sequence": sequence, "video_frames": frame_count,
                         "gt_lines": gt_lines, "width": width, "height": height,
                         "issue": issue})
    return rows


def numeric(row: dict, key: str, default=float("nan")) -> float:
    try:
        value = float(row.get(key, default))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def trace_diagnostics(record: dict) -> dict:
    trace_path = Path(record.get("trace_path", ""))
    quality = []
    fallback = 0
    statuses = defaultdict(int)
    if trace_path.is_file():
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                quality.append(numeric(item, "quality", 1.0))
                fallback += int(bool(item.get("fallback_used", False)))
                statuses[str(item.get("status", "unknown"))] += 1
        except (OSError, ValueError, TypeError):
            pass
    q = np.asarray(quality, dtype=float)
    return {
        "quality_mean": float(np.mean(q)) if q.size else float("nan"),
        "quality_p10": float(np.percentile(q, 10)) if q.size else float("nan"),
        "quality_low_fraction": float(np.mean(q <= 0.40)) if q.size else float("nan"),
        "fallback_trace_count": int(fallback),
        "statuses": dict(statuses),
    }


def classify(record: dict, tags: dict) -> tuple[list[str], dict]:
    tag_row = tags.get(record.get("sequence", ""), {})
    scene = [value for value in str(tag_row.get("scene_tags", "")).split(";") if value]
    diagnostics = trace_diagnostics(record)
    auc_value = numeric(record, "auc")
    fps_value = numeric(record, "e2e_fps")
    if np.isfinite(auc_value) and auc_value < 0.40:
        scene.append("low_auc")
    if np.isfinite(fps_value) and fps_value < 30.0:
        scene.append("slow")
    if diagnostics["quality_low_fraction"] == diagnostics["quality_low_fraction"] and diagnostics["quality_low_fraction"] > 0.20:
        scene.append("low_quality")
    if diagnostics["fallback_trace_count"] >= 20:
        scene.append("fallback_overuse")
    return sorted(set(scene)), diagnostics


def build_diagnostics(records: list[dict], tags: dict) -> tuple[list[dict], dict]:
    rows = []
    for record in records:
        scene, diagnostic = classify(record, tags)
        row = dict(record)
        row["scene_tags_online"] = ";".join(scene)
        row.update(diagnostic)
        rows.append(row)
    clusters = defaultdict(list)
    for row in rows:
        for tag in row["scene_tags_online"].split(";"):
            if tag:
                clusters[tag].append(row)
    summary = {}
    for tag, items in sorted(clusters.items()):
        summary[tag] = {
            "n": len(items),
            "mean_auc": float(np.mean([numeric(item, "auc") for item in items])),
            "mean_sr": float(np.mean([numeric(item, "sr") for item in items])),
            "mean_e2e_fps": float(np.mean([numeric(item, "e2e_fps") for item in items])),
            "low_auc_count": sum(numeric(item, "auc") < 0.40 for item in items),
        }
    return rows, summary


def write_diagnostic_artifacts(out_root: Path, rows: list[dict], clusters: dict,
                               data_audit_rows: list[dict] | None = None) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["sequence", "auc", "sr", "e2e_fps", "n_frames", "n_scored",
              "scene_tags_online", "quality_mean", "quality_p10",
              "quality_low_fraction", "fallback_trace_count", "updates_frozen",
              "updates_frozen_frame", "active_search_factor", "active_fallback_search_factor"]
    with (out_root / "scenario_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_root / "scenario_summary.json").write_text(
        json.dumps({"generated_at": utc_now(), "clusters": clusters, "rows": rows},
                   ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    if data_audit_rows is not None:
        audit_fields = ["sequence", "video_frames", "gt_lines", "width", "height", "issue", "detail"]
        with (out_root / "data_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=audit_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data_audit_rows)
        (out_root / "data_audit.json").write_text(
            json.dumps({"generated_at": utc_now(), "rows": data_audit_rows},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    # The controller's matrix is intentionally diagnostic: it combines causal
    # runtime signals with the optional offline scene tags, but never becomes a
    # submission-time lookup table.
    with (out_root / "failure_matrix_130.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lowest = sorted(rows, key=lambda row: numeric(row, "auc"))[:20]
    slowest = sorted(rows, key=lambda row: numeric(row, "e2e_fps"))[:20]
    lines = ["# Autonomous precision diagnostics", "", f"Generated: {utc_now()}", "",
             "## Lowest AUC", "", "| Sequence | AUC | SR | FPS | Tags |", "|---|---:|---:|---:|---|"]
    for row in lowest:
        lines.append(f"| {row['sequence']} | {numeric(row, 'auc'):.4f} | {numeric(row, 'sr'):.4f} | {numeric(row, 'e2e_fps'):.2f} | {row['scene_tags_online']} |")
    lines += ["", "## Slowest sequences", "", "| Sequence | FPS | P95 latency | AUC |", "|---|---:|---:|---:|"]
    for row in slowest:
        lines.append(f"| {row['sequence']} | {numeric(row, 'e2e_fps'):.2f} | {numeric(row, 'e2e_latency_p95_ms'):.2f} ms | {numeric(row, 'auc'):.4f} |")
    lines += ["", "## Cluster means", "", "| Tag | N | AUC | SR | FPS |", "|---|---:|---:|---:|---:|"]
    for tag, item in clusters.items():
        lines.append(f"| {tag} | {item['n']} | {item['mean_auc']:.4f} | {item['mean_sr']:.4f} | {item['mean_e2e_fps']:.2f} |")
    (out_root / "failure_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    latency_rows = [row for row in rows if np.isfinite(numeric(row, "e2e_fps"))]
    latency = {
        "n_sequences": len(latency_rows),
        "mean_e2e_fps": float(np.mean([numeric(row, "e2e_fps") for row in latency_rows])) if latency_rows else float("nan"),
        "min_e2e_fps": float(np.min([numeric(row, "e2e_fps") for row in latency_rows])) if latency_rows else float("nan"),
        "max_e2e_p95_ms": float(np.max([numeric(row, "e2e_latency_p95_ms") for row in latency_rows])) if latency_rows else float("nan"),
        "sequences": [{"sequence": row.get("sequence"), "e2e_fps": row.get("e2e_fps"),
                       "e2e_latency_p95_ms": row.get("e2e_latency_p95_ms"),
                       "fallback_trace_count": row.get("fallback_trace_count")} for row in latency_rows],
    }
    (out_root / "latency_summary.json").write_text(
        json.dumps(latency, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    (out_root / "route_policy.json").write_text(json.dumps({
        "enabled": False,
        "status": "diagnostic_only",
        "reason": "No promoted OOF policy; sequence names and GT are forbidden at inference",
        "features": ["quality", "response_entropy", "anchor_similarity", "spherical_motion", "scale_change", "geometry_risk"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_next_experiment(rows: list[dict], out_root: Path) -> dict:
    """Return one deterministic next action; never invent a new axis per loop."""
    if not rows:
        return {"status": "blocked", "reason": "no completed sequence metrics"}
    low = sorted(rows, key=lambda row: numeric(row, "auc"))
    worst = low[0]
    tags = set(str(worst.get("scene_tags_online", "")).split(";"))
    if "large" in tags or numeric(worst, "active_search_factor") == 2.0:
        axis = "large_target_recovery"
        flags = ["--search-factor-mode", "large_fov", "--large-fov-fallback-search-factor", "5.0"]
    elif "small" in tags or "seam" in tags or "fallback_overuse" in tags:
        axis = "small_seam_recovery"
        flags = ["--polar-rectify", "--polar-max-frame", "20",
                 "--small-template-factor", "1.5", "--fallback-search-factor", "3.25"]
    elif "scale" in tags or "low_quality" in tags:
        axis = "scale_memory_guard"
        flags = ["--auto-freeze-scale-threshold", "0.25", "--auto-freeze-scale-window", "40",
                 "--auto-freeze-max-frame", "100"]
    else:
        axis = "baseline_recheck"
        flags = []
    experiment_id = f"{axis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    proposal = {
        "experiment_id": experiment_id,
        "axis": axis,
        "trigger_sequence": worst.get("sequence"),
        "trigger_auc": numeric(worst, "auc"),
        "flags": flags,
        "promotion": {
            "single_auc_gain": 0.10,
            "normal_regression_max": 0.01,
            "cluster_auc_gain": 0.05,
            "cluster_win_rate": 0.60,
            "full_auc": 0.80,
            "full_sr": 0.80,
            "full_e2e_fps": 30.0,
        },
        "created_at": utc_now(),
    }
    (out_root / "next_experiment.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    return proposal


def write_snapshot(args, out_root: Path, records: list[dict]) -> dict:
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=False).stdout.strip()
    git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=False).stdout.splitlines()
    data_sequences = []
    for block in ("train_real", "train_sim"):
        root = Path(args.data) / block
        if root.is_dir():
            data_sequences.extend(f"{block}/{p.name}" for p in sorted(root.iterdir())
                                  if p.is_dir() and (p / "video.mp4").is_file())
    payload = {
        "snapshot_id": f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "captured_at": utc_now(),
        "data_root": str(Path(args.data).resolve()),
        "n_data_sequences": len(data_sequences),
        "data_sequences_sha256": hashlib.sha256("\n".join(data_sequences).encode()).hexdigest(),
        "result_root": str(Path(args.results).resolve()),
        "completed_metrics_count": len(records),
        "git_head": git_head,
        "git_status": git_status,
        "devices": device_probe(),
        "processes": process_snapshot(),
        "weights": {
            "xml": {"path": str(Path(args.xml).resolve()), "sha256": sha256_file(Path(args.xml))},
            "high_xml": {"path": str(Path(args.high_xml).resolve()) if args.high_xml else None,
                         "sha256": sha256_file(Path(args.high_xml)) if args.high_xml else None},
        },
        "safety": {"docker_push": False, "delete_files": False, "gt_routing": False},
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "autonomous_run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_policy(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _flag(flags: list[str], name: str, value) -> None:
    if value is None or value is False:
        return
    if value is True:
        flags.append(name)
    else:
        flags.extend([name, str(value)])


def policy_flags(policy: dict) -> list[str]:
    """Translate the approved policy JSON into batch-runner arguments."""
    flags: list[str] = []
    for name, key in (
        ("--quality-threshold", "quality_threshold"),
        ("--quality-run", "quality_run"),
        ("--switch-deadline", "switch_deadline"),
        ("--search-factor", "search_factor"),
        ("--search-factor-mode", "search_factor_mode"),
        ("--large-fov-fallback-search-factor", "large_fov_fallback_search_factor"),
        ("--template-factor", "template_factor"),
        ("--fallback-search-factor", "fallback_search_factor"),
        ("--fallback-quality-threshold", "fallback_quality_threshold"),
        ("--fallback-min-gain", "fallback_min_gain"),
        ("--fallback-run", "fallback_run"),
        ("--fallback-start-frame", "fallback_start_frame"),
        ("--fallback-cooldown", "fallback_cooldown"),
        ("--auto-freeze-scale-threshold", "auto_freeze_scale_threshold"),
        ("--auto-freeze-scale-window", "auto_freeze_scale_window"),
        ("--auto-freeze-quality-slope", "auto_freeze_quality_slope"),
        ("--auto-freeze-scale-step-p95", "auto_freeze_scale_step_p95"),
        ("--auto-freeze-quality-floor", "auto_freeze_quality_floor"),
        ("--auto-freeze-scale-step-median-max", "auto_freeze_scale_step_median_max"),
        ("--auto-freeze-max-frame", "auto_freeze_max_frame"),
        ("--small-template-factor", "small_template_factor"),
        ("--small-template-width", "small_template_width"),
        ("--polar-latitude-threshold", "polar_latitude_threshold"),
        ("--polar-aspect-max", "polar_aspect_max"),
        ("--polar-small-width", "polar_small_width"),
        ("--polar-max-frame", "polar_max_frame"),
    ):
        if key in policy:
            _flag(flags, name, policy[key])
    for name, key in (
        ("--motion-adaptive", "motion_adaptive"),
        ("--seam-recenter", "seam_recenter"),
        ("--polar-rectify", "polar_rectify"),
    ):
        if policy.get(key):
            flags.append(name)
    if policy.get("polar_require_initial", True) is False:
        flags.append("--no-polar-require-initial")
    if policy.get("small_template_require_initial", True) is False:
        flags.append("--no-small-template-require-initial")
    return flags


def choose_control(rows: list[dict], trigger: str) -> str | None:
    candidates = [row for row in rows if row.get("sequence") != trigger]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-numeric(row, "auc"), row.get("sequence", "")))
    return str(candidates[0].get("sequence"))


def choose_cluster(rows: list[dict], trigger: str, limit: int = 8) -> list[str]:
    trigger_row = next((row for row in rows if row.get("sequence") == trigger), None)
    trigger_tags = set(str(trigger_row.get("scene_tags_online", "")).split(";")) if trigger_row else set()
    candidates = []
    for row in rows:
        if row.get("sequence") == trigger:
            continue
        tags = set(str(row.get("scene_tags_online", "")).split(";"))
        overlap = len(trigger_tags & tags)
        candidates.append((overlap, numeric(row, "auc"), str(row.get("sequence"))))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = [trigger] + [item[2] for item in candidates[: max(0, limit - 1)]]
    return selected


def acceptance(rows: list[dict], expected: int | None = None,
               data_audit_rows: list[dict] | None = None) -> dict:
    finite = [row for row in rows if np.isfinite(numeric(row, "auc"))]
    anomalies = []
    for row in finite:
        if numeric(row, "auc") < 0.0 or numeric(row, "auc") > 1.0:
            anomalies.append({"sequence": row.get("sequence"), "reason": "auc_out_of_range"})
        if numeric(row, "sr") < 0.0 or numeric(row, "sr") > 1.0:
            anomalies.append({"sequence": row.get("sequence"), "reason": "sr_out_of_range"})
        if numeric(row, "e2e_fps") <= 0.0:
            anomalies.append({"sequence": row.get("sequence"), "reason": "invalid_fps"})
    data_issues = [row for row in (data_audit_rows or []) if row.get("issue") != "ok"]
    anomalies.extend({"sequence": row.get("sequence"), "reason": row.get("issue")}
                     for row in data_issues)
    result = {
        "completed": len(finite), "expected": expected,
        "anomalies": anomalies,
        "data_issue_count": len(data_issues),
        "mean_auc": float(np.mean([numeric(row, "auc") for row in finite])) if finite else float("nan"),
        "mean_sr": float(np.mean([numeric(row, "sr") for row in finite])) if finite else float("nan"),
        "mean_e2e_fps": float(np.mean([numeric(row, "e2e_fps") for row in finite])) if finite else float("nan"),
    }
    result["full_pass"] = bool(
        expected is not None and len(finite) >= expected and not anomalies and
        result["mean_auc"] > 0.8 and result["mean_sr"] > 0.8 and result["mean_e2e_fps"] > 30.0
    )
    result["status"] = "pass" if result["full_pass"] else ("anomaly" if anomalies else "in_progress")
    return result


def run_one_experiment(args, proposal: dict, rows: list[dict], policy: dict, out_root: Path) -> tuple[int, Path]:
    if process_snapshot():
        print("[autonomous] GPU tracker process already active; refusing concurrent launch", file=sys.stderr)
        return 3, out_root
    batch = PROJECT_ROOT / "scripts" / "run_sutrack_b224_openvino_batch.py"
    experiment_root = out_root / "experiments" / proposal["experiment_id"]
    experiment_root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(batch), "--xml", str(Path(args.xml).resolve()),
               "--data", str(Path(args.data).resolve()), "--out", str(experiment_root),
               "--device", args.device]
    if args.high_xml:
        command += ["--high-xml", str(Path(args.high_xml).resolve())]
    command += policy_flags(policy)
    trigger = str(proposal.get("trigger_sequence", ""))
    if args.seqs:
        command += ["--seqs", args.seqs]
    elif args.apply_scope == "micro":
        control = choose_control(rows, trigger)
        selected = [value for value in (trigger, control) if value]
        if selected:
            command += ["--seqs", ",".join(selected)]
    elif args.apply_scope == "cluster":
        selected = choose_cluster(rows, trigger)
        if selected:
            command += ["--seqs", ",".join(selected)]
    if args.max_frames is not None:
        command += ["--max-frames", str(args.max_frames)]
    command += proposal.get("flags", [])
    (experiment_root / "experiment.json").write_text(json.dumps({
        "experiment_id": proposal["experiment_id"], "command": command,
        "started_at": utc_now(), "proposal": proposal,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[autonomous] launching:", shlex.join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    (experiment_root / "experiment_complete.json").write_text(json.dumps({
        "experiment_id": proposal["experiment_id"], "finished_at": utc_now(),
        "returncode": completed.returncode,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(completed.returncode), experiment_root


def make_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--results", required=True, help="current result root to snapshot/diagnose")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--failure-matrix", default=str(DEFAULT_FAILURE_MATRIX))
    ap.add_argument("--xml", required=True)
    ap.add_argument("--high-xml", default=None)
    ap.add_argument("--device", choices=["CPU", "GPU"], default="GPU")
    ap.add_argument("--seqs", default=None, help="optional comma-separated sequences for --apply")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--apply-scope", choices=["micro", "cluster", "full"], default="micro")
    ap.add_argument("--max-frames", type=int, default=450)
    ap.add_argument("--apply", action="store_true", help="launch the proposed next experiment")
    ap.add_argument("--max-iterations", type=int, default=1)
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    out_root = Path(args.out).resolve()
    policy = load_policy(Path(args.policy).resolve())
    tags = read_failure_tags(Path(args.failure_matrix) if args.failure_matrix else None)
    result_roots = [Path(args.results).resolve()]
    data_audit_rows = audit_data(Path(args.data).resolve())
    last_code = 0
    for iteration in range(max(1, int(args.max_iterations))):
        records = discover_metrics_many(result_roots)
        snapshot = write_snapshot(args, out_root, records)
        rows, clusters = build_diagnostics(records, tags)
        write_diagnostic_artifacts(out_root, rows, clusters, data_audit_rows)
        proposal = choose_next_experiment(rows, out_root)
        decision = acceptance(records, expected=130, data_audit_rows=data_audit_rows)
        (out_root / "promotion.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        (out_root / "full130_summary.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        valid_path = PROJECT_ROOT / "data360" / "official_split" / "seqlist_official_valid.txt"
        valid_sequences = {line.strip().replace("\\", "/") for line in valid_path.read_text(encoding="utf-8").splitlines()
                           if line.strip()} if valid_path.is_file() else set()
        valid_rows = [row for row in records if row.get("sequence") in valid_sequences]
        valid_decision = acceptance(valid_rows, expected=len(valid_sequences) or 35,
                                    data_audit_rows=data_audit_rows)
        (out_root / "valid35_summary.json").write_text(
            json.dumps(valid_decision, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
        print(json.dumps({
            "iteration": iteration + 1,
            "snapshot_id": snapshot["snapshot_id"],
            "completed": len(records), "data_sequences": snapshot["n_data_sequences"],
            "devices": snapshot["devices"].get("devices", []),
            "acceptance": decision,
            "next_experiment": proposal,
        }, ensure_ascii=False))
        if proposal.get("status") == "blocked":
            return 2
        if not args.apply:
            return 0
        if args.apply_scope == "full":
            args.max_frames = None
        last_code, experiment_root = run_one_experiment(args, proposal, rows, policy, out_root)
        if experiment_root != out_root:
            result_roots.append(experiment_root)
            experiment_rows = discover_metrics(experiment_root)
            experiment_decision = acceptance(experiment_rows, data_audit_rows=data_audit_rows)
            (experiment_root / "promotion.json").write_text(
                json.dumps(experiment_decision, ensure_ascii=False, indent=2, allow_nan=True),
                encoding="utf-8")
        if last_code != 0:
            return last_code
    return last_code


if __name__ == "__main__":
    raise SystemExit(main())
