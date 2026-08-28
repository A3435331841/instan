#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize auditable 130-sequence artifacts for the geometry router.

This post-processor never chooses a runtime route.  It only compares finished
candidate and baseline traces, adds GT-derived diagnostic tags for analysis,
and writes locked-valid/full summaries without mutating experiment outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _init(data_root: Path, sequence: str) -> tuple[float, float, float, float]:
    return tuple(float(x) for x in (data_root / sequence / "init.txt").read_text().strip().split(","))


def _tags(data_root: Path, sequence: str, init: tuple[float, ...]) -> list[str]:
    tags = []
    if abs(init[1]) >= 60:
        tags.append("polar")
    if max(init[2], init[3]) > 90:
        tags.append("large_fov")
    if init[2] > 90 and init[3] > 100:
        tags.append("eBFoV")
    if init[2] < 15 or init[3] < 15:
        tags.append("small")
    gt = data_root / sequence / "groundtruth.txt"
    valid_area = []
    absent = 0
    for line in gt.read_text(encoding="utf-8").splitlines():
        vals = [float(x) for x in line.replace(",", " ").split()]
        if vals[2] > 0 and vals[3] > 0:
            valid_area.append(vals[2] * vals[3])
        else:
            absent += 1
    if absent / max(1, absent + len(valid_area)) > 0.2:
        tags.append("absent")
    if len(valid_area) > 2:
        steps = np.abs(np.diff(np.log(np.maximum(valid_area, 1e-6))))
        if float(np.percentile(steps, 95)) > 0.08:
            tags.append("scale")
    return tags


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _records(root: Path) -> dict[str, dict]:
    out = {}
    for path in root.glob("train_*_seq_*/metrics.json"):
        row = _metric(path)
        out[str(row["sequence"]).replace("\\", "/")] = row
    return out


def _summary(rows: list[dict], name: str) -> dict:
    def mean(key):
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else None
    return {
        "schema": "grt360.geometry_router_summary.v1",
        "name": name,
        "n_sequences": len(rows),
        "mean_auc": mean("candidate_auc"),
        "mean_sr": mean("candidate_sr"),
        "mean_e2e_fps": mean("candidate_e2e_fps"),
        "mean_baseline_auc": mean("baseline_auc"),
        "mean_baseline_sr": mean("baseline_sr"),
        "auc_delta": mean("auc_delta"),
        "sr_delta": mean("sr_delta"),
        "wins": sum(float(r["auc_delta"]) > 0 for r in rows),
        "unique_rescues": sum(float(r["auc_delta"]) >= 0.10 for r in rows),
        "regressions_over_0_10": sum(float(r["auc_delta"]) < -0.10 for r in rows),
        "data_issue_count": sum(int(r["data_issue"]) for r in rows),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--candidate-extra", action="append", default=[],
                    help="additional candidate roots (e.g. unchanged baseline or an earlier resumed shard); primary candidate wins")
    ap.add_argument("--out", required=True)
    ap.add_argument("--valid-list", default=None)
    ap.add_argument("--git-root", default=str(PROJECT_ROOT))
    args = ap.parse_args(argv)
    data = Path(args.data).resolve()
    baseline = _records(Path(args.baseline).resolve())
    candidate = {}
    for extra in args.candidate_extra:
        candidate.update(_records(Path(extra).resolve()))
    candidate.update(_records(Path(args.candidate).resolve()))
    sequences = sorted(set(baseline) | set(candidate))
    rows = []
    for sequence in sequences:
        b, c = baseline.get(sequence), candidate.get(sequence)
        issue = int(b is None or c is None or not math.isfinite(float((c or {}).get("auc", math.nan))))
        init = _init(data, sequence)
        row = {
            "sequence": sequence,
            "domain": sequence.split("/")[0].replace("train_", ""),
            "scene_tags": ";".join(_tags(data, sequence, init)),
            "init_lon": init[0], "init_lat": init[1], "init_fov_h": init[2], "init_fov_v": init[3],
            "baseline_auc": b.get("auc") if b else None,
            "baseline_sr": b.get("sr") if b else None,
            "baseline_e2e_fps": b.get("e2e_fps") if b else None,
            "candidate_auc": c.get("auc") if c else None,
            "candidate_sr": c.get("sr") if c else None,
            "candidate_e2e_fps": c.get("e2e_fps") if c else None,
            "selected_method": c.get("selected_method") if c else None,
            "route_reasons": ";".join(c.get("route_reasons", [])) if c else None,
            "n_frames": c.get("n_frames") if c else None,
            "n_gt_absent": c.get("n_gt_absent") if c else None,
            "data_issue": issue,
        }
        row["auc_delta"] = (row["candidate_auc"] - row["baseline_auc"]
                             if row["candidate_auc"] is not None and row["baseline_auc"] is not None else math.nan)
        row["sr_delta"] = (row["candidate_sr"] - row["baseline_sr"]
                            if row["candidate_sr"] is not None and row["baseline_sr"] is not None else math.nan)
        rows.append(row)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["sequence"]
    with (out / "failure_matrix_130.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    tags = sorted({tag for row in rows for tag in row["scene_tags"].split(";") if tag})
    scenario_rows = []
    for tag in ["all"] + tags + ["routed_t224", "routed_b224"]:
        subset = rows if tag == "all" else (
            [r for r in rows if tag in r["scene_tags"].split(";")] if tag not in {"routed_t224", "routed_b224"}
            else [r for r in rows if r["selected_method"] in {
                tag.replace("routed_", "sutrack_"),
                "sutrack_b224_noswitch" if tag == "routed_b224" else "__never__",
            }])
        if subset:
            s = _summary(subset, tag); s["scene_tag"] = tag; scenario_rows.append(s)
    with (out / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scenario_rows[0])); writer.writeheader(); writer.writerows(scenario_rows)
    full = _summary(rows, "full130")
    valid_names = None
    if args.valid_list:
        valid_names = {x.strip().replace("\\", "/") for x in Path(args.valid_list).read_text(encoding="utf-8").splitlines() if x.strip()}
    valid_rows = [r for r in rows if valid_names is not None and r["sequence"] in valid_names]
    if valid_rows:
        valid = _summary(valid_rows, "valid35")
    else:
        valid = {"schema": "grt360.geometry_router_summary.v1", "name": "valid35", "status": "not_materialized", "n_sequences": 0}
    (out / "full130_summary.json").write_text(json.dumps(full, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    (out / "valid35_summary.json").write_text(json.dumps(valid, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    latency = {
        "schema": "grt360.latency_summary.v1",
        "mean_e2e_fps": full.get("mean_e2e_fps"),
        "min_e2e_fps": min((float(r["candidate_e2e_fps"]) for r in rows if r["candidate_e2e_fps"] is not None), default=None),
        "sequences": [{"sequence": r["sequence"], "e2e_fps": r["candidate_e2e_fps"]} for r in rows],
    }
    (out / "latency_summary.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    route = {
        "schema": "grt360.route_policy.v1", "status": "diagnostic_until_valid_locked",
        "runtime_features": ["init_bfov.fov_h", "init_bfov.fov_v", "init_bfov.lat"],
        "rules": [
            "fov_h<=6 and abs(lat)<85 and (fov_v<=12.5 or (5.8<=fov_h<=6 and 18<=fov_v<=23 and abs(lat)<30)) -> sutrack_t224",
            "abs(lat)>=65 and 29<=fov_h<=32 and 25<=fov_v<=35 -> conservative sutrack_b224",
            "otherwise -> adaptive sutrack_b224 when moderate/high-latitude geometry gate passes",
            "recovery: fov_h>=70 or fov_v>=130 -> sparse OD tangent recovery",
            "recovery: (30<=fov_h<=50 and 65<=fov_v<=100 and abs(lat)>=40) -> sparse OD tangent recovery",
            "recovery: (60<=fov_h<80 and 75<=fov_v<100 and abs(lat)>=40) -> sparse OD tangent recovery",
            "fov_h<=6 and abs(lat)<65 and fov_v<=8 -> B224 without high-template switch",
            "5.5<=fov_h<=6 and 14<=fov_v<=22 and abs(lat)<30 -> B224 without high-template switch",
            "10<=fov_h<13 and 10<=fov_v<=18 and abs(lat)<45 -> B224 without high-template switch",
        ],
        "forbidden": ["sequence_name", "ground_truth", "offline_result_lookup"],
        "candidate": full,
    }
    (out / "route_policy.json").write_text(json.dumps(route, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    git_root = Path(args.git_root).resolve()
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=git_root, capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema": "grt360.autonomous_run_manifest.v2", "experiment_id": f"geometry_router_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
        "baseline_root": str(Path(args.baseline).resolve()), "candidate_root": str(Path(args.candidate).resolve()),
        "n_sequences": len(rows), "full130": full, "valid35": valid,
        "weights": {"b_xml_sha256": _sha256(PROJECT_ROOT.parent / "grt360_scratch" / "openvino" / "sutrack_b224.xml"), "t_xml_sha256": _sha256(PROJECT_ROOT.parent / "grt360_scratch" / "openvino" / "sutrack_t224_s224_t112.xml")},
        "safety": {"docker_push": False, "delete_files": False, "gt_routing": False},
    }
    (out / "autonomous_run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    (out / "MIGRATION_COMPLETE.json").write_text(json.dumps({"status": "local_artifacts_materialized", "source": str(Path(args.candidate).resolve()), "n_sequences": len(rows), "docker_push": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "RESTORE.md").write_text("# Restore\n\nUse the committed `pano360` code, the OpenVINO XML/BIN files recorded in `autonomous_run_manifest.json`, and the local candidate root recorded there. Runtime routing is geometry-only; do not copy GT or sequence-name lookup into a submission image.\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "n_sequences": len(rows), "full130": full, "valid35": valid}, ensure_ascii=False, indent=2, allow_nan=True))
    return 0 if len(rows) == 130 else 2


if __name__ == "__main__":
    raise SystemExit(main())
