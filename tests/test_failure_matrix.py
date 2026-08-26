# -*- coding: utf-8 -*-
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.evaluation.failure_matrix import (  # noqa: E402
    AuditThresholds,
    audit_sequence,
    discover_sequence_records,
    summarize_bakeoff,
    write_failure_artifacts,
)


def _make_record(root: Path, sequence: str, auc: float, fps: float, boxes) -> None:
    out = root / sequence
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "sequence": sequence,
        "auc": auc,
        "sr": min(1.0, auc + 0.1),
        "fps": fps,
        "resolution": "100x50",
    }
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    np.savetxt(out / "results_erp.txt", np.asarray(boxes), delimiter=",")


def test_audit_tags_and_lost_segments(tmp_path):
    data = tmp_path / "data"
    sequence = "train_sim/seq_0001"
    seq_dir = data / sequence
    seq_dir.mkdir(parents=True)
    # Tiny, polar, seam-adjacent target with a large scale change.
    gt = [
        "175,80,5,5",
        "176,81,5,5",
        "177,82,5,5",
        "178,83,20,20",
    ]
    (seq_dir / "groundtruth.txt").write_text("\n".join(gt), encoding="utf-8")
    root = tmp_path / "baseline"
    boxes = [[97, 1, 2, 2], [40, 20, 2, 2], [40, 20, 2, 2], [40, 20, 2, 2]]
    _make_record(root, sequence, 0.1, 25.0, boxes)
    baseline = discover_sequence_records(root)[sequence]
    row = audit_sequence(sequence, data, baseline, {}, AuditThresholds())
    tags = set(row["scene_tags"].split(";"))
    assert {"small", "polar", "seam", "scale", "hard", "drift"} <= tags
    assert row["lost_segment_count"] == 1
    assert row["longest_lost_segment"] == 3
    assert row["first_lost_frame"] == 1


def test_bakeoff_preserves_unique_scene_expert(tmp_path):
    rows = []
    for index in range(4):
        rows.append({
            "sequence": f"train_sim/seq_{index:04d}",
            "scene_tags": "polar;hard",
            "baseline_auc": 0.2,
            "baseline_sr": 0.2,
            "baseline_fps": 30.0,
            "expert_auc": 0.35,
            "expert_sr": 0.4,
            "expert_fps": 10.0,
            "expert_auc_delta": 0.15,
        })
    bakeoff = summarize_bakeoff(rows, ["expert"], "odtrack")
    expert = bakeoff["methods"]["expert"]
    assert expert["retain"] is True
    assert "scene_expert" in expert["retain_reasons"]
    assert "three_unique_rescues" in expert["retain_reasons"]
    assert expert["clusters"]["polar"]["promote"] is True
    paths = write_failure_artifacts(rows, bakeoff, tmp_path / "out")
    assert all(path.is_file() for path in paths.values())


def test_discovery_prefers_newest_duplicate(tmp_path):
    sequence = "train_real/seq_0001"
    old = tmp_path / "old"
    new = tmp_path / "new"
    _make_record(old, sequence, 0.2, 20.0, [[0, 0, 1, 1]])
    _make_record(new, sequence, 0.6, 30.0, [[0, 0, 1, 1]])
    # Ensure deterministic mtimes even when traversal order differs.
    old_metric = old / sequence / "metrics.json"
    new_metric = new / sequence / "metrics.json"
    os.utime(old_metric, ns=(1_000_000_000, 1_000_000_000))
    os.utime(new_metric, ns=(2_000_000_000, 2_000_000_000))
    records = discover_sequence_records(tmp_path)
    assert records[sequence]["auc"] == 0.6
