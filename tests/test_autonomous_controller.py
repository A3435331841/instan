#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure unit checks for the autonomous controller (no GPU or dataset needed)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.autonomous_precision_controller import (
    acceptance,
    build_diagnostics,
    choose_cluster,
    choose_control,
    choose_next_experiment,
    discover_metrics,
    policy_flags,
)


def _metric(root: Path, sequence: str, auc: float, sr: float, fps: float):
    path = root / sequence.replace("/", "_")
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({
        "sequence": sequence, "auc": auc, "sr": sr, "e2e_fps": fps,
        "n_frames": 10, "n_scored": 9,
    }), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        _metric(root, "train_sim/seq_0001", 0.25, 0.30, 35.0)
        _metric(root, "train_sim/seq_0002", 0.85, 0.90, 40.0)
        records = discover_metrics(root)
        assert len(records) == 2
        tags = {"train_sim/seq_0001": {"scene_tags": "small;seam"}}
        rows, clusters = build_diagnostics(records, tags)
        assert "small" in clusters and "low_auc" in clusters
        assert choose_control(rows, "train_sim/seq_0001") == "train_sim/seq_0002"
        assert choose_cluster(rows, "train_sim/seq_0001", limit=2)[0] == "train_sim/seq_0001"
        proposal = choose_next_experiment(rows, root)
        assert proposal["axis"] == "small_seam_recovery"
        result = acceptance(records, expected=2)
        assert result["full_pass"] is False
        flags = policy_flags({"motion_adaptive": True, "seam_recenter": True,
                              "quality_threshold": 0.4})
        assert "--motion-adaptive" in flags and "--seam-recenter" in flags
    print("ALL AUTONOMOUS CONTROLLER TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
