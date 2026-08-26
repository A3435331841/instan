# -*- coding: utf-8 -*-
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.data.training_manifest import (  # noqa: E402
    augmentation_policy,
    build_training_manifest,
    write_training_manifest,
)


def test_failure_balancing_and_no_leakage():
    rows = []
    train = []
    for index in range(20):
        sequence = f"train_{'real' if index % 2 == 0 else 'sim'}/seq_{index:04d}"
        train.append(sequence)
        tags = "polar;hard" if index < 5 else "seam" if index < 10 else ""
        rows.append({"sequence": sequence, "scene_tags": tags, "baseline_auc": 0.2 + index / 100})
    # A validation-only row exists in the matrix but must not enter the manifest.
    rows.append({"sequence": "train_real/seq_valid", "scene_tags": "hard", "baseline_auc": 0.1})
    records = build_training_manifest(rows, train, n_folds=5)
    assert len(records) == 20
    assert all(record["sequence"] != "train_real/seq_valid" for record in records)
    assert set(record["oof_fold"] for record in records) == set(range(5))
    hard = next(record for record in records if "hard" in record["scene_tags"])
    easy = next(record for record in records if not record["scene_tags"])
    assert hard["sample_weight_raw"] > easy["sample_weight_raw"]
    assert abs(sum(record["sample_weight"] for record in records) / len(records) - 1.0) < 1e-9


def test_augmentation_policy_targets_failure_modes():
    hard = augmentation_policy(["polar", "small", "fast", "absent"])
    easy = augmentation_policy([])
    assert hard["pole_rotation"] > easy["pole_rotation"]
    assert hard["small_target_degrade"] > easy["small_target_degrade"]
    assert hard["motion_blur"] > easy["motion_blur"]
    assert hard["temporal_occlusion"] > easy["temporal_occlusion"]


def test_manifest_artifacts():
    rows = [{
        "sequence": f"train_real/seq_{index:04d}",
        "scene_tags": "hard" if index % 2 else "seam",
        "baseline_auc": 0.3,
    } for index in range(10)]
    train = [row["sequence"] for row in rows]
    records = build_training_manifest(rows, train, n_folds=5)
    with tempfile.TemporaryDirectory() as tmp:
        paths = write_training_manifest(records, tmp)
        assert all(path.is_file() for path in paths.values())
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        assert summary["n_sequences"] == 10
        assert summary["valid35_included"] is False


if __name__ == "__main__":
    test_failure_balancing_and_no_leakage()
    test_augmentation_policy_targets_failure_modes()
    test_manifest_artifacts()
    print("ALL TRAINING MANIFEST TESTS PASSED")
