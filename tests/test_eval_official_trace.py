# -*- coding: utf-8 -*-
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_official as official  # noqa: E402


def test_run_sequence_emits_trace_and_e2e_latency():
    if official.cv is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sequence = "train_sim/seq_0001"
        seq_dir = root / sequence
        seq_dir.mkdir(parents=True)
        gt_line = "0,0,20,20"
        (seq_dir / "groundtruth.txt").write_text(
            "\n".join([gt_line] * 6) + "\n", encoding="utf-8")
        (seq_dir / "init.txt").write_text(gt_line + "\n", encoding="utf-8")
        video = official.cv.VideoWriter(
            str(seq_dir / "video.mp4"),
            official.cv.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 50))
        assert video.isOpened()
        for index in range(6):
            image = np.full((50, 100, 3), index * 10, dtype=np.uint8)
            video.write(image)
        video.release()
        result = official.run_sequence(
            sequence, root,
            lambda gt_erp: official.GtEchoTracker(gt_erp=gt_erp))
        metrics, _, _, _, _, _, _, traces, latency = result
        assert len(traces) == 6
        assert metrics["e2e_fps"] > 0.0
        assert metrics["tracker_latency_p95_ms"] >= 0.0
        assert latency["e2e_latency_p95_ms"] >= 0.0
        json.dumps(traces, allow_nan=True)


if __name__ == "__main__":
    test_run_sequence_emits_trace_and_e2e_latency()
    print("EVAL OFFICIAL TRACE TEST PASSED")
