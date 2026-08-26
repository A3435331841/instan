# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.pipeline.adaptive_spherical import (  # noqa: E402
    AdaptiveRouterConfig,
    AdaptiveSphericalTracker,
    AdaptiveStatus,
    ExpertBudget,
)


class FakeTracker:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.index = 0
        self.init_calls = 0
        self.last_init = None

    def init(self, frame, bbox):
        self.init_calls += 1
        self.last_init = list(bbox)

    def track(self, frame):
        output = self.outputs[min(self.index, len(self.outputs) - 1)]
        self.index += 1
        return output


class FakeRedetector:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def search(self, frame, erp_downscale=2):
        self.calls += 1
        return self.result


def frame(width=200, height=100):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[40:60, 90:110] = 255
    return image


def output(box, quality):
    return {"target_bbox": box, "quality": quality}


def test_expert_budget_caps_long_run_fraction():
    budget = ExpertBudget(0.2, capacity=1.0)
    consumed = []
    for _ in range(100):
        budget.advance()
        consumed.append(budget.consume())
    assert sum(consumed) <= 21
    assert budget.call_fraction <= 0.21


def test_latency_controller_reduces_expert_budget():
    tracker = AdaptiveSphericalTracker(
        FakeTracker([output([0, 0, 10, 10], 1.0)]),
        config=AdaptiveRouterConfig(
            expert_max_fraction=0.20, target_frame_ms=33.333,
            latency_ema_alpha=1.0))
    tracker._update_latency_budget(main_ms=28.0, expert_ms=58.0)
    assert 0.08 < tracker.budget.fraction < 0.10


def test_public_contract_and_geometry_route():
    # Box center near the ERP seam, so a geometry route should be recorded.
    main = FakeTracker([output([195, 40, 10, 20], 0.9)])
    expert = FakeTracker([output([195, 40, 10, 20], 0.8)])
    tracker = AdaptiveSphericalTracker(
        main, expert,
        config=AdaptiveRouterConfig(expert_max_fraction=1.0, suspect_run=1))
    tracker.init(frame(), [90, 40, 20, 20])
    result = tracker.track(frame())
    required = {"target_bbox", "quality", "status", "response_entropy",
                "anchor_similarity", "latency_ms", "expert_used"}
    assert required <= set(result)
    assert "seam" in result["route_reasons"]
    assert expert.init_calls == 1
    assert result["status"] in {status.value for status in AdaptiveStatus}


def test_lost_redetect_verify_and_recover():
    box = [90, 40, 20, 20]
    main = FakeTracker([
        output(box, 0.05),
        output(box, 0.05),
        output(box, 0.9),
        output(box, 0.9),
    ])
    redetector = FakeRedetector((box, 0.9))
    config = AdaptiveRouterConfig(
        suspect_run=1, lost_run=2, redetect_interval=1,
        verify_frames=2, anchor_min_similarity=0.4,
        geometry_risk=1.0,
    )
    tracker = AdaptiveSphericalTracker(main, redetector=redetector, config=config)
    tracker.init(frame(), box)
    first = tracker.track(frame())
    second = tracker.track(frame())
    assert first["status"] == "suspect"
    assert second["status"] == "lost"
    third = tracker.track(frame())
    assert third["status"] == "verify"
    assert "global_redetect" in third["route_reasons"]
    fourth = tracker.track(frame())
    fifth = tracker.track(frame())
    assert fourth["status"] == "verify"
    assert fifth["status"] == "normal"
    assert main.init_calls >= 2


def test_expert_can_replace_failed_main():
    good_box = [90, 40, 20, 20]
    bad_box = [10, 10, 20, 20]
    main = FakeTracker([output(bad_box, 0.1)])
    expert = FakeTracker([output(good_box, 0.9)])
    tracker = AdaptiveSphericalTracker(
        main, expert,
        config=AdaptiveRouterConfig(
            expert_max_fraction=1.0, anchor_min_similarity=0.4,
            suspect_run=1, verify_frames=1, geometry_risk=1.0,
        ),
        expert_name="precision_expert",
    )
    tracker.init(frame(), good_box)
    result = tracker.track(frame())
    assert result["target_bbox"] == good_box
    assert result["expert_used"] == "precision_expert"
    assert "expert_switch" in result["route_reasons"]


def test_expert_episode_keeps_temporal_state():
    good_box = [90, 40, 20, 20]
    bad_box = [10, 10, 20, 20]
    main = FakeTracker([output(bad_box, 0.1), output(good_box, 0.9)])
    expert = FakeTracker([output(good_box, 0.9)] * 3)
    tracker = AdaptiveSphericalTracker(
        main, expert,
        config=AdaptiveRouterConfig(
            expert_max_fraction=1.0, expert_episode_frames=3,
            anchor_min_similarity=0.4, suspect_run=1,
            verify_frames=3, geometry_risk=1.0,
        ),
        expert_name="episode_expert",
    )
    tracker.init(frame(), good_box)
    first = tracker.track(frame())
    second = tracker.track(frame())
    third = tracker.track(frame())
    assert first["expert_episode_remaining"] == 2
    assert "expert_episode" in second["route_reasons"]
    assert third["expert_episode_remaining"] == 0
    assert expert.init_calls == 1
    assert expert.index == 3
    assert main.init_calls == 2  # sequence init + hand-back after episode


if __name__ == "__main__":
    test_expert_budget_caps_long_run_fraction()
    test_latency_controller_reduces_expert_budget()
    test_public_contract_and_geometry_route()
    test_lost_redetect_verify_and_recover()
    test_expert_can_replace_failed_main()
    test_expert_episode_keeps_temporal_state()
    print("ALL ADAPTIVE SPHERICAL TESTS PASSED")
