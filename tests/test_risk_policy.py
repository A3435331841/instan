# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.pipeline.risk_policy import LinearRiskPolicy


def test_linear_risk_policy_roundtrip():
    policy = LinearRiskPolicy.from_dict({
        "feature_names": ["quality", "motion"],
        "mean": [0.5, 0.0], "std": [0.25, 1.0],
        "weights": [-2.0, 1.0], "bias": 0.0, "threshold": 0.5,
    })
    easy = policy.score({"quality": 1.0, "motion": 0.0})
    hard = policy.score({"quality": 0.0, "motion": 1.0})
    assert hard > easy
    assert policy.should_probe({"quality": 0.0, "motion": 1.0})


if __name__ == "__main__":
    test_linear_risk_policy_roundtrip()
    print("RISK POLICY TEST PASSED")
