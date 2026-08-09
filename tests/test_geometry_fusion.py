# -*- coding: utf-8 -*-
"""Tests for the prediction-only geometry expert router."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fuse_geometry_experts import (  # noqa: E402
    circular_blend_box,
    fuse_sequence,
    seam_weight,
)


def test_seam_weight_geometry():
    assert seam_weight([45, 0, 10, 10], 100, search_factor=2) == 0.0
    assert seam_weight([96, 0, 8, 8], 100, search_factor=4) > 0.9


def test_circular_blend_uses_short_arc():
    box = circular_blend_box([94, 2, 4, 4], [2, 2, 4, 4], 0.5, 100)
    center = (box[0] + box[2] / 2.0) % 100
    assert center < 1.0 or center > 99.0


def test_fusion_preserves_first_frame():
    baseline = np.array([[10, 1, 5, 5], [11, 1, 5, 5]], dtype=float)
    seam = np.array([[10, 1, 5, 5], [90, 1, 5, 5]], dtype=float)
    fused, weights = fuse_sequence(baseline, seam, 100)
    assert np.array_equal(fused[0], baseline[0])
    assert weights[0] == 0.0 and weights[1] == 0.0
    assert np.allclose(fused[1], baseline[1], atol=1e-12)


def main():
    tests = [test_seam_weight_geometry, test_circular_blend_uses_short_arc,
             test_fusion_preserves_first_frame]
    for test in tests:
        test()
        print('PASS', test.__name__)
    print(f'{len(tests)}/{len(tests)} passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
