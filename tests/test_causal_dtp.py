import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.causal_dtp import CausalDTPRouter


def test_seam_crossing_uses_circular_motion():
    router = CausalDTPRouter(1000, 500, velocity_alpha=1.0)
    router.update([[980, 200, 40, 40], [980, 200, 40, 40], [980, 200, 40, 40]])
    box, _, _ = router.update([[5, 200, 40, 40], [5, 200, 40, 40], [5, 200, 40, 40]])
    assert abs(((box[0] + box[2] / 2.0) % 1000) - 25) < 1e-6


def test_unreliable_teacher_allows_student_recovery():
    router = CausalDTPRouter(1000, 500, teacher_margin=0.01, hold_frames=0)
    router.update([[100, 200, 40, 40], [100, 200, 40, 40], [100, 200, 40, 40]])
    box, expert, rel = router.update([[500, 20, 300, 300], [110, 201, 40, 40], [900, 400, 5, 5]])
    assert expert == 1
    assert np.all(np.isfinite(box))
    assert rel[1] > rel[0]
