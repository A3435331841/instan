import numpy as np

from scripts.fuse_external_results import fuse_sequence


def _boxes(values):
    return np.asarray([[float(x), 10.0, 20.0, 20.0] for x in values])


def test_fusion_keeps_periodic_centers_finite():
    od = _boxes([1910, 1915, 0, 5])
    ue = _boxes([1910, 1915, 0, 5])
    lf = od.copy()
    fused, expert = fuse_sequence(od, ue, lf, 1920)
    assert np.all(np.isfinite(fused))
    assert np.all((fused[:, 0] >= 0) & (fused[:, 0] < 1920))
    assert np.all(expert == 0)


def test_confidence_router_can_recover_from_low_confidence():
    od = _boxes([100, 102, 500, 504])
    ue = _boxes([100, 102, 104, 106])
    lf = od.copy()
    confidence = np.array([1.0, 1.0, 0.05, 0.05])
    fused, expert = fuse_sequence(
        od, ue, lf, 1920, confidence_threshold=0.2,
        min_low_confidence_run=1, switch_margin=2.0)
    assert expert[2] == 1
    assert abs(fused[2, 0] - ue[2, 0]) < 1e-6


def test_confidence_length_is_strict():
    boxes = _boxes([1, 2, 3])
    try:
        fuse_sequence(boxes, boxes, boxes, 1920, od_confidence=[1.0])
    except ValueError as exc:
        assert 'confidence length' in str(exc)
    else:
        raise AssertionError('expected confidence length validation')
