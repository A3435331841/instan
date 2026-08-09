# -*- coding: utf-8 -*-
"""GRT-360 Commit 5b 测试：Soft S² Motion Prior 及其 pipeline 集成。

运行: python tests/test_motion_prior.py
覆盖:
  1. 一致方向 -> 得分几乎不打折
  2. 大角距离 -> 得分显著压低
  3. 快目标 sigma 自适应：同角距离下高速目标折扣更小
  4. 角距离计算正确（0/90/180 度）
  5. d_min 阈值内不打折
  6. pipeline 集成：开启 motion_prior 时 tracker 被构造、无开启时不受影响
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.geometry.motion_prior import SoftS2MotionPrior
from panotrack.geometry.sphere import lonlat_to_unit, normalize
from panotrack.pipeline.pipeline import PanoTracker


def _vec(lon, lat):
    x, y, z = lonlat_to_unit(float(lon), float(lat))
    return normalize(np.array([x, y, z]))


def test_aligned_no_penalty():
    mp = SoftS2MotionPrior()
    v = _vec(0.0, 0.0)
    s, d = mp(0.9, v, v, angular_speed_deg=0.0)
    assert d == 0.0
    assert abs(s - 0.9) < 1e-9


def test_large_distance_penalizes():
    mp = SoftS2MotionPrior(lambda_=1.0, sigma_base=15.0)
    s_small, _ = mp(0.9, _vec(0.0, 0.0), _vec(5.0, 0.0), angular_speed_deg=0.0)
    s_big, _ = mp(0.9, _vec(0.0, 0.0), _vec(60.0, 0.0), angular_speed_deg=0.0)
    assert s_small > s_big
    assert s_big < 0.5


def test_fast_target_adaptive_sigma():
    mp = SoftS2MotionPrior()
    obs = _vec(30.0, 0.0)
    s_slow, _ = mp(0.9, _vec(0.0, 0.0), obs, angular_speed_deg=0.0)
    s_fast, _ = mp(0.9, _vec(0.0, 0.0), obs, angular_speed_deg=60.0)
    # 高速目标 sigma 更大，折扣更小
    assert s_fast > s_slow


def test_angular_distance_values():
    mp = SoftS2MotionPrior()
    assert mp.angular_distance_deg(_vec(0, 0), _vec(0, 0)) < 1e-6
    assert abs(mp.angular_distance_deg(_vec(0, 0), _vec(90, 0)) - 90.0) < 1e-6
    assert abs(mp.angular_distance_deg(_vec(0, 0), _vec(180, 0)) - 180.0) < 1e-6
    # 极区穿越：经度 0 到 180 在球面 = 半圆 180°
    assert abs(mp.angular_distance_deg(_vec(0, 0), _vec(179.9, 0)) - 179.9) < 1e-3


def test_d_min_threshold():
    mp = SoftS2MotionPrior(d_min=5.0)
    s, _ = mp(0.9, _vec(0.0, 0.0), _vec(3.0, 0.0), angular_speed_deg=0.0)
    assert abs(s - 0.9) < 1e-9  # 阈值内不打折


def test_pipeline_flag_creates_prior():
    on = PanoTracker(config={'motion_prior': True})
    assert on._motion_prior is not None
    off = PanoTracker(config={'motion_prior': False})
    assert off._motion_prior is None
    default = PanoTracker()
    assert default._motion_prior is None  # 默认关闭


def test_pipeline_prior_params():
    t = PanoTracker(config={'motion_prior': True, 'mp_sigma_base': 30.0,
                            'mp_lambda': 2.0})
    assert t._motion_prior.sigma_base == 30.0
    assert t._motion_prior.lambda_ == 2.0


def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    fail = 0
    for t in tests:
        try:
            t()
            print(f'  PASS  {t.__name__}')
        except Exception as e:
            fail += 1
            print(f'  FAIL  {t.__name__}: {e}')
    print(f'\n{len(tests) - fail}/{len(tests)} passed')
    return fail


if __name__ == '__main__':
    sys.exit(1 if _run() else 0)