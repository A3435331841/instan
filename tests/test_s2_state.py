# -*- coding: utf-8 -*-
"""GRT-360 Commit 3 测试：S² 单位球向量状态 + Rodrigues 旋转。

运行: python tests/test_s2_state.py
覆盖:
  1. S² 状态基本：unit norm、predict/update 契约、bfov 往返
  2. 子午线跨界（+179° lon -> -179°）无速度冲击
  3. 近极点穿越（+86° -> +88° -> +89°）方向正确
  4. 静止目标：速度收敛、位置不漂移
  5. 突然跳变离群观测：position 平滑、不至于灾难
  6. FoV 平滑
  7. 对跖点守卫：rotation_vec_between 不 NaN
  8. Rodrigues 旋转正确性（绕 z 轴 90°）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.geometry.bfov import BFoV
from panotrack.geometry.sphere import (
    rodrigues_rotate, rotation_vec_between, normalize, unit_to_lonlat,
    lonlat_to_unit,
)
from panotrack.pipeline.state import SphericalState


def _wfov(lon, lat, fh=30.0, fv=20.0):
    return BFoV(lon=lon, lat=lat, fov_h=fh, fov_v=fv)


def test_rodrigues_rotates_around_z():
    """绕 z 轴旋转 90°（右手系）：+x -> +y。"""
    v = np.array([1.0, 0.0, 0.0])
    out = rodrigues_rotate(v, np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
    assert np.allclose(normalize(out), np.array([0.0, 1.0, 0.0]), atol=1e-9), out


def test_rotation_vec_between_antipodal_guard():
    """对跖点：rotation_vec_between 不产生 NaN，角度为 π。"""
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([-1.0, 0.0, 0.0])
    r = rotation_vec_between(p, q)
    assert np.all(np.isfinite(r)), r
    assert float(np.linalg.norm(r)) > 3.14  # 接近 π


def test_unit_vector_norm():
    """unit_vector 恒为单位向量。"""
    st = SphericalState(_wfov(10.0, -20.0))
    p = st.unit_vector
    assert np.allclose(np.linalg.norm(p), 1.0, atol=1e-12)
    st.update(_wfov(12.0, -21.0))
    assert np.allclose(np.linalg.norm(st.unit_vector), 1.0, atol=1e-12)


def test_bfov_roundtrip():
    """lonlat_to_unit / unit_to_lonlat 往返一致。"""
    s = _wfov(170.0, 45.0)
    x, y, z = lonlat_to_unit(s.lon, s.lat)
    lon, lat = unit_to_lonlat(x, y, z)
    assert abs(lon - s.lon) < 1e-9 and abs(lat - s.lat) < 1e-9


def test_meridian_crossing_no_speed_spike():
    """+179° -> +179.5° -> -179.5°（跨 ±180°）不应产生速度冲击。"""
    st = SphericalState(_wfov(179.0, 0.0), pos_alpha=0.8, vel_alpha=0.4)
    st.update(_wfov(179.5, 0.0))
    st.update(_wfov(-179.5, 0.0))
    # 经度差实际仅 1°，速度应为小量（远小于 179° 的冲击）
    assert st.angular_speed_deg < 5.0, st.angular_speed_deg
    # 后验应已平滑到 -179.5 附近
    post = st.bfov
    assert abs(post.lon - (-179.5)) < 2.0, post


def test_pole_approach():
    """+86° -> +88° -> +89° 纬度递增，方向正确、不 NaN。"""
    st = SphericalState(_wfov(0.0, 85.9), pos_alpha=0.9, vel_alpha=0.5)
    st.update(_wfov(0.0, 87.9))
    st.update(_wfov(0.0, 88.9))
    post = st.bfov
    assert np.isfinite(post.lon) and np.isfinite(post.lat)
    assert post.lat > 88.0, post  # 明显向极点移动


def test_stationary_converges():
    """目标静止：观测 = 固定位置，位置应收敛不漂移。"""
    st = SphericalState(_wfov(-30.0, 10.0), pos_alpha=0.8, vel_alpha=0.5)
    for _ in range(50):
        st.update(_wfov(-30.0, 10.0))
    post = st.bfov
    assert abs(post.lon - (-30.0)) < 1e-3
    assert abs(post.lat - 10.0) < 1e-3
    assert st.angular_speed_deg < 1e-3  # 速度衰减到 0


def test_sudden_outlier_smoothed():
    """突然大幅跳变观测：位置平滑，不应瞬时跳到离群点。"""
    st = SphericalState(_wfov(0.0, 0.0), pos_alpha=0.5, vel_alpha=0.3)
    st.update(_wfov(0.0, 0.0))
    st.update(_wfov(80.0, 0.0))  # 大幅离群
    # 位置只移了 pa=0.5 的比例，不会瞬间到 80°
    assert abs(st.bfov.lon) < 60.0, st.bfov


def test_fov_smooth():
    """FoV 指数平滑。"""
    st = SphericalState(_wfov(0.0, 0.0, fh=30.0, fv=20.0), fov_alpha=0.5)
    st.update(_wfov(0.0, 0.0, fh=40.0, fv=20.0))
    assert abs(st.bfov.fov_h - 35.0) < 1e-6  # 0.5*30 + 0.5*40


def test_predict_does_not_mutate():
    """predict 不修改内部状态（可重复调用一致）。"""
    st = SphericalState(_wfov(0.0, 0.0), pos_alpha=0.8, vel_alpha=0.4)
    st.update(_wfov(2.0, 0.0))
    p1 = st.predict()
    p2 = st.predict()
    assert p1.lon == p2.lon and p1.lat == p2.lat
    # 后验仍为 update 后的值
    assert abs(st.bfov.lon) < 2.0


def test_prediction_error_deg():
    """prediction_error_deg 量纲正确。"""
    st = SphericalState(_wfov(0.0, 0.0), pos_alpha=0.8, vel_alpha=0.4)
    st.update(_wfov(2.0, 0.0))
    err = st.prediction_error_deg(_wfov(2.0, 0.0))
    assert 0.0 <= err <= 180.0
    assert np.isfinite(err)


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