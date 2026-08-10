# -*- coding: utf-8 -*-
"""GRT-360 Commit 5 测试：GeometryDescriptor + GeometryRouter。

运行: python tests/test_router.py
覆盖:
  1. GeometryDescriptor 各分量范围与方向
  2. 近极/赤道的纬度描述差异
  3. seam 距离（跨子午线）
  4. risk 处于 [0,1] 且近极/大速度/高不确定度更高
  5. Router softmax 输出归一化（和为 1）
  6. Router 从 torch 导出的参数字典加载
  7. fuse_weights MoE 叠加
  8. 描述子维度不匹配报错
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.geometry.descriptor import GeometryDescriptor
from panotrack.geometry.router import GeometryRouter


def test_descriptor_shape_and_dtype():
    d = GeometryDescriptor()
    g = d.descriptor(lon=0.0, lat=0.0)
    assert g.shape == (8,)
    assert g.dtype == np.float64
    assert 0.0 <= g.min() and g.max() <= 1.0


def test_descriptor_pole_vs_equator():
    d = GeometryDescriptor()
    eq = d.descriptor(lon=0.0, lat=0.0)
    near_pole = d.descriptor(lon=0.0, lat=85.0)
    # 近极：sin(|lat|) 更大、sec 拉伸更大
    assert near_pole[0] > eq[0]
    assert near_pole[1] > eq[1]


def test_descriptor_seam_distance():
    d = GeometryDescriptor()
    center = d.descriptor(lon=0.0, lat=0.0)
    seam = d.descriptor(lon=180.0, lat=0.0)
    # seam 距离：中心处大，子午线处接近 0
    assert center[2] > seam[2]
    assert seam[2] < 1.0


def test_descriptor_area_and_speed_norm():
    d = GeometryDescriptor()
    g = d.descriptor(lon=0.0, lat=0.0, angular_speed_deg=200.0,
                     motion_uncertainty_deg=200.0, bbox_area=2.0)
    # 归一化截断到 1.0
    assert g[5] == 1.0
    assert g[6] == 1.0
    assert g[3] == 1.0


def test_descriptor_risk_range_and_direction():
    d = GeometryDescriptor()
    r_easy = d.risk(lon=0.0, lat=0.0, angular_speed_deg=1.0,
                    motion_uncertainty_deg=1.0, tracker_confidence=0.95)
    r_hard = d.risk(lon=0.0, lat=85.0, angular_speed_deg=80.0,
                    motion_uncertainty_deg=80.0, tracker_confidence=0.1)
    assert 0.0 <= r_easy <= 1.0
    assert 0.0 <= r_hard <= 1.0
    assert r_hard > r_easy


def test_router_softmax_normalized():
    r = GeometryRouter(n_experts=3, seed=0)
    g = GeometryDescriptor().descriptor(lon=10.0, lat=20.0)
    a = r(g)
    assert a.shape == (3,)
    assert abs(float(a.sum()) - 1.0) < 1e-9
    assert (a >= 0).all()


def test_router_expert_pick():
    r = GeometryRouter(n_experts=3, seed=1)
    g = GeometryDescriptor().descriptor(lon=0.0, lat=0.0)
    k = r.pick_expert(g)
    assert 0 <= k < 3


def test_router_load_torch_params():
    # 模拟 torch 训练导出：形状相同的参数 dict
    rng = np.random.default_rng(7)
    params = {
        'W1': rng.normal(0, 0.5, (16, 8)),
        'b1': rng.normal(0, 0.1, (16,)),
        'W2': rng.normal(0, 0.5, (4, 16)),
        'b2': rng.normal(0, 0.1, (4,)),
    }
    r = GeometryRouter(n_experts=4, params=params)
    g = GeometryDescriptor().descriptor(lon=30.0, lat=-15.0)
    a = r(g)
    assert a.shape == (4,)
    assert abs(float(a.sum()) - 1.0) < 1e-9


def test_router_fuse_weights():
    r = GeometryRouter(n_experts=2, seed=3)
    g = GeometryDescriptor().descriptor(lon=0.0, lat=0.0)
    base = np.ones((4, 4))
    deltas = [np.zeros((4, 4)), np.ones((4, 4))]
    fused = r.fuse_weights(g, base, deltas)
    a = r(g)
    # fused = base + alpha[1]*1
    expected = base + a[1]
    assert np.allclose(fused, expected)


def test_router_describe_roundtrip():
    r = GeometryRouter(n_experts=3, seed=5)
    params = r.describe()
    r2 = GeometryRouter(n_experts=3, params=params)
    g = GeometryDescriptor().descriptor(lon=0.0, lat=0.0)
    assert np.allclose(r(g), r2(g))


def test_router_wrong_dim_raises():
    r = GeometryRouter(n_experts=3, seed=0)
    try:
        r(np.zeros(7))
    except ValueError:
        return
    raise AssertionError('7 维描述子应抛 ValueError')


def test_router_fuse_mismatch_raises():
    r = GeometryRouter(n_experts=3, seed=0)
    g = GeometryDescriptor().descriptor(lon=0.0, lat=0.0)
    try:
        r.fuse_weights(g, np.zeros(3), [np.zeros(3)])  # 只有 1 个 delta
    except ValueError:
        return
    raise AssertionError('expert 数与 delta 数不匹配应抛 ValueError')


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