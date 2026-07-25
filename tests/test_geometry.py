# -*- coding: utf-8 -*-
"""panotrack.geometry 单元测试（纯 assert，用 python tests/test_geometry.py 运行）。

覆盖：球面工具、BFoV 互转、tangent 正逆投影往返（中心偏差 < 2 像素）、
跨界（lon≈±179°）、极点（lat≈85°）、纯 numpy 双线性 remap、RemapCache。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panotrack.geometry.sphere import wrap_lon, delta_lon, lonlat_to_unit, unit_to_lonlat
from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov
from panotrack.geometry.projection import (
    tangent_remap, remap_image, local_bbox_to_erp, RemapCache,
)

W, H = 1024, 512  # 测试用全景尺寸


def _center_x_err(x1, w1, x2, w2, width):
    """两个 ERP 框中心的水平偏差（跨界感知，结果 ∈ [0, W/2]）。"""
    d = abs((x1 + w1 / 2.0) - (x2 + w2 / 2.0)) % width
    return min(d, width - d)


def test_sphere():
    # wrap_lon：归一化到 (-180, 180]
    assert wrap_lon(0.0) == 0.0
    assert wrap_lon(180.0) == 180.0
    assert wrap_lon(-180.0) == 180.0
    assert abs(wrap_lon(181.0) - (-179.0)) < 1e-9
    assert abs(wrap_lon(-181.0) - 179.0) < 1e-9
    assert abs(wrap_lon(540.0) - 180.0) < 1e-9
    assert abs(wrap_lon(-540.0) - 180.0) < 1e-9
    arr = wrap_lon(np.array([179.0, 180.5, -270.0, 720.0]))
    assert np.allclose(arr, [179.0, -179.5, 90.0, 0.0])
    # delta_lon：环绕差分
    assert abs(delta_lon(350.0) - (-10.0)) < 1e-9
    assert abs(delta_lon(-350.0) - 10.0) < 1e-9
    assert abs(delta_lon(10.0) - 10.0) < 1e-9
    assert np.allclose(delta_lon(np.array([190.0, -190.0])), [-170.0, 170.0])
    # lonlat_to_unit：基本方向
    x, y, z = lonlat_to_unit(0.0, 0.0)
    assert np.allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-12)
    x, y, z = lonlat_to_unit(0.0, 90.0)
    assert np.allclose([x, y, z], [0.0, 1.0, 0.0], atol=1e-12)
    # 往返：随机点 + 特殊点（极点、±180 经线附近）
    rng = np.random.default_rng(0)
    lons = np.concatenate([rng.uniform(-180, 180, 200), [179.9, -179.9, 180.0, 0.0]])
    lats = np.concatenate([rng.uniform(-89, 89, 200), [85.0, -85.0, 90.0, -90.0]])
    vx, vy, vz = lonlat_to_unit(lons, lats)
    lon2, lat2 = unit_to_lonlat(vx, vy, vz)
    assert np.allclose(np.linalg.norm(np.stack([vx, vy, vz]), axis=0), 1.0, atol=1e-12)
    dlat = np.abs(lat2 - lats)
    dlon = np.abs((np.asarray(lon2) - lons + 180.0) % 360.0 - 180.0)
    assert np.max(dlat) < 1e-9
    assert np.max(dlon[np.abs(lats) < 89.0]) < 1e-9  # 极点处经度不定，跳过
    print("[OK] sphere")


def test_bfov_roundtrip():
    # 中心与视场角：赤道普通框
    b = bfov_from_erp_bbox(400, 200, 80, 60, W, H)
    assert isinstance(b, BFoV) and b.rotation == 0.0
    exp_lon = 440.0 / W * 360.0 - 180.0
    exp_lat = 90.0 - 230.0 / H * 180.0
    assert abs(b.lon - exp_lon) < 1e-6
    assert abs(b.lat - exp_lat) < 1e-6
    assert 24.0 < b.fov_h < 32.0 and 18.0 < b.fov_v < 24.0
    # 跨界框：中心经度应 wrap 到 ≈+178.6
    bc = bfov_from_erp_bbox(996, 200, 48, 32, W, H)
    assert abs(bc.lon - 178.59375) < 1e-6
    assert 14.0 < bc.fov_h < 20.0
    # BFoV -> ERP 框：三种场景中心偏差 < 2 像素
    cases = [
        ("equator", (400, 200, 80, 60)),
        ("crossing", (996, 200, 48, 32)),
        ("pole", (400, 2, 96, 24)),  # 中心 lat≈85.1
    ]
    for name, (x, y, w, h) in cases:
        bfov = bfov_from_erp_bbox(x, y, w, h, W, H)
        x2, y2, w2, h2 = erp_bbox_from_bfov(bfov, W, H)
        ex = _center_x_err(x, w, x2, w2, W)
        ey = abs((y + h / 2.0) - (y2 + h2 / 2.0))
        assert ex < 2.0, f"{name}: x 中心偏差 {ex:.3f}px"
        assert ey < 2.0, f"{name}: y 中心偏差 {ey:.3f}px"
        assert 0.0 <= x2 < W and w2 > 0 and h2 > 0
    # 跨界框 x+w 允许超 W
    bfov = bfov_from_erp_bbox(996, 200, 48, 32, W, H)
    x2, y2, w2, h2 = erp_bbox_from_bfov(bfov, W, H)
    assert x2 + w2 > x2  # 框有效
    print("[OK] bfov roundtrip")


def test_tangent_remap_basic():
    bfov = BFoV(lon=-25.3125, lat=9.140625, fov_h=30.0, fov_v=22.0)
    mx, my = tangent_remap(bfov, 128, 96, W, H)
    assert mx.shape == (96, 128) and my.shape == (96, 128)
    assert mx.dtype == np.float32 and my.dtype == np.float32
    assert mx.min() >= 0.0 and mx.max() < W
    assert my.min() >= 0.0 and my.max() <= H - 1
    # 切图中心应对准 BFoV 中心（像素索引坐标 = 连续坐标 - 0.5）
    cu = ((-25.3125 + 180.0) / 360.0 * W - 0.5) % W
    cv = (90.0 - 9.140625) / 180.0 * H - 0.5
    cx_val = float(mx[48, 63] + mx[48, 64]) / 2  # 中心四像素均值
    cy_val = float(my[47, 64] + my[48, 64]) / 2
    dx = abs(cx_val - cu) % W
    assert min(dx, W - dx) < 0.5
    assert abs(cy_val - cv) < 0.5
    # fov > 90° 走 eBFoV 分支：有限且不炸
    big = BFoV(lon=0.0, lat=0.0, fov_h=120.0, fov_v=100.0)
    mx2, my2 = tangent_remap(big, 64, 64, W, H)
    assert np.isfinite(mx2).all() and np.isfinite(my2).all()
    assert mx2.min() >= 0.0 and mx2.max() < W
    # rotation 保留参数：传入不报错
    mx3, _ = tangent_remap(BFoV(0.0, 0.0, 60.0, 40.0, rotation=15.0), 32, 32, W, H)
    assert mx3.shape == (32, 32)
    print("[OK] tangent_remap basic")


def test_full_roundtrip():
    """ERP 框 -> BFoV -> 切图 -> 局部框 -> 逆投影回 ERP：中心偏差 < 2 像素。"""
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)  # 合成 ERP 图
    cases = [
        ("equator", (400, 200, 80, 60)),
        ("crossing", (996, 200, 48, 32)),       # 中心 lon≈+178.6
        ("crossing2", (0, 300, 30, 40)),        # 中心 lon≈-174.7，跨左边界方向
        ("pole", (400, 2, 96, 24)),             # 中心 lat≈85.1
        ("pole_cross", (990, 4, 60, 24)),       # 极点 + 跨界组合
    ]
    # 随机模糊用例（避开极点奇异区）
    for _ in range(20):
        w = float(rng.uniform(20, 120))
        h = float(rng.uniform(15, 80))
        x = float(rng.uniform(0, W - 1))
        y = float(rng.uniform(H * 0.15, H * 0.85 - h))
        cases.append(("fuzz", (x, y, w, h)))
    for name, (x, y, w, h) in cases:
        bfov = bfov_from_erp_bbox(x, y, w, h, W, H)
        mx, my = tangent_remap(bfov, 128, 128, W, H)
        patch = remap_image(img, mx, my)  # 切图
        assert patch.shape == (128, 128, 3) and patch.dtype == np.uint8
        # 局部框居中放置（模拟跟踪器输出），逆投影回 ERP
        lw = lh = 40.0
        lx = ly = (128.0 - lw) / 2.0
        x2, y2, w2, h2 = local_bbox_to_erp(lx, ly, lw, lh, mx, my, W, H)
        ex = _center_x_err(x, w, x2, w2, W)
        ey = abs((y + h / 2.0) - (y2 + h2 / 2.0))
        assert ex < 2.0, f"{name} {(x, y, w, h)}: x 中心偏差 {ex:.3f}px"
        assert ey < 2.0, f"{name} {(x, y, w, h)}: y 中心偏差 {ey:.3f}px"
        assert 0.0 <= x2 < W and w2 > 0 and h2 > 0
    print("[OK] full roundtrip (center err < 2px, incl. crossing/pole)")


def test_remap_image():
    img = np.arange(W * H * 3, dtype=np.uint8).reshape(H, W, 3)
    # 恒等映射：输出与输入完全一致
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    out = remap_image(img, gx, gy)
    assert out.dtype == np.uint8 and np.array_equal(out, img)
    # 右移 1 像素 + 水平回绕：最后一列取第一列
    out = remap_image(img, gx + 1.0, gy)
    assert np.array_equal(out[:, :-1], img[:, 1:])
    assert np.array_equal(out[:, -1], img[:, 0])
    # 垂直 clamp：my = H-1 全取最后一行
    out = remap_image(img, gx, np.full_like(gy, H - 1.0))
    assert np.array_equal(out, np.broadcast_to(img[-1:], (H, W, 3)))
    # 双线性插值：手工可算的小图
    small = np.array([[0, 10], [20, 30]], dtype=np.uint8)
    val = remap_image(small, np.array([[0.5]]), np.array([[0.5]]))
    assert val.shape == (1, 1) and val[0, 0] == 15  # 0.25*(0+10+20+30)
    val = remap_image(small, np.array([[0.3]]), np.array([[0.0]]))
    assert val[0, 0] == 3  # 0.7*0 + 0.3*10 = 3
    # 灰度 (H,W) 支持
    gray = np.arange(16, dtype=np.uint8).reshape(4, 4)
    gout = remap_image(gray, np.array([[1.5, 0.0]]), np.array([[1.5, 3.0]]))
    assert gout.shape == (1, 2) and gout.dtype == np.uint8
    assert gout[0, 0] == 8  # round(0.25*(5+6+9+10)) = round(7.5) = 8
    assert gout[0, 1] == 12
    print("[OK] remap_image")


def test_remap_cache():
    cache = RemapCache()  # 默认容量 64
    assert cache.capacity == 64
    b1 = BFoV(lon=10.2, lat=5.1, fov_h=30.0, fov_v=20.0)
    b2 = BFoV(lon=10.9, lat=5.9, fov_h=30.9, fov_v=20.9)  # 同一量化桶
    b3 = BFoV(lon=40.0, lat=5.1, fov_h=30.0, fov_v=20.0)  # 不同桶
    m1 = cache.get_remap(b1, 128, 128, W, H)
    m2 = cache.get_remap(b2, 128, 128, W, H)  # 应命中：结果与 b1 完全一致
    assert np.array_equal(m1[0], m2[0]) and np.array_equal(m1[1], m2[1])
    ref = tangent_remap(b1, 128, 128, W, H)
    assert np.array_equal(m1[0], ref[0]) and np.array_equal(m1[1], ref[1])
    m3 = cache.get_remap(b3, 128, 128, W, H)  # 不同桶：结果应不同
    assert not np.array_equal(m1[0], m3[0])
    # LRU 淘汰：容量 2 写入 3 个键，最旧的被淘汰
    small = RemapCache(capacity=2)
    small.get_remap(BFoV(0.0, 0.0, 30.0, 20.0), 64, 64, W, H)
    small.get_remap(BFoV(20.0, 0.0, 30.0, 20.0), 64, 64, W, H)
    small.get_remap(BFoV(40.0, 0.0, 30.0, 20.0), 64, 64, W, H)
    assert len(small._cache) == 2
    # 命中副本隔离：修改返回值不污染缓存
    m1[0][:] = 0.0
    m1b = cache.get_remap(b1, 128, 128, W, H)
    assert np.array_equal(m1b[0], ref[0])
    print("[OK] RemapCache")


def test_robustness():
    # 跨界切图：中心 lon≈±179 时 map 与切图均正常
    img = np.random.default_rng(1).integers(0, 256, size=(H, W, 3), dtype=np.uint8)
    for lon in (179.0, -179.0, 180.0):
        bfov = BFoV(lon=lon, lat=10.0, fov_h=40.0, fov_v=30.0)
        mx, my = tangent_remap(bfov, 128, 128, W, H)
        assert np.isfinite(mx).all() and np.isfinite(my).all()
        patch = remap_image(img, mx, my)
        assert patch.shape == (128, 128, 3)
        # 跨界时 map_x 应同时出现接近 0 与接近 W 的值（接缝穿过切图）
        assert mx.max() > W - 20 and mx.min() < 20
        # 居中局部框逆投影不炸
        x, y, w, h = local_bbox_to_erp(44, 44, 40, 40, mx, my, W, H)
        assert 0.0 <= x < W and w > 0 and h > 0
        assert _center_x_err(x, w, (lon + 180.0) / 360.0 * W % W, 0.0, W) < 2.0
    # 极点切图：lat=85 正常输出，顶部 clamp 生效
    bfov = BFoV(lon=0.0, lat=85.0, fov_h=60.0, fov_v=30.0)
    mx, my = tangent_remap(bfov, 128, 128, W, H)
    assert np.isfinite(mx).all() and np.isfinite(my).all()
    assert my.max() <= H - 1 and my.min() >= 0.0
    patch = remap_image(img, mx, my)
    assert patch.shape == (128, 128, 3) and patch.dtype == np.uint8
    # 中心恰在极点（lat=90）也不炸
    mx, my = tangent_remap(BFoV(lon=0.0, lat=90.0, fov_h=40.0, fov_v=40.0), 64, 64, W, H)
    assert np.isfinite(mx).all() and np.isfinite(my).all()
    print("[OK] robustness (crossing/pole)")


if __name__ == "__main__":
    test_sphere()
    test_bfov_roundtrip()
    test_tangent_remap_basic()
    test_full_roundtrip()
    test_remap_image()
    test_remap_cache()
    test_robustness()
    print("ALL TESTS PASSED")
