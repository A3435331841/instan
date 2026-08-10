# -*- coding: utf-8 -*-
"""GRT-360 Commit 6 测试：Spherical Multi-view Reacquisition（redetect_v3）。

运行: python tests/test_redetect_v3.py
覆盖:
  1. 视角数量在 6-12 区间
  2. 视角覆盖整个球面（经度环绕 + 纬度分带，含 seam 跨界视角）
  3. 在 ERP 帧中放置模板目标，能正确找回位置
  4. 无模板 / 空帧返回 None
  5. 模板池多模板时可命中
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector


def _make_frame(H=180, W=360, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 40, (H, W, 3), dtype=np.uint8)
    return frame


def _checker_block(cells=4, size=6, low=120, high=255):
    """结构化棋盘格模板：真实目标通常含低频结构，下采样后仍高度自相关。"""
    n = cells * size
    w = np.fromfunction(
        lambda i, j: ((i // size + j // size) % 2),
        (n, n),
    ).astype(np.float64)
    w = low + (high - low) * w
    return np.stack([w, w, w], axis=-1).astype(np.uint8)


def _place_target(frame, cx, cy, r=12, seed=1):
    """在 ERP 帧 (cx, cy) 处放置结构化棋盘格目标块（cx/cy 为块中心）。"""
    H, W = frame.shape[:2]
    block = _checker_block(cells=4, size=max(3, r // 2))
    n = block.shape[0]
    r = n // 2
    for dy in range(n):
        for dx in range(n):
            x = int((cx - r + dx) % W)
            y = int(cy - r + dy)
            if 0 <= y < H:
                frame[y, x] = block[dy, dx]
    return frame, block, (n, n)


def test_view_count_in_range():
    d = SphericalMultiViewRedetector(get_templates=lambda: [])
    assert 6 <= d.n_views <= 12


def test_view_centers_cover_sphere():
    d = SphericalMultiViewRedetector(get_templates=lambda: [])
    centers = d.view_centers()
    lons = [c[0] for c in centers]
    lats = [c[1] for c in centers]
    # 经度环绕覆盖：跨度接近 360°
    assert max(lons) - min(lons) >= 270.0
    # 纬度分带覆盖三带
    assert set(round(l) for l in lats) == {-45, 0, 45}


def test_view_centers_unique():
    d = SphericalMultiViewRedetector(get_templates=lambda: [])
    centers = d.view_centers()
    assert len(set(centers)) == len(centers)


def test_search_finds_placed_target():
    W, H = 360, 180
    frame = _make_frame(H, W)
    frame, block, (bw, bh) = _place_target(frame, cx=90, cy=60)
    tpl = (block, (bw, bh))
    d = SphericalMultiViewRedetector(get_templates=lambda: [tpl],
                                     min_score=0.3)
    res = d.search(frame, erp_downscale=2)
    assert res is not None, '应能找回目标'
    (x, y, w, h), score = res
    # 目标中心应接近放置位置 (90, 60)
    cx_found = x + w / 2.0
    cy_found = y + h / 2.0
    assert abs(cx_found - 90) < 20, cx_found
    assert abs(cy_found - 60) < 20, cy_found


def test_search_none_no_templates():
    d = SphericalMultiViewRedetector(get_templates=lambda: [])
    assert d.search(_make_frame()) is None


def test_search_flat_frame_returns_none_or_low():
    # 平坦帧无目标，任何命中分应低于 min_score -> None
    frame = np.full((180, 360, 3), 10, dtype=np.uint8)
    tpl = (np.full((20, 20, 3), 200, dtype=np.uint8), (20, 20))
    d = SphericalMultiViewRedetector(get_templates=lambda: [tpl],
                                     min_score=0.5)
    assert d.search(frame) is None


def test_search_multi_template_pool():
    W, H = 360, 180
    frame = _make_frame(H, W)
    frame, block, (bw, bh) = _place_target(frame, cx=200, cy=90)
    tpl = (block, (bw, bh))
    # 多模板池：一个无关模板 + 目标模板，应命中目标模板
    rng = np.random.default_rng(9)
    noise_tpl = (rng.integers(0, 255, (20, 20, 3), dtype=np.uint8), (20, 20))
    d = SphericalMultiViewRedetector(get_templates=lambda: [noise_tpl, tpl],
                                     min_score=0.3)
    res = d.search(frame, erp_downscale=2)
    assert res is not None


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