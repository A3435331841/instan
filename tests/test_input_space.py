# -*- coding: utf-8 -*-
"""GRT-360 Commit 2 测试：input-space capability + guard。

运行: python tests/test_input_space.py
覆盖:
  1. 局部切图跟踪器声明的 input_space == 'local_patch'
  2. 全帧跟踪器声明的 input_space == 'erp_full'
  3. get_tracker_input_space 对未知名称返回 None
  4. PanoTracker 拒绝 'erp_full' 跟踪器（抛 ValueError）
  5. PanoTracker 接受 'local_patch' 跟踪器（不抛）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.trackers.factory import get_tracker_input_space
from panotrack.trackers.base import BaseTracker
from panotrack.pipeline.pipeline import PanoTracker

# 局部切图跟踪器（可被 PanoTracker 使用）
_LOCAL = ('ncc', 'ncc_v2', 'vittrack_onnx', 'vittrack_cv2')
# 全帧 ERP 跟踪器（必须走 full-frame runner）
_ERP = ('direct_erp', 'lightfc_onnx', 'lightfc_cpu')


def test_base_default_local_patch():
    """BaseTracker 默认 input_space == 'local_patch'（向后兼容）。"""
    assert BaseTracker.input_space == 'local_patch'


def test_local_trackers_local_patch():
    for name in _LOCAL:
        assert get_tracker_input_space(name) == 'local_patch', name


def test_erp_trackers_erp_full():
    for name in _ERP:
        assert get_tracker_input_space(name) == 'erp_full', name


def test_unknown_returns_none():
    assert get_tracker_input_space('does_not_exist') is None
    assert get_tracker_input_space('') is None
    assert get_tracker_input_space(None) is None


def test_case_insensitive():
    assert get_tracker_input_space('NCC') == 'local_patch'
    assert get_tracker_input_space('LightFC_ONNX') == 'erp_full'


def test_pano_tracker_rejects_erp_full():
    """PanoTracker 用 erp_full 跟踪器初始化应抛 ValueError。"""
    for name in _ERP:
        try:
            PanoTracker({'tracker': name}).init(
                _synthetic_frame(), (0, 0, 50, 50))
            raise AssertionError(f'{name} 应被拒绝')
        except ValueError:
            pass  # 期望


def test_pano_tracker_accepts_local():
    """PanoTracker 用 local_patch 跟踪器初始化不应抛 input-space 错误。"""
    pt = PanoTracker({'tracker': 'ncc'})
    try:
        pt.init(_synthetic_frame(), (0, 0, 50, 50))
    except AssertionError:
        import numpy as np
        # 合成帧可能因图案太弱触发 NCC 内部断言，输入空间守卫本身应通过
        pass


def _synthetic_frame():
    import numpy as np
    h, w = 256, 512
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (xx * 3 + yy) % 256
    arr[..., 1] = (xx + yy * 5) % 256
    arr[..., 2] = (yy * 7) % 256
    return arr


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