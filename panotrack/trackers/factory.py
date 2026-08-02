# -*- coding: utf-8 -*-
"""跟踪器工厂：按名称创建 BaseTracker 实例。

支持的跟踪器:
  - 'ncc': 经典 FFT-NCC 模板匹配
  - 'ncc_v2': 增强 NCC(多尺度+自适应搜索)
  - 'vittrack_cv2': VitTrack via cv2.TrackerVit (本地Windows验证用,非部署级)
  - 'vittrack_onnx': VitTrack via onnxruntime (生产部署,需docker linux)
  - 'direct_erp': 直接ERP跟踪，绕过BFoV框架 (推荐用于小目标和漂移敏感场景)
  - 'lightfc_onnx': LightFC via onnxruntime 双子图 (生产部署,CPU 实时)
  - 'lightfc_cpu': LightFC via torch CPU (本地验证用)
"""
from .base import BaseTracker
from .ncc import NCCTracker
from .ncc_v2 import NCCTrackerV2
from .vittrack_onnx import VitTrackONNX
from .direct_erp import DirectERPTracker


def create_tracker(name='ncc', **kwargs):
    """按名称创建局部透视图单目标跟踪器。"""
    key = (name or '').lower()
    if key == 'ncc':
        return NCCTracker(**kwargs)
    if key == 'ncc_v2':
        v2_kwargs = dict(kwargs)
        if 'search_scale' in v2_kwargs:
            v2_kwargs['search_scale_init'] = v2_kwargs.pop('search_scale')
        return NCCTrackerV2(**v2_kwargs)
    if key == 'vittrack_onnx':
        try:
            return VitTrackONNX(**kwargs)
        except Exception as e:
            print(f'[Factory] vittrack_onnx init failed, fallback to ncc_v2: {e}',
                  file=__import__('sys').stderr)
            return NCCTrackerV2(**kwargs)
    if key == 'vittrack_cv2':
        # 仅在 Windows 上有 cv2 可用时才能工作;docker linux 无 cv2 会回退
        try:
            from .__local_vittrack_cv2 import VitTrackCV2Tracker
            return VitTrackCV2Tracker(**kwargs)
        except ImportError as e:
            print(f'[Factory] vittrack_cv2 unavailable ({e}), fallback to ncc_v2',
                  file=__import__('sys').stderr)
            return NCCTrackerV2(**kwargs)
    if key == 'direct_erp':
        # 直接 ERP 跟踪，绕过 BFoV 框架，避免漂移
        return DirectERPTracker(**kwargs)
    if key == 'lightfc_onnx':
        # LightFC via onnxruntime(双子图):生产部署,CPU 实时,无 torch 依赖
        try:
            from .lightfc_onnx import LightFCONNX
            return LightFCONNX(**kwargs)
        except Exception as e:
            print(f'[Factory] lightfc_onnx init failed, fallback to ncc_v2: {e}',
                  file=__import__('sys').stderr)
            return NCCTrackerV2(**kwargs)
    if key == 'lightfc_cpu':
        # LightFC via torch CPU:本地验证用
        try:
            from .lightfc_cpu import LightFCTracker
            return LightFCTracker(**kwargs)
        except Exception as e:
            print(f'[Factory] lightfc_cpu init failed, fallback to ncc_v2: {e}',
                  file=__import__('sys').stderr)
            return NCCTrackerV2(**kwargs)
    if key == 'lightfc':
        raise NotImplementedError(
            "请用 'lightfc_onnx'(onnxruntime 生产)或 'lightfc_cpu'(torch 本地)")
    raise ValueError(
        f"未知跟踪器名称: {name!r}，当前可用: 'ncc', 'ncc_v2', 'vittrack_onnx', "
        f"'vittrack_cv2', 'direct_erp', 'lightfc_onnx', 'lightfc_cpu'"
    )
