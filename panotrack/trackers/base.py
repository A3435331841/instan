# -*- coding: utf-8 -*-
"""局部透视图单目标跟踪器抽象基类（模块 B 契约）。"""
from abc import ABC, abstractmethod


class BaseTracker(ABC):
    """局部透视图（tangent 切图）上的单目标跟踪器统一接口。

    实现类必须线程无关、无全局状态；init/update 均在局部透视图坐标系下进行，
    不感知 ERP 全景与跨界回绕（由上层 pipeline 负责）。
    """

    @abstractmethod
    def init(self, image, bbox):
        """用首帧局部透视图与目标框初始化跟踪器。

        参数:
            image: np.ndarray，(H, W, 3) uint8 RGB，局部透视图（非全景）。
            bbox: (x, y, w, h) 浮点，局部像素框。
        返回:
            None
        """

    @abstractmethod
    def update(self, image):
        """在新一帧局部透视图上更新目标状态。

        参数:
            image: np.ndarray，(H, W, 3) uint8 RGB，局部透视图。
        返回:
            dict：{'bbox': (x, y, w, h) 浮点局部像素框,
                   'score': float ∈ [0, 1] 峰值置信度,
                   'psr': float 峰值旁瓣比,
                   'apce': float 平均峰值相关能量}
        """
