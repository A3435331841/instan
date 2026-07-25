# -*- coding: utf-8 -*-
"""球面跟踪状态：恒定角速度 + 指数平滑（模块 E）。

SphericalState 维护目标在球面上的 (lon, lat, fov_h, fov_v) 后验估计，
predict() 按恒定角速度外推下一帧窗口，update(measured) 用观测 BFoV
做指数平滑校正并在线估计角速度。经度差分一律走 delta_lon 环绕差分，
跨界（±180°）运动不会产生速度冲击。
"""
import numpy as np

from panotrack.geometry.bfov import BFoV
from panotrack.geometry.sphere import wrap_lon, delta_lon

_LAT_LIM = 89.9  # 纬度钳制，避免恰好在极点处切平面退化


class SphericalState:
    """球面目标状态：恒定角速度 + 指数平滑 + 阻尼。

    属性 bfov 为当前后验估计；predict() 返回外推一步的 BFoV；
    update(measured) 用观测 BFoV 校正位置/FoV 并更新角速度。
    """

    def __init__(self, bfov=None, pos_alpha=0.8, vel_alpha=0.4, fov_alpha=0.5,
                 damping=0.9):
        """创建球面状态。

        参数: bfov 初始 BFoV（可为 None，首次 update 时建立状态，速度置 0）；
              pos_alpha 位置平滑系数（越大越贴近观测）；
              vel_alpha 角速度平滑系数；fov_alpha FoV 平滑系数；
              damping 速度阻尼系数（0~1），每帧速度乘以 damping，防止漂移累积。
        返回: None
        """
        self.pos_alpha = float(pos_alpha)
        self.vel_alpha = float(vel_alpha)
        self.fov_alpha = float(fov_alpha)
        self.damping = float(np.clip(damping, 0.0, 1.0))
        self._lon = 0.0
        self._lat = 0.0
        self._fov_h = 1.0
        self._fov_v = 1.0
        self._vlon = 0.0  # 角速度（度/帧）
        self._vlat = 0.0
        self._ready = False
        if bfov is not None:
            self._set(bfov)

    def _set(self, bfov):
        """直接写入状态（私有），速度清零。"""
        self._lon = wrap_lon(float(bfov.lon))
        self._lat = float(np.clip(bfov.lat, -_LAT_LIM, _LAT_LIM))
        self._fov_h = max(float(bfov.fov_h), 1e-3)
        self._fov_v = max(float(bfov.fov_v), 1e-3)
        self._vlon = 0.0
        self._vlat = 0.0
        self._ready = True

    @property
    def bfov(self):
        """当前后验 BFoV（未就绪时抛 RuntimeError）。"""
        if not self._ready:
            raise RuntimeError('SphericalState 未初始化，请先 update(measured)')
        return BFoV(lon=self._lon, lat=self._lat,
                    fov_h=self._fov_h, fov_v=self._fov_v)

    def predict(self):
        """按恒定角速度外推下一帧 BFoV（lon 自动 wrap，lat clamp，速度阻尼）。

        参数: 无。
        返回: BFoV 预测窗口。
        """
        if not self._ready:
            raise RuntimeError('SphericalState 未初始化，请先 update(measured)')
        # 应用阻尼，防止漂移累积
        vlon = self._vlon * self.damping
        vlat = self._vlat * self.damping
        lon = wrap_lon(self._lon + vlon)
        lat = float(np.clip(self._lat + vlat, -_LAT_LIM, _LAT_LIM))
        return BFoV(lon=lon, lat=lat, fov_h=self._fov_h, fov_v=self._fov_v)

    def update(self, measured):
        """用观测 BFoV 校正状态（环绕差分 + 指数平滑 + 阻尼）。

        参数: measured 观测到的目标 BFoV。
        返回: None
        """
        if not self._ready:
            self._set(measured)
            return
        pa, va, fa = self.pos_alpha, self.vel_alpha, self.fov_alpha
        dlon = delta_lon(float(measured.lon) - self._lon)   # 环绕差分
        dlat = float(measured.lat) - self._lat
        # 角速度 = 后验间位移的指数平滑，并应用阻尼
        self._vlon = (self._vlon * self.damping) * (1.0 - va) + va * dlon
        self._vlat = (self._vlat * self.damping) * (1.0 - va) + va * dlat
        self._lon = wrap_lon(self._lon + pa * dlon)
        self._lat = float(np.clip(self._lat + pa * dlat, -_LAT_LIM, _LAT_LIM))
        self._fov_h = max((1.0 - fa) * self._fov_h + fa * float(measured.fov_h), 1e-3)
        self._fov_v = max((1.0 - fa) * self._fov_v + fa * float(measured.fov_v), 1e-3)
