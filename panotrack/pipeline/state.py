# -*- coding: utf-8 -*-
"""球面跟踪状态（GRT-360 Commit 3）：S² 单位球向量 + 旋转速度。

SphericalState 用单位球向量 p = [cos(lat)cos(lon), sin(lat), cos(lat)sin(lon)]
表示目标朝向，用三维旋转向量 ω = axis*angle（弧度/帧）表示角速度，
在 S² 流形上做指数平滑（Rodrigues 旋转插值），彻底规避经纬度表示在
子午线跨界（±179°）与近极点处（+86°→+89°）的切平面退化问题。

predict() 按当前旋转速度外推一步；update(measured) 用观测 BFoV 校正
位置与速度。运动仅作为 soft prior（位置校正仍由观测主导），速度用
(1-va)*damping*ω + va*r_obs 平滑，天然支持 360° 大横移与极区穿越。
"""
import numpy as np

from panotrack.geometry.bfov import BFoV
from panotrack.geometry.sphere import (
    lonlat_to_unit, unit_to_lonlat, normalize, rotate_with_rotvec,
    rotation_vec_between,
)

# 输出纬度钳制：避免在极点处切平面退化（S² 内部表示不受此限，仅输出收敛）
_LAT_LIM = 89.9


class SphericalState:
    """S² 球面目标状态：单位球向量 + 旋转速度 + 指数平滑。

    属性 bfov 为当前后验估计（经纬度/FoV）；predict() 返回按旋转速度外推
    一步的 BFoV；update(measured) 用观测 BFoV 校正位置并更新旋转速度。
    新增 unit_vector / angular_speed_deg / prediction_error_deg 供
    Geometry Descriptor 与 Soft S² Motion Prior 使用。
    """

    def __init__(self, bfov=None, pos_alpha=0.8, vel_alpha=0.4, fov_alpha=0.15,
                 damping=0.9):
        """创建 S² 球面状态。

        参数: bfov 初始 BFoV（可为 None，首次 update 时建立状态，速度置 0）；
              pos_alpha 位置平滑系数（越大越贴近观测）；
              vel_alpha 角速度平滑系数；fov_alpha FoV 平滑系数；
              damping 速度阻尼系数（0~1），每帧对保留速度乘 damping，防漂移。
        返回: None
        """
        self.pos_alpha = float(pos_alpha)
        self.vel_alpha = float(vel_alpha)
        self.fov_alpha = float(fov_alpha)
        self.damping = float(np.clip(damping, 0.0, 1.0))
        self._p = np.array([1.0, 0.0, 0.0])   # 后验单位球向量 (3,)
        self._omega = np.zeros(3)             # 旋转速度向量 (3,) 弧度/帧
        self._fov_h = 1.0
        self._fov_v = 1.0
        self._ready = False
        if bfov is not None:
            self._set(bfov)

    def _set(self, bfov):
        """直接写入状态（私有），速度清零。"""
        x, y, z = lonlat_to_unit(float(bfov.lon), float(bfov.lat))
        self._p = normalize(np.array([x, y, z]))
        self._omega = np.zeros(3)
        self._fov_h = max(float(bfov.fov_h), 1e-3)
        self._fov_v = max(float(bfov.fov_v), 1e-3)
        self._ready = True

    @property
    def bfov(self):
        """当前后验 BFoV（未就绪时抛 RuntimeError）。"""
        if not self._ready:
            raise RuntimeError('SphericalState 未初始化，请先 update(measured)')
        lon, lat = unit_to_lonlat(*self._p)
        return BFoV(lon=lon, lat=float(np.clip(lat, -_LAT_LIM, _LAT_LIM)),
                    fov_h=self._fov_h, fov_v=self._fov_v)

    @property
    def unit_vector(self):
        """当前后验单位球向量 (3,)（未就绪时抛 RuntimeError）。"""
        if not self._ready:
            raise RuntimeError('SphericalState 未初始化，请先 update(measured)')
        return self._p.copy()

    @property
    def angular_speed_deg(self):
        """当前角速度大小（度/帧，未就绪返回 0.0）。"""
        if not self._ready:
            return 0.0
        return float(np.linalg.norm(self._omega) * 180.0 / np.pi)

    def prediction_error_deg(self, measured):
        """预测方向与观测方向间的球面角误差（度，诊断/可靠性用）。

        参数: measured 观测 BFoV。
        返回: float 度，∈ [0, 180]。
        """
        if not self._ready:
            return 0.0
        p_pred = normalize(rotate_with_rotvec(self._p, self._omega))
        x, y, z = lonlat_to_unit(float(measured.lon), float(measured.lat))
        q = normalize(np.array([x, y, z]))
        r = rotation_vec_between(p_pred, q)
        return float(np.linalg.norm(r) * 180.0 / np.pi)

    def _predict_point(self):
        """按当前旋转速度外推一步的球面方向（私有，不修改状态）。"""
        return normalize(rotate_with_rotvec(self._p, self._omega))

    def predict(self):
        """按当前旋转速度外推下一帧 BFoV（不修改内部状态）。

        参数: 无。
        返回: BFoV 预测窗口。
        """
        if not self._ready:
            raise RuntimeError('SphericalState 未初始化，请先 update(measured)')
        p_pred = self._predict_point()
        lon, lat = unit_to_lonlat(*p_pred)
        return BFoV(lon=lon, lat=float(np.clip(lat, -_LAT_LIM, _LAT_LIM)),
                    fov_h=self._fov_h, fov_v=self._fov_v)

    def update(self, measured):
        """用观测 BFoV 校正状态（S² 流形指数平滑 + 旋转速度阻尼）。

        参数: measured 观测到的目标 BFoV。
        返回: None
        说明: r_obs = 后验 -> 观测的最短旋转向量；位置按 pa*r_obs 做
              Rodrigues 旋转插值，旋转速度按 (1-va)*damping*ω + va*r_obs
              平滑更新。motion 只作 soft prior，位置仍由观测主导。
        """
        if not self._ready:
            self._set(measured)
            return
        pa, va, fa = self.pos_alpha, self.vel_alpha, self.fov_alpha
        x, y, z = lonlat_to_unit(float(measured.lon), float(measured.lat))
        q = normalize(np.array([x, y, z]))
        r_obs = rotation_vec_between(self._p, q)   # 弧度
        self._omega = (self._omega * self.damping) * (1.0 - va) + va * r_obs
        self._p = normalize(rotate_with_rotvec(self._p, r_obs * pa))
        self._fov_h = max((1.0 - fa) * self._fov_h + fa * float(measured.fov_h), 1e-3)
        self._fov_v = max((1.0 - fa) * self._fov_v + fa * float(measured.fov_v), 1e-3)