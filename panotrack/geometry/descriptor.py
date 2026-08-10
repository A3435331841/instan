# -*- coding: utf-8 -*-
"""GRT-360 Geometry Descriptor：目标在球面的几何上下文特征向量。

g_t = [sin(|lat|),                                # 近极风险（极点处 →1）
       clip(1/cos(lat), 1, D_MAX),                # 极区 sec 拉伸（纬度畸变）
       seam_distance,                             # 到 ±180° 子午线 seam 的距离
       bbox_area,                                 # 目标角面积占比（归一化）
       log_aspect,                                # 框纵横比对数的绝对值
       angular_speed,                             # S² 角速度（度/帧，归一化）
       motion_uncertainty,                        # 运动先验不确定性（预测误差）
       tracker_confidence]                        # 跟踪器响应峰值

前 5 维为静态几何（目标自身的球面/框形态），后 3 维为动态几何（状态估计
质量与跟踪置信度）。该描述子同时供 Geometry Router（gating）与
ReliabilityGate 的 geometry_risk 计算使用。
"""
import numpy as np

_D_MAX = 8.0          # 极区 sec 拉伸截断上限
_SPEED_NORM = 90.0    # 角速度归一化基准（度/帧）
_ERR_NORM = 90.0      # 运动不确定性归一化基准（度）


class GeometryDescriptor:
    """从目标几何与 S² 状态估计生成 8 维几何描述子。"""

    def descriptor(self, lon, lat, angular_speed_deg=0.0,
                   motion_uncertainty_deg=0.0, tracker_confidence=0.5,
                   bbox_area=0.01, log_aspect=0.0):
        """计算几何描述子。

        参数: lon, lat 目标中心经纬度（度）；angular_speed_deg 角速度（度/帧）；
              motion_uncertainty_deg 运动预测误差（度）；tracker_confidence 响应峰值；
              bbox_area 目标角面积占比（0~1）；log_aspect 框纵横比对数。
        返回: (8,) float 归一化描述子。
        """
        lat = float(np.clip(lat, -89.99, 89.99))
        sec = float(np.clip(1.0 / np.cos(np.deg2rad(lat)), 1.0, _D_MAX))
        seam = float(180.0 - abs(float(np.clip(lon, -180.0, 180.0))))
        return np.array([
            float(np.sin(np.deg2rad(abs(lat)))),          # 近极风险
            sec,                                          # 极区拉伸
            float(np.clip(seam / 180.0, 0.0, 1.0)),       # seam 距离（归一化）
            float(np.clip(bbox_area, 0.0, 1.0)),          # 面积占比
            float(np.clip(log_aspect, -3.0, 3.0)),        # 纵横比对数
            float(np.clip(angular_speed_deg / _SPEED_NORM, 0.0, 1.0)),
            float(np.clip(motion_uncertainty_deg / _ERR_NORM, 0.0, 1.0)),
            float(np.clip(tracker_confidence, 0.0, 1.0)),
        ], dtype=np.float64)

    def risk(self, lon, lat, angular_speed_deg=0.0,
             motion_uncertainty_deg=0.0, tracker_confidence=0.5):
        """几何风险标量 ∈[0,1]，供 ReliabilityGate 惩罚项。

        组合：近极风险 + 极区拉伸 + seam 临近 + 高角速度 + 高运动不确定性，
        且跟踪自信度低时风险更高。
        """
        g = self.descriptor(lon, lat, angular_speed_deg,
                            motion_uncertainty_deg, tracker_confidence)
        sin_lat, sec, seam = g[0], g[1], g[2]
        speed, uncert, conf = g[5], g[6], g[7]
        risk = (0.25 * sin_lat
                + 0.15 * float(np.clip((sec - 1.0) / (_D_MAX - 1.0), 0.0, 1.0))
                + 0.15 * float(np.clip(1.0 - seam, 0.0, 1.0))
                + 0.20 * speed
                + 0.25 * uncert)
        # 置信度低放大约束：低 confidence 使风险整体抬升
        risk += 0.20 * (1.0 - conf)
        return float(np.clip(risk, 0.0, 1.0))