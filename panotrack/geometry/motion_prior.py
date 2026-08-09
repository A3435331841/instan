# -*- coding: utf-8 -*-
"""GRT-360 Soft S² Motion Prior：用球面运动预测一致性软调制跟踪得分。

核心公式（log-space 更稳）：
    score_final = score_visual * exp(-λ * d_sphere² / σ²)

其中 d_sphere 为 S² 单位球向量表示的预测方向与观测方向之间的球面角距离
（度），σ 随目标角速度自适应（快目标允许更大偏移），λ 控制先验强度。

与硬 gate（直接丢弃不符运动的高响应）不同，Soft prior 只按预测一致性把
得分打一个光滑的折扣：慢目标若观测漂移过大，得分被显著压低（置信判定
随之下降，触发 FoV 扩大重试），而极快速目标（如 360° 大横移）仍保留
较高的响应权重，不会因短时运动不确定性被误杀。

该模块为纯 numpy 实现，供 PanoTracker 在 feature-flag 'motion_prior' 开启时
集成，也可独立用于可靠性门控的 c_motion 分量。
"""
import numpy as np

from panotrack.geometry.sphere import normalize, rotation_vec_between


class SoftS2MotionPrior:
    """Soft S² motion prior：按预测-观测球面角距离软调制视觉得分。"""

    def __init__(self, lambda_=1.0, sigma_base=15.0, sigma_per_speed=0.5,
                 d_min=0.0):
        """创建 soft motion prior。

        参数: lambda_ 先验强度（越大折扣越强）；sigma_base 静止目标的角距离
              容忍基（度）；sigma_per_speed 每 (度/帧) 角速度对 σ 的增量；
              d_min 低于该角距离（度）不打折（数值保护）。
        返回: None
        """
        self.lambda_ = float(lambda_)
        self.sigma_base = float(sigma_base)
        self.sigma_per_speed = float(sigma_per_speed)
        self.d_min = float(d_min)

    @staticmethod
    def angular_distance_deg(pred_vec, obs_vec):
        """两单位球向量间的球面角距离（度，∈[0,180]）。"""
        p = normalize(np.asarray(pred_vec, dtype=np.float64))
        q = normalize(np.asarray(obs_vec, dtype=np.float64))
        r = rotation_vec_between(p, q)
        return float(np.linalg.norm(r) * 180.0 / np.pi)

    def __call__(self, visual_score, pred_vec, obs_vec, angular_speed_deg=0.0):
        """软调制视觉得分。

        参数: visual_score 视觉跟踪得分（0~1）；pred_vec 预测单位球方向 (3,)；
              obs_vec 观测单位球方向 (3,)；angular_speed_deg 当前角速度（度/帧）。
        返回: (modulated_score, d_deg) —— 调制后得分与原角距离。
        """
        d = self.angular_distance_deg(pred_vec, obs_vec)
        if d <= self.d_min:
            return float(visual_score), d
        sigma = self.sigma_base + self.sigma_per_speed * float(angular_speed_deg)
        sigma = max(sigma, 1e-3)
        gauss = float(np.exp(-self.lambda_ * (d * d) / (2.0 * sigma * sigma)))
        return float(visual_score) * gauss, d

    def score(self, visual_score, pred_vec, obs_vec, angular_speed_deg=0.0):
        """同 __call__，仅返回调制后得分（便捷）。"""
        return self.__call__(visual_score, pred_vec, obs_vec,
                             angular_speed_deg)[0]