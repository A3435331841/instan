# -*- coding: utf-8 -*-
"""ODTrack + 可靠性门控 + 球面重捕获 wrapper（Step 3，决赛方案）。

一句话：ODTrack 跟丢之后会一路错到底——时序记忆被污染后没有任何自愈
机制（0047 那条 2334 帧的序列就只对了 39 帧）。这个 wrapper 就是给它
补上"判丢 + 全图找回"的能力。组件（可靠性门控、模板记忆、多视角重检测）
都是之前做 PanoTracker 时留下的，这里只是接到 ODTrack 上。

`OdtrackRecaptureTracker` 在 ODTrack 上游 tracker 外加一层系统级状态机，
不修改上游任何代码：

  NORMAL: 逐帧 ODTrack track -> ReliabilityGate 判可靠
      - 高可靠: 输出框 + TemplateMemory 门控写入（anchor 永不被覆盖）
      - 低可靠连续 run_len 帧: 进入 LOST
  LOST:   冻结模板更新；每 search_interval 帧执行一次
      SphericalMultiViewRedetector 全局搜索
      - 命中且通过 VERIFY（anchor 一致 + 分数门槛 + 运动合理）:
          用候选框重新 initialize ODTrack（清空被污染的记忆）-> OBSERVE
      - 未命中/未过 VERIFY: 保持 LOST（宁可丢，不可锁错）
  OBSERVE: 连续 observe_frames 帧保持高可靠 -> NORMAL；否则回 LOST

判丢信号（运行时可得，无真值）：
  - C_visual: ODTrack IoU-head 响应图峰值（tracker.last_pred_iou，若存在）
  - C_anchor: 预测框 crop 与 anchor 模板的 NCC 相似度（外观强证据）
  - C_motion: 预测中心与圆周恒速外推的偏差（seam 感知）
  - C_scale:  框面积 log 变化（相对滑动 EMA）
  - geometry_risk: 极区 |lat|>55° / seam 距离 <12%W（causal_dtp 公式）

VERIFY 的 anchor 强校验是防误锁的关键——丢着不找只是丢分，锁错对象
会让后面全错，我们不想在答辩时解释"为什么跟了一个长得像的"。

接口对齐 BaseTracker：init(image, bbox) / update(image) -> dict。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from panotrack.pipeline.memory import ReliabilityGate, TemplateMemory
from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector


def _center(box: Sequence[float], width: float) -> float:
    return (float(box[0]) + 0.5 * float(box[2])) % float(width)


def _circ_delta(a: float, b: float, width: float) -> float:
    return ((float(a) - float(b) + width / 2.0) % width) - width / 2.0


def _circ_dist(a: float, b: float, width: float) -> float:
    return abs(_circ_delta(a, b, width))


def _spherical_angle_deg(lon1, lat1, lon2, lat2):
    """两经纬度间的球面角距（度，Haversine 形式，高纬/极点正确）。"""
    lon1, lat1 = np.deg2rad(lon1), np.deg2rad(lat1)
    lon2, lat2 = np.deg2rad(lon2), np.deg2rad(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * np.degrees(np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))))


def _geometry_risk(box: Sequence[float], width: float, height: float) -> float:
    """极区/接缝几何风险（与 causal_dtp._geometry_risk 一致，0~1）。"""
    cy = float(box[1]) + 0.5 * float(box[3])
    lat = abs(90.0 - 180.0 * np.clip(cy, 0.0, height) / height)
    pole = np.clip((lat - 55.0) / 35.0, 0.0, 1.0)
    cx = _center(box, width)
    seam_dist = min(cx, width - cx)
    seam = 1.0 - np.clip(seam_dist / max(1.0, 0.12 * width), 0.0, 1.0)
    aspect = abs(np.log(max(float(box[2]), 1.0) / max(float(box[3]), 1.0)))
    return float(np.clip(0.50 * pole + 0.35 * seam
                         + 0.15 * np.clip(aspect / 4.0, 0.0, 1.0), 0.0, 1.0))


def _template_similarity(tpl_a, tpl_b, size=32):
    """两个 (img, (w,h)) 模板的 NCC 相似度（与 memory.py 一致，避免循环导入）。"""
    from PIL import Image

    def gray(arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            return (arr[..., 0].astype(np.float32) * 0.299
                    + arr[..., 1].astype(np.float32) * 0.587
                    + arr[..., 2].astype(np.float32) * 0.114)
        return np.ascontiguousarray(arr, dtype=np.float32)

    a = np.asarray(Image.fromarray(np.clip(gray(tpl_a[0]), 0, 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR), dtype=np.float64)
    b = np.asarray(Image.fromarray(np.clip(gray(tpl_b[0]), 0, 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR), dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _crop_wrap(frame: np.ndarray, box: Sequence[float]) -> np.ndarray:
    """按跨界约定裁剪框区域（x 回绕），返回 (H,W,3) uint8。"""
    H, W = frame.shape[:2]
    x, y, w, h = (float(v) for v in box)
    iw, ih = max(2, int(round(w))), max(2, int(round(h)))
    ix, iy = int(round(x)), int(round(y))
    iy = int(np.clip(iy, 0, max(0, H - ih)))
    cols = np.mod(ix + np.arange(iw), W)
    return np.ascontiguousarray(frame[iy:iy + ih][:, cols])


class OdtrackRecaptureTracker:
    """ODTrack 外挂：可靠性门控 + 球面重捕获（不修改上游代码）。"""

    STATUS_NORMAL = 'ok'
    STATUS_LOST = 'lost'
    STATUS_OBSERVE = 'recovered'

    def __init__(self, odtrack_tracker, gate=None, memory=None,
                 redetector=None, run_len=5, search_interval=5,
                 observe_frames=3, anchor_min_sim=0.5, recapture_min_score=0.45,
                 motion_max_deg=90.0, w_motion=1.0, w_scale=1.0, w_ncc=0.5,
                 w_geom=1.0, threshold=0.55):
        """创建重捕获 wrapper。

        参数: odtrack_tracker 上游 ODTrack 实例（提供 initialize/track）；
              gate ReliabilityGate（None 默认，仅供模板记忆门控）；
              memory TemplateMemory（None 默认，需在 init 后 set_anchor）；
              redetector SphericalMultiViewRedetector（None 默认，
              get_templates 接 memory.get_bank）；
              run_len 连续低可靠判 lost 帧数；search_interval LOST 态搜索间隔；
              observe_frames 找回后观察帧数；anchor_min_sim VERIFY 锚点相似度
              下限；recapture_min_score 重捕获 NCC 分数下限；
              motion_max_deg 找回候选与失锁前位置的最大球面角距（度）；
              w_motion/w_scale/w_ncc/w_geom/threshold 判丢可靠性权重与阈值
              （默认值 = 本地 60/60 离线标定结果 w_ncc=0.5 一档，
              公式与 scripts/score_offline_gate.py 完全一致，可直接迁移）。
        """
        self.tracker = odtrack_tracker
        self.run_len = max(2, int(run_len))
        self.search_interval = max(1, int(search_interval))
        self.observe_frames = max(1, int(observe_frames))
        self.anchor_min_sim = float(anchor_min_sim)
        self.recapture_min_score = float(recapture_min_score)
        self.motion_max_deg = float(motion_max_deg)
        # 判丢权重/阈值（与离线标定同公式，见 _reliability）
        self.w_motion = float(w_motion)
        self.w_scale = float(w_scale)
        self.w_ncc = float(w_ncc)
        self.w_geom = float(w_geom)
        self.threshold = float(threshold)

        # memory 的 gate 只用于模板记忆写入门控（accept_thr 语义），
        # 与判丢可靠性是两回事，两者独立配置
        if memory is None:
            memory = TemplateMemory(gate=gate if gate is not None
                                    else ReliabilityGate())
        self.memory = memory
        self.redetector = redetector if redetector is not None \
            else SphericalMultiViewRedetector(self.memory.get_bank,
                                              min_score=self.recapture_min_score)
        # 运行时状态
        self._erp_w = 0
        self._erp_h = 0
        self._low_run = 0
        self._search_counter = 0
        self._observe = 0
        self._lost_anchor_pos = None   # 失锁前最后位置 (lon_deg, lat_deg)
        self._last_box = None
        self._status = self.STATUS_NORMAL
        self._last_reliability = 1.0
        # 运动外推（圆周恒速）
        self._vel = np.zeros(2, dtype=float)
        self._prev_center = None
        # 尺度 EMA
        self._scale_ema = None

    # ------------------------------------------------------------ 内部工具

    def _c_motion(self, box: Sequence[float]) -> float:
        """预测中心与圆周恒速外推的偏差 -> 一致性（0~1）。"""
        width, height = self._erp_w, self._erp_h
        diag = max(2.0, float(np.hypot(box[2], box[3])))
        center = np.array([_center(box, width),
                           float(box[1]) + 0.5 * float(box[3])])
        if self._prev_center is None:
            self._prev_center = center
            return 1.0
        pred = self._prev_center + self._vel
        err = float(np.hypot(_circ_delta(center[0], pred[0], width),
                             center[1] - pred[1]))
        self._vel = 0.7 * self._vel + 0.3 * (
            np.array([_circ_delta(center[0], self._prev_center[0], width),
                      center[1] - self._prev_center[1]]))
        self._prev_center = center
        return float(np.clip(1.0 - err / diag, 0.0, 1.0))

    def _c_scale(self, box: Sequence[float]) -> float:
        area = max(4.0, float(box[2] * box[3]))
        if self._scale_ema is None:
            self._scale_ema = area
            return 1.0
        ratio = area / self._scale_ema
        self._scale_ema = 0.9 * self._scale_ema + 0.1 * area
        return float(np.clip(np.exp(-abs(np.log(ratio)) / 0.4), 0.0, 1.0))

    def _c_anchor(self, frame: np.ndarray, box: Sequence[float]) -> float:
        if self.memory.anchor is None:
            return 0.5
        crop = _crop_wrap(frame, box)
        sim = _template_similarity((crop, (box[2], box[3])), self.memory.anchor[0])
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    def _reliability(self, frame: np.ndarray, box: Sequence[float]):
        """融合可靠性 R（低 = 判丢）。

        公式与 scripts/score_offline_gate.py 完全一致（线性 logit + sigmoid，
        权重/阈值来自离线 60/60 标定）：
          logit = w_motion*(C_motion-0.5) + w_scale*(C_scale-0.5)
                  + w_ncc*(C_anchor-0.5) - w_geom*geometry_risk
        C_visual（last_pred_iou）在 wrapper 里不参与——离线标定时它只有
        10 条序列的证据，权重设为 0；若服务器重跑全量 confidence 后
        标定出它的权重，加进公式即可。

        返回: (r, c_motion, c_scale)。motion/scale 分量有状态副作用
        （推进速度/尺度 EMA），必须只在每帧调用一次，返回值供
        TemplateMemory.add 复用。
        """
        c_anchor = self._c_anchor(frame, box)
        c_motion = self._c_motion(box)
        c_scale = self._c_scale(box)
        geom = _geometry_risk(box, self._erp_w, self._erp_h)
        logit = (self.w_motion * (c_motion - 0.5)
                 + self.w_scale * (c_scale - 0.5)
                 + self.w_ncc * (c_anchor - 0.5)
                 - self.w_geom * geom)
        r = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
        return r, c_motion, c_scale

    def _verify(self, frame: np.ndarray, candidate: Sequence[float],
                score: float) -> bool:
        """重捕获候选 VERIFY：分数 + anchor 一致 + 运动合理。"""
        if score < self.recapture_min_score:
            return False
        if self.memory.anchor is not None:
            crop = _crop_wrap(frame, candidate)
            sim = _template_similarity((crop, (candidate[2], candidate[3])),
                                       self.memory.anchor[0])
            if (sim + 1.0) / 2.0 < self.anchor_min_sim:
                return False
        if self._lost_anchor_pos is not None:
            lon = _center(candidate, self._erp_w) / self._erp_w * 360.0 - 180.0
            lat = 90.0 - (float(candidate[1]) + 0.5 * float(candidate[3])
                          ) / self._erp_h * 180.0
            ang = _spherical_angle_deg(
                self._lost_anchor_pos[0], self._lost_anchor_pos[1], lon, lat)
            if ang > self.motion_max_deg:
                return False
        return True

    # ------------------------------------------------------------ 契约接口

    def init(self, image: np.ndarray, bbox: Sequence[float]):
        """首帧初始化：设置 anchor 模板与 ODTrack 初始状态。"""
        self._erp_h, self._erp_w = image.shape[:2]
        self._last_box = [float(v) for v in bbox]
        self._low_run = 0
        self._search_counter = 0
        self._observe = 0
        self._lost_anchor_pos = None
        self._status = self.STATUS_NORMAL
        self._vel = np.zeros(2, dtype=float)
        self._prev_center = None
        self._scale_ema = None
        self.memory.set_anchor(
            (_crop_wrap(image, bbox), (float(bbox[2]), float(bbox[3]))))
        self.tracker.initialize(
            np.concatenate((image, image, image), axis=1),
            {'init_bbox': [float(bbox[0]) % self._erp_w + self._erp_w,
                           float(bbox[1]), float(bbox[2]), float(bbox[3])]})

    def update(self, image: np.ndarray) -> dict:
        """跟踪新帧，返回 {'bbox','score','status','fov','reliability'}。"""
        H, W = image.shape[:2]
        assert W == self._erp_w and H == self._erp_h
        tiled = np.concatenate((image, image, image), axis=1)

        if self._status == self.STATUS_LOST:
            self._search_counter += 1
            if self._search_counter % self.search_interval == 0:
                found = self.redetector.search(image, erp_downscale=2)
                if found is not None:
                    candidate, score = found
                    if self._verify(image, candidate, score):
                        # 重新 initialize：清空被污染的时序记忆
                        self.tracker.initialize(
                            tiled, {'init_bbox': [
                                float(candidate[0]) % W + W, float(candidate[1]),
                                float(candidate[2]), float(candidate[3])]})
                        self._last_box = [float(v) for v in candidate]
                        self._status = self.STATUS_OBSERVE
                        self._observe = 0
                        self._low_run = 0
                        self._lost_anchor_pos = None
                        # 运动外推与尺度 EMA 重置（不再用失锁前的旧状态）
                        self._vel = np.zeros(2, dtype=float)
                        self._prev_center = None
                        self._scale_ema = None
                        if self.memory.anchor is None:
                            self.memory.set_anchor(
                                (_crop_wrap(image, candidate),
                                 (float(candidate[2]), float(candidate[3]))))
                        return {'bbox': tuple(self._last_box), 'score': score,
                                'status': self.STATUS_OBSERVE,
                                'fov': (0.0, 0.0), 'reliability': 1.0}
            # 未找回：沿用上次框（lost 占位）
            return {'bbox': tuple(self._last_box), 'score': 0.0,
                    'status': self.STATUS_LOST, 'fov': (0.0, 0.0),
                    'reliability': 0.0}

        output = self.tracker.track(tiled)
        box = [float(v) for v in output['target_bbox']]
        box[0] %= W
        r, c_motion, c_scale = self._reliability(image, box)
        self._last_box = box

        if self._status == self.STATUS_OBSERVE:
            self._observe += 1
            if r >= self.threshold:
                if self._observe >= self.observe_frames:
                    self._status = self.STATUS_NORMAL
                return {'bbox': tuple(box), 'score': float(r),
                        'status': self.STATUS_OBSERVE, 'fov': (0.0, 0.0),
                        'reliability': r}
            self._status = self.STATUS_LOST
            self._low_run = 0
            self._lost_anchor_pos = None
            return {'bbox': tuple(box), 'score': float(r),
                    'status': self.STATUS_LOST, 'fov': (0.0, 0.0),
                    'reliability': r}

        # NORMAL
        if r < self.threshold:
            self._low_run += 1
            if self._low_run >= self.run_len:
                self._status = self.STATUS_LOST
                self._search_counter = 0
                # 记录失锁前位置（球面经纬度）供 VERIFY 运动约束
                lon = _center(box, W) / W * 360.0 - 180.0
                lat = 90.0 - (box[1] + 0.5 * box[3]) / H * 180.0
                self._lost_anchor_pos = (lon, lat)
                return {'bbox': tuple(box), 'score': float(r),
                        'status': self.STATUS_LOST, 'fov': (0.0, 0.0),
                        'reliability': r}
        else:
            self._low_run = 0
            # 门控写入模板记忆（anchor 永不被覆盖）
            self.memory.add(
                (_crop_wrap(image, box), (box[2], box[3])),
                c_visual=float(getattr(self.tracker, 'last_pred_iou', r)),
                c_anchor=self._c_anchor(image, box),
                c_motion=c_motion,
                c_scale=c_scale,
                geometry_risk=_geometry_risk(box, W, H))
        return {'bbox': tuple(box), 'score': float(r),
                'status': self.STATUS_NORMAL, 'fov': (0.0, 0.0),
                'reliability': r}
