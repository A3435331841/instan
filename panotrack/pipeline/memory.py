# -*- coding: utf-8 -*-
"""GRT-360 Commit 4：Reliability Gate + Reliable Compressed Temporal Memory。

两个核心组件：
1. ReliabilityGate —— 综合视觉/锚点/运动/尺度/历史多分量与几何风险，输出
   单帧可靠性 R ∈ [0,1]，作为是否接受当前帧进入模板记忆与状态更新的门控。
   R = sigmoid(a1*z(C_visual) + a2*z(C_anchor) + a3*z(C_motion)
               + a4*z(C_scale) + a5*z(C_history) - a6*geometry_risk)
   其中 z(·) 为分量去噪标准化（稳健 min-max），C_* 为可靠性证据分量：
     C_visual  跟踪器响应峰值（观测强度）
     C_anchor  与 immutable anchor 模板的相似度（一致性）
     C_motion  S² 运动先验一致性（预测误差小则高）
     C_scale   目标尺度一致性（相对历史尺度变化小则高）
     C_history 历史可靠性指数滑动平均（长期稳定性）
     geometry_risk 几何风险（近极点/大角速度/遮挡/大尺度膨胀）

2. TemplateMemory —— 三层模板记忆：
     Anchor  不可变基准模板（首帧或首次高质量观测，作几何一致性校验锚点）
     Short   最近 2-4 个高质量 keyframe 模板（短期外观变化）
     Long    去冗余长期模板（与既有模板相似度低于阈值才入，超上限挤掉最旧）
   供全局重检测（Spherical Multi-view Reacquisition）取用多视角模板池。
"""
import numpy as np
from PIL import Image

_EPS = 1e-8


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-float(np.clip(x, -30.0, 30.0))))


def _robust_z(v, center=0.5, scale=0.25):
    """稳健标准化：z = (clip(v,0,1) - center) / scale（抑制极端值）。"""
    c = float(np.clip(v, 0.0, 1.0))
    return (c - center) / scale


def _template_similarity(tpl_a, tpl_b, size=32):
    """两个模板的归一化相似度（resize 到固定尺寸后的 NCC，∈[-1,1]）。

    参数: tpl_a, tpl_b —— (img, (w,h)) 模板元组；size 比较分辨率。
    返回: float 相似度。
    """
    img_a, img_b = tpl_a[0], tpl_b[0]
    a = np.asarray(_to_gray_f32(img_a))
    b = np.asarray(_to_gray_f32(img_b))
    a = _resize_gray(a, size, size)
    b = _resize_gray(b, size, size)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float((a * b).sum() / (na * nb))


def _to_gray_f32(img):
    arr = np.asarray(img)
    if arr.ndim == 3:
        return (arr[..., 0].astype(np.float32) * 0.299
                + arr[..., 1].astype(np.float32) * 0.587
                + arr[..., 2].astype(np.float32) * 0.114)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _resize_gray(gray, out_w, out_h):
    im = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    return np.asarray(im.resize((int(out_w), int(out_h)), Image.BILINEAR))


class ReliabilityGate:
    """综合可靠性门控，输出单帧 R ∈ [0,1] 并据阈值接受/拒绝。

    默认权重对齐规格：视觉 0.5、锚点 0.3、运动 0.3、尺度 0.2、历史 0.2，
    几何风险 0.6（惩罚项）。权重可在构造时覆盖。
    """

    def __init__(self, a=(0.5, 0.3, 0.3, 0.2, 0.2), a_risk=0.6,
                 accept_thr=0.5):
        """创建可靠性门控。

        参数: a (a1..a5) 五个正向分量的权重；a_risk 几何风险惩罚权重；
              accept_thr 接受阈值（R >= 阈值才接受）。
        返回: None
        """
        self.a = tuple(float(x) for x in a)
        self.a_risk = float(a_risk)
        self.accept_thr = float(np.clip(accept_thr, 0.0, 1.0))
        self._history = 0.5  # 历史可靠性 EMA
        self._hist_alpha = 0.2

    # ------------------------------------------------------------ 分量评分

    @staticmethod
    def c_visual(score):
        """视觉分量：跟踪器响应峰值（0~1 归一化）。"""
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def c_anchor(anch_sim):
        """锚点一致性分量：与 immutable anchor 模板的相似度 ∈[0,1]。"""
        return float(np.clip((anch_sim + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def c_motion(pred_err_deg, sigma=10.0):
        """运动一致性分量：预测误差越大越低（高斯衰减）。"""
        return float(np.exp(-(pred_err_deg * pred_err_deg) / (2.0 * sigma * sigma)))

    @staticmethod
    def c_scale(scale_ratio, thr=0.4):
        """尺度一致性分量：相对历史尺度变化小则高（对称高斯）。"""
        r = float(np.clip(scale_ratio, 0.0, 4.0))
        return float(np.exp(-((r - 1.0) ** 2) / (2.0 * thr * thr)))

    @staticmethod
    def c_history(ema):
        """历史可靠性分量：EMA 本身。"""
        return float(np.clip(ema, 0.0, 1.0))

    # ------------------------------------------------------------ 主接口

    def reliability(self, c_visual=0.5, c_anchor=0.5, c_motion=1.0,
                    c_scale=1.0, geometry_risk=0.0):
        """计算综合可靠性 R ∈ [0,1] 并更新历史 EMA。

        参数: c_visual/c_anchor/c_motion/c_scale 各分量 ∈[0,1]（未提供用默认）；
              geometry_risk 几何风险 ∈[0,1]（惩罚项）。
        返回: float R。
        """
        a = self.a
        logit = (a[0] * _robust_z(c_visual)
                 + a[1] * _robust_z(c_anchor)
                 + a[2] * _robust_z(c_motion)
                 + a[3] * _robust_z(c_scale)
                 + a[4] * _robust_z(self._history)
                 - self.a_risk * geometry_risk)
        r = _sigmoid(logit)
        self._history = (1.0 - self._hist_alpha) * self._history \
            + self._hist_alpha * r
        return r

    def accept(self, reliability):
        """据阈值判断是否接受（R >= accept_thr）。"""
        return float(reliability) >= self.accept_thr

    def reset_history(self):
        """重置历史 EMA 到中性值。"""
        self._history = 0.5


class TemplateMemory:
    """三层模板记忆：Anchor(immutable) / Short(最近 keyframes) / Long(去冗余)。

    模板格式 (img, (w_erp, h_erp))，与 PanoTracker._template 一致。
    add() 走 ReliabilityGate 门控；Long 层以相似度阈值去冗余，超上限挤掉最旧。
    get_bank() 返回全部可用模板（供多视角重检测器做多模板候选池）。
    """

    def __init__(self, gate=None, short_cap=3, long_cap=8,
                 dedup_thr=0.75, min_quality=0.5):
        """创建模板记忆。

        参数: gate ReliabilityGate 实例（None 则新建默认）；short_cap Short 层
              容量（2-4）；long_cap Long 层容量；dedup_thr 去冗余相似度阈值
              （相似度高于此视为重复不入 Long）；min_quality 最低可靠性。
        返回: None
        """
        self.gate = gate if gate is not None else ReliabilityGate()
        self.short_cap = int(short_cap)
        self.long_cap = int(long_cap)
        self.dedup_thr = float(dedup_thr)
        self.min_quality = float(min_quality)
        self.anchor = None       # (template, reliability)
        self.short = []          # list[(template, reliability, frame_idx)]
        self.long = []           # list[(template, reliability, frame_idx)]
        self._frame = 0

    @property
    def has_anchor(self):
        return self.anchor is not None

    def set_anchor(self, template, reliability=1.0):
        """设置不可变 anchor（首帧或首次高质量观测）。仅在未设置时生效。"""
        if template is None:
            return
        if self.anchor is None:
            self.anchor = (template, float(reliability))

    def reset(self):
        """清空全部记忆（anchor 保留不变，因其 immutable）。"""
        self.short = []
        self.long = []
        self.gate.reset_history()

    def add(self, template, c_visual=0.5, c_anchor=0.5, c_motion=1.0,
            c_scale=1.0, geometry_risk=0.0, sim=None):
        """门控式加入模板：只有可靠性达 min_quality 且通过 accept 才入记忆。

        参数: template (img, (w,h))；c_* 与 geometry_risk 供 ReliabilityGate；
              sim 可选显式锚点相似度（None 则实时计算）。
        返回: 布尔，是否被接受入记忆（未被门控拒绝）。
        """
        if template is None:
            return False
        if self.anchor is not None and sim is None:
            sim = _template_similarity(template, self.anchor[0])
        if sim is None:
            sim = c_anchor
        r = self.gate.reliability(
            c_visual=c_visual, c_anchor=sim, c_motion=c_motion,
            c_scale=c_scale, geometry_risk=geometry_risk)
        if r < self.min_quality or not self.gate.accept(r):
            return False
        self._remember(template, r)
        return True

    def _remember(self, template, r):
        """写入记忆：Short 层滚动存最近 keyframes；Long 层去冗余。"""
        entry = (template, float(r), self._frame)
        self._frame += 1
        self.short.append(entry)
        if len(self.short) > self.short_cap:
            self.short.pop(0)
        # Long 层去冗余：与已有模板相似度高于阈值视为重复
        dup = False
        for tpl, _, _ in self.long:
            if _template_similarity(template, tpl) >= self.dedup_thr:
                dup = True
                break
        if not dup:
            self.long.append(entry)
            if len(self.long) > self.long_cap:
                self.long.pop(0)  # 挤掉最旧

    def get_bank(self):
        """返回全部去重模板池（anchor + short + long，按可靠性降序）。

        返回: list[template]，供全局重检测器多模板候选。
        """
        seen = set()
        out = []
        items = []
        if self.anchor is not None:   # anchor 为 (template, reliability) 二元组
            items.append((self.anchor[0], self.anchor[1], -1))
        items += self.short + self.long   # short/long 为三元组
        for tpl, _r, _f in items:
            if tpl is None:
                continue
            key = id(tpl[0])
            if key in seen:
                continue
            seen.add(key)
            out.append(tpl)
        return out

    def best(self):
        """返回当前最优模板（最高可靠性，优先 anchor）。"""
        if self.anchor is not None:
            return self.anchor[0]
        cands = [(r, tpl) for tpl, r, _ in self.short + self.long]
        if not cands:
            return None
        _, tpl = max(cands, key=lambda x: x[0])
        return tpl

    def anchor_similarity(self, template):
        """模板与 anchor 的相似度（未设 anchor 返回 0.5 中性）。"""
        if self.anchor is None:
            return 0.5
        return _template_similarity(template, self.anchor[0])