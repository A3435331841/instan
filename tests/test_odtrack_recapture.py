# -*- coding: utf-8 -*-
"""OdtrackRecaptureTracker CPU 冒烟测试（纯 assert，不依赖 GPU/上游 ODTrack）。

用合成 ERP 帧 + 可编程 FakeTracker 验证状态机：
  1. NORMAL：高可靠帧正常输出，模板记忆写入；
  2. 失锁检测：连续低可靠 run_len 帧 -> LOST；
  3. 重捕获：LOST 下 redetect_v3 在帧中命中移动后的目标 -> VERIFY -> OBSERVE；
  4. 恢复：找回后连续高可靠 observe_frames 帧 -> NORMAL；
  5. anchor 永不被覆盖（内存检查）。

运行: python tests/test_odtrack_recapture.py
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.odtrack.recapture import OdtrackRecaptureTracker  # noqa: E402


class FakeTracker:
    """可编程假 ODTrack：按剧本返回框与 last_pred_iou。

    reinit_boxes 支持 reinit 后多帧剧本（列表的列表）；为 None 时
    退化为单框 reinit_box 语义。
    """

    def __init__(self, boxes, ious, reinit_box=None, reinit_iou=0.8,
                 reinit_boxes=None):
        self.boxes = list(boxes)
        self.ious = list(ious)
        self.reinit_box = reinit_box
        self.reinit_iou = float(reinit_iou)
        self.reinit_boxes = reinit_boxes
        self.idx = 0
        self.last_pred_iou = 1.0
        self.n_init = 0

    def initialize(self, tiled, info):
        self.n_init += 1
        if self.n_init > 1:
            self.idx = 0
            if self.reinit_boxes is not None:
                self.boxes = [list(b) for b in self.reinit_boxes]
                self.ious = [self.reinit_iou] * len(self.boxes)
            elif self.reinit_box is not None:
                self.boxes = [self.reinit_box]
                self.ious = [self.reinit_iou]
        self.last_pred_iou = 1.0

    def track(self, tiled):
        if self.idx >= len(self.boxes):
            box = self.boxes[-1]
            iou = self.ious[-1]
        else:
            box = self.boxes[self.idx]
            iou = self.ious[self.idx]
        self.idx += 1
        self.last_pred_iou = iou
        return {'target_bbox': box}


def make_frame(W=512, H=256, target_cx=None, target_cy=120, target_size=24):
    """合成 ERP 帧：经纬网格渐变背景 + 可选红色方块目标。"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lon = (xx / W * 360 - 180) / 180 * np.pi
    lat = (90 - yy / H * 180) / 180 * np.pi
    bg_r = ((np.sin(lon) * 0.5 + 0.5) * 200 + 20).astype(np.uint8)
    bg_g = ((np.cos(lat) * 0.5 + 0.5) * 160 + 30).astype(np.uint8)
    bg_b = ((np.sin(lon + lat) * 0.5 + 0.5) * 120 + 40).astype(np.uint8)
    img = np.stack([bg_r, bg_g, bg_b], axis=2)
    if target_cx is not None:
        r = target_size // 2
        img[target_cy - r:target_cy + r, target_cx - r:target_cx + r] = \
            np.array([255, 60, 60], dtype=np.uint8)
    return img


def test_recapture_state_machine():
    W, H = 512, 256
    # 剧本：前 5 帧正确跟（高 iou）；第 6-10 帧跟错（低 iou -> 判 lost）；
    # 丢失期间目标移动到 (240,120)（球面角距约 42°，在 motion 约束内），
    # FakeTracker reinit 后返回该位置。
    init_box = [168.0, 108.0, 24.0, 24.0]
    boxes = [init_box] * 5
    # 失锁段：交替跳变框（真实错跟是持续漂移，静止错框靠 C_visual 才能检出，
    # 而 C_visual 在离线标定里权重为 0，这里用跳变模拟漂移）
    boxes += [[60.0 + 40 * (i % 2), 60.0 + 40 * ((i + 1) % 2), 24.0, 24.0]
               for i in range(5)]
    ious = [0.8] * 5 + [0.05] * 5
    fake = FakeTracker(boxes, ious, reinit_box=[228.0, 108.0, 24.0, 24.0])

    tracker = OdtrackRecaptureTracker(fake, run_len=5, search_interval=1,
                                      observe_frames=3, motion_max_deg=120.0)
    frames = [make_frame(W, H, target_cx=180 + i * 12 if i < 5 else 240)
              for i in range(20)]
    tracker.init(frames[0], init_box)
    assert tracker.memory.has_anchor, 'anchor 应在 init 时设置'

    statuses = []
    for i in range(1, 16):
        out = tracker.update(frames[i])
        statuses.append(out['status'])
        print(f'frame {i:2d}: status={out["status"]:9s} '
              f'box=({out["bbox"][0]:7.1f},{out["bbox"][1]:7.1f}) '
              f'r={out["reliability"]:.2f}')
    # 0-4 正常（帧1-4 是 NORMAL），5-9 低可靠 -> 帧 10 判 LOST
    assert statuses[0] == 'ok' and statuses[3] == 'ok', '正常段应为 ok'
    assert 'lost' in statuses, '连续低可靠应进入 lost'
    # 重捕获：lost 后 redetector 命中移动目标 -> recovered -> 观察期 -> ok
    assert 'recovered' in statuses, '重捕获应找回目标'
    assert statuses[-1] == 'ok', '观察期后应恢复 normal'
    # anchor 未被覆盖
    anchor = tracker.memory.anchor[0][0]
    assert anchor.shape[0] == 24 and anchor.shape[1] == 24, 'anchor 应保持首帧尺寸'
    # 找回后 reinit 至少发生一次
    assert fake.n_init >= 2, '重捕获应重新 initialize ODTrack'
    print('PASS test_recapture_state_machine')


def test_verify_rejects_far_jump():
    """VERIFY 应拒绝超出运动约束的远跳候选（防误锁特性）。"""
    W, H = 512, 256
    from panotrack.pipeline.memory import ReliabilityGate, TemplateMemory
    from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector
    mem = TemplateMemory(gate=ReliabilityGate(a=(1.5, 0.3, 0.3, 0.2, 0.2)))
    rd = SphericalMultiViewRedetector(mem.get_bank, min_score=0.95)
    # 框漂移 + 低可靠 -> 快速进入 LOST（设置 _lost_anchor_pos）
    fake = FakeTracker([[60.0 + 40 * (i % 2), 60.0 + 40 * ((i + 1) % 2), 24.0, 24.0]
                           for i in range(20)], [0.05] * 20)
    tracker = OdtrackRecaptureTracker(fake, run_len=3, search_interval=1,
                                      motion_max_deg=60.0,
                                      memory=mem, redetector=rd)
    frames = [make_frame(W, H, target_cx=180, target_cy=120) for _ in range(6)]
    tracker.init(frames[0], [168.0, 108.0, 24.0, 24.0])
    for i in range(1, 5):
        tracker.update(frames[i])
    assert tracker._status == tracker.STATUS_LOST, '测试前置：应已进入 LOST'
    # 失锁前位置中心 (72,72)；候选 (300,120) -> 球面角距约 173° > 60°
    accepted = tracker._verify(frames[5], [288.0, 108.0, 24.0, 24.0], 0.80)
    assert not accepted, '远跳候选应被 VERIFY 拒绝'
    # 失锁位置本身（中心 72,72）应通过
    accepted = tracker._verify(frames[5], [60.0, 60.0, 24.0, 24.0], 0.80)
    assert accepted, '近距候选应通过 VERIFY'
    print('PASS test_verify_rejects_far_jump')


def test_lost_placeholder_when_no_target():
    """目标消失后 LOST 状态下输出 lost 占位且不崩溃。

    注意：合成帧的渐变背景纹理简单，低 NCC 阈值可能产生假阳性命中；
    这里用高 min_score 的 redetector 聚焦验证状态机（真实数据上
    假阳性由 VERIFY 的 anchor 校验把关）。
    """
    W, H = 512, 256
    from panotrack.pipeline.redetect_v3 import SphericalMultiViewRedetector
    from panotrack.pipeline.memory import ReliabilityGate, TemplateMemory
    # 默认 gate 权重对"响应极低"不够敏感（这正是 Step 2 离线标定要解决的）；
    # 测试显式传入高 C_visual 权重的 gate，聚焦验证状态机逻辑。
    mem = TemplateMemory(gate=ReliabilityGate(a=(1.5, 0.3, 0.3, 0.2, 0.2)))
    rd = SphericalMultiViewRedetector(mem.get_bank, min_score=0.95)
    fake = FakeTracker([[60.0 + 40 * (i % 2), 60.0 + 40 * ((i + 1) % 2), 24.0, 24.0]
                           for i in range(20)],
                       [0.05] * 20)  # 跳变漂移 + 全程低可靠（目标消失）
    tracker = OdtrackRecaptureTracker(fake, run_len=3, search_interval=1,
                                      memory=mem, redetector=rd)
    frames = [make_frame(W, H, target_cx=None) for _ in range(8)]
    tracker.init(frames[0], [168.0, 108.0, 24.0, 24.0])
    lost_seen = 0
    for i in range(1, 8):
        out = tracker.update(frames[i])
        if out['status'] == 'lost':
            lost_seen += 1
    assert lost_seen >= 3, '目标消失应持续 lost（宁可丢不可锁错）'
    print('PASS test_lost_placeholder_when_no_target')


def test_observe_fail_returns_to_lost():
    """找回后观察期内再次失锁 -> 回 LOST（防"锁错对象"的第二道闸）。

    剧本：复用 test_recapture_state_machine 已验证的重捕获触发条件
    （motion_max_deg=120，目标在 240），但 reinit 后 ODTrack 继续跟错
    （reinit_box 在错误位置 + 低可靠），观察期第一帧 R 低应立即回 LOST，
    而不是硬撑到观察期结束。
    """
    W, H = 512, 256
    init_box = [168.0, 108.0, 24.0, 24.0]
    bad = [[60.0, 100.0, 24.0, 24.0], [100.0, 60.0, 24.0, 24.0]]
    bad_boxes = [bad[i % 2] for i in range(5)]
    fake = FakeTracker([init_box] * 5 + bad_boxes,
                       [0.8] * 5 + [0.05] * 5,
                       reinit_boxes=[[60.0, 100.0, 24.0, 24.0],
                                    [100.0, 60.0, 24.0, 24.0],
                                    [60.0, 100.0, 24.0, 24.0],
                                    [100.0, 60.0, 24.0, 24.0]],
                       reinit_iou=0.05)
    tracker = OdtrackRecaptureTracker(fake, run_len=3, search_interval=1,
                                      observe_frames=3, motion_max_deg=120.0)
    frames = [make_frame(W, H, target_cx=240) for _ in range(15)]
    tracker.init(frames[0], init_box)
    statuses = []
    for i in range(1, 15):
        out = tracker.update(frames[i])
        statuses.append(out['status'])
    assert 'recovered' in statuses, '重捕获应触发'
    idx = statuses.index('recovered')
    assert 'lost' in statuses[idx:], '观察期失锁应立即回 LOST'
    print('PASS test_observe_fail_returns_to_lost')


if __name__ == '__main__':
    test_recapture_state_machine()
    test_lost_placeholder_when_no_target()
    test_verify_rejects_far_jump()
    test_observe_fail_returns_to_lost()
    print('ALL TESTS PASSED')
