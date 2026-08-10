# -*- coding: utf-8 -*-
"""GRT-360 Commit 4 测试：ReliabilityGate + TemplateMemory。

运行: python tests/test_memory.py
覆盖:
  1. ReliabilityGate 高可信观测 -> 高 R，低可信 -> 低 R
  2. accept 阈值门控
  3. 几何风险惩罚压低 R
  4. 历史 EMA 平滑（连续高信使 R 上升）
  5. TemplateMemory 设 anchor 后不可覆盖
  6. Short 层滚动容量
  7. Long 层去冗余（相似模板不重复入）
  8. get_bank 去重 + best 返回
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.pipeline.memory import ReliabilityGate, TemplateMemory


def _patch(seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (32, 32, 3), dtype=np.uint8), (32, 32))


def test_gate_high_vs_low():
    gate = ReliabilityGate()
    r_high = gate.reliability(c_visual=0.95, c_anchor=0.9, c_motion=1.0,
                              c_scale=1.0, geometry_risk=0.0)
    gate.reset_history()
    r_low = gate.reliability(c_visual=0.1, c_anchor=0.1, c_motion=0.2,
                             c_scale=0.3, geometry_risk=0.8)
    assert r_high > 0.6, r_high
    assert r_low < 0.4, r_low
    assert r_high > r_low


def test_gate_risk_penalizes():
    gate = ReliabilityGate()
    r_no = gate.reliability(c_visual=0.8, c_anchor=0.8, c_motion=0.9,
                            c_scale=0.9, geometry_risk=0.0)
    gate.reset_history()
    r_risk = gate.reliability(c_visual=0.8, c_anchor=0.8, c_motion=0.9,
                              c_scale=0.9, geometry_risk=0.9)
    assert r_no > r_risk


def test_gate_accept_threshold():
    gate = ReliabilityGate(accept_thr=0.5)
    r = gate.reliability(c_visual=0.95, c_anchor=0.95, c_motion=1.0,
                         c_scale=1.0, geometry_risk=0.0)
    assert gate.accept(r)
    gate.reset_history()
    r = gate.reliability(c_visual=0.05, c_anchor=0.05, c_motion=0.1,
                         c_scale=0.2, geometry_risk=0.9)
    assert not gate.accept(r)


def test_gate_history_smoothing():
    gate = ReliabilityGate()
    # 连续高信：R 应逐步上升（EMA 累积）
    rs = [gate.reliability(c_visual=0.9, c_anchor=0.9, c_motion=1.0,
                           c_scale=1.0, geometry_risk=0.0) for _ in range(5)]
    assert rs[-1] > rs[0], rs


def test_memory_anchor_immutable():
    mem = TemplateMemory()
    a0 = _patch(0)
    mem.set_anchor(a0)
    mem.set_anchor(_patch(1))  # 不应覆盖
    assert mem.anchor[0][0] is a0[0]


def test_memory_short_rolling_cap():
    mem = TemplateMemory(short_cap=3)
    for i in range(5):
        mem.set_anchor(_patch(0))
        mem.add(_patch(i), c_visual=0.9, c_anchor=0.9, c_motion=1.0,
                c_scale=1.0, geometry_risk=0.0)
    assert len(mem.short) <= 3


def test_memory_long_dedup():
    mem = TemplateMemory(dedup_thr=0.7)
    mem.set_anchor(_patch(0))
    t = _patch(7)
    mem.add(t, c_visual=0.9, c_anchor=0.9, c_motion=1.0, c_scale=1.0,
            geometry_risk=0.0)
    # 与 t 高度相似（同 seed 同帧）—— 应被去重
    mem.add(t, c_visual=0.9, c_anchor=0.9, c_motion=1.0, c_scale=1.0,
            geometry_risk=0.0)
    same = [e for e in mem.long if e[0][0] is t[0]]
    assert len(same) == 1, len(same)


def test_memory_gate_rejects_low():
    mem = TemplateMemory(min_quality=0.5)
    mem.set_anchor(_patch(0))
    ok = mem.add(_patch(1), c_visual=0.05, c_anchor=0.05, c_motion=0.1,
                 c_scale=0.2, geometry_risk=0.9)
    assert not ok
    assert len(mem.short) == 0 and len(mem.long) == 0


def test_memory_bank_dedup_and_best():
    mem = TemplateMemory(short_cap=4, long_cap=8)
    anchor = _patch(0)
    mem.set_anchor(anchor)
    for i in range(1, 4):
        mem.add(_patch(100 + i), c_visual=0.9, c_anchor=0.9, c_motion=1.0,
                c_scale=1.0, geometry_risk=0.0)
    bank = mem.get_bank()
    # anchor 一定在 bank 中且无重复
    assert any(e is anchor for e in bank)
    assert len(bank) == len({id(e[0]) for e in bank})
    assert mem.best() is anchor


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