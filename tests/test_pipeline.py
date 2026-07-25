# -*- coding: utf-8 -*-
"""模块 E（panotrack.pipeline）单元测试：纯 assert 脚本。

运行方式：在项目根目录 D:\\instan\\pano360 下执行 `python tests/test_pipeline.py`。

覆盖质量要求：
  1. SphericalState：恒定角速度预测的经度环绕（±180° 无跳变）与纬度钳制；
  2. GlobalRedetector：含跨界目标的全图找回（位置精度）与无目标/低分返回 None；
  3. PanoTracker 契约：update 返回字段齐全、status 合法、bbox 满足跨界约定；
  4. crossing 场景 30 帧端到端：SR@0.5（双口径）≥ 0.9；
  5. occlusion 场景 30 帧端到端：丢失后 10 帧内出现 recovered。
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.data.synth import generate_sequence, target_texture, _background
from panotrack.data.io import load_sequence
from panotrack.geometry.bfov import BFoV
from panotrack.evaluation.metrics import ope_evaluate
from panotrack.pipeline.state import SphericalState
from panotrack.pipeline.redetect import GlobalRedetector
from panotrack.pipeline.pipeline import PanoTracker

W, H = 1024, 512
_N_FRAMES = 30          # 轻量：端到端用 30 帧
_TMP = tempfile.mkdtemp(prefix='test_pipeline_', dir=str(Path(__file__).resolve().parents[1] / 'runs'))


def _make_seq(scenario, n=_N_FRAMES):
    """生成 n 帧测试序列并加载（私有）。"""
    out = Path(_TMP) / scenario
    seq_dir = generate_sequence(out, n_frames=n, w=W, h=H, scenario=scenario, seed=3)
    return load_sequence(seq_dir)


def _run(frames, gt):
    """OPE 跑完整序列，返回 (preds, results)（私有）。"""
    tr = PanoTracker()
    tr.init(frames[0], tuple(gt[0]))
    preds = [tuple(gt[0])]
    results = []
    for i in range(1, len(frames)):
        r = tr.update(frames[i])
        preds.append(r['bbox'])
        results.append(r)
    return np.array(preds, dtype=float), results


# ---------------------------------------------------------------- 1. SphericalState

def test_state_wrap_and_clamp():
    """经度环绕预测连续、纬度钳制不越界。"""
    st = SphericalState(BFoV(lon=178.0, lat=0.0, fov_h=10.0, fov_v=10.0))
    st.update(BFoV(lon=179.0, lat=0.0, fov_h=10.0, fov_v=10.0))
    st.update(BFoV(lon=-179.0, lat=0.0, fov_h=10.0, fov_v=10.0))   # 跨界 +2°
    p = st.predict()
    assert -180.0 < p.lon <= 180.0, f'预测经度未环绕到 (-180,180]: {p.lon}'
    assert abs(p.lon - (-178.0)) < 1.0, f'跨界预测不连续: lon={p.lon}（应约 -178）'
    # 纬度钳制：观测/预测均不超过 ±89.9
    st.update(BFoV(lon=0.0, lat=95.0, fov_h=10.0, fov_v=10.0))
    assert st.bfov.lat <= 89.9, f'纬度未钳制: {st.bfov.lat}'
    assert abs(st.predict().lat) <= 89.9
    print('[OK] SphericalState 环绕与钳制')


# ---------------------------------------------------------------- 2. GlobalRedetector

def test_redetector():
    """跨界目标可找回、无目标/无模板返回 None。"""
    bg = _background(W, H, seed=1)
    tex = target_texture(80, 40)
    frame = bg.copy()
    x0, y0 = W - 42, 100                       # x0+w > W：跨界放置
    cols = (x0 + np.arange(80)) % W
    frame[y0:y0 + 40][:, cols] = tex
    rd = GlobalRedetector(lambda: (tex, (80.0, 40.0)), min_score=0.6)
    found = rd.search(frame, erp_downscale=2)
    assert found is not None, '跨界目标未找回'
    (bx, by, bw, bh), score = found
    assert 0.0 <= bx < W, f'找回框 x 越界: {bx}'
    assert abs(bx - x0) <= 8.0 and abs(by - y0) <= 8.0, \
        f'找回位置偏差过大: ({bx:.1f},{by:.1f}) vs ({x0},{y0})'
    assert (bw, bh) == (80.0, 40.0), '找回框尺寸应等于模板原尺寸'
    # 无目标（纯背景）高分门限下返回 None；无模板返回 None
    assert GlobalRedetector(lambda: (tex, (80.0, 40.0)),
                            min_score=0.98).search(bg, erp_downscale=2) is None
    assert GlobalRedetector(lambda: None).search(frame, erp_downscale=2) is None
    print('[OK] GlobalRedetector 跨界找回与 None 分支')


# ---------------------------------------------------------------- 3. PanoTracker 契约字段

def test_contract_fields():
    """update 返回 {'bbox','score','status','fov'}，取值合法且满足跨界约定。"""
    frames, gt = _make_seq('equator', n=12)
    _, results = _run(frames, gt)
    assert len(results) == 11
    for r in results:
        assert set(r.keys()) == {'bbox', 'score', 'status', 'fov'}, f'字段缺失: {r.keys()}'
        x, y, w, h = r['bbox']
        assert 0.0 <= x < W, f'bbox x 不满足跨界约定 [0,W): {x}'
        assert w > 0.0 and h > 0.0, 'bbox 尺寸必须为正'
        assert 0.0 <= r['score'] <= 1.0, f'score 越界: {r["score"]}'
        assert r['status'] in ('ok', 'lost', 'recovered'), f'非法 status: {r["status"]}'
        fh, fv = r['fov']
        assert fh > 0.0 and fv > 0.0, 'fov 必须为正'
    print('[OK] PanoTracker 契约字段')


# ---------------------------------------------------------------- 4. crossing 端到端

def test_crossing_sr():
    """crossing 30 帧：SR@0.5 双口径 ≥ 0.9，且确实发生跨界输出。"""
    frames, gt = _make_seq('crossing')
    preds, results = _run(frames, gt)
    m = ope_evaluate(preds, np.array(gt, dtype=float), W)
    assert m['sr'] >= 0.9 and m['sr_dual'] >= 0.9, \
        f'crossing SR 不达标: sr={m["sr"]:.3f} sr_dual={m["sr_dual"]:.3f}'
    assert any(r['bbox'][0] + r['bbox'][2] > W for r in results), '未出现跨界输出框'
    print(f'[OK] crossing SR: sr={m["sr"]:.3f} sr_dual={m["sr_dual"]:.3f}')


# ---------------------------------------------------------------- 5. occlusion 找回

def test_occlusion_recovery():
    """occlusion 30 帧：出现 lost，且首个 lost 后 10 帧内出现 recovered。"""
    frames, gt = _make_seq('occlusion')
    _, results = _run(frames, gt)
    statuses = [r['status'] for r in results]
    assert 'lost' in statuses, '遮挡场景未出现 lost 判定'
    i0 = statuses.index('lost')
    rec = [i for i in range(i0 + 1, len(statuses)) if statuses[i] == 'recovered']
    assert rec, '丢失后未找回（无 recovered）'
    assert rec[0] - i0 <= 10, f'找回超时: lost@f{i0 + 1} -> recovered@f{rec[0] + 1}'
    print(f'[OK] occlusion 找回: lost@f{i0 + 1} -> recovered@f{rec[0] + 1}')


if __name__ == '__main__':
    test_state_wrap_and_clamp()
    test_redetector()
    test_contract_fields()
    test_crossing_sr()
    test_occlusion_recovery()
    print('ALL PIPELINE TESTS PASSED')
