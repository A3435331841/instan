# -*- coding: utf-8 -*-
"""模块 B（panotrack.trackers）单元测试：纯 assert 脚本。

运行方式：在项目根目录 D:\\instan\\pano360 下执行 `python tests/test_trackers.py`。

覆盖质量要求：
  1. 契约接口与工厂函数（签名/返回字段/lightfc 预留）；
  2. 合成局部透视图序列（平移 + 尺度变化 + 噪声的纹理块）50 帧不漂移，
     IoU>0.5 占比 ≥ 95%；
  3. 遮挡帧 score 显著下降（丢失判定信号可用）；
  4. update 单帧耗时：255×255 搜索图 < 50ms 量级；
  5. 无全局状态，多实例并发运行结果可复现。
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.trackers.base import BaseTracker
from panotrack.trackers.ncc import NCCTracker
from panotrack.trackers.factory import create_tracker


# ---------------------------------------------------------------- 测试辅助

def _bilinear_resize(img, out_h, out_w):
    """numpy 双线性缩放 (h,w) -> (out_h,out_w)（测试辅助）。"""
    h, w = img.shape
    ys = np.clip((np.arange(out_h) + 0.5) * (h / out_h) - 0.5, 0.0, h - 1.0)
    xs = np.clip((np.arange(out_w) + 0.5) * (w / out_w) - 0.5, 0.0, w - 1.0)
    y0 = np.floor(ys).astype(np.intp)
    x0 = np.floor(xs).astype(np.intp)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    return (img[np.ix_(y0, x0)] * (1.0 - wy) * (1.0 - wx)
            + img[np.ix_(y0, x1)] * (1.0 - wy) * wx
            + img[np.ix_(y1, x0)] * wy * (1.0 - wx)
            + img[np.ix_(y1, x1)] * wy * wx)


def _iou(b1, b2):
    """普通 (x,y,w,h) IoU（测试辅助，不涉及跨界）。"""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix0 = max(x1, x2)
    iy0 = max(y1, y2)
    ix1 = min(x1 + w1, x2 + w2)
    iy1 = min(y1 + h1, y2 + h2)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def synth_sequence(n_frames=51, W=400, H=300, seed=0, occlude_frames=()):
    """生成合成局部透视图序列（自包含，不依赖其他模块）。

    背景为低频随机场 + 亮度渐变；目标为棋盘/条纹/噪声混合的独特纹理块，
    沿亚像素轨迹平移并缓慢缩放；每帧叠加高斯噪声；可选帧用灰色块完全遮挡目标。
    返回 (frames, gts)：uint8 (H,W,3) RGB 列表与 (x,y,w,h) 浮点 GT 列表。
    """
    rng = np.random.default_rng(seed)
    low = rng.normal(0.0, 1.0, (max(2, H // 8), max(2, W // 8))).astype(np.float32)
    bg = _bilinear_resize(low, H, W) * 14.0
    yy, xx = np.mgrid[0:H, 0:W]
    bg = bg + 80.0 + 0.10 * xx.astype(np.float32) + 0.06 * yy.astype(np.float32)
    tw, th = 44, 34
    ty, tx = np.mgrid[0:th, 0:tw]
    tex = ((((tx // 4 + ty // 4) % 2) * 140.0).astype(np.float32)
           + 55.0 * np.sin(tx / 2.9) + 40.0 * np.cos(ty / 3.7)
           + rng.normal(0.0, 28.0, (th, tw))).astype(np.float32)
    frames, gts = [], []
    for t in range(n_frames):
        scale = 1.0 + 0.004 * t          # 缓变尺度：50 帧内 1.00 -> 1.20
        w_t = tw * scale
        h_t = th * scale
        cx = 80.0 + 2.3 * t              # 亚像素平移
        cy = 140.0 + 30.0 * float(np.sin(t / 8.0))
        img = bg + rng.normal(0.0, 5.0, (H, W)).astype(np.float32)
        twi = max(4, int(round(w_t)))
        thi = max(4, int(round(h_t)))
        patch = _bilinear_resize(tex, thi, twi)
        x0 = int(round(cx - w_t / 2.0))
        y0 = int(round(cy - h_t / 2.0))
        img[y0:y0 + thi, x0:x0 + twi] = patch
        if t in occlude_frames:
            ow = int(round(w_t * 1.5))
            oh = int(round(h_t * 1.5))
            ox = int(round(cx - ow / 2.0))
            oy = int(round(cy - oh / 2.0))
            img[oy:oy + oh, ox:ox + ow] = (
                128.0 + rng.normal(0.0, 2.0, (oh, ow))).astype(np.float32)
        rgb = np.clip(img, 0, 255).astype(np.uint8)
        frames.append(np.repeat(rgb[:, :, None], 3, axis=2))
        gts.append((cx - w_t / 2.0, cy - h_t / 2.0, w_t, h_t))
    return frames, gts


# ---------------------------------------------------------------- 测试用例

def test_factory_and_contract():
    """工厂函数、默认参数、抽象基类与 update 返回契约。"""
    tr = create_tracker('ncc')
    assert isinstance(tr, NCCTracker) and isinstance(tr, BaseTracker)
    tr2 = create_tracker('ncc', lr=0.05)
    assert abs(tr2.lr - 0.05) < 1e-12
    # lightfc：预留接口，必须抛 NotImplementedError 且消息符合契约
    try:
        create_tracker('lightfc')
        raise AssertionError('lightfc 应抛出 NotImplementedError')
    except NotImplementedError as e:
        assert 'LightFC' in str(e) and '接口已预留' in str(e)
    # 未知名称
    try:
        create_tracker('xxx')
        raise AssertionError('未知名称应抛出 ValueError')
    except ValueError:
        pass
    # 默认参数与契约逐字一致
    d = NCCTracker()
    assert d.context == 1.0 and d.scales == (0.98, 1.0, 1.02)
    assert d.lr == 0.02 and d.search_scale == 2.0 and d.template_size == 127
    # 抽象基类不可实例化
    try:
        BaseTracker()
        raise AssertionError('抽象基类不应可实例化')
    except TypeError:
        pass
    # update 返回契约字段与取值范围
    frames, gts = synth_sequence(n_frames=3, seed=7)
    d.init(frames[0], gts[0])
    res = d.update(frames[1])
    assert set(res.keys()) == {'bbox', 'score', 'psr', 'apce'}
    assert len(res['bbox']) == 4
    assert all(np.isfinite(v) for v in res['bbox'])
    assert 0.0 <= res['score'] <= 1.0
    assert np.isfinite(res['psr']) and np.isfinite(res['apce'])
    # 贴边初始化鲁棒性：不崩溃且返回有限框
    edge = NCCTracker()
    edge.init(frames[0], (2.0, 2.0, 44.0, 34.0))
    r2 = edge.update(frames[1])
    assert all(np.isfinite(v) for v in r2['bbox'])
    # init 可重复调用（状态复位）
    d.init(frames[0], gts[0])
    res = d.update(frames[1])
    assert all(np.isfinite(v) for v in res['bbox'])
    print('ok: factory & contract')


def test_no_drift_50_frames():
    """平移+尺度+噪声序列 50 帧不漂移：IoU>0.5 占比 ≥ 95%。"""
    frames, gts = synth_sequence(n_frames=51, seed=1)
    tr = NCCTracker()
    tr.init(frames[0], gts[0])
    ious, scores = [], []
    for t in range(1, 51):
        res = tr.update(frames[t])
        ious.append(_iou(res['bbox'], gts[t]))
        scores.append(res['score'])
    ious = np.asarray(ious)
    frac = float(np.mean(ious > 0.5))
    print(f'[drift] IoU>0.5 占比 {frac:.3f}, min IoU {ious.min():.3f}, '
          f'mean IoU {ious.mean():.3f}, mean score {np.mean(scores):.3f}')
    assert frac >= 0.95, f'50 帧内漂移: IoU>0.5 占比 {frac:.3f} < 0.95'
    assert float(ious.min()) > 0.3, '个别帧定位严重偏离'
    print('ok: no drift 50 frames')


def test_occlusion_score_drop():
    """遮挡帧 score 显著下降，验证丢失判定信号可用。"""
    occ = {30, 31, 32, 33, 34}
    frames, gts = synth_sequence(n_frames=51, seed=2, occlude_frames=occ)
    tr = NCCTracker()
    tr.init(frames[0], gts[0])
    scores = {}
    for t in range(1, 51):
        scores[t] = tr.update(frames[t])['score']
    s_occ = np.array([scores[t] for t in sorted(occ)])
    s_ok = np.array([s for t, s in scores.items() if t not in occ])
    print(f'[occlusion] 正常帧 score 均值 {s_ok.mean():.3f} (min {s_ok.min():.3f}), '
          f'遮挡帧 score 均值 {s_occ.mean():.3f} (max {s_occ.max():.3f})')
    assert s_occ.mean() < 0.45, '遮挡帧 score 未降至常用丢失阈值以下'
    assert s_ok.mean() - s_occ.mean() > 0.25, '遮挡前后 score 差异不显著'
    assert float(s_occ.max()) < float(s_ok.min()), '遮挡帧与正常帧 score 分布重叠'
    print('ok: occlusion score drop')


def test_update_speed():
    """update 单帧耗时：255×255 搜索图中位数 < 50ms。"""
    frames, gts = synth_sequence(n_frames=26, seed=3)
    tr = NCCTracker()
    assert tr.search_size == 255  # 与质量要求中的 255×255 搜索图对应
    tr.init(frames[0], gts[0])
    for t in range(1, 4):  # 预热
        tr.update(frames[t])
    n = 20
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        tr.update(frames[4 + (i % 20)])
        times.append(time.perf_counter() - t0)
    dt = float(np.median(times))  # 中位数抗系统瞬时抖动
    print(f'[speed] update 中位耗时 {dt * 1000:.1f} ms/帧 '
          f'(min {min(times) * 1000:.1f}, max {max(times) * 1000:.1f}, '
          f'搜索图 {tr.search_size}x{tr.search_size})')
    assert dt < 0.05, f'update 过慢: 中位 {dt * 1000:.1f} ms >= 50 ms'
    print('ok: update speed')


def _run_sequence(seed, n_frames=21):
    """完整跑一段序列，返回逐帧结果（并发测试辅助）。"""
    frames, gts = synth_sequence(n_frames=n_frames, seed=seed)
    tr = NCCTracker()
    tr.init(frames[0], gts[0])
    return [tr.update(f) for f in frames[1:]]


def test_concurrent_instances():
    """无全局状态：多实例并发运行与串行结果一致。"""
    ref_a = _run_sequence(11)
    ref_b = _run_sequence(22)
    out = {}
    ta = threading.Thread(target=lambda: out.__setitem__('a', _run_sequence(11)))
    tb = threading.Thread(target=lambda: out.__setitem__('b', _run_sequence(22)))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    for ref, got in zip(ref_a, out['a']):
        assert np.allclose(ref['bbox'], got['bbox'])
        assert np.isclose(ref['score'], got['score'])
    for ref, got in zip(ref_b, out['b']):
        assert np.allclose(ref['bbox'], got['bbox'])
        assert np.isclose(ref['score'], got['score'])
    # 两实例状态相互独立：不同序列结果确实不同
    assert not np.allclose(ref_a[0]['bbox'], ref_b[0]['bbox'])
    print('ok: concurrent instances')


if __name__ == '__main__':
    test_factory_and_contract()
    test_no_drift_50_frames()
    test_occlusion_score_drop()
    test_update_speed()
    test_concurrent_instances()
    print('ALL TRACKER TESTS PASSED')
