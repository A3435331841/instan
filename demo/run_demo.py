# -*- coding: utf-8 -*-
"""端到端 demo：4 个合成场景跑 PanoTracker 并输出验收产物。

运行方式（项目根目录 D:\\instan\\pano360 下）：
    python demo/run_demo.py                 # 跑全部 4 个场景（约 4~6 分钟）
    python demo/run_demo.py pole equator    # 只跑指定场景

每个场景输出到 runs/<scenario>/：
    seq/            合成序列（frames/*.png + gt.txt，可供 CLI 复用）
    results.txt     逐帧跟踪框 x,y,w,h（保留 2 位小数，首帧为初始化 GT 框）
    metrics.json    ope_evaluate 全量指标 + 丢失/找回统计 + 达标判定
    demo.gif        可视化（绿=ok，红=lost，黄=recovered）

验收口径（CONTRACTS.md）：
    equator/crossing SR@0.5 >= 0.9（普通与 dual 双口径）；pole SR@0.5 >= 0.7；
    occlusion 丢失后 10 帧内找回。全部达标退出码 0，否则 1。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panotrack.data.synth import generate_sequence
from panotrack.data.io import load_sequence
from panotrack.data.viz import draw_bbox, save_gif
from panotrack.evaluation.metrics import ope_evaluate
from panotrack.pipeline.pipeline import PanoTracker

SCENARIOS = ('equator', 'crossing', 'pole', 'occlusion')
N_FRAMES, W, H, SEED = 60, 1024, 512, 3
RUNS = Path(__file__).resolve().parents[1] / 'runs'

# 状态颜色：绿 ok / 红 lost / 黄 recovered
_COLOR = {'ok': (0, 200, 0), 'lost': (220, 0, 0), 'recovered': (230, 200, 0)}


def _pass_rule(scenario, m, recovery_latency):
    """按场景判定达标（私有）。"""
    if scenario in ('equator', 'crossing'):
        return m['sr'] >= 0.9 and m['sr_dual'] >= 0.9
    if scenario == 'pole':
        return m['sr'] >= 0.7 and m['sr_dual'] >= 0.7
    if scenario == 'occlusion':
        return recovery_latency is not None and recovery_latency <= 10
    return False


def run_scenario(scenario):
    """单场景端到端：生成序列 -> 跟踪 -> 写 results/metrics/gif -> 打印指标。

    参数: scenario 场景名（equator/crossing/pole/occlusion）。
    返回: (是否达标, 指标 dict)。
    """
    out_dir = RUNS / scenario
    seq_dir = out_dir / 'seq'
    if not (seq_dir / 'gt.txt').exists():
        generate_sequence(seq_dir, n_frames=N_FRAMES, w=W, h=H,
                          scenario=scenario, seed=SEED)
    frames, gt = load_sequence(seq_dir)

    tr = PanoTracker()
    tr.init(frames[0], tuple(gt[0]))
    preds = [tuple(float(v) for v in gt[0])]
    statuses = ['ok']
    t0 = time.perf_counter()
    for i in range(1, len(frames)):
        r = tr.update(frames[i])
        preds.append(r['bbox'])
        statuses.append(r['status'])
    fps = (len(frames) - 1) / max(time.perf_counter() - t0, 1e-9)

    m = ope_evaluate(np.array(preds), np.array(gt, dtype=float), W)
    n_lost = sum(1 for s in statuses if s == 'lost')
    n_rec = sum(1 for s in statuses if s == 'recovered')
    latency = None
    if 'lost' in statuses:
        i0 = statuses.index('lost')
        rec = [i for i in range(i0 + 1, len(statuses)) if statuses[i] == 'recovered']
        if rec:
            latency = rec[0] - i0          # 首个 lost 到首个 recovered 的帧差

    # results.txt：逐帧跟踪框（%.2f，首帧为初始化框）
    with open(out_dir / 'results.txt', 'w', encoding='utf-8') as f:
        for b in preds:
            f.write(f'{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f}\n')

    ok = _pass_rule(scenario, m, latency)
    metrics = {
        'scenario': scenario, 'n_frames': len(frames),
        'sr': m['sr'], 'auc': m['auc'],
        'sr_dual': m['sr_dual'], 'auc_dual': m['auc_dual'],
        'n_lost': n_lost, 'n_recovered': n_rec,
        'recovery_latency': latency, 'fps': fps, 'pass': ok,
    }
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # demo.gif：按状态上色画框
    vis = [draw_bbox(fr, preds[i], color=_COLOR[statuses[i]], thickness=3)
           for i, fr in enumerate(frames)]
    save_gif(vis, out_dir / 'demo.gif', fps=10)

    lat = '-' if latency is None else str(latency)
    print(f'{scenario:10s} sr={m["sr"]:.3f} sr_dual={m["sr_dual"]:.3f} '
          f'auc={m["auc"]:.3f} auc_dual={m["auc_dual"]:.3f} '
          f'lost={n_lost} recovered={n_rec} latency={lat} '
          f'fps={fps:.1f} pass={ok}')
    return ok, metrics


def main(argv):
    """跑指定（或全部）场景，打印达标表并按结果退出。

    参数: argv 场景名列表（空 = 全部 4 场景）。
    返回: None；全部达标 sys.exit(0)，否则 sys.exit(1)。
    """
    scenarios = argv or list(SCENARIOS)
    for sc in scenarios:
        assert sc in SCENARIOS, f'未知场景: {sc!r}，应为 {SCENARIOS}'
    all_ok = True
    for sc in scenarios:
        ok, _ = run_scenario(sc)
        all_ok = all_ok and ok
    print('PASS: 全部场景达标' if all_ok else 'FAIL: 存在未达标场景')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main(sys.argv[1:])
