# -*- coding: utf-8 -*-
"""test_lora_moe —— Geometry-LoRA MoE torch 训练模块测试。

运行: python tests/test_lora_moe.py
依赖: torch（仅训练端需要；生产推理 requirements 不含）。
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panotrack.geometry.train_lora_moe import (  # noqa: E402
    LoRAMoEModel, LoRAMoE, GeometryGating,
    synth_geometry_data, fit, export_model, load_params,
)
from panotrack.geometry.router import GeometryRouter  # noqa: E402


PASS = 0


def check(name, cond):
    global PASS
    status = 'PASS' if cond else 'FAIL'
    print(f'  [{status}] {name}')
    PASS += 1
    if not cond:
        raise AssertionError(f'FAILED: {name}')


def test_gating_forward():
    """gating 输出 K 维 softmax，和为 1。"""
    m = GeometryGating(in_dim=8, hidden=32, n_experts=3)
    g = torch.zeros(4, 8)
    logits = m(g)
    alpha = torch.softmax(logits, dim=-1)
    check('gating 输出形状 (4,3)', tuple(alpha.shape) == (4, 3))
    check('softmax 和为 1', np.allclose(alpha.detach().sum(-1).numpy(), 1.0))


def test_lora_zero_init():
    """LoRA B=0 初始化 → 融合权重应≈0（不破坏原跟踪器）。"""
    m = LoRAMoE(base_dim=1024, rank=8, n_experts=3)
    alpha = torch.full((1, 3), 1.0 / 3.0)
    fused = m.forward(alpha)
    check('零初始化融合权重≈0', torch.allclose(fused, torch.zeros_like(fused), atol=1e-6))


def test_synth_data_shape():
    X, y, proto = synth_geometry_data(n=1000, seed=0, n_experts=3)
    check('X 形状 (1000,8)', X.shape == (1000, 8))
    check('y 在 [0,2]', set(np.unique(y)).issubset({0, 1, 2}))
    check('proto 形状 (3,8)', proto.shape == (3, 8))


def test_fit_and_export():
    """端到端训练 + 导出 + numpy GeometryRouter 加载。"""
    X, y, _ = synth_geometry_data(n=512, seed=1, n_experts=3)
    model, hist = fit(X, y, n_experts=3, base_dim=64, rank=4, epochs=20,
                      lr=1e-2, seed=0, verbose=False)
    check('训练产生历史', len(hist) == 20)
    # 损失应下降
    first = hist[1]
    last = hist[20]
    check(f'损失下降 ({first:.3f}->{last:.3f})', last < first)

    # 导出到临时目录
    with tempfile.TemporaryDirectory() as td:
        g_mean = X.mean(0)
        g_std = X.std(0) + 1e-8
        p, s = export_model(model, td, g_mean, g_std, name='router')
        check('导出文件存在', os.path.exists(p) and os.path.exists(s))

        # numpy GeometryRouter 加载 gating 参数
        params, deltas = load_params(p)
        router = GeometryRouter(n_experts=3, params=params)
        check('router 参数维度 W1 (32,8)', router.params['W1'].shape == (32, 8))
        check('deltas 数量=3', len(deltas) == 3)
        check('delta 形状 (64,)', deltas[0].shape == (64,))

        # 描述子 gating 权重和为 1
        g = X[0]
        alpha = router.forward(g)
        check('numpy gating 和为 1', np.isclose(alpha.sum(), 1.0))

        # MoE 融合：给定 base + delta，验证 alpha 加权叠加
        base = np.ones(64)
        fused = router.fuse_weights(g, base, deltas)
        expected = base.copy()
        for k, d in enumerate(deltas):
            expected = expected + alpha[k] * d
        check('fuse_weights 与手算一致', np.allclose(fused, expected))


def test_export_roundtrip_equals_torch():
    """导出参数与 torch 前向对齐：同一描述子，numpy 与 torch gating 输出一致。"""
    X, y, _ = synth_geometry_data(n=64, seed=2, n_experts=3)
    model, _ = fit(X, y, n_experts=3, base_dim=32, rank=2, epochs=5,
                   lr=1e-2, seed=0, verbose=False)
    params = model.export_router_params()
    router = GeometryRouter(n_experts=3, params=params)

    g = torch.tensor(X[:5], dtype=torch.float32)
    torch_alpha = torch.softmax(model.gating(g), dim=-1).detach().numpy()
    for i in range(5):
        np_alpha = router.forward(X[i])
        check(f'g[{i}] torch/numpy 对齐', np.allclose(np_alpha, torch_alpha[i], atol=1e-5))


def test_cli_help():
    """CLI 可运行（--help 不报错）。"""
    import subprocess
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'panotrack', 'geometry', 'train_lora_moe.py')
    r = subprocess.run([sys.executable, script, '--help'],
                       capture_output=True, text=True)
    check('CLI --help 退出码 0', r.returncode == 0)


def main():
    print('=== test_lora_moe ===')
    try:
        import torch  # noqa: F401
        globals()['torch'] = torch
    except Exception as e:
        print('SKIP: torch 不可用（仅训练端需要，生产推理不含）:', e)
        return 0
    for fn in [test_synth_data_shape, test_gating_forward, test_lora_zero_init,
               test_fit_and_export, test_export_roundtrip_equals_torch, test_cli_help]:
        fn()
    print(f'\nPASS {PASS} checks')
    return 0


if __name__ == '__main__':
    sys.exit(main())