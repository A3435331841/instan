# -*- coding: utf-8 -*-
"""GRT-360 Geometry Router：把几何描述子映射为 expert 选择权重（gating）。

α = softmax(MLP(g_t)) ∈ R^K，K 为 Geometry-LoRA experts 数量。

几何路由器是 GRT-360 的核心创新点之一：它读取 8 维几何描述子 g_t（近极风险、
极区 sec 拉伸、seam 距离、角面积、纵横比、角速度、运动不确定性、跟踪置信度），
经由一个 2 层 MLP 输出各 expert 的软权重 α。在推理时，融合专家权重按下式叠加
到基础权重上（MoE ／ 低秩适配）：

    W_eff = W_0 + Σ_k α_k · ΔW_k

由于本仓库跟踪器以 numpy / 模板匹配为主（无逐层可训练线性层），Router 的
gating 输出 α 实际用于对多模板匹配响应做加权融合（soft selection），并可在
torch 训练脚本中导出 (W1,b1,W2,b2) 参数后由本类加载，实现"训练用 torch、
推理用 numpy"的自包含部署。
"""
import numpy as np


class GeometryRouter:
    """从 8 维几何描述子输出 K 维 softmax gating 权重。"""

    def __init__(self, n_experts=3, seed=0, params=None):
        """创建几何路由器。

        参数: n_experts 专家数量 K（输出维度）；seed 随机种子（初始权重）；
              params 可选预训练参数字典（torch 训练后导出）：
              {'W1': (H,8), 'b1': (H,), 'W2': (K,H), 'b2': (K,)}。
        返回: None
        """
        self.n_experts = int(n_experts)
        self.params = params if params is not None else self._init_params(seed)

    def _init_params(self, seed):
        rng = np.random.default_rng(seed)
        h = 32
        return {
            'W1': rng.normal(0.0, 0.5, (h, 8)).astype(np.float64),
            'b1': rng.normal(0.0, 0.1, (h,)).astype(np.float64),
            'W2': rng.normal(0.0, 0.5, (self.n_experts, h)).astype(np.float64),
            'b2': rng.normal(0.0, 0.1, (self.n_experts,)).astype(np.float64),
        }

    def set_params(self, params):
        """加载预训练权重（torch 导出 dict）。返回 self 便于链式调用。"""
        self.params = params
        return self

    def forward(self, g):
        """前向：g (8,) 描述子 -> α (K,) softmax 权重（和为 1）。

        参数: g 8 维几何描述子（GeometryDescriptor 输出）。
        返回: (K,) float gating 权重，Σα = 1。
        """
        g = np.asarray(g, dtype=np.float64).reshape(-1)
        if g.shape[0] != 8:
            raise ValueError(f'描述子应为 8 维，实际 {g.shape[0]}')
        p = self.params
        h = np.tanh(g @ p['W1'].T + p['b1'])
        logits = h @ p['W2'].T + p['b2']
        return self._softmax(logits)

    def __call__(self, g):
        return self.forward(g)

    @staticmethod
    def _softmax(logits):
        logits = np.asarray(logits, dtype=np.float64)
        z = logits - logits.max()
        e = np.exp(z)
        s = e.sum()
        if s < 1e-12:
            return np.full_like(logits, 1.0 / logits.size)
        return e / s

    def pick_expert(self, g):
        """返回 gating 权重最大的 expert 下标（仅用于可视化/冷启动）。"""
        return int(np.argmax(self.forward(g)))

    def fuse_weights(self, g, base_weights, deltas):
        """MoE 低秩叠加：W_eff = W0 + Σ_k α_k(g) · ΔW_k。

        参数: g 当前几何描述子 (8,)；base_weights 基础权重（标量或任意形状数组）；
              deltas list[ΔW_k]，长度需等于 n_experts，每项形状与 base_weights 一致。
        返回: 融合后的权重（形状同 base_weights）。
        """
        if len(deltas) != self.n_experts:
            raise ValueError(f'需要 {self.n_experts} 个 expert 增量，'
                             f'实际 {len(deltas)}')
        alpha = self.forward(g)
        out = np.asarray(base_weights, dtype=np.float64).copy()
        for k, d in enumerate(deltas):
            out = out + alpha[k] * np.asarray(d)
        return out

    # ------------------------------------------------------------ 便捷
    def describe(self):
        """返回参数字典（供 torch 训练导出 / 序列化）。"""
        return {k: np.asarray(v, dtype=np.float64) for k, v in self.params.items()}