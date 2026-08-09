# -*- coding: utf-8 -*-
"""GRT-360 Geometry-LoRA MoE —— torch 训练模块。

目标：为 GeometryRouter 学习一组几何专家（gating）与低秩增量（LoRA 风格），
使不同几何情形（近极 / seam / 快速运动 / 高置信）能路由到不同 expert，从而
让跟踪器在 360° ERP 全景下获得更好的角域自适应。

设计（与仓库 numpy 推理端一一对应，自包含部署）：
- 训练用 torch（本模块），推理用 numpy（panotrack.geometry.router.GeometryRouter）。
- GeometryRouter 的 gating MLP 参数 (W1,b1,W2,b2) 由本模块训练并导出为 numpy dict，
  通过 GeometryRouter(params=...) 加载 —— 实现"训练用 torch、推理用 numpy"。
- LoRA 增量 ΔW_k 用于对"模板匹配响应权重"做低秩叠加（MoE 融合），托盘在推理端
  fuse_weights() 中按 α_k 加权叠加。

特征归一化统计（从训练数据估算，用于推理端与 GeometryDescriptor 对齐）：
本模块在 fit 时记录 g_mean/g_std，供 offline export 一并写出，供推理端反归一化
后再送入 GeometryRouter（保证训练/推理描述子同分布）。

依赖说明：torch 仅在本训练模块需要；生产推理 requirements.txt 不含 torch，
Docker 镜像也不含 torch。本模块为**离线训练工具**，不进入运行时推理路径。
"""
import os
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- 网络定义

class GeometryGating(nn.Module):
    """几何路由 gating：g(8,) -> α(K,) softmax。

    与 numpy GeometryRouter 结构一致：h = tanh(g@W1^T+b1)，logits = h@W2^T+b2。
    训练时额外输出归一化统计供导出。
    """

    def __init__(self, in_dim=8, hidden=32, n_experts=3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_experts)
        self.n_experts = n_experts

    def forward(self, g):
        h = torch.tanh(self.fc1(g))
        return self.fc2(h)


class LoRAMoE(nn.Module):
    """Geometry-LoRA MoE 融合头。

    把 gating 权重 α 与 K 个低秩增量 ΔW_k 结合，对基础权重 W0 做加权叠加：
        W_eff = W0 + Σ_k α_k · ΔW_k
    在 torch 端以"逐 expert 的低秩参数 (A_k, B_k)"学习，导出时展开为 ΔW_k = A_k@B_k。

    参数:
        base_dim       基础权重维度（推理端融合对象的维度，如模板匹配窗口展平长度）
        rank           低秩秩大小
        n_experts      专家数量
    """

    def __init__(self, base_dim, rank=8, n_experts=3):
        super().__init__()
        self.base_dim = int(base_dim)
        self.rank = int(rank)
        self.n_experts = int(n_experts)
        # 每个 expert 一个低秩向量分解：ΔW_k = A_k(base_dim,rank) @ b_k(rank,)
        self.A = nn.Parameter(torch.zeros(n_experts, base_dim, rank))
        self.b = nn.Parameter(torch.zeros(n_experts, rank))
        # 零初始化：初始时 ΔW_k = A@b = 0，MoE 退化为纯 base，保证不破坏原跟踪器
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.b)

    def deltas(self):
        """返回 list[torch.Tensor]，每项形状 (base_dim,) 的 ΔW_k。"""
        return [a @ b for a, b in zip(self.A, self.b)]

    def forward(self, alpha):
        """按 gating 权重横向叠加为融合权重（每行一个样本，维度 base_dim）。

        alpha: (B, n_experts)；返回 (B, base_dim)。
        """
        b = alpha.shape[0]
        fused = torch.zeros(b, self.base_dim, device=alpha.device, dtype=alpha.dtype)
        for k in range(self.n_experts):
            dk = self.A[k] @ self.b[k]      # (base_dim,)
            fused = fused + alpha[:, k:k + 1] * dk.unsqueeze(0)
        return fused


class LoRAMoEModel(nn.Module):
    """组合模型：GeometryGating + LoRAMoE，端到端可训练。"""

    def __init__(self, in_dim=8, hidden=32, n_experts=3, base_dim=1024, rank=8,
                 use_geometry_labels=True):
        super().__init__()
        self.gating = GeometryGating(in_dim, hidden, n_experts)
        self.experts = LoRAMoE(base_dim, rank, n_experts)
        self.use_geometry_labels = use_geometry_labels

    def forward(self, g, base_hint=None):
        """g: (B, 8) 几何描述子 -> (alpha, fused)。

        base_hint: 可选，用于辅助损失（见 fit 中 loss 设计）。
        """
        logits = self.gating(g)
        alpha = torch.softmax(logits, dim=-1)
        fused = self.experts.forward(alpha)
        return alpha, fused

    # ---------------------------------------------------------- 导出
    def export_router_params(self):
        """导出 gating 参数为大写数组 dict，供 numpy GeometryRouter 加载。"""
        W1 = self.gating.fc1.weight.detach().cpu().numpy()   # (hidden, in)
        b1 = self.gating.fc1.bias.detach().cpu().numpy()
        W2 = self.gating.fc2.weight.detach().cpu().numpy()   # (n_experts, hidden)
        b2 = self.gating.fc2.bias.detach().cpu().numpy()
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def export_deltas(self):
        """导出 K 个 ΔW_k（展开后的 numpy 数组，每个形状 (base_dim,))。"""
        return [a @ b for a, b in zip(self.experts.A.detach().cpu().numpy(),
                                      self.experts.b.detach().cpu().numpy())]


# ---------------------------------------------------------------- 数据合成

def synth_geometry_data(n=8192, seed=0, n_experts=3):
    """合成几何描述子训练数据。

    按专家预设的几何原型采样，给每个样本打上"最匹配 expert"软标签，用于
    监督 gating 学到有意义的几何分群。描述子为 8 维，范围大致 [0,1]：
        [0] sin(|lat|)     近极风险
        [1] sec 拉伸       极区畸变
        [2] seam 距离      到 seam 的归一化距离
        [3] bbox 面积占比
        [4] 纵横比对数
        [5] 角速度归一化
        [6] 运动不确定性归一化
        [7] 跟踪置信度

    expert 原型（几何语义）：
        E0 近极专家：lat 高、sec 高、seam 中性
        E1 快速运动专家：角速度高、运动不确定性高、置信度中低
        E2 高置信专家：置信度高、seam 远、低速
    返回: X (n,8) 特征, y (n,) 硬标签(0..K-1), proto (K,8) 原型。
    """
    rng = np.random.default_rng(seed)
    proto = np.array([
        [0.85, 0.90, 0.50, 0.30, 0.10, 0.30, 0.30, 0.60],   # E0 近极
        [0.30, 0.35, 0.50, 0.40, 0.20, 0.90, 0.85, 0.40],   # E1 快速
        [0.20, 0.25, 0.90, 0.60, 0.30, 0.15, 0.15, 0.90],   # E2 高置信
    ], dtype=np.float64)[:n_experts]
    k = proto.shape[0]
    labels = rng.integers(0, k, size=n)
    noise = rng.normal(0.0, 0.12, (n, 8))
    X = proto[labels] + noise
    X = np.clip(X, 0.0, 1.0)
    return X.astype(np.float64), labels, proto


# ---------------------------------------------------------------- 训练

def fit(X, y, n_experts=3, base_dim=1024, rank=8, hidden=32,
        epochs=300, lr=1e-3, batch=256, seed=0, device=None,
        gating_weight=1.0, task_weight=1.0, verbose=True):
    """端到端训练 gating + LoRA MoE。

    损失 = 交叉熵(gating, 几何标签) + 任务损失(融合权重与目标 W_eff 的 MSE)。
    任务损失目标：随机生成一个"目标融合权重"，逼近 MoE 融合输出，使 ΔW 有实际
    低秩语义（而非退化到全零）。

    参数:
        X (n,8) 几何描述子；y (n,) 硬标签。
        base_dim 融合权重维度（LoRA 输出维度）。
        返回: (model, history)。history 为 dict(epoch -> loss)。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n = X.shape[0]
    if X.shape[1] != 8:
        raise ValueError(f'描述子应为 8 维，实际 {X.shape[1]}')

    model = LoRAMoEModel(in_dim=8, hidden=hidden, n_experts=n_experts,
                         base_dim=base_dim, rank=rank).to(device)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 任务损失目标：随机生成 target W_eff（与 base_dim 同维），让融合权重逼近它
    target = torch.tensor(np.random.uniform(-1.0, 1.0, (base_dim,)),
                          dtype=torch.float32, device=device)

    history = {}
    nbatch = max(1, int(np.ceil(n / batch)))
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(n)
        el = 0.0
        for bi in range(nbatch):
            idx = perm[bi * batch:(bi + 1) * batch]
            g = torch.tensor(X[idx], device=device)
            lb = torch.tensor(y[idx], device=device)

            alpha, fused = model(g)
            loss_ce = ce(model.gating(g), lb)
            loss_task = mse(fused, target.expand(fused.shape[0], -1))
            loss = gating_weight * loss_ce + task_weight * loss_task

            opt.zero_grad()
            loss.backward()
            opt.step()
            el += float(loss.item()) * len(idx)
        avg = el / n
        history[ep] = avg
        if verbose and (ep % 50 == 0 or ep == 1 or ep == epochs):
            acc = (model.gating(g).argmax(-1) == lb).float().mean().item()
            print(f'[epoch {ep}/{epochs}] loss={avg:.4f} gating_acc={acc:.3f}')
    return model, history


# ---------------------------------------------------------------- 导出

def export_model(model, out_dir, g_mean=None, g_std=None, name='router'):
    """把训练好的模型导出为 numpy dict（router.json 同构），供推理端加载。

    输出到 out_dir:
        {name}_params.npz   含 W1,b1,W2,b2（gating MLP）+ delta0..deltaK-1（ΔW_k）
        {name}_stats.npz    含 g_mean/g_std（描述子归一化统计，可为空）
    返回: (params_path, stats_path)
    """
    os.makedirs(out_dir, exist_ok=True)
    params = model.export_router_params()
    deltas = model.export_deltas()
    np.savez(os.path.join(out_dir, f'{name}_params.npz'),
             W1=params['W1'], b1=params['b1'],
             W2=params['W2'], b2=params['b2'],
             **{f'delta{k}': d for k, d in enumerate(deltas)})
    stats_path = os.path.join(out_dir, f'{name}_stats.npz')
    np.savez(stats_path,
             g_mean=(g_mean if g_mean is not None else np.zeros(8)),
             g_std=(g_std if g_std is not None else np.ones(8)))
    return os.path.join(out_dir, f'{name}_params.npz'), stats_path


def load_params(path):
    """从 export_model 产物加载 gating 参数字典（numpy 版 GeometryRouter 可直接用）。"""
    z = np.load(path)
    keys = ['W1', 'b1', 'W2', 'b2']
    if not all(k in z for k in keys):
        raise ValueError(f'{path} 缺少 gating 参数（{keys}）')
    params = {k: z[k] for k in keys}
    deltas = [z[f'delta{k}'] for k in range(len([n for n in z.files if n.startswith('delta')]))]
    return params, deltas


# ---------------------------------------------------------------- CLI

def _main():
    import argparse
    ap = argparse.ArgumentParser(description='GRT-360 Geometry-LoRA MoE 训练')
    ap.add_argument('--n', type=int, default=8192, help='合成样本数')
    ap.add_argument('--experts', type=int, default=3, help='专家数量 K')
    ap.add_argument('--base-dim', type=int, default=1024, help='融合权重维度')
    ap.add_argument('--rank', type=int, default=8, help='LoRA 秩')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='runs/lora_moe', help='导出目录')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    X, y, proto = synth_geometry_data(args.n, args.seed, args.experts)
    model, hist = fit(X, y, n_experts=args.experts, base_dim=args.base_dim,
                      rank=args.rank, epochs=args.epochs, lr=args.lr, seed=args.seed)
    # 归一化统计
    g_mean = X.mean(0)
    g_std = X.std(0) + 1e-8
    p, s = export_model(model, args.out, g_mean, g_std)
    print('[export] params:', p)
    print('[export] stats :', s)
    print('[proto ] 专家原型:\n', proto)


if __name__ == '__main__':
    _main()