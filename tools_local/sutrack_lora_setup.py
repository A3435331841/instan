#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoRA 注入 SUTrack Fast-iTPN backbone 的 stage-3 attention 层。

SUTrack-B224 架构:
  - Fast_iTPN 3-stage pyramid: stage1(dim=128,3conv) + stage2(dim=256,3conv) + stage3(dim=512,24attn)
  - stage3 (blocks 8-31) 每个 Block 有:
      attn.q_proj / k_proj / v_proj (Linear 512→512, no bias, subln=True)
      attn.proj (Linear 512→512, with bias)
      mlp.w1 / w2 / w3 (SwiGLU: 512→1536 / 512→1536 / 1536→512)
  - 只对 stage3 的 attention Linear 注入 LoRA（conv 层不注入）

用法:
  python tools_local/sutrack_lora_setup.py  # 测试注入是否成功
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# UETrack 源码路径（包含 Fast_iTPN 定义）
UETRACK_ROOT = str(Path(__file__).resolve().parents[1] / "artifacts" / "server_snapshot" / "upstream" / "UETrack")
# SUTrack 权重
SUTRACK_CKPT = str(Path(__file__).resolve().parents[1] / "artifacts" / "hf" / "sutrack_b224" / "SUTRACK_ep0180.pth.tar")


class LoraLinear(nn.Module):
    """LoRA 包装的 nn.Linear: y = frozen_W·x + scale·B(A(x))"""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        # 正常赋值，让 nn.Module 注册为子模块（确保 .cuda() 能移动）
        self.original = original
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return out + lora_out

    def __getattr__(self, name):
        """代理原始 Linear 的属性（weight, bias, in_features 等）。"""
        # 优先让 nn.Module 找 _parameters / _buffers / _modules
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        # 再代理到原始 Linear（从 _modules 中取）
        try:
            orig = super(nn.Module, self).__getattribute__('_modules')['original']
            return getattr(orig, name)
        except (KeyError, AttributeError):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


def apply_lora_to_sutrack(model, rank=8, alpha=16.0, stage3_start=8):
    """递归遍历 SUTrack Fast_iTPN 模型，对 stage3 的 attention Linear 注入 LoRA。

    目标层:
      - blocks.{stage3_start:}.attn.q_proj (subln=True 时存在)
      - blocks.{stage3_start:}.attn.k_proj
      - blocks.{stage3_start:}.attn.v_proj
      - blocks.{stage3_start:}.attn.proj
      - blocks.{stage3_start:}.mlp.w1 / w2 / w3 (SwiGLU)

    返回注入的 LoRA Linear 数量。
    """
    count = 0

    def _inject(module, name_prefix=""):
        nonlocal count
        for name, child in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name

            # 检查是否是 stage3 的 Block（有 attn 子模块）
            if hasattr(child, 'attn') and child.attn is not None:
                attn = child.attn
                # 注入 attention 的 q/k/v/proj
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'proj']:
                    if hasattr(attn, proj_name) and isinstance(getattr(attn, proj_name), nn.Linear):
                        setattr(attn, proj_name, LoraLinear(getattr(attn, proj_name), rank=rank, alpha=alpha))
                        count += 1

                # 注入 MLP (SwiGLU: w1, w2, w3)
                if hasattr(child, 'mlp') and child.mlp is not None:
                    mlp = child.mlp
                    for w_name in ['w1', 'w2', 'w3']:
                        if hasattr(mlp, w_name) and isinstance(getattr(mlp, w_name), nn.Linear):
                            setattr(mlp, w_name, LoraLinear(getattr(mlp, w_name), rank=rank, alpha=alpha))
                            count += 1
            else:
                _inject(child, full_name)

    _inject(model)
    return count


def load_sutrack_model(sutrack_workspace=None, checkpoint=None, config_name="sutrack_b224"):
    """加载 SUTrack 模型并注入 LoRA。

    Args:
        sutrack_workspace: SUTrack 源码路径（服务器上默认 /data/sutrack_src_20260825/SUTrack）
        checkpoint: SUTrack checkpoint 路径
        config_name: 配置名（sutrack_b224 或 sutrack_t224）

    Returns:
        (model, cfg, lora_params, head_params)
    """
    if sutrack_workspace is None:
        sutrack_workspace = UETRACK_ROOT
    if checkpoint is None:
        checkpoint = SUTRACK_CKPT

    sys.path.insert(0, sutrack_workspace)

    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.test.tracker.sutrack import SUTRACK as SUTRACK_MODEL

    # 加载配置
    config_path = Path(sutrack_workspace) / "experiments" / "sutrack" / f"{config_name}.yaml"
    update_config_from_file(str(config_path))

    # 禁用 iTPN 预训练（我们加载完整 checkpoint）
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""

    # 构建模型
    from copy import deepcopy
    cfg_copy = deepcopy(cfg)
    model = SUTRACK_MODEL(cfg_copy)

    # 加载权重
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("net", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded SUTrack checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")

    # 注入 LoRA（只对 encoder.stage3 的 attention/MLP 层）
    n_lora = apply_lora_to_sutrack(model.encoder.body, rank=8, alpha=16.0, stage3_start=8)
    print(f"Injected LoRA into {n_lora} Linear layers in stage3 attention/MLP")

    # 冻结非 LoRA 参数
    lora_params, head_params = [], []
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            lora_params.append(param)
        elif "decoder" in name or "task_decoder" in name:
            param.requires_grad_(True)
            head_params.append(param)
        else:
            param.requires_grad_(False)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: total={total/1e6:.1f}M, trainable={trainable/1e6:.1f}M "
          f"(LoRA={sum(p.numel() for p in lora_params)/1e3:.1f}K, head={sum(p.numel() for p in head_params)/1e6:.1f}M)")
    print(f"Trainable ratio: {trainable/total*100:.2f}%")

    return model, cfg, lora_params, head_params


if __name__ == "__main__":
    print("=== SUTrack LoRA Setup Test ===")
    model, cfg, lora_p, head_p = load_sutrack_model()
    print("\nSUTrack LoRA setup complete!")

    # 显示 LoRA 层名称
    print("\nLoRA layers:")
    for name, param in model.named_parameters():
        if "lora_" in name:
            print(f"  {name}: {param.shape}")
