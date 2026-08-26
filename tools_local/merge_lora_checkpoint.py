#!/usr/bin/env python3
"""将 LoRA 权重合并回原始模型，保存独立 checkpoint 供 eval_official.py 直接加载。

用法:
    cd /data/sutrack_src_20260825/SUTrack
    CUDA_VISIBLE_DEVICES=0 python /data/pano360/tools_local/merge_lora_checkpoint.py \
        --lora-ckpt /data/sutrack_lora_training/sutrack_lora_ep0005.pth \
        --out /data/weights/SUTRACK_b224_lora_ep0005.pth.tar
"""
import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    args = parser.parse_args()

    sutrack_ws = os.environ.get("SUTRACK_WORKSPACE", "/data/sutrack_src_20260825/SUTrack")
    sutrack_ckpt = os.environ.get("SUTRACK_CKPT", "/data/weights/SUTRACK_b224_ep0180.pth.tar")

    sys.path.insert(0, sutrack_ws)
    os.chdir(sutrack_ws)

    import torch
    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.models.sutrack.sutrack import build_sutrack

    # 构建模型
    print("Building SUTrack model...")
    update_config_from_file(os.path.join(sutrack_ws, "experiments", "sutrack", "sutrack_b224.yaml"))
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    sutrack_encoder.is_main_process = lambda: False
    model = build_sutrack(deepcopy(cfg))

    # 加载原始权重
    print(f"Loading original checkpoint: {sutrack_ckpt}")
    ckpt = torch.load(sutrack_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("net", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)

    # 注入 LoRA
    sys.path.insert(0, str(PROJECT_ROOT / "tools_local"))
    from sutrack_lora_setup import apply_lora_to_sutrack
    n_lora = apply_lora_to_sutrack(model.encoder.body, rank=args.rank, alpha=args.alpha)
    print(f"Injected LoRA into {n_lora} layers")

    # 加载 LoRA 权重
    print(f"Loading LoRA checkpoint: {args.lora_ckpt}")
    lora_ckpt = torch.load(args.lora_ckpt, map_location="cpu", weights_only=False)
    lora_state = lora_ckpt.get("net", lora_ckpt)
    if hasattr(lora_state, "state_dict"):
        lora_state = lora_state.state_dict()
    model.load_state_dict(lora_state, strict=False)

    # 合并 LoRA 到原始权重
    print("Merging LoRA weights into original...")
    merged = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # y = W*x + (alpha/r)*B*A*x = (W + (alpha/r)*B*A)*x
            scale = args.alpha / args.rank
            delta = (module.lora_B @ module.lora_A) * scale
            module.original.weight.data += delta.data
            merged += 1
    print(f"Merged {merged} LoRA layers")

    # 保存合并后的 checkpoint（去掉 LoRA 参数，保留原始结构）
    # 用 build_sutrack 重新构建干净模型
    clean_model = build_sutrack(deepcopy(cfg))
    # 从 merged model 提取干净 state_dict（排除 lora_A, lora_B）
    clean_state = {}
    for k, v in model.state_dict().items():
        if 'lora_A' not in k and 'lora_B' not in k:
            clean_state[k] = v
    missing, unexpected = clean_model.load_state_dict(clean_state, strict=False)
    print(f"Clean model: missing={len(missing)}, unexpected={len(unexpected)}")

    # 保存
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'net': clean_model.state_dict()}, str(out_path))
    print(f"Saved merged checkpoint: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
