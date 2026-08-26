#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SUTrack-B224 LoRA 微调独立训练脚本。

与 lora_train.py（ODTrack）不同，本脚本直接调用 SUTrack 的 forward 接口，
避免重新实现其复杂的 encoder-decoder pipeline。

用法（服务器上）:
    cd /data/sutrack_src_20260825/SUTrack
    CUDA_VISIBLE_DEVICES=0 nohup python /data/pano360/tools_local/sutrack_lora_train.py \
        > /data/sutrack_lora_train.log 2>&1 < /dev/null &

前置条件:
    1. SUTrack 源码在 /data/sutrack_src_20260825/SUTrack
    2. SUTrack-B224 权重在 /data/weights/SUTRACK_b224_ep0180.pth.tar
    3. 官方训练集已转为 GOT-10k 格式: /data/finetune/official_got10k
"""
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn

# === 路径配置（服务器上修改） ===
SUTRACK_WORKSPACE = os.environ.get("SUTRACK_WORKSPACE", "/data/sutrack_src_20260825/SUTrack")
SUTRACK_CKPT = os.environ.get("SUTRACK_CKPT", "/data/weights/SUTRACK_b224_ep0180.pth.tar")
DATA_ROOT = os.environ.get("FINETUNE_DATA", "/data/finetune/official_got10k")
OUTPUT_DIR = os.environ.get("LORA_OUTPUT", "/data/sutrack_lora_training")
SPLIT_FILE = os.environ.get("SPLIT_FILE", "")

# === 超参 ===
LR = 5e-5              # LoRA lr（比 ODTrack 的 1e-4 更保守）
HEAD_LR = 1e-5         # decoder head lr
EPOCHS = 5
BATCH_SIZE = 4
SEARCH_SIZE = 224       # SUTrack-B224 默认
TEMPLATE_SIZE = 112     # SUTrack-B224 默认
SEARCH_FACTOR = 4.0
TEMPLATE_FACTOR = 2.0
LORA_RANK = 8
LORA_ALPHA = 16.0
WARMUP_EPOCHS = 1
SAVE_EVERY = 1          # 每 epoch 保存（快速验证）
SAMPLES_PER_EPOCH = 3000


def setup_lora():
    """构建带 LoRA 的 SUTrack 模型。"""
    sys.path.insert(0, SUTRACK_WORKSPACE)
    os.chdir(SUTRACK_WORKSPACE)

    from lib.config.sutrack.config import cfg, update_config_from_file
    import lib.models.sutrack.encoder as sutrack_encoder
    from lib.models.sutrack.sutrack import build_sutrack

    config_path = os.path.join(SUTRACK_WORKSPACE, "experiments", "sutrack", "sutrack_b224.yaml")
    update_config_from_file(config_path)
    cfg.MODEL.ENCODER.PRETRAIN_TYPE = ""
    sutrack_encoder.is_main_process = lambda: False

    model = build_sutrack(cfg)

    # 加载权重
    ckpt = torch.load(SUTRACK_CKPT, map_location="cpu", weights_only=False)
    state = ckpt.get("net", ckpt)
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded SUTrack: missing={len(missing)}, unexpected={len(unexpected)}")

    # 注入 LoRA
    sys.path.insert(0, os.path.dirname(__file__))
    from sutrack_lora_setup import apply_lora_to_sutrack
    n_lora = apply_lora_to_sutrack(model.encoder.body, rank=LORA_RANK, alpha=LORA_ALPHA, stage3_start=8)
    print(f"Injected LoRA into {n_lora} layers")

    # 冻结
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
    print(f"Total={total/1e6:.1f}M, Trainable={trainable/1e6:.1f}M ({trainable/total*100:.2f}%)")

    model.cuda()
    return model, cfg, lora_params, head_params


def sample_training_pair(seq_dir, rng):
    """从子序列中随机采样 template + search 帧对。"""
    gt_file = seq_dir / "groundtruth.txt"
    lines = gt_file.read_text().strip().splitlines()
    boxes = []
    for ln in lines:
        f = [float(v) for v in ln.replace(",", " ").split()]
        boxes.append(f)

    n = len(boxes)
    if n < 3:
        return None

    # GOT-10k 格式帧从 1 开始（00000001.jpg）
    search_idx = rng.randint(2, n - 1)
    max_gap = min(search_idx - 1, 50)
    template_idx = max(1, search_idx - rng.randint(1, max_gap))

    t_img = cv.imread(str(seq_dir / f"{template_idx:08d}.jpg"))
    s_img = cv.imread(str(seq_dir / f"{search_idx:08d}.jpg"))
    if t_img is None or s_img is None:
        return None
    t_img = cv.cvtColor(t_img, cv.COLOR_BGR2RGB)
    s_img = cv.cvtColor(s_img, cv.COLOR_BGR2RGB)

    return {
        "template_img": t_img,
        "search_img": s_img,
        "template_box": boxes[template_idx],
        "search_box": boxes[search_idx],
        "search_idx": search_idx,
    }


def crop_resize(img, box, out_size, factor=1.0):
    """以 box 为中心裁剪 factor 倍区域并 resize 到 out_size。"""
    H, W = img.shape[:2]
    cx = box[0] + box[2] / 2
    cy = box[1] + box[3] / 2
    crop_size = max(box[2], box[3]) * factor
    crop_size = max(crop_size, 16)

    x1 = int(max(0, cx - crop_size / 2))
    y1 = int(max(0, cy - crop_size / 2))
    x2 = int(min(W, cx + crop_size / 2))
    y2 = int(min(H, cy + crop_size / 2))

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8), (0, 0, 1.0)

    scale = out_size / max(patch.shape[:2])
    resized = cv.resize(patch, (out_size, out_size))
    return resized, (x1, y1, scale)


def normalize(img):
    """ImageNet 归一化 + HWC->CHW。"""
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def train():
    device = "cuda"
    rng = random.Random(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Setting up SUTrack model with LoRA...")
    model, cfg, lora_params, head_params = setup_lora()

    # 检查模型结构
    print(f"\nModel encoder body type: {type(model.encoder.body).__name__}")
    print(f"Number of blocks: {len(model.encoder.body.blocks)}")
    # 打印 stage3 的 LoRA 层
    lora_count = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_count += 1
    print(f"LoRA parameters: {lora_count} tensors")

    # 优化器
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": LR},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 数据
    data_root = Path(DATA_ROOT)
    if not data_root.is_dir():
        print(f"[error] Data root {DATA_ROOT} does not exist!")
        print("Run prepare_finetune_data.py first:")
        print("  python scripts/prepare_finetune_data.py --data /data/traindata/train --out /data/finetune/official_got10k")
        return

    # 读取 split 文件过滤序列
    if SPLIT_FILE and Path(SPLIT_FILE).is_file():
        allowed = set(ln.strip() for ln in Path(SPLIT_FILE).read_text().splitlines() if ln.strip())
        seq_dirs = sorted([d for d in data_root.iterdir() if d.is_dir() and d.name in allowed])
    else:
        seq_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    print(f"Training sequences: {len(seq_dirs)}")

    if len(seq_dirs) == 0:
        print("[error] No training sequences found!")
        return

    # 损失函数
    l1_loss = nn.L1Loss()

    def giou_loss(pred_box, gt_box):
        """简化 GIoU loss。"""
        p = torch.stack([pred_box[..., 0] - pred_box[..., 2]/2,
                         pred_box[..., 1] - pred_box[..., 3]/2,
                         pred_box[..., 0] + pred_box[..., 2]/2,
                         pred_box[..., 1] + pred_box[..., 3]/2], dim=-1)
        g = torch.stack([gt_box[..., 0] - gt_box[..., 2]/2,
                         gt_box[..., 1] - gt_box[..., 3]/2,
                         gt_box[..., 0] + gt_box[..., 2]/2,
                         gt_box[..., 1] + gt_box[..., 3]/2], dim=-1)
        xi1 = torch.max(p[..., 0], g[..., 0])
        yi1 = torch.max(p[..., 1], g[..., 1])
        xi2 = torch.min(p[..., 2], g[..., 2])
        yi2 = torch.min(p[..., 3], g[..., 3])
        inter = (xi2 - xi1).clamp(min=0) * (yi2 - yi1).clamp(min=0)
        p_area = (p[..., 2] - p[..., 0]) * (p[..., 3] - p[..., 1])
        g_area = (g[..., 2] - g[..., 0]) * (g[..., 3] - g[..., 1])
        union = p_area + g_area - inter
        iou = inter / (union + 1e-8)
        ci1 = torch.min(p[..., 0], g[..., 0])
        ci2 = torch.max(p[..., 2], g[..., 2])
        cj1 = torch.min(p[..., 1], g[..., 1])
        cj2 = torch.max(p[..., 3], g[..., 3])
        c_area = (ci2 - ci1) * (cj2 - cj1) + 1e-8
        return 1.0 - iou + (c_area - union) / c_area

    print(f"Starting training: {EPOCHS} epochs, {len(seq_dirs)} seqs, {SAMPLES_PER_EPOCH} samples/epoch")
    global_step = 0

    for epoch in range(EPOCHS):
        rng.shuffle(seq_dirs)
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        # 采样
        batch_data = []
        attempts = 0
        while len(batch_data) < SAMPLES_PER_EPOCH and attempts < SAMPLES_PER_EPOCH * 3:
            seq_dir = rng.choice(seq_dirs)
            pair = sample_training_pair(seq_dir, rng)
            if pair is not None:
                batch_data.append(pair)
            attempts += 1

        n_batches_target = len(batch_data) // BATCH_SIZE

        for batch_start in range(0, len(batch_data) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = batch_data[batch_start:batch_start + BATCH_SIZE]

            template_imgs = []
            search_imgs = []
            template_annos = []
            gt_centers = []
            gt_sizes = []

            for sample in batch:
                t_resized, (tx, ty, tscale) = crop_resize(sample["template_img"],
                                           sample["template_box"],
                                           TEMPLATE_SIZE, TEMPLATE_FACTOR)
                s_resized, (sx, sy, sscale) = crop_resize(sample["search_img"],
                                                          sample["search_box"],
                                                          SEARCH_SIZE, SEARCH_FACTOR)

                t_norm = normalize(t_resized)  # (3, H, W)
                s_norm = normalize(s_resized)  # (3, H, W)
                template_imgs.append(t_norm)
                search_imgs.append(s_norm)

                # 模板框归一化坐标（用于 create_mask 生成前景/背景 mask）
                tbox = sample["template_box"]
                t_anno_cx = (tbox[0] + tbox[2]/2 - tx) * tscale / TEMPLATE_SIZE
                t_anno_cy = (tbox[1] + tbox[3]/2 - ty) * tscale / TEMPLATE_SIZE
                t_anno_w = tbox[2] * tscale / TEMPLATE_SIZE
                t_anno_h = tbox[3] * tscale / TEMPLATE_SIZE
                template_annos.append([t_anno_cx - t_anno_w/2, t_anno_cy - t_anno_h/2,
                                       t_anno_w, t_anno_h])

                # 搜索框 GT 归一化坐标
                box = sample["search_box"]
                gt_cx = (box[0] + box[2]/2 - sx) * sscale / SEARCH_SIZE
                gt_cy = (box[1] + box[3]/2 - sy) * sscale / SEARCH_SIZE
                gt_w = box[2] * sscale / SEARCH_SIZE
                gt_h = box[3] * sscale / SEARCH_SIZE
                gt_centers.append([gt_cx, gt_cy])
                gt_sizes.append([gt_w, gt_h])

            # SUTrack-B224 patch_embed 期望 6ch 输入
            # 训练时 template+search 拼接为 6ch；推理时复制自身为 6ch
            template_tensor = torch.stack(template_imgs).to(device)  # (B, 3, 112, 112)
            search_tensor = torch.stack(search_imgs).to(device)      # (B, 3, 224, 224)
            # 复制为 6ch（与推理侧一致）
            template_tensor = torch.cat((template_tensor, template_tensor), dim=1)  # (B, 6, 112, 112)
            search_tensor = torch.cat((search_tensor, search_tensor), dim=1)        # (B, 6, 224, 224)
            template_anno = torch.tensor(template_annos, dtype=torch.float32).to(device)
            gt_center = torch.tensor(gt_centers, dtype=torch.float32).to(device)
            gt_size = torch.tensor(gt_sizes, dtype=torch.float32).to(device)

            # 前向：使用 SUTrack 的 forward 接口
            with torch.amp.autocast("cuda", enabled=True):
                try:
                    # SUTrack forward: mode="encoder" -> xz (feature list)
                    # 然后 mode="decoder" -> (pred_dict, task_pred)
                    out_enc = model(
                        template_list=[template_tensor],
                        search_list=[search_tensor],
                        template_anno_list=[template_anno],
                        text_src=None,
                        task_index=0,
                        mode="encoder"
                    )
                    pred_dict, task_pred = model(feature=out_enc, mode="decoder")

                    # 解码预测
                    pred_boxes = pred_dict["pred_boxes"]  # (B, 1, 4) in [0,1] cxcywh
                    pred_center = pred_boxes[:, 0, :2]    # (B, 2) — 取所有 batch
                    pred_size = pred_boxes[:, 0, 2:]      # (B, 2)

                    # 损失
                    loss_center = l1_loss(pred_center, gt_center)
                    loss_size = l1_loss(pred_size, gt_size)
                    pred_box = torch.cat([pred_center, pred_size], dim=-1)
                    gt_box = torch.cat([gt_center, gt_size], dim=-1)
                    loss_giou = giou_loss(pred_box, gt_box).mean()

                    loss = 5.0 * loss_center + 5.0 * loss_size + 2.0 * loss_giou

                except Exception as e:
                    print(f"  Forward failed: {e}")
                    continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.1)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} "
              f"batches={n_batches} time={elapsed:.0f}s "
              f"lr_lora={optimizer.param_groups[0]['lr']:.2e}")

        # 保存
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == EPOCHS - 1:
            ckpt_out = os.path.join(OUTPUT_DIR, f"sutrack_lora_ep{epoch+1:04d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": avg_loss,
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
            }, ckpt_out)
            print(f"  saved: {ckpt_out}")

    print("Training complete!")


if __name__ == "__main__":
    train()
