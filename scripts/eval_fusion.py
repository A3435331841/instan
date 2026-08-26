#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多 tracker 离线融合评测器。

思路：同一序列分别跑多个 tracker，逐帧按置信度+一致性融合预测。
用法（服务器上）:
    python scripts/eval_fusion.py --data /data/traindata/train \
        --seqs train_real/seq_0002 --trackers sutrack_b224,sutrack_t224 \
        --out /data/runs/fusion_test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.geometry.bfov import BFoV, bfov_from_erp_bbox, erp_bbox_from_bfov
from panotrack.evaluation.metrics import auc, dual_iou, iou_xywh, success_rate

try:
    import cv2 as cv
except ImportError:
    cv = None


def iou(a, b):
    """两个 xywh 框的 IoU。"""
    return iou_xywh(a, b)


def fuse_frame(preds):
    """融合多 tracker 的单帧预测。

    preds: list of (bbox, confidence, tracker_name)
    规则：
      1. 过滤低置信 (<0.1)
      2. 若 ≥2 个预测 IoU>0.5（一致组），取组内最高置信
      3. 若无一致组，取全局最高置信
    """
    valid = [(b, c, n) for b, c, n in preds if c > 0.1 and b[2] > 0]
    if not valid:
        return [0, 0, 0, 0], 0.0, "none"

    if len(valid) == 1:
        return valid[0][0], valid[0][1], valid[0][2]

    # 找最大一致组
    best_group = [valid[0]]
    for i in range(len(valid)):
        group = [valid[i]]
        for j in range(len(valid)):
            if i != j and iou(valid[i][0], valid[j][0]) > 0.5:
                group.append(valid[j])
        if len(group) > len(best_group):
            best_group = group

    # 组内取最高置信
    best = max(best_group, key=lambda x: x[1])
    return best[0], best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seqs", required=True, help="逗号分隔 block/seq")
    ap.add_argument("--trackers", required=True, help="逗号分隔 tracker 名")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    # 这里只做离线融合分析（读取已有评测结果的逐帧输出）
    # 实际的多 tracker 同跑需要在 eval_official.py 基础上扩展
    print("此脚本为融合分析工具。多 tracker 融合需要逐帧预测文件。")
    print("建议方法：")
    print("1. 分别跑各 tracker 保存逐帧 results_erp.txt")
    print("2. 用 fusion_postprocess.py 合并逐帧结果")
    print("3. 评分合并后的结果")


if __name__ == "__main__":
    main()
