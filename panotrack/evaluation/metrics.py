"""评测指标：IoU / dual IoU / 成功率 / AUC / OPE 评估。"""
import numpy as np


def iou_xywh(b1, b2):
    """普通轴对齐框 IoU（不处理跨界）。

    参数：b1, b2 —— (x, y, w, h) 像素框。
    返回：float，IoU ∈ [0, 1]。
    """
    x1, y1, w1, h1 = (float(v) for v in b1)
    x2, y2, w2, h2 = (float(v) for v in b2)
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def dual_iou(b1, b2, width):
    """360VOT dual IoU：b1 水平平移 ±width 后与 b2 的 IoU 取最大（处理跨界）。

    参数：b1, b2 —— (x, y, w, h) 像素框；width —— ERP 图像宽 W。
    返回：float，三种平移（0, -W, +W）下的最大 IoU。
    """
    best = iou_xywh(b1, b2)
    for shift in (-float(width), float(width)):
        b1s = (b1[0] + shift, b1[1], b1[2], b1[3])
        best = max(best, iou_xywh(b1s, b2))
    return best


def success_rate(ious, thr=0.5):
    """成功率：IoU >= thr 的帧占比。

    参数：ious —— 逐帧 IoU 序列；thr —— 成功阈值。
    返回：float ∈ [0, 1]。
    """
    arr = np.asarray(ious, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= thr))


def auc(ious):
    """AUC：阈值 0~1 步长 0.05（共 21 点）的 SR 均值。

    参数：ious —— 逐帧 IoU 序列。
    返回：float ∈ [0, 1]。
    """
    thrs = np.linspace(0.0, 1.0, 21)
    return float(np.mean([success_rate(ious, t) for t in thrs]))


def ope_evaluate(pred, gt, width):
    """OPE 评估：逐帧普通/dual IoU 及对应 SR、AUC。首帧（初始化帧）不计入统计。

    参数：pred, gt —— (N, 4) 预测与真值框；width —— ERP 图像宽 W。
    返回：dict {'sr','auc','sr_dual','auc_dual','ious','ious_dual'}，
         ious/ious_dual 为长度 N-1 的 float 列表，其余为 float。
    """
    pred = np.asarray(pred, dtype=float)
    gt = np.asarray(gt, dtype=float)
    ious = [iou_xywh(p, g) for p, g in zip(pred[1:], gt[1:])]
    ious_dual = [dual_iou(p, g, width) for p, g in zip(pred[1:], gt[1:])]
    return {
        'sr': success_rate(ious),
        'auc': auc(ious),
        'sr_dual': success_rate(ious_dual),
        'auc_dual': auc(ious_dual),
        'ious': ious,
        'ious_dual': ious_dual,
    }
