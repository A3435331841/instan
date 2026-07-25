"""OPE 运行器：驱动跟踪器在序列上逐帧运行。"""
import numpy as np


def run_tracker_on_sequence(tracker, frames, gt):
    """OPE 协议：用 gt[0] 初始化 tracker，逐帧 update，收集预测框。

    参数：tracker —— 具有 init(image, bbox) / update(image)->dict(含 'bbox') 的对象；
         frames —— 图像列表，(H,W,3) uint8；gt —— (N,4) 真值框。
    返回：np.ndarray (N,4)，第 0 行为 gt[0]，其余为 tracker 输出 bbox。
    """
    gt = np.asarray(gt, dtype=float)
    if len(frames) != len(gt):
        raise ValueError('frames 与 gt 帧数不一致')
    tracker.init(frames[0], tuple(gt[0]))
    preds = np.zeros((len(frames), 4), dtype=float)
    preds[0] = gt[0]
    for i in range(1, len(frames)):
        out = tracker.update(frames[i])
        preds[i] = np.asarray(out['bbox'], dtype=float)
    return preds
