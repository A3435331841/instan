"""metrics 手算验证测试：python tests/test_metrics.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panotrack.evaluation.metrics import (
    iou_xywh, dual_iou, success_rate, auc, ope_evaluate,
)
from panotrack.evaluation.runner import run_tracker_on_sequence
import numpy as np

EPS = 1e-9


def close(a, b):
    assert abs(a - b) < EPS, f'{a} != {b}'


# ---- iou_xywh：手算样例 ----
# 重叠 5x5=25，并集 100+100-25=175
close(iou_xywh((0, 0, 10, 10), (5, 5, 10, 10)), 25.0 / 175.0)
# 完全重合
close(iou_xywh((3, 3, 8, 8), (3, 3, 8, 8)), 1.0)
# 不相交
close(iou_xywh((0, 0, 10, 10), (20, 0, 10, 10)), 0.0)
# 包含关系：交集=小框 25，并集=大框 100
close(iou_xywh((0, 0, 10, 10), (2, 2, 5, 5)), 25.0 / 100.0)
# 零面积框
close(iou_xywh((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

# ---- dual_iou：跨界手算样例 ----
# W=100，b1=(95,10,10,10) 跨右边界（实际覆盖 x∈[95,100)∪[0,5)），b2=(0,10,10,10)
# 普通 IoU=0；b1 左移 100 后 (-5,10,10,10) 与 b2 交 5*10=50，并 100+100-50=150 → 1/3
close(iou_xywh((95, 10, 10, 10), (0, 10, 10, 10)), 0.0)
close(dual_iou((95, 10, 10, 10), (0, 10, 10, 10), 100), 50.0 / 150.0)
# 交换参数对称（b2 左移 100 后与 b1 同样交 50）
close(dual_iou((0, 10, 10, 10), (95, 10, 10, 10), 100), 50.0 / 150.0)
# 不跨界时 dual 不劣于普通（此处等于普通值 64/136）
close(dual_iou((10, 10, 10, 10), (12, 12, 10, 10), 100), 64.0 / 136.0)
# 完全相同的跨界框：平移 ±W 后完全重合 → 1.0
close(dual_iou((95, 10, 10, 10), (95, 10, 10, 10), 100), 1.0)

# ---- success_rate ----
close(success_rate([0.6, 0.4, 0.5], 0.5), 2.0 / 3.0)   # 0.5 计为成功（>=）
close(success_rate([0.1, 0.2]), 0.0)
close(success_rate([]), 0.0)

# ---- auc：21 个阈值点（0, 0.05, ..., 1.0）----
# 全 1：所有阈值都成功 → 1.0
close(auc([1.0, 1.0]), 1.0)
# 恒定 0.31：阈值 0~0.30 共 7 点成功 → 7/21 = 1/3（避开 0.3 的浮点边界）
close(auc([0.31, 0.31, 0.31]), 7.0 / 21.0)
# 全 0：仅阈值 0 成功（0>=0）→ 1/21
close(auc([0.0, 0.0]), 1.0 / 21.0)

# ---- ope_evaluate ----
gt = np.array([[0, 0, 10, 10]] * 5, dtype=float)
pred = gt.copy()
r = ope_evaluate(pred, gt, 100)
assert set(r.keys()) == {'sr', 'auc', 'sr_dual', 'auc_dual', 'ious', 'ious_dual'}
assert len(r['ious']) == 4 and len(r['ious_dual']) == 4   # 首帧不计入
close(r['sr'], 1.0) and close(r['auc'], 1.0)
close(r['sr_dual'], 1.0) and close(r['auc_dual'], 1.0)

# 跨界序列：gt 在左边缘，pred 跨界回绕表示同一目标
gt2 = np.array([[0, 10, 10, 10]] * 4, dtype=float)
pred2 = np.array([[95, 10, 10, 10]] * 4, dtype=float)
r2 = ope_evaluate(pred2, gt2, 100)
close(r2['sr'], 0.0)                 # 普通口径全失败
close(r2['sr_dual'], 0.0)            # dual IoU=1/3 < 0.5
close(r2['auc_dual'], 7.0 / 21.0)    # 阈值 0~0.30 共 7 点
for v in r2['ious_dual']:
    close(v, 1.0 / 3.0)


# ---- runner：假 tracker 验证 OPE 协议 ----
class _EchoTracker:
    """init 记录框，update 每次把框右移 1 像素。"""

    def init(self, image, bbox):
        self.box = tuple(bbox)

    def update(self, image):
        self.box = (self.box[0] + 1.0, *self.box[1:])
        return {'bbox': self.box, 'score': 1.0}


frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]
gt3 = np.array([[1, 2, 3, 3]] * 4, dtype=float)
preds = run_tracker_on_sequence(_EchoTracker(), frames, gt3)
assert preds.shape == (4, 4)
assert (preds[0] == gt3[0]).all()              # 首帧回显 GT
assert (preds[3] == [4, 2, 3, 3]).all()        # 每帧 +1

print('test_metrics: 全部通过')
