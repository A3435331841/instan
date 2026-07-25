# -*- coding: utf-8 -*-
"""YOLOv8n ONNX 检测器 + 简单关联的 360° ERP 跟踪器。

方向 A：检测器辅助跟踪
- 用 YOLOv8n ONNX + cv2.dnn 在整幅 ERP 帧上做检测
- 帧间用 IoU + 中心点距离做简单关联
- 不依赖 BFoV 切图，避免小目标被稀释
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class YOLOv8nDetector:
    """YOLOv8n ONNX 检测器，用 cv2.dnn 推理（兼容 Windows Python 3.13）。"""

    def __init__(self, model_path: str, conf_threshold: float = 0.35,
                 iou_threshold: float = 0.45):
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold
        self._classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    @staticmethod
    def _preprocess(img: np.ndarray, target_size: int = 640) -> np.ndarray:
        """将 uint8 RGB 图像预处理为模型输入 blob。"""
        return cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(target_size, target_size),
                                     mean=(0, 0, 0), swapRB=True, crop=False)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> list[int]:
        """贪心 NMS，返回保留框的下标。"""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= self._iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img: np.ndarray) -> list[dict]:
        """检测单帧，返回检测框列表（ERP 坐标）。"""
        h, w = img.shape[:2]
        blob = self._preprocess(img)
        self._net.setInput(blob)
        outputs = self._net.forward()
        # YOLOv8 输出: [1, 84, 8400] (xywh + conf + 80 class)
        pred = outputs[0].T  # (8400, 84)
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        confs = scores.max(axis=1)
        cls_ids = scores.argmax(axis=1)
        mask = confs >= self._conf_threshold
        boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]
        if len(boxes) == 0:
            return []
        # xywh -> xyxy
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        # 映射回 ERP 坐标
        scale = min(w / 640, h / 640)
        boxes_xyxy = boxes_xyxy * scale
        keep = self._nms(boxes_xyxy, confs)
        results = []
        for i in keep:
            cls_id = int(cls_ids[i])
            cls_name = self._classes[cls_id] if cls_id < len(self._classes) else str(cls_id)
            results.append({
                'bbox': (float(x1[i] * scale), float(y1[i] * scale),
                         float(bw[i] * scale), float(bh[i] * scale)),
                'score': float(confs[i]),
                'class_id': cls_id,
                'class_name': cls_name,
            })
        return results


class DetectionTracker:
    """基于 YOLO 检测 + 简单关联的跟踪器。"""

    def __init__(self, model_path: str, conf_threshold: float = 0.35):
        self._detector = YOLOv8nDetector(model_path, conf_threshold=conf_threshold)
        self._last_boxes: list[dict] = []
        self._lost_count = 0
        self._max_lost = 5
        self._target_class_id: int | None = None
        self._last_w, self._last_h = 0, 0

    def init(self, frame: np.ndarray, bbox: tuple[float, float, float, float]):
        """用首帧初始化。"""
        dets = self._detector.detect(frame)
        if not dets:
            raise RuntimeError('YOLOv8n 未检测到任何目标，无法初始化')
        # 找与 GT 框 IoU 最大的检测框作为目标
        gt_box = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
        best_i, best_iou = -1, 0.0
        for i, d in enumerate(dets):
            db = d['bbox']
            det_box = np.array([db[0], db[1], db[0] + db[2], db[1] + db[3]])
            iou = self._iou(gt_box, det_box)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i < 0 or best_iou < 0.1:
            # 没有好的匹配，用中心点最近
            gt_cx, gt_cy = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2
            best_i = min(range(len(dets)),
                         key=lambda i: (dets[i]['bbox'][0] + dets[i]['bbox'][2]/2 - gt_cx)**2 +
                                       (dets[i]['bbox'][1] + dets[i]['bbox'][3]/2 - gt_cy)**2)
        target = dets[best_i]
        self._target_class_id = target['class_id']
        self._last_boxes = [target]
        self._last_w, self._last_h = bbox[2], bbox[3]
        self._lost_count = 0

    def update(self, frame: np.ndarray) -> dict:
        """跟踪新一帧。"""
        dets = self._detector.detect(frame)
        # 过滤同类
        if self._target_class_id is not None:
            same_class = [d for d in dets if d['class_id'] == self._target_class_id]
            if same_class:
                dets = same_class
        if not dets:
            self._lost_count += 1
            if self._last_boxes:
                last = self._last_boxes[0]
                return {'bbox': last['bbox'], 'score': 0.0,
                        'status': 'lost', 'fov': (0.0, 0.0)}
            return {'bbox': (0.0, 0.0, 1.0, 1.0), 'score': 0.0,
                    'status': 'lost', 'fov': (0.0, 0.0)}
        # 与上一帧做关联
        if self._last_boxes:
            cost = np.full((len(self._last_boxes), len(dets)), 1e5)
            for i, prev in enumerate(self._last_boxes):
                for j, cur in enumerate(dets):
                    iou = self._iou(
                        np.array([prev['bbox'][0], prev['bbox'][1],
                                  prev['bbox'][0] + prev['bbox'][2],
                                  prev['bbox'][1] + prev['bbox'][3]]),
                        np.array([cur['bbox'][0], cur['bbox'][1],
                                  cur['bbox'][0] + cur['bbox'][2],
                                  cur['bbox'][1] + cur['bbox'][3]])
                    )
                    pcx = prev['bbox'][0] + prev['bbox'][2] / 2
                    pcy = prev['bbox'][1] + prev['bbox'][3] / 2
                    ccx = cur['bbox'][0] + cur['bbox'][2] / 2
                    ccy = cur['bbox'][1] + cur['bbox'][3] / 2
                    W = frame.shape[1]
                    dx = min(abs(pcx - ccx), W - abs(pcx - ccx))
                    dy = abs(pcy - ccy)
                    dist = np.sqrt(dx * dx + dy * dy)
                    cost[i, j] = -iou + 0.01 * dist
            row_ind, col_ind = linear_sum_assignment(cost)
            best_j = col_ind[0] if len(col_ind) > 0 else 0
            best_cost = cost[0, best_j] if len(cost) > 0 else 1e5
            if best_cost < 0.5:
                target = dets[best_j]
                self._last_boxes = [target]
                self._lost_count = 0
                self._last_w, self._last_h = target['bbox'][2], target['bbox'][3]
                return {'bbox': target['bbox'], 'score': target['score'],
                        'status': 'ok', 'fov': (0.0, 0.0)}
        target = max(dets, key=lambda d: d['score'])
        self._last_boxes = [target]
        self._lost_count += 1
        if self._lost_count <= self._max_lost:
            return {'bbox': target['bbox'], 'score': target['score'] * 0.5,
                    'status': 'lost', 'fov': (0.0, 0.0)}
        return {'bbox': target['bbox'], 'score': target['score'],
                'status': 'ok', 'fov': (0.0, 0.0)}

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个 xyxy 框的 IoU（考虑水平回绕）。"""
        W = max(a[2] - a[0], b[2] - b[0]) * 2
        inter = (min(a[2], b[2]) - max(a[0], b[0])) * (min(a[3], b[3]) - max(a[1], b[1]))
        inter = max(0.0, inter)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        iou1 = inter / (area_a + area_b - inter + 1e-6)
        b_shift = b + np.array([W, 0, W, 0])
        inter2 = (min(a[2], b_shift[2]) - max(a[0], b_shift[0])) * \
                  (min(a[3], b_shift[3]) - max(a[1], b_shift[1]))
        inter2 = max(0.0, inter2)
        iou2 = inter2 / (area_a + area_b - inter2 + 1e-6)
        return max(iou1, iou2)
