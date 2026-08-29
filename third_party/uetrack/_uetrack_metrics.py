# -*- coding: utf-8 -*-
"""Compute AUC / SR@0.5 / FPS for UETrack predictions on the ERP dataset."""
import json
import numpy as np

DATA = '/data/projects/instan/data360'
RES = '/data/projects/instan_check/uetrack_output/test/tracking_results/uetrack/uetrack_base'
SEQS = ['0008', '0036', '0116']


def iou(a, b):
    a_x2, a_y2 = a[0] + a[2], a[1] + a[3]
    b_x2, b_y2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(a_x2, b_x2) - max(a[0], b[0]))
    iy = max(0.0, min(a_y2, b_y2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def calc(gt_boxes, pred_boxes):
    overlaps = np.array([iou(g, p) for g, p in zip(gt_boxes, pred_boxes)])
    sr = float(np.mean(overlaps > 0.5))
    thresholds = np.arange(0.0, 1.0, 0.01)
    sr_curve = np.array([np.mean(overlaps >= t) for t in thresholds])
    auc = float(np.mean(sr_curve))
    return sr, auc, overlaps


def main():
    all_sr, all_auc, all_fps = [], [], []
    for seq in SEQS:
        with open('%s/%s/label.json' % (DATA, seq)) as f:
            label = json.load(f)
        keys = sorted(label.keys())
        gt = []
        for k in keys:
            b = label[k]['bbox']
            gt.append([b['cx'] - b['w'] / 2.0, b['cy'] - b['h'] / 2.0, b['w'], b['h']])
        pred = np.loadtxt('%s/%s.txt' % (RES, seq)).reshape(-1, 4)
        times = np.loadtxt('%s/%s_time.txt' % (RES, seq))
        fps = len(times) / float(np.sum(times))
        sr, auc, overlaps = calc(gt, pred)
        all_sr.append(sr)
        all_auc.append(auc)
        all_fps.append(fps)
        print('%s: frames=%d fps=%.2f AUC=%.4f SR@0.5=%.4f (mean_holdover=%.4f)' % (
            seq, len(gt), fps, auc, sr, float(np.mean(overlaps))))

    print('---- AVERAGE ----')
    print('mean fps=%.2f  AUC=%.4f  SR@0.5=%.4f' % (
        float(np.mean(all_fps)), float(np.mean(all_auc)), float(np.mean(all_sr))))


if __name__ == '__main__':
    main()