#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 Arena 协议 CPU/GPU 冒烟用合成 ERP 测试集（video.mp4 + init.txt）。

用法:
  python scripts/make_arena_smoke_dataset.py [OUT_DIR]

生成 2 条序列到 OUT_DIR（默认 ./smoke_dataset）:
  seq_0001  目标匀速右移（正常移动，10 帧）
  seq_0002  目标跨越 0/360 度接缝（左移回绕，8 帧）

配合镜像验证（CPU 冒烟，无 GPU 也可跑）:
  docker run --rm -v <OUT_DIR>:/mnt/dataset:ro -v <OUT_DIR>_result:/mnt/result \
      <镜像> --force-cpu

预期: 退出码 0；seq_0001.txt 的 clon 单调递增（目标右移），
seq_0002.txt 的 clon 从 +171 穿缝到 -138（跨接缝正确回绕）；
行数 = 帧数，首行 = init.txt BFoV。
"""
import math
import os
import sys

import cv2
import numpy as np


def make_seq(out_root, name, tx0, ty0, tx1, ty1, n_frames=10, W=1280, H=640):
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    yy, xx = np.mgrid[0:H, 0:W]
    bg = np.zeros((H, W, 3), np.uint8)
    bg[..., 0] = (xx * 180 // W).astype(np.uint8)
    bg[..., 1] = (yy * 180 // H).astype(np.uint8)
    bg[..., 2] = ((xx + yy) * 90 // (W + H)).astype(np.uint8)
    bg[::40, :] = (bg[::40, :].astype(np.int16) + 40).clip(0, 255).astype(np.uint8)
    bg[:, ::40] = (bg[:, ::40].astype(np.int16) + 40).clip(0, 255).astype(np.uint8)

    tw, th = 96, 96
    cx0, cy0 = tx0 + tw / 2.0, ty0 + th / 2.0
    clon = (cx0 / W - 0.5) * 360.0
    clat = (0.5 - cy0 / H) * 180.0
    coslat = max(math.cos(math.radians(clat)), 1e-6)
    fov_h = (tw / W * 360.0) * coslat
    fov_v = th / H * 180.0
    with open(os.path.join(out, 'init.txt'), 'w') as f:
        f.write('%s,%s,%s,%s\n' % (round(clon, 3), round(clat, 3),
                                    round(fov_h, 3), round(fov_v, 3)))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(os.path.join(out, 'video.mp4'), fourcc, 10.0, (W, H))
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        tx = int(round(tx0 + (tx1 - tx0) * t))
        ty = int(round(ty0 + (ty1 - ty0) * t))
        frame = bg.copy()
        target = np.zeros((th, tw, 3), np.uint8)
        for r in range(0, th, 16):
            for c in range(0, tw, 16):
                if (r // 16 + c // 16) % 2 == 0:
                    target[r:r + 16, c:c + 16] = (0, 200, 255)
                else:
                    target[r:r + 16, c:c + 16] = (60, 60, 60)
        x2 = min(tx + tw, W)
        frame[ty:ty + th, tx:x2] = target[:, :x2 - tx]
        vw.write(frame)
    vw.release()
    print('made', name, 'frames', n_frames,
          'init', round(clon, 2), round(clat, 2))


def main():
    out_root = sys.argv[1] if len(sys.argv) > 1 else './smoke_dataset'
    make_seq(out_root, 'seq_0001', 100, 200, 700, 230, n_frames=10)
    make_seq(out_root, 'seq_0002', 1200, 300, 100, 320, n_frames=8)
    print('DATA READY ->', os.path.abspath(out_root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
