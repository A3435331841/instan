#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arena_protocol 协议层验证：不加载 ODTrack，用 mock tracker 验证

输入/输出格式与平台规范完全一致：
  - 输入 /mnt/dataset/<seq>/video.mp4 + init.txt (BFoV)
  - 输出 /mnt/result/<seq>.txt（每行 clon,clat,fov_h,fov_v，行号=帧号）
  - 丢失帧 0,0,0,0；首行=init BFoV；多序列遍历；seqlist.txt 支持。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# arena_protocol.py 位于 integrations/odtrack/，可被 tests/ 与镜像共享
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'integrations', 'odtrack'))
import arena_protocol as ap


class MockTracker:
    """按真实 ODTrack 接口行为的假跟踪器：init 后每帧返回一个小偏移框。"""
    def __init__(self, params):
        self.state = None
        self.last_pred_iou = 0.9

    def initialize(self, image, info):
        self.state = list(info['init_bbox'])

    def track(self, image):
        # 三平铺帧：中间副本宽度 = W；模拟向右偏移 10px
        w = image.shape[1] // 3
        x, y, bw, bh = self.state
        self.state = [x + 10, y, bw, bh]
        return {'target_bbox': self.state}


def make_seq(root, name, n_frames, init_bfov, w=960, h=480):
    import cv2
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(d / 'video.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), 100, dtype=np.uint8)
        cv2.rectangle(frame, (int(w*0.4), int(h*0.35)), (int(w*0.4)+80, int(h*0.35)+60),
                      (0, 200, 255), -1)
        out.write(frame)
    out.release()
    (d / 'init.txt').write_text(','.join(f'{v:.3f}' for v in init_bfov) + '\n', encoding='utf-8')


def main():
    tmp = Path(tempfile.mkdtemp(prefix='arena_verify_'))
    ds = tmp / 'dataset'
    rs = tmp / 'result'
    ds.mkdir(parents=True)

    # 两条序列 + 一条无视频目录（应被忽略）
    make_seq(ds, 'seq_0001', 6, (0.0, 10.0, 20.0, 15.0))
    make_seq(ds, 'seq_0002', 4, (5.0, -5.0, 18.0, 12.0))
    (ds / 'empty_dir').mkdir()

    # 用 mock tracker 跑协议层（绕过 ODTrack 权重加载）
    seqs = ap.list_sequences(ds)
    assert seqs == ['seq_0001', 'seq_0002'], f'list_sequences wrong: {seqs}'

    rows1, n1 = ap._track_one_sequence(ds / 'seq_0001', 'seq_0001', None, MockTracker)
    assert n1 == 6, f'seq_0001 frames: {n1} (expect 6)'
    assert len(rows1) == 6
    assert rows1[0] == (0.0, 10.0, 20.0, 15.0), 'first row must be init BFoV'
    # 后续帧应是 ERP 框向右偏移后的 BFoV：clon 应随 x 增大而增大
    lons = [r[0] for r in rows1[1:]]
    assert lons[0] < lons[1] < lons[2], f'clon should increase: {lons}'
    # BFoV 合法性
    for (clon, clat, fh, fv) in rows1:
        assert -180 <= clon <= 180, clon
        assert -90 <= clat <= 90, clat
        assert 0 < fh <= 180 and 0 < fv <= 180, (fh, fv)

    rows2, n2 = ap._track_one_sequence(ds / 'seq_0002', 'seq_0002', None, MockTracker)
    assert n2 == 4 and len(rows2) == 4

    # 丢失帧：高阈值强制全部标记丢失
    rows_lost, _ = ap._track_one_sequence(ds / 'seq_0001', 'seq_0001', None, MockTracker,
                                          lost_iou_threshold=0.99)
    assert rows_lost[0] == (0.0, 10.0, 20.0, 15.0)
    assert all(r == (0.0, 0.0, 0.0, 0.0) for r in rows_lost[1:]), rows_lost

    # seqlist.txt（含 BOM）
    (ds / 'seqlist.txt').write_bytes(b'\xef\xbb\xbfseq_0002\n')
    assert ap.list_sequences(ds) == ['seq_0002'], ap.list_sequences(ds)
    (ds / 'seqlist.txt').unlink()

    # 写入与格式
    rs.mkdir(parents=True)
    ap.write_bfov_rows(rs / 'seq_0001.txt', rows1)
    lines = (rs / 'seq_0001.txt').read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) == 6
    for ln in lines:
        parts = ln.split(',')
        assert len(parts) == 4, ln
        vals = [float(p) for p in parts]
        assert all(np.isfinite(vals))
    print('--- seq_0001.txt (mock) ---')
    print('\n'.join(lines))
    print('ALL PROTOCOL TESTS PASSED')
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
