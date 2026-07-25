"""synth 合成数据测试：python tests/test_synth.py

抽查：按 GT 裁剪出的区域与目标纹理逐像素一致（遮挡帧为纯色块）。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from panotrack.data.synth import generate_sequence, target_texture
from panotrack.data.io import load_sequence
from panotrack.data.viz import draw_bbox, save_gif

W, H, N = 1024, 512, 60
SCENARIOS = ['equator', 'crossing', 'pole', 'occlusion']


def crop_wrap(frame, box):
    """按跨界约定裁剪：x 模 W 回绕，返回 (h, w, 3) 区域。"""
    x, y, w, h = box
    ix, iy, iw, ih = (int(round(v)) for v in (x, y, w, h))
    cols = (ix + np.arange(iw)) % frame.shape[1]
    return frame[iy:iy + ih, cols]


tmp = tempfile.mkdtemp(prefix='synth_test_')

for sc in SCENARIOS:
    out_dir = os.path.join(tmp, sc)
    ret = generate_sequence(out_dir, n_frames=N, w=W, h=H, scenario=sc, seed=1)
    assert ret == out_dir

    # 文件结构：60 帧 PNG + gt.txt 60 行
    pngs = sorted(os.listdir(os.path.join(out_dir, 'frames')))
    assert len(pngs) == N and pngs[0] == '000000.png' and pngs[-1] == '000059.png'
    frames, gt = load_sequence(out_dir)
    assert len(frames) == N and gt.shape == (N, 4)
    assert frames[0].shape == (H, W, 3) and frames[0].dtype == np.uint8
    # GT 的 x 在 [0, W) 内
    assert (gt[:, 0] >= 0).all() and (gt[:, 0] < W).all()

    # 逐帧检查：GT 裁剪区域与目标纹理严格一致，或为纯色遮挡块
    occluded = []
    for t in range(N):
        crop = crop_wrap(frames[t], gt[t])
        iw, ih = int(round(gt[t, 2])), int(round(gt[t, 3]))
        if np.all(crop == crop[0, 0]):
            occluded.append(t)
        else:
            assert np.array_equal(crop, target_texture(iw, ih)), \
                f'{sc} 第 {t} 帧 GT 区域与目标纹理不一致'

    if sc == 'crossing':
        # 必须存在跨界帧：x + w > W
        assert (gt[:, 0] + gt[:, 2] > W).any(), 'crossing 未跨越右边界'
        # 目标纹理确实回绕到左边缘
        t = int(np.argmax(gt[:, 0] + gt[:, 2] > W))
        assert np.array_equal(crop_wrap(frames[t], gt[t]),
                              target_texture(int(gt[t, 2]), int(gt[t, 3])))
    elif sc == 'pole':
        # 接近顶行且按 1/cos(lat) 拉宽压扁：最小 y 很小，最大宽明显超过基准 60
        assert gt[:, 1].min() < 30, 'pole 未接近顶行'
        assert gt[:, 2].max() > 120, 'pole 未按 1/cos(lat) 拉宽'
        # 宽度增大时高度应变小
        tmax = int(np.argmax(gt[:, 2]))
        assert gt[tmax, 3] < 40
    elif sc == 'occlusion':
        # 中段 5~10 帧被遮挡（纯色块），且遮挡帧连续
        assert 5 <= len(occluded) <= 10, f'遮挡帧数异常: {len(occluded)}'
        assert occluded == list(range(occluded[0], occluded[-1] + 1)), '遮挡帧不连续'
        assert occluded[0] >= N // 3 and occluded[-1] <= 2 * N // 3, '遮挡不在中段'
    else:
        assert not occluded, f'{sc} 不应出现遮挡帧'

# ---- viz.draw_bbox：跨界拆两段，且不修改原图 ----
img = np.zeros((100, 200, 3), dtype=np.uint8)
out = draw_bbox(img, (190, 10, 20, 30))   # 覆盖 x∈[190,200)∪[0,10)
assert img.sum() == 0, 'draw_bbox 修改了原图'
# 上边横线：右段末尾与左段开头（跨界两段的证据）
assert (out[10, 199] == (0, 255, 0)).all()
assert (out[10, 0] == (0, 255, 0)).all()
assert (out[11, 195] == (0, 255, 0)).all()   # 线宽 2
assert (out[12, 100] == 0).all()             # 框外不受影响
# 不跨界普通框
out2 = draw_bbox(img, (50, 50, 20, 20), color=(255, 0, 0))
assert (out2[50, 60] == (255, 0, 0)).all() and (out2[60, 50] == (255, 0, 0)).all()

# ---- save_gif ----
gif_path = os.path.join(tmp, 'demo.gif')
small = [np.zeros((16, 32, 3), dtype=np.uint8) + i * 40 for i in range(3)]
save_gif(small, gif_path, fps=5)
assert os.path.getsize(gif_path) > 0

print('test_synth: 全部通过')
