"""合成 ERP 序列生成器（自包含，不依赖 panotrack.geometry）。

背景为经纬网格渐变 + 固定噪声；目标为确定性纹理矩形，在 ERP 像素域运动。
GT 与画面严格逐帧一致：先算出整数框再绘制，gt.txt 记录同一整数框。
"""
import os
import numpy as np
from PIL import Image

# 遮挡块颜色：目标纹理的蓝色通道恒为 0 或 255，故该灰色不可能与纹理混淆
OCCLUDER_COLOR = (200, 200, 200)

BASE_W = 60.0   # 目标基准宽（像素）
BASE_H = 40.0   # 目标基准高（像素）


def target_texture(w, h):
    """生成确定性目标纹理：横向红渐变 + 纵向绿渐变 + 蓝棋盘。

    参数：w, h —— 纹理宽高（像素，>=1）。
    返回：(h, w, 3) uint8 数组。
    """
    u = np.arange(w)[None, :]
    v = np.arange(h)[:, None]
    r = (u * 255) // max(w - 1, 1)
    g = (v * 255) // max(h - 1, 1)
    b = (((u // 4) + (v // 4)) % 2) * 255
    img = np.stack([
        np.broadcast_to(r, (h, w)),
        np.broadcast_to(g, (h, w)),
        np.broadcast_to(b, (h, w)),
    ], axis=-1)
    return img.astype(np.uint8)


def _background(w, h, seed):
    """生成静态背景：经纬网格渐变 + 固定噪声纹理。

    参数：w, h —— ERP 宽高；seed —— 噪声随机种子。
    返回：(h, w, 3) uint8 数组。
    """
    rng = np.random.default_rng(seed)
    lon = np.arange(w)[None, :] / w          # 0~1
    lat = np.arange(h)[:, None] / h          # 0~1（顶到底）
    base = np.zeros((h, w, 3), dtype=float)
    base[..., 0] = 30 + lon * 70
    base[..., 1] = 30 + lat * 70
    base[..., 2] = 60
    # 经纬网格线（每 30° 一条）
    grid_col = (np.arange(w) % max(w // 12, 1)) == 0
    grid_row = (np.arange(h) % max(h // 6, 1)) == 0
    base[:, grid_col, :] += 30
    base[grid_row, :, :] += 30
    noise = rng.integers(0, 20, size=(h, w, 1))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _motion(scenario, t, w, h):
    """按场景给出第 t 帧的目标浮点框 (x, y, bw, bh)。

    参数：scenario —— 场景名；t —— 帧号；w, h —— ERP 宽高。
    返回：(x, y, bw, bh) 浮点；x 允许 >= w（绘制与 GT 时模 w 回绕）。
    """
    bw, bh = BASE_W, BASE_H
    if scenario == 'equator':
        # 赤道附近匀速右移 + 轻微正弦上下
        x = 120.0 + 4.0 * t
        y = h / 2 - bh / 2 + 20.0 * np.sin(t / 8.0)
    elif scenario == 'crossing':
        # 从右边缘附近出发，中途跨越右边界回绕
        x = (w - 90.0) + 4.0 * t
        y = h / 2 - bh / 2
    elif scenario == 'pole':
        # 向北极（顶行）移动，按 1/cos(lat) 拉宽、cos(lat) 压扁
        y = max(2.0, 150.0 - 3.0 * t)
        cy = y + bh / 2
        lat = 90.0 - cy / h * 180.0          # 目标中心纬度（度）
        k = 1.0 / max(np.cos(np.radians(lat)), 0.2)   # 拉宽系数，上限 5
        bw = BASE_W * k
        bh = max(6.0, BASE_H / k)
        x = w / 2 - bw / 2
    elif scenario == 'occlusion':
        # 赤道附近匀速右移（中段被遮挡块盖住）
        x = 300.0 + 3.0 * t
        y = h / 2 - bh / 2
    else:
        raise ValueError(f'未知场景: {scenario!r}，应为 equator/crossing/pole/occlusion')
    return x, y, bw, bh


def generate_sequence(out_dir, n_frames=60, w=1024, h=512, scenario='equator', seed=0):
    """生成合成 ERP 序列：frames/%06d.png 与 gt.txt（每行 x,y,w,h）。

    参数：out_dir —— 输出目录；n_frames —— 帧数；w, h —— ERP 宽高（W=2H）；
         scenario —— 'equator'/'crossing'/'pole'/'occlusion'；seed —— 背景噪声种子。
    返回：out_dir 路径字符串。
    说明：crossing 目标跨右边界回绕绘制，GT 的 x+w 可超过 W；
         occlusion 中段 7 帧目标被遮挡块完全盖住，被盖帧 GT 与遮挡框一致（即目标框）。
    """
    frames_dir = os.path.join(out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    bg = _background(w, h, seed)
    # occlusion：中段连续 7 帧（5~10 帧要求内）
    occl_frames = set(range(n_frames // 2 - 3, n_frames // 2 + 4)) if scenario == 'occlusion' else set()

    gt_lines = []
    for t in range(n_frames):
        x, y, bw, bh = _motion(scenario, t, w, h)
        # 先取整再绘制，保证 GT 与画面逐像素一致
        ix, iy, iw, ih = (int(round(v)) for v in (x, y, bw, bh))
        iy = max(0, min(iy, h - ih))         # 垂直方向不跨界
        frame = bg.copy()
        cols = (ix + np.arange(iw)) % w      # 水平回绕
        if t in occl_frames:
            frame[iy:iy + ih, cols] = OCCLUDER_COLOR
        else:
            frame[iy:iy + ih, cols] = target_texture(iw, ih)
        Image.fromarray(frame).save(os.path.join(frames_dir, f'{t:06d}.png'))
        gx = ix % w                          # x ∈ [0, W)，x+w 跨界时可超 W
        gt_lines.append(f'{gx:.2f},{iy:.2f},{iw:.2f},{ih:.2f}')

    with open(os.path.join(out_dir, 'gt.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(gt_lines) + '\n')
    return out_dir
