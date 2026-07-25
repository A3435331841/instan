"""可视化：跨界感知画框与 GIF 导出。"""
import numpy as np
from PIL import Image


def draw_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """在图像上画矩形框，跨界（x+w > W）时自动拆成左右两段绘制。

    参数：img —— (H,W,3) uint8；bbox —— (x,y,w,h) 跨界约定框；
         color —— RGB 颜色；thickness —— 线宽（像素）。
    返回：绘制后的新图（不修改原图）。
    """
    out = img.copy()
    H, W = out.shape[:2]
    x, y, bw, bh = (float(v) for v in bbox)
    x = x % W
    x2 = x + bw
    # 水平拆段：右段 [x, min(x2,W))，跨界余量回绕到左边缘
    segs = [(x, min(x2, float(W)))]
    if x2 > W:
        segs.append((0.0, x2 - W))
    y1 = max(0, int(round(y)))
    y2 = min(H, int(round(y + bh)))
    if y2 <= y1:
        return out
    for xa, xb in segs:
        xa = max(0, int(round(xa)))
        xb = min(W, int(round(xb)))
        if xb <= xa:
            continue
        t = min(thickness, xb - xa, y2 - y1)
        out[y1:y1 + t, xa:xb] = color          # 上边
        out[y2 - t:y2, xa:xb] = color          # 下边
        out[y1:y2, xa:xa + t] = color          # 左边
        out[y1:y2, xb - t:xb] = color          # 右边
    return out


def save_gif(frames, out_path, fps=10):
    """用 PIL 把帧序列保存为 GIF。

    参数：frames —— list[np.ndarray (H,W,3) uint8]；out_path —— 输出路径；fps —— 帧率。
    返回：None。
    """
    if not frames:
        raise ValueError('frames 为空')
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
