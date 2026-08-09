# -*- coding: utf-8 -*-
"""360VOT 测试集序列加载器（HuggingFace gated 仓库 xuyzshaun/360VOTS）。

适配 360VOT-test/<seq>.zip 解压后的本地目录，容错点：
  - zip 解压后多套一层目录（find_sequences 递归发现，load 端自动下钻）；
  - 帧目录命名：img / frames / imgs / JPEGImages，或序列目录本身即帧目录；
  - 标注文件名：groundtruth.txt / groundtruth_rect.txt / bbox.txt / gt.txt 等，
    否则取目录内首个能解析出 4 列数字的 .txt/.csv；
  - 标注格式：[x1 y1 w h] 逐帧，空格 / 逗号 / Tab 分隔，表头与非数字行自动跳过；
  - 帧文件名零填充与非零填充均可（自然排序，2.png < 10.png）。
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

GT_JSON_NAME = 'label.json'
FRAME_DIR_NAMES = ('img', 'frames', 'imgs', 'JPEGImages', 'image')
GT_FILE_NAMES = ('groundtruth.txt', 'groundtruth_rect.txt', 'bbox.txt',
                 'gt.txt', 'groundtruth.csv', 'bbox.csv')
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
_MAX_DEPTH = 4          # find_sequences 递归发现的最大层深
_SKIP_DIRS = {'__MACOSX'}          # macOS zip 产物
_NUM_SPLIT = re.compile(r'[\s,;]+')
_NUM_PART = re.compile(r'(\d+)')


def _natural_key(name):
    """文件名自然排序键：数字段按数值比较（私有）。"""
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_PART.split(name)]


def _list_images(d):
    """目录一层内的图像文件，按自然文件名排序（私有）。"""
    d = Path(d)
    if not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file()
             and p.suffix.lower() in IMG_EXTS and not p.name.startswith('._')]
    return sorted(files, key=lambda p: _natural_key(p.name))


def parse_gt_file(path):
    """解析 [x1 y1 w h] 逐帧标注文件。

    参数：path —— 标注文件路径（空格/逗号/Tab 分隔，允许表头、注释与空行）。
    返回：(N,4) float 数组；无有效行时返回形状 (0,4) 的空数组。
    """
    rows = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                vals = [float(v) for v in _NUM_SPLIT.split(line)[:4]]
            except ValueError:
                continue                    # 表头/注释行
            if len(vals) < 4:
                continue
            rows.append(vals)
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def parse_label_json(path):
    """解析 HuggingFace 版 label.json 标注。

    格式：{帧文件名: {'bbox': {'cx','cy','w','h','rotation'}, 'bfov': {...}, ...}}。
    bbox 为中心点格式，转换为 [x1 y1 w h]。
    返回：(names, gt) —— names 为帧文件名列表，gt 为 (N,4) float [x1 y1 w h]。
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    names, rows = [], []
    for name, item in data.items():
        bb = item.get('bbox') if isinstance(item, dict) else None
        if not bb:
            continue
        cx, cy, w, h = float(bb['cx']), float(bb['cy']), float(bb['w']), float(bb['h'])
        names.append(name)
        rows.append([cx - w / 2.0, cy - h / 2.0, w, h])
    return names, np.asarray(rows, dtype=float).reshape(-1, 4)


def _find_gt_file(d):
    """在目录内定位标注文件：label.json 优先，其次候选名，最后首个可解析的
    .txt/.csv（私有）。"""
    d = Path(d)
    p = d / GT_JSON_NAME
    if p.is_file():
        return p
    for name in GT_FILE_NAMES:
        p = d / name
        if p.is_file():
            return p
    if not d.is_dir():
        return None
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in ('.txt', '.csv') \
                and not p.name.startswith('._'):
            try:
                if len(parse_gt_file(p)) > 0:
                    return p
            except OSError:
                continue
    return None


def _find_frames_dir(d):
    """定位帧目录：候选子目录名优先，否则目录本身（私有）。"""
    d = Path(d)
    for name in FRAME_DIR_NAMES:
        if _list_images(d / name):
            return d / name
    if _list_images(d):
        return d
    return None


def _looks_like_sequence(d):
    """目录是否直接构成一个 360VOT 序列（含标注文件且有帧，私有）。"""
    return _find_gt_file(d) is not None and _find_frames_dir(d) is not None


def find_sequences(root):
    """发现 root 下所有已解压的 360VOT 序列目录。

    参数：root —— 数据集根目录（如 data360；zip 多套一层目录也能发现，
         root 本身即序列目录时返回其自身）。
    返回：list[Path]，按序列目录名排序；root 不存在时返回空列表。
    """
    root = Path(root)
    found = []
    if not root.is_dir():
        return found

    def walk(d, depth):
        if _looks_like_sequence(d):
            found.append(d)
            return                        # 序列目录不再下钻
        if depth >= _MAX_DEPTH:
            return
        for sub in sorted(p for p in d.iterdir()
                          if p.is_dir() and not p.name.startswith('.')
                          and p.name not in FRAME_DIR_NAMES
                          and p.name not in _SKIP_DIRS):
            walk(sub, depth + 1)

    walk(root, 0)
    return sorted(found, key=lambda p: p.name)


def _resolve_seq_dir(seq_dir):
    """序列目录内部多套一层时自动下钻：当前层无标注且无帧且只有一个
    非帧子目录时进入该子目录（最多 3 层，私有）。"""
    d = Path(seq_dir)
    for _ in range(3):
        if _find_gt_file(d) is not None or _list_images(d):
            return d
        subs = [p for p in d.iterdir() if p.is_dir()
                and p.name not in FRAME_DIR_NAMES and p.name not in _SKIP_DIRS
                and not p.name.startswith('.')]
        if len(subs) != 1:
            return d
        d = subs[0]
    return d


def _resolve_sequence(seq_dir):
    """解析序列的帧路径列表与 GT 数组（私有）。

    返回：(paths, gt, resolved_dir)；paths 为帧文件路径列表（自然排序），
    gt 为 (N,4) float [x1 y1 w h]（未缩放），两者按帧名对齐。
    """
    d = _resolve_seq_dir(seq_dir)
    gt_path = _find_gt_file(d)
    if gt_path is None:
        raise FileNotFoundError(f'未找到标注文件（尝试 {GT_JSON_NAME} 或 {GT_FILE_NAMES}）: {d}')
    frames_dir = _find_frames_dir(d)
    if frames_dir is None:
        raise FileNotFoundError(
            f'未找到帧目录（尝试 {FRAME_DIR_NAMES} 或目录本身）: {d}')
    if gt_path.name == GT_JSON_NAME:
        names, gt_all = parse_label_json(gt_path)
        by_name = {p.name: p for p in _list_images(frames_dir)}
        pairs = sorted(((by_name[n], g) for n, g in zip(names, gt_all)
                        if n in by_name),
                       key=lambda pg: _natural_key(pg[0].name))
        paths = [p for p, _ in pairs]
        gt = np.asarray([g for _, g in pairs], dtype=float).reshape(-1, 4)
    else:
        gt = parse_gt_file(gt_path)
        paths = _list_images(frames_dir)
    return paths, gt, d


def load_vot360_annotations(seq_dir, max_frames=None):
    """Load frame paths and aligned ``[x, y, w, h]`` annotations only.

    Unlike :func:`load_vot360_sequence`, this helper does not decode image
    pixels.  It is intended for scoring predictions produced by external
    trackers, where only the first image header is needed to obtain ERP width.
    """
    paths, gt, resolved_dir = _resolve_sequence(seq_dir)
    if max_frames is not None:
        paths = paths[:int(max_frames)]
        gt = gt[:int(max_frames)]
    if len(paths) != len(gt):
        raise ValueError(
            f'帧数({len(paths)})与 GT 行数({len(gt)})不一致: {resolved_dir}')
    if not paths:
        raise FileNotFoundError(f'序列为空（无帧图像）: {resolved_dir}')
    return paths, np.asarray(gt, dtype=float).reshape(-1, 4)


def load_vot360_sequence(seq_dir, downscale=1.0, max_frames=None):
    """加载一个 360VOT 序列：帧图像 + [x1 y1 w h] 真值。

    参数：seq_dir —— 序列目录（内部多套一层目录时自动下钻）；
         downscale —— 帧与 GT 同步缩放比例，默认 1.0（原始 4K）；
             0.5 可显著提速省内存，IoU 尺度不变（dual IoU 的 width 用缩放后帧宽）；
         max_frames —— 调试截断帧数（None 全量）。
    返回：(frames, gt)；frames 为 list[np.ndarray (H,W,3) uint8 RGB]（按文件名
         自然排序），gt 为 (N,4) float（随帧同步缩放）。
    异常：帧目录/标注缺失抛 FileNotFoundError；帧数与 GT 行数不一致抛 ValueError。
    注意：大序列请改用 iter_vot360_sequence 流式加载以省内存。
    """
    frames, rows = [], []
    for _, frame, row in iter_vot360_sequence(seq_dir, downscale, max_frames):
        frames.append(frame)
        rows.append(row)
    gt = np.asarray(rows, dtype=float).reshape(-1, 4)
    return frames, gt


def iter_vot360_sequence(seq_dir, downscale=1.0, max_frames=None):
    """流式加载 360VOT 序列：逐帧 yield (idx, frame, gt_row)，大序列省内存。

    参数与异常同 load_vot360_sequence；gt_row 为 (4,) float [x1 y1 w h]
    （随帧同步缩放）。
    """
    paths, gt, d = _resolve_sequence(seq_dir)
    if max_frames is not None:
        paths = paths[:int(max_frames)]
        gt = gt[:int(max_frames)]
    if len(paths) != len(gt):
        raise ValueError(f'帧数({len(paths)})与 GT 行数({len(gt)})不一致: {d}')
    if not paths:
        raise FileNotFoundError(f'序列为空（无帧图像）: {d}')
    if downscale != 1.0:
        if not 0.0 < float(downscale) <= 1.0:
            raise ValueError(f'downscale 须在 (0,1]: {downscale}')
        gt = gt * float(downscale)
    for i, p in enumerate(paths):
        img = Image.open(p).convert('RGB')
        if downscale != 1.0:
            w, h = img.size
            img = img.resize((max(1, round(w * float(downscale))),
                              max(1, round(h * float(downscale)))),
                             Image.Resampling.BILINEAR)
        yield i, np.array(img, dtype=np.uint8), gt[i]
