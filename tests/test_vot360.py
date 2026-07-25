# -*- coding: utf-8 -*-
"""360VOT 加载器（panotrack.data.vot360）与批量评测脚本（scripts/eval_360vot.py）测试。

运行方式：在项目根目录 D:\\instan\\pano360 下执行 `python tests/test_vot360.py`。

覆盖：
  A. find_sequences：标准布局 + zip 解压多套一层目录布局 + 非序列目录过滤
     + root 本身即序列目录 + 不存在目录返回空；
  B. load_vot360_sequence：候选帧目录（img/frames）与候选标注名
     （groundtruth.txt/bbox.txt）、空格/逗号/Tab 分隔、表头行跳过、
     自然文件名排序（2.png < 10.png）、从上层目录自动下钻、downscale 同步缩放、
     max_frames 截断、帧数/GT 行数不一致与缺标注的异常；
  C. eval_360vot.eval_sequence / write_summary：小序列上产出合法指标与落盘文件。
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from panotrack.data.synth import generate_sequence
from panotrack.data.io import load_sequence
from panotrack.data.vot360 import find_sequences, load_vot360_sequence, parse_gt_file

import eval_360vot

# 两个独立临时根：源序列（synth 布局）不得出现在 360VOT 数据集根内，
# 否则其 frames/+gt.txt 布局本身也会被 find_sequences 合法发现
SRC = Path(tempfile.mkdtemp(prefix='test_vot360_src_', dir=str(ROOT / 'runs')))
TMP = Path(tempfile.mkdtemp(prefix='test_vot360_', dir=str(ROOT / 'runs')))
N_FRAMES, W, H = 12, 512, 256


def _save_frames(frames, d, names, ext='png'):
    """把帧列表按给定文件名落盘（私有）。"""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    for fr, n in zip(frames, names):
        img = Image.fromarray(fr)
        if ext == 'jpg':
            img.save(d / f'{n}.jpg', quality=95)
        else:
            img.save(d / f'{n}.png')


# ---------------- 构造源序列与两套 360VOT 风格布局 ----------------
src_dir = Path(generate_sequence(str(SRC / 'src'), n_frames=N_FRAMES, w=W, h=H,
                                 scenario='equator', seed=5))
src_frames, src_gt = load_sequence(src_dir)
assert len(src_frames) == N_FRAMES and src_gt.shape == (N_FRAMES, 4)

# 变体 A：多套一层目录 + img/ + groundtruth.txt（空格分隔 + 表头行）+ jpg 帧
dir_a = TMP / 'A_extra' / '013' / '013'
_save_frames(src_frames, dir_a / 'img', [f'{i:06d}' for i in range(N_FRAMES)],
             ext='jpg')
with open(dir_a / 'groundtruth.txt', 'w', encoding='utf-8') as f:
    f.write('x1 y1 w h\n')                            # 表头行，应被跳过
    for r in src_gt:
        f.write(f'{r[0]:.2f} {r[1]:.2f} {r[2]:.2f} {r[3]:.2f}\n')

# 变体 B：frames/ + bbox.txt（逗号分隔）+ 非零填充文件名（验自然排序）
dir_b = TMP / 'B' / '007'
_save_frames(src_frames, dir_b / 'frames', [str(i + 1) for i in range(N_FRAMES)])
with open(dir_b / 'bbox.txt', 'w', encoding='utf-8') as f:
    for r in src_gt:
        f.write(f'{r[0]:.2f},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f}\n')

# 干扰目录：txt 无法解析出 4 列数字，不应被发现为序列
decoy = TMP / 'B' / 'not_a_seq'
decoy.mkdir(parents=True)
(decoy / 'notes.txt').write_text('hello world, not a gt file', encoding='utf-8')

# ---------------- A. find_sequences ----------------
seqs = find_sequences(TMP)
names = [p.name for p in seqs]
assert names == ['007', '013'], f'find_sequences 结果错误: {names}'
assert [p.name for p in find_sequences(dir_b)] == ['007'], \
    'root 本身即序列目录时应返回其自身'
assert find_sequences(TMP / 'no_such_dir') == [], '不存在目录应返回空列表'

# ---------------- B. load_vot360_sequence ----------------
# 变体 A：多套一层 + 表头跳过 + img 目录 + 空格分隔 + jpg
fa, ga = load_vot360_sequence(dir_a)
assert len(fa) == N_FRAMES and ga.shape == (N_FRAMES, 4)
assert fa[0].shape == (H, W, 3) and fa[0].dtype == np.uint8
assert np.allclose(ga, src_gt, atol=1e-2), 'GT 解析（空格分隔+表头跳过）错误'
# 从多套一层的上层目录传入也能自动下钻
fa2, ga2 = load_vot360_sequence(dir_a.parent)
assert len(fa2) == N_FRAMES and np.allclose(ga2, ga)

# 变体 B：自然排序（1.png,2.png,...,12.png 帧序必须与 GT 逐帧对齐）
fb, gb = load_vot360_sequence(dir_b)
assert len(fb) == N_FRAMES and gb.shape == (N_FRAMES, 4)
for i in range(N_FRAMES):
    assert np.array_equal(fb[i], src_frames[i]), \
        f'第 {i} 帧顺序错乱（自然排序失败）'
assert np.allclose(gb, src_gt, atol=1e-2), 'GT 解析（逗号分隔）错误'

# downscale：帧与 GT 同步缩放
fd, gd = load_vot360_sequence(dir_b, downscale=0.5)
assert fd[0].shape == (H // 2, W // 2, 3) and fd[0].dtype == np.uint8
assert np.allclose(gd, src_gt * 0.5, atol=1e-2), 'GT 未随帧同步缩放'

# max_frames 截断
fm, gm = load_vot360_sequence(dir_b, max_frames=5)
assert len(fm) == 5 and gm.shape == (5, 4)

# parse_gt_file：Tab 分隔 + 注释行 + 空行 + 行尾空白
tab_file = TMP / 'tab.txt'
tab_file.write_text('# comment\n\n10\t20\t30\t40\n50\t60\t70\t80\t\n',
                    encoding='utf-8')
arr = parse_gt_file(tab_file)
assert arr.shape == (2, 4) and arr[1, 0] == 50.0 and arr[0, 3] == 40.0

# 帧数与 GT 行数不一致 -> ValueError
bad = TMP / 'B' / '009'
_save_frames(src_frames[:5], bad / 'frames', [str(i + 1) for i in range(5)])
with open(bad / 'bbox.txt', 'w', encoding='utf-8') as f:
    for r in src_gt[:6]:
        f.write(f'{r[0]:.2f},{r[1]:.2f},{r[2]:.2f},{r[3]:.2f}\n')
try:
    load_vot360_sequence(bad)
    raise AssertionError('帧数/GT 行数不一致应抛 ValueError')
except ValueError:
    pass

# 缺标注文件 -> FileNotFoundError
empty = TMP / 'B' / '011'
(empty / 'frames').mkdir(parents=True)
try:
    load_vot360_sequence(empty)
    raise AssertionError('缺标注应抛 FileNotFoundError')
except FileNotFoundError:
    pass

# ---------------- C. eval 脚本核心函数 ----------------
out_dir = TMP / 'eval_out'
m = eval_360vot.eval_sequence(dir_b, out_dir=out_dir, downscale=1.0)
for k in ('sequence', 'n_frames', 'sr', 'sr_dual', 'auc', 'auc_dual', 'fps',
          'n_lost', 'n_recovered'):
    assert k in m, f'指标缺键 {k}'
assert m['sequence'] == '007' and m['n_frames'] == N_FRAMES
for k in ('sr', 'sr_dual', 'auc', 'auc_dual'):
    assert 0.0 <= m[k] <= 1.0, f'{k} 越界: {m[k]}'
assert m['fps'] > 0.0
assert m['sr_dual'] >= 0.5, f'equator 小序列双口径 SR 过低: {m["sr_dual"]:.3f}'

# 落盘校验：results.txt 行数与首帧初始化框、metrics.json 与返回值一致
lines = (out_dir / '007' / 'results.txt').read_text(encoding='utf-8') \
    .strip().splitlines()
assert len(lines) == N_FRAMES, f'results.txt 行数 {len(lines)} != {N_FRAMES}'
first = [float(v) for v in lines[0].split(',')]
assert np.allclose(first, src_gt[0], atol=1e-2), 'results.txt 首帧应为初始化 GT 框'
mj = json.loads((out_dir / '007' / 'metrics.json').read_text(encoding='utf-8'))
assert abs(mj['sr_dual'] - m['sr_dual']) < 1e-9 and mj['n_frames'] == N_FRAMES

# max_frames 参数贯通到评测
m5 = eval_360vot.eval_sequence(dir_b, out_dir=out_dir, max_frames=5)
assert m5['n_frames'] == 5

# select_sequences：all / 列表 / 忽略前导零
# （009 布局合法会被发现，仅在 load 时因帧数/GT 不一致抛 ValueError，eval 会跳过）
assert [p.name for p in eval_360vot.select_sequences(TMP, 'all')] == ['007', '009', '013']
assert [p.name for p in eval_360vot.select_sequences(TMP, '13')] == ['013']
assert [p.name for p in eval_360vot.select_sequences(TMP, '001')] == []

# write_summary：csv（表头 + 各序列 + MEAN 行）与终端表格
csv_path, table = eval_360vot.write_summary([m, m5], out_dir)
csv_lines = csv_path.read_text(encoding='utf-8').strip().splitlines()
assert csv_lines[0].split(',') == list(eval_360vot.SUMMARY_COLS)
assert len(csv_lines) == 1 + 2 + 1, 'csv 应为表头 + 2 序列 + MEAN 行'
assert csv_lines[-1].startswith('MEAN')
assert 'SR_dual' in table and '007' in table and 'MEAN(2)' in table

print('PASS: test_vot360 全部通过')


if __name__ == '__main__':
    shutil.rmtree(SRC, ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)
