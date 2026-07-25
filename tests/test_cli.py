"""tests/test_cli.py —— 模块 D（文件协议 / trax 协议 / CLI）单元测试。

策略：PanoTracker（契约模块 E）尚未实现，这里通过向 sys.modules 注入
假的 panotrack.pipeline.pipeline 模块（猴子补丁）来验证协议读写正确性。
纯 assert 脚本，直接运行:  python tests/test_cli.py   （工作目录为项目根）
"""
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import types

import numpy as np
from PIL import Image

# 保证可从项目根导入 panotrack
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LINE_RE = re.compile(r'^-?\d+\.\d{2},-?\d+\.\d{2},-?\d+\.\d{2},-?\d+\.\d{2}$')

# 假 tracker 实例记录表，用于断言协议层行为
_records = []


class FakePanoTracker:
    """假 PanoTracker：记录调用，update 返回可预测的移动框。"""

    def __init__(self, config=None):
        self.config = config
        self.bbox = None
        self.n_update = 0
        self.init_frame_shape = None
        _records.append(self)

    def init(self, frame, bbox):
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
        self.init_frame_shape = frame.shape
        self.bbox = tuple(float(v) for v in bbox)

    def update(self, frame):
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        self.n_update += 1
        x, y, w, h = self.bbox
        # 每帧右移 1.5 像素；末帧故意跨界 (x + w > W) 验证跨界格式输出
        self.bbox = (x + 1.5, y + 0.25, w, h)
        return {'bbox': self.bbox, 'score': 0.9, 'status': 'ok',
                'fov': (60.0, 40.0)}


def _install_fake_pipeline():
    """向 sys.modules 注入假 pipeline 模块（延迟 import 会命中缓存）。"""
    mod = types.ModuleType('panotrack.pipeline.pipeline')
    mod.PanoTracker = FakePanoTracker
    sys.modules['panotrack.pipeline.pipeline'] = mod
    return mod


def _remove_fake_pipeline():
    sys.modules.pop('panotrack.pipeline.pipeline', None)


def _make_seq(tmpdir, n=4, w=64, h=32):
    """造 n 帧 (h,w,3) uint8 PNG 序列与 init 文件，返回 (frames_dir, init_file)。"""
    frames_dir = os.path.join(tmpdir, 'frames')
    os.makedirs(frames_dir)
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(frames_dir, f'{i:06d}.png'))
    init_file = os.path.join(tmpdir, 'init.txt')
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write('10,8,12,10\n')
    return frames_dir, init_file


def test_help_runs():
    """python -m panotrack.cli --help 可运行（模块 E 缺失也不影响）。"""
    env = dict(os.environ, PYTHONIOENCODING='utf-8')  # 强制子进程 UTF-8 输出
    r = subprocess.run([sys.executable, '-m', 'panotrack.cli', '--help'],
                       cwd=ROOT, capture_output=True, text=True,
                       encoding='utf-8', errors='replace', env=env)
    assert r.returncode == 0, r.stderr
    assert '--frames' in r.stdout and '--visualize' in r.stdout
    print('PASS test_help_runs')


def test_file_protocol():
    """文件协议：读取帧/init、逐帧输出 2 位小数、stdout 干净、统计正确。"""
    _install_fake_pipeline()
    try:
        from panotrack.io.file_protocol import run_file_protocol
        with tempfile.TemporaryDirectory() as td:
            frames_dir, init_file = _make_seq(td, n=4)
            out_file = os.path.join(td, 'results.txt')
            cfg = {'tracker': 'fake', 'tag': 'ut'}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                stats = run_file_protocol(frames_dir, init_file, out_file, cfg)
            assert buf.getvalue() == '', 'stdout 必须保持干净'

            with open(out_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            assert len(lines) == 4, f'应有 4 行输出, 实际 {len(lines)}'
            for ln in lines:
                assert LINE_RE.match(ln), f'行格式错误: {ln!r}'
            assert lines[0] == '10.00,8.00,12.00,10.00', lines[0]  # 首行为 init 框
            # 第 3 次 update 后 x = 10 + 4.5 = 14.5 ... 验证递增趋势即可
            xs = [float(ln.split(',')[0]) for ln in lines]
            assert xs == sorted(xs) and xs[-1] > xs[0]

            assert stats['n_frames'] == 4
            assert stats['elapsed_sec'] >= 0 and stats['fps'] > 0
            assert os.path.samefile(stats['out_file'], out_file)

            tk = _records[-1]
            assert tk.config == cfg, 'config 必须原样传给 PanoTracker'
            assert tk.n_update == 3, 'init 后应 update N-1 次'
            assert tk.init_frame_shape == (32, 64, 3)

            # 追加语义：再跑一次，行数翻倍
            run_file_protocol(frames_dir, init_file, out_file, cfg)
            with open(out_file, 'r', encoding='utf-8') as f:
                assert len(f.read().splitlines()) == 8
    finally:
        _remove_fake_pipeline()
    print('PASS test_file_protocol')


def test_cli_main():
    """CLI main()：argv 驱动 + --config JSON，结果文件正确。"""
    _install_fake_pipeline()
    try:
        from panotrack.cli import main
        with tempfile.TemporaryDirectory() as td:
            frames_dir, init_file = _make_seq(td, n=3)
            out_file = os.path.join(td, 'results.txt')
            cfg_file = os.path.join(td, 'cfg.json')
            with open(cfg_file, 'w', encoding='utf-8') as f:
                f.write('{"tracker": "fake", "patch_size": 255}')
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(['--frames', frames_dir, '--init', init_file,
                           '--out', out_file, '--config', cfg_file])
            assert rc == 0
            assert buf.getvalue() == '', 'CLI 运行时 stdout 必须干净'
            with open(out_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            assert len(lines) == 3
            assert _records[-1].config['patch_size'] == 255
    finally:
        _remove_fake_pipeline()
    print('PASS test_cli_main')


def test_trax_protocol():
    """trax 占位协议：握手/init/frame/quit 行协议往返正确。"""
    _install_fake_pipeline()
    try:
        from panotrack.io.trax_protocol import run_trax_protocol
        with tempfile.TemporaryDirectory() as td:
            frames_dir, _ = _make_seq(td, n=2)
            f0 = os.path.join(frames_dir, '000000.png')
            f1 = os.path.join(frames_dir, '000001.png')
            cmds = f'init {f0} 10 8 12 10\nframe {f1}\nbadcmd\nquit\n'
            instream = io.StringIO(cmds)
            outstream = io.StringIO()
            rc = run_trax_protocol(instream, outstream, {'tracker': 'fake'})
            assert rc == 0
            lines = outstream.getvalue().splitlines()
            assert lines[0].startswith('panotrack-trax') and 'ready' in lines[0]
            assert lines[1] == 'ok init'
            m = re.match(r'^bbox (-?\d+\.\d{2}) (-?\d+\.\d{2}) '
                         r'(-?\d+\.\d{2}) (-?\d+\.\d{2})$', lines[2])
            assert m, f'bbox 行格式错误: {lines[2]!r}'
            assert float(m.group(1)) == 11.5 and float(m.group(2)) == 8.25
            assert lines[3].startswith('error')
            assert lines[4] == 'ok bye'
    finally:
        _remove_fake_pipeline()
    print('PASS test_trax_protocol')


def test_crossing_bbox_output():
    """跨界框 (x + w > W) 按契约原样输出，不截断不报错。"""
    _install_fake_pipeline()
    try:
        from panotrack.io.file_protocol import run_file_protocol
        with tempfile.TemporaryDirectory() as td:
            frames_dir, init_file = _make_seq(td, n=2, w=64, h=32)
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write('60,10,12,8\n')  # x + w = 72 > W=64, 跨界
            out_file = os.path.join(td, 'results.txt')
            run_file_protocol(frames_dir, init_file, out_file, None)
            with open(out_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()
            assert lines[0] == '60.00,10.00,12.00,8.00'
            assert float(lines[1].split(',')[0]) + 12 > 64  # 跨界保持
    finally:
        _remove_fake_pipeline()
    print('PASS test_crossing_bbox_output')


if __name__ == '__main__':
    test_help_runs()
    test_file_protocol()
    test_cli_main()
    test_trax_protocol()
    test_crossing_bbox_output()
    print('\nALL TESTS PASSED')
