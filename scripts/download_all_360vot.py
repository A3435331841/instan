#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量下载 360VOT 测试集全部 120 序列(断点续传 + 失败重试)。

用法:
    python scripts/download_all_360vot.py            # 下载全部缺失序列
    python scripts/download_all_360vot.py --seqs 5,6,7   # 只补指定序列

行为:
  - 已存在(含 label.json)的序列自动跳过 → 中断后重跑即断点续传;
  - 每个序列最多重试 3 次(间隔 10s/30s),单序列失败不阻塞其余;
  - 下载后自动解压到 data360/<seq>/ 并删除 zip;
  - 进度逐行写 stderr(日志重定向到文件即可),结束时打印失败清单。
"""
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = 'xuyzshaun/360VOTS'
PREFIX = '360VOT-test'            # VOT 格式(带 label.json),4 位编号 0001~0120
OUT = Path('data360')
RETRIES = 3
RETRY_WAIT = (10, 30)             # 第 n 次重试前等待秒数

TOKEN_FILE = Path.home() / '.cache' / 'huggingface' / 'token'


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _get_token():
    tok = os.environ.get('HF_TOKEN')
    if not tok and TOKEN_FILE.is_file():
        tok = TOKEN_FILE.read_text(encoding='utf-8').strip()
    return tok or None


def already_have(seq):
    """序列是否已就绪:目录下存在 label.json 即视为已有。"""
    d = OUT / seq
    return d.is_dir() and bool(list(d.rglob('label.json')))


def _safe_extract(zip_path, dst):
    """解压并防路径穿越。"""
    dst = Path(dst).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            try:
                (dst / m).resolve().relative_to(dst)
            except ValueError:
                raise ValueError(f'zip 内含越界路径: {m!r}')
        zf.extractall(dst)


def download_one(seq):
    """下载并解压单个序列;返回 True/False。"""
    filename = f'{PREFIX}/{seq}.zip'
    for attempt in range(1, RETRIES + 1):
        try:
            _log(f'[下载] {seq} (第{attempt}次尝试) ...')
            p = hf_hub_download(repo_id=REPO_ID, filename=filename,
                                repo_type='dataset', local_dir=str(OUT),
                                token=_get_token())
            # zip 顶层目录即 <seq>/ ;解压到 OUT 即得到 OUT/<seq>/label.json
            _safe_extract(p, OUT)
            os.remove(p)
            _log(f'  完成 {seq}, 已解压并删除 zip')
            return True
        except (HfHubHTTPError, OSError, zipfile.BadZipFile) as e:
            code = getattr(getattr(e, 'response', None), 'status_code', None)
            if code == 404:
                _log(f'  失败 {seq}: 仓库内不存在 {filename}(HTTP 404),跳过')
                return False
            if attempt < RETRIES:
                wait = RETRY_WAIT[min(attempt - 1, len(RETRY_WAIT) - 1)]
                _log(f'  失败 {seq}({type(e).__name__}: {str(e)[:120]}), '
                     f'{wait}s 后重试')
                time.sleep(wait)
            else:
                _log(f'  失败 {seq}({type(e).__name__}: {str(e)[:160]}),重试耗尽')
        except Exception as e:
            _log(f'  失败 {seq}({type(e).__name__}: {str(e)[:160]})')
            if attempt < RETRIES:
                wait = RETRY_WAIT[min(attempt - 1, len(RETRY_WAIT) - 1)]
                time.sleep(wait)
            else:
                _log(f'  {seq} 重试耗尽')
    return False


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description='批量下载 360VOT 全部 120 序列')
    p.add_argument('--seqs', default=None,
                   help='逗号分隔序列号(如 5,6,7);缺省为 0001~0120 全量')
    args = p.parse_args(argv)

    token = _get_token()
    if not token:
        _log('未找到 HF 令牌:请设置 HF_TOKEN 或写入 ~/.cache/huggingface/token')
        return 3

    if args.seqs:
        seqs = [s.strip().zfill(4) for s in args.seqs.split(',') if s.strip()]
    else:
        seqs = [f'{i:04d}' for i in range(1, 121)]

    todo = [s for s in seqs if not already_have(s)]
    skipped = [s for s in seqs if s not in todo]
    _log(f'计划下载 {len(todo)} 个序列, 跳过已存在 {len(skipped)} 个: {",".join(skipped) or "无"}')
    _log(f'输出目录: {OUT.resolve()}')

    ok, failed = [], []
    t0 = time.time()
    for i, seq in enumerate(todo, 1):
        (ok if download_one(seq) else failed).append(seq)
        _log(f'进度 [{i}/{len(todo)}] 成功 {len(ok)} 失败 {len(failed)}')

    _log(f'全部结束: 成功 {len(ok)}/{len(todo)}, 失败 {len(failed)}, '
         f'总耗时 {(time.time()-t0)/60:.1f} 分钟')
    if failed:
        _log('失败序列: ' + ','.join(failed))
        _log('重跑本脚本即可续传剩余序列(已成功的会跳过)')
        return 1
    _log('全部下载完成!下一步评测: python scripts/eval_360vot.py --data data360 --seqs all --downscale 0.5')
    return 0


if __name__ == '__main__':
    sys.exit(main())
