# -*- coding: utf-8 -*-
"""下载 360VOT 测试集序列（HuggingFace gated 仓库 xuyzshaun/360VOTS）。

前置条件（仓库需申请权限，一次申请长期有效）：
  1. 注册/登录 HuggingFace：https://huggingface.co/join
  2. 打开仓库页 https://huggingface.co/datasets/xuyzshaun/360VOTS
     点击 "Request access" 提交申请，等待通过（邮件/页面确认）。
  3. 创建 Read 权限令牌：https://huggingface.co/settings/tokens
  4. 设置访问令牌（二选一）：
       a. 写入环境变量 HF_TOKEN（禁止写进代码/命令行历史）：
            Windows 当前会话:  set HF_TOKEN=hf_xxxxxxxx
            Windows 永久:      setx HF_TOKEN hf_xxxxxxxx   （重开终端生效）
            Linux/macOS:       export HF_TOKEN=hf_xxxxxxxx
       b. 或写入标准令牌文件 ~/.cache/huggingface/token
          （huggingface-cli login 同款位置，本脚本会自动读取）
  5. 安装下载依赖：pip install huggingface_hub
     （未安装时本脚本打印本提示并以退出码 2 退出；仅本下载脚本需要，
       跟踪/评测代码仍只依赖 numpy/Pillow/scipy）

用法（工作目录 D:\\instan\\pano360）：
  python scripts/download_360vot.py                          # 默认 5 个代表序列
  python scripts/download_360vot.py --seqs 001,002,003 --out data360 --extract

注意：
  - 仓库内 360VOT-test 前缀下文件为 4 位编号（0001.zip~0120.zip），
    360VOS-test 前缀下为 3 位编号（001.zip~120.zip）；脚本按 --prefix 自动选择。
  - 默认直连官方端点 huggingface.co（hf-mirror.com 对 gated 仓库不可用）；
    如需镜像请自行 export HF_ENDPOINT=https://hf-mirror.com。
进度与错误信息走 stderr，摘要走 stdout。
退出码：0 全部成功；1 部分/全部失败；2 缺少 huggingface_hub；3 缺少 HF_TOKEN。
"""
import os

# 不默认设置 HF_ENDPOINT：直连官方端点（hf-mirror 对 gated 仓库会 404/连接失败）
# 若用户自行设置了 HF_ENDPOINT，huggingface_hub 会尊重该值。

import argparse
import sys
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print('缺少依赖 huggingface_hub，请先执行：\n'
          '    pip install huggingface_hub\n'
          '然后重试本脚本。（该依赖仅供下载使用，不进入项目 requirements）',
          file=sys.stderr)
    sys.exit(2)

REPO_ID = 'xuyzshaun/360VOTS'
DEFAULT_SEQS = ('001', '002', '003', '004', '005')   # 默认 5 个代表序列
DEFAULT_PREFIX = '360VOT-test'      # 仓库内测试集路径前缀：<prefix>/<seq>.zip
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOKEN_GUIDE = """\
未检测到 HF_TOKEN 环境变量。360VOTS 是 gated（需申请权限）数据集，请按以下步骤操作：
  1. 注册/登录 HuggingFace：https://huggingface.co/join
  2. 打开数据集仓库页：https://huggingface.co/datasets/xuyzshaun/360VOTS
     点击 "Request access"（同意条款并提交申请），等待邮件/页面确认通过。
  3. 创建访问令牌（Read 权限即可）：https://huggingface.co/settings/tokens
  4. 把令牌写入环境变量 HF_TOKEN（不要写进代码）：
       Windows 当前会话:  set HF_TOKEN=hf_xxxxxxxx
       Windows 永久:      setx HF_TOKEN hf_xxxxxxxx   （重开终端生效）
       Linux/macOS:       export HF_TOKEN=hf_xxxxxxxx
  5. 重新运行本脚本。"""


def _log(msg):
    """日志一律走 stderr（私有）。"""
    print(msg, file=sys.stderr, flush=True)


def _normalize_seq(s, width=3):
    """序列号规范化：纯数字补零到 width 位（默认 3 位，'1' -> '001'，私有）。

    width 由 --prefix 决定：360VOT-test 下文件为 4 位编号（0001.zip~0120.zip），
    360VOS-test 下为 3 位编号（001.zip~120.zip）。
    """
    s = s.strip()
    return s.zfill(width) if s.isdigit() else s


def _safe_extract(zip_path, dst):
    """解压 zip 到 dst，防路径穿越（私有）。"""
    dst = Path(dst).resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            try:
                (dst / m).resolve().relative_to(dst)
            except ValueError:
                raise ValueError(f'zip 内含越界路径: {m!r}')
        zf.extractall(dst)


def download_sequence(seq, out, prefix, token, extract):
    """下载单个序列 zip（可选解压并删 zip，私有）。

    参数：seq 序列号；out 输出根目录；prefix 仓库内路径前缀；
         token HF 令牌；extract 是否解压后删除 zip。
    返回：True 成功 / False 失败（错误信息已打印到 stderr）。
    """
    filename = f'{prefix}/{seq}.zip'
    _log(f'[下载] {REPO_ID}:{filename} -> {out}')
    try:
        path = hf_hub_download(repo_id=REPO_ID, filename=filename,
                               repo_type='dataset', local_dir=str(out),
                               token=token)
    except HfHubHTTPError as e:
        code = getattr(getattr(e, 'response', None), 'status_code', None)
        if code in (401, 403):
            _log(f'  失败：无访问权限（HTTP {code}）。请确认仓库页申请已通过、'
                 f'HF_TOKEN 有效；未申请请先按脚本docstring步骤申请。')
        elif code == 404:
            _log(f'  失败：仓库内不存在 {filename}（HTTP 404）。'
                 f'可用 --prefix 调整路径前缀（当前 {prefix!r}）。')
        else:
            _log(f'  失败：HTTP {code}: {e}')
        return False
    except Exception as e:          # 网络错误/镜像不可用等
        _log(f'  失败：{type(e).__name__}: {e}')
        return False
    _log(f'  完成：{path}')
    if extract:
        try:
            _safe_extract(path, Path(path).parent)
            os.remove(path)
            _log('  已解压并删除 zip')
        except Exception as e:
            _log(f'  解压失败（zip 保留在 {path}）：{e}')
            return False
    return True


def main(argv=None):
    """解析参数并批量下载序列。

    参数：argv 命令行参数（None 取 sys.argv）。
    返回：退出码（0 全部成功；1 有失败；3 缺 HF_TOKEN）。
    """
    p = argparse.ArgumentParser(
        description='下载 360VOT 测试集序列（HF gated 仓库 xuyzshaun/360VOTS）')
    p.add_argument('--seqs', default=','.join(DEFAULT_SEQS),
                   help=f'逗号分隔序列号，默认 {",".join(DEFAULT_SEQS)}')
    p.add_argument('--out', default=str(PROJECT_ROOT / 'data360'),
                   help='输出根目录（默认 <项目>/data360）')
    p.add_argument('--prefix', default=DEFAULT_PREFIX,
                   help=f'仓库内路径前缀（默认 {DEFAULT_PREFIX}，形如 <prefix>/001.zip；'
                        f'360VOT-test 为 4 位编号 0001.zip~0120.zip）')
    p.add_argument('--extract', action='store_true',
                   help='下载后解压并删除 zip（推荐，评测脚本只识别解压后的目录）')
    args = p.parse_args(argv)

    token = os.environ.get('HF_TOKEN')
    if not token:
        # 兜底：读取标准 HF 令牌文件（huggingface-cli login 同款位置）
        tok_file = Path.home() / '.cache' / 'huggingface' / 'token'
        if tok_file.is_file():
            token = tok_file.read_text(encoding='utf-8').strip()
    if not token:
        print(TOKEN_GUIDE, file=sys.stderr)
        return 3

    # 编号位数：360VOT-test 为 4 位（0001.zip），其余默认 3 位（001.zip）
    width = 4 if 'VOT-test' in args.prefix else 3
    seqs = [_normalize_seq(s, width) for s in args.seqs.split(',') if s.strip()]
    out = Path(args.out)
    ok, failed = [], []
    for seq in seqs:
        (ok if download_sequence(seq, out, args.prefix, token, args.extract)
         else failed).append(seq)

    print(f'成功 {len(ok)}/{len(seqs)}：' + (', '.join(ok) if ok else '无'))
    if failed:
        _log('失败序列：' + ', '.join(failed))
    elif not args.extract:
        print('提示：未加 --extract，zip 尚未解压；解压后即可评测。')
    if ok and args.extract:
        print(f'下一步评测：python scripts/eval_360vot.py --data {out} '
              f'--seqs all --downscale 0.5')
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
