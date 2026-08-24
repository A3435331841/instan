#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方训练集 train/valid 留出集划分（2026-08-24）。

纪律（对应 docs/EXECUTION_PLAN_ZH_2026-08-22.md 的 official-valid 约定）：
  - valid 留出集（15 real + 20 sim）绝不参与任何训练/微调；
  - 划分用固定种子确定性生成，任何人重跑得到同一结果；
  - 划分文件入库（split.json + 两个 seqlist），供训练与评测代码共同消费。

用法:
    python scripts/make_official_split.py                    # 从本地 zip 读取序列清单
    python scripts/make_official_split.py --root <解压目录>   # 从解压后的 train/ 目录读取
"""
import argparse
import json
import random
import zipfile
from pathlib import Path

SEED = 20260824
N_VALID = {"train_real": 15, "train_sim": 20}   # 其余进训练集
DEFAULT_ZIP = Path(r"D:\instan\初赛数据\ys_panotracking_train.zip")
OUT_DIR = Path(__file__).resolve().parents[1] / "data360" / "official_split"


def sequences_from_zip(zip_path: Path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    seqs = set()
    for n in names:
        parts = n.split("/")
        if len(parts) >= 3 and parts[2].startswith("seq_"):
            seqs.add((parts[1], parts[2]))
    return sorted(seqs)


def sequences_from_root(root: Path):
    seqs = set()
    for block in ("train_real", "train_sim"):
        d = root / block
        if not d.is_dir():
            continue
        for sub in sorted(d.iterdir()):
            if sub.is_dir() and sub.name.startswith("seq_"):
                seqs.add((block, sub.name))
    return sorted(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    ap.add_argument("--root", type=Path, default=None,
                    help="解压后的 train/ 目录（优先于 --zip）")
    args = ap.parse_args()

    if args.root is not None:
        seqs = sequences_from_root(args.root)
        src = str(args.root)
    else:
        seqs = sequences_from_zip(args.zip)
        src = str(args.zip)

    by_block = {}
    for block, seq in seqs:
        by_block.setdefault(block, []).append(seq)

    rng = random.Random(SEED)
    split = {"train": [], "valid": []}
    for block, items in sorted(by_block.items()):
        items = sorted(items)
        valid_n = min(N_VALID.get(block, 0), len(items))
        valid_pick = sorted(rng.sample(items, valid_n))
        valid_set = set(valid_pick)
        split["valid"].extend({"block": block, "seq": s} for s in valid_pick)
        split["train"].extend(
            {"block": block, "seq": s} for s in items if s not in valid_set)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_path = OUT_DIR / "split.json"
    split_path.write_text(json.dumps({
        "seed": SEED,
        "source": src,
        "n_train": len(split["train"]),
        "n_valid": len(split["valid"]),
        "per_block": {b: len(v) for b, v in by_block.items()},
        **split,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "seqlist_official_train.txt").write_text(
        "\n".join(f"{x['block']}/{x['seq']}" for x in split["train"]) + "\n",
        encoding="utf-8")
    (OUT_DIR / "seqlist_official_valid.txt").write_text(
        "\n".join(f"{x['block']}/{x['seq']}" for x in split["valid"]) + "\n",
        encoding="utf-8")

    print(f"source: {src}")
    for b, items in sorted(by_block.items()):
        v = sum(1 for x in split["valid"] if x["block"] == b)
        print(f"{b}: 共 {len(items)} -> train {len(items)-v} / valid {v}")
    print(f"train {len(split['train'])} + valid {len(split['valid'])} = {len(seqs)}")
    print(f"written: {split_path}")


if __name__ == "__main__":
    main()
