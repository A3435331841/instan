#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the frozen prelim ODTrack and final recapture scheme on a small suite.

The server is no longer reachable and this workstation has no CUDA adapter.
Therefore this report intentionally consumes the already completed, same-
protocol GPU runs copied into ``server_exit_20260827/runs`` rather than
pretending to generate a new CPU score.  It also adds SUTrack-B224 as a
reference because it became the later single-model race leader.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUITE = (
    "train_real/seq_0002",  # recapture win candidate
    "train_real/seq_0003",  # normal high-quality control
    "train_real/seq_0004",  # difficult real scene
    "train_real/seq_0012",  # known recapture regression
    "train_real/seq_0047",  # long sequence / recovery stress
    "train_sim/seq_0047",   # difficult sim control
)


def metrics_map(root: Path) -> dict[str, dict]:
    result = {}
    for path in root.rglob("metrics.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            result[str(item["sequence"])] = {**item, "source": str(path)}
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    return result


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827\runs")
    parser.add_argument("--out", default=r"D:\instan\pano360\reports\decision_vs_final_comparison_20260827")
    parser.add_argument("--seqs", default=None, help="comma-separated sequence names; defaults to the six-case suite")
    args = parser.parse_args(argv)
    runs = Path(args.runs).resolve()
    out = Path(args.out).resolve()
    seqs = tuple(item.strip() for item in args.seqs.split(",")) if args.seqs else SUITE
    prelim = metrics_map(runs / "_all130_gpu0")
    final = metrics_map(runs / "odtrack_recapture_representative_20260825")
    b224 = metrics_map(runs / "phase2_sutrack_20260826" / "sutrack_b224_all130")
    rows = []
    for sequence in seqs:
        p, f, b = prelim.get(sequence), final.get(sequence), b224.get(sequence)
        if not p or not f:
            continue
        rows.append({
            "sequence": sequence,
            "prelim_auc": p.get("auc"), "prelim_sr": p.get("sr"), "prelim_fps": p.get("fps"),
            "final_auc": f.get("auc"), "final_sr": f.get("sr"), "final_fps": f.get("fps"),
            "delta_auc": f.get("auc", 0.0) - p.get("auc", 0.0),
            "delta_sr": f.get("sr", 0.0) - p.get("sr", 0.0),
            "delta_fps": f.get("fps", 0.0) - p.get("fps", 0.0),
            "b224_auc": None if not b else b.get("auc"),
            "b224_sr": None if not b else b.get("sr"),
            "b224_fps": None if not b else b.get("fps"),
            "prelim_source": p["source"], "final_source": f["source"],
        })
    if not rows:
        raise SystemExit("no aligned metrics found for the requested suite")
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    mean = {key: sum(float(row[key]) for row in rows if row.get(key) is not None) / len(rows)
            for key in ("prelim_auc", "prelim_sr", "prelim_fps", "final_auc", "final_sr", "final_fps",
                        "delta_auc", "delta_sr", "delta_fps")}
    lines = [
        "# 初赛方案 vs 决赛方案：小序列对照",
        "",
        "初赛方案定义为冻结的 ODTrack ERP 三平铺；决赛方案定义为 ODTrack + 可靠性门控 + 球面重捕获。",
        "以下结果来自服务器已完成的同一评分协议 GPU 运行，并非本地无 GPU 的伪造 CPU 复测。",
        "",
        "| 序列 | 初赛 AUC/SR/FPS | 决赛 AUC/SR/FPS | ΔAUC | ΔSR | ΔFPS | B224参考 AUC/SR/FPS |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| `{row['sequence']}` | {fmt(row['prelim_auc'])}/{fmt(row['prelim_sr'])}/{fmt(row['prelim_fps'])} "
                     f"| {fmt(row['final_auc'])}/{fmt(row['final_sr'])}/{fmt(row['final_fps'])} "
                     f"| {row['delta_auc']:+.4f} | {row['delta_sr']:+.4f} | {row['delta_fps']:+.2f} "
                     f"| {fmt(row['b224_auc'])}/{fmt(row['b224_sr'])}/{fmt(row['b224_fps'])} |")
    lines += [
        "",
        f"**{len(rows)}条宏平均**：初赛 AUC/SR/FPS = `{mean['prelim_auc']:.4f}/{mean['prelim_sr']:.4f}/{mean['prelim_fps']:.2f}`；"
        f"决赛 = `{mean['final_auc']:.4f}/{mean['final_sr']:.4f}/{mean['final_fps']:.2f}`；"
        f"增量 = `{mean['delta_auc']:+.4f}/{mean['delta_sr']:+.4f}/{mean['delta_fps']:+.2f}`。",
        "",
        "## 判读",
        "",
        "- `seq_0002` 是决赛方案唯一明显正向样本（AUC约+0.028），说明重捕获机制有局部价值。",
        "- `seq_0012`、`sim/seq_0047` 等出现明显回退，说明旧版门控/重捕获不能直接作为全场景提交方案。",
        "- B224 作为后续单模型参考，通常比两套 ODTrack 方案更稳，但速度需要单 GPU 优化。",
        "- 结论只用于下一轮方案取舍，不改变全量验收门槛。",
    ]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps({"suite": list(seqs), "rows": rows, "mean": mean,
                                                        "sources": {"prelim": str(runs / "_all130_gpu0"),
                                                                    "final": str(runs / "odtrack_recapture_representative_20260825"),
                                                                    "b224": str(runs / "phase2_sutrack_20260826" / "sutrack_b224_all130")}},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "mean": mean, "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
