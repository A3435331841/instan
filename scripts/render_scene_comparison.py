#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a readable scene/expert comparison from failure_matrix.json."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SCENES = ("hard", "polar", "seam", "small", "large", "fast", "scale", "absent", "drift")
TECHNIQUE = {
    "hard": "逐帧失锁归因；先做重捕获/身份验证，再做主干训练",
    "polar": "切平面投影、极区旋转增强、球面状态",
    "seam": "圆周裁剪/三平铺、dual-IoU、接缝一致性",
    "small": "高分辨率搜索、短时特征聚合、小目标增强",
    "large": "宽ROI+紧ROI双分支、动态搜索尺度",
    "fast": "S²速度先验、多假设搜索中心",
    "scale": "log-FoV滤波、尺度增强、动态搜索窗",
    "absent": "冻结记忆、全局重捕获、re-init验证",
    "drift": "anchor身份校验、低置信门控、模板污染阻断",
}


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _tags(row):
    return set(str(row.get("scene_tags", "")).split(";")) - {""}


def _method_prefixes(rows):
    keys = set().union(*(row.keys() for row in rows))
    return sorted(key[:-4] for key in keys if key.endswith("_auc") and key not in {"baseline_auc", "best_auc"})


def _mean(rows, key):
    values = [_number(row.get(key)) for row in rows]
    values = [value for value in values if math.isfinite(value)]
    return (float(np.mean(values)), len(values)) if values else (math.nan, 0)


def build_scene_rows(rows):
    methods = _method_prefixes(rows)
    scene_rows = []
    for scene in SCENES:
        selected = [row for row in rows if scene in _tags(row)]
        if not selected:
            continue
        baseline_auc, _ = _mean(selected, "baseline_auc")
        baseline_sr, _ = _mean(selected, "baseline_sr")
        candidates = []
        for method in methods:
            auc, coverage = _mean(selected, f"{method}_auc")
            delta, _ = _mean(selected, f"{method}_auc_delta")
            if coverage:
                candidates.append({"method": method, "auc": auc, "delta": delta, "coverage": coverage})
        candidates.sort(key=lambda item: (item["delta"], item["coverage"]), reverse=True)
        t224 = next((item for item in candidates if item["method"] == "sutrack_t224"), None)
        non_t = next((item for item in candidates if item["method"] != "sutrack_t224"), None)
        scene_rows.append({
            "scene": scene,
            "n_sequences": len(selected),
            "odtrack_auc": baseline_auc,
            "odtrack_sr": baseline_sr,
            "t224_delta_auc": t224["delta"] if t224 else math.nan,
            "t224_coverage": t224["coverage"] if t224 else 0,
            "best_expert": non_t["method"] if non_t else "-",
            "best_expert_delta_auc": non_t["delta"] if non_t else math.nan,
            "best_expert_coverage": non_t["coverage"] if non_t else 0,
            "recommended_technique": TECHNIQUE[scene],
        })
    return scene_rows


def render(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = build_scene_rows(rows)
    scene_csv = output_dir / "scene_comparison.csv"
    with scene_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scene_rows)

    hardest = sorted(rows, key=lambda row: _number(row.get("baseline_auc")))[:30]
    markdown = [
        "# GRT-360 场景—方案对比",
        "",
        "说明：ODTrack 和 SUTRACK-T224 的覆盖为全量时可直接比较；其他方案目前多为medium子集，表中括号是该场景的可比序列数，不能外推为全量成绩。",
        "",
        "| 场景 | 序列数 | OD AUC/SR | T224 ΔAUC (n) | 最强已测专家 ΔAUC (n) | 建议技术 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in scene_rows:
        od = f"{row['odtrack_auc']:.3f}/{row['odtrack_sr']:.3f}"
        t = "-" if not math.isfinite(row["t224_delta_auc"]) else f"{row['t224_delta_auc']:+.3f} ({row['t224_coverage']})"
        expert = "-" if not math.isfinite(row["best_expert_delta_auc"]) else (
            f"{row['best_expert']} {row['best_expert_delta_auc']:+.3f} ({row['best_expert_coverage']})")
        markdown.append(f"| {row['scene']} | {row['n_sequences']} | {od} | {t} | {expert} | {row['recommended_technique']} |")
    markdown.extend([
        "",
        "## ODTrack最困难的30条",
        "",
        "| 序列 | 场景 | OD AUC | T224 AUC | 当前最优已测方案 | 增益 |",
        "|---|---|---:|---:|---|---:|",
    ])
    for row in hardest:
        t_auc = _number(row.get("sutrack_t224_auc"))
        best_auc = _number(row.get("best_auc"))
        best_delta = _number(row.get("best_auc_delta"))
        t_text = "-" if not math.isfinite(t_auc) else f"{t_auc:.3f}"
        best_text = str(row.get("best_method", "-"))
        delta_text = "-" if not math.isfinite(best_delta) else f"{best_delta:+.3f}"
        markdown.append(f"| {row['sequence']} | {row.get('scene_tags','-')} | {row['baseline_auc']:.3f} | {t_text} | {best_text} | {delta_text} |")
    markdown_path = output_dir / "scene_comparison.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return scene_csv, markdown_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-matrix", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    rows = json.loads(Path(args.failure_matrix).read_text(encoding="utf-8"))
    scene_csv, markdown = render(rows, args.out)
    print(scene_csv)
    print(markdown)


if __name__ == "__main__":
    main()
