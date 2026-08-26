#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRT-360 技术方案 HTML 生成器：读 docs/schemes/schemes.json 渲染全部方案页面 + 索引页。
用法: python scripts/gen_schemes_html.py
输出: docs/schemes/<id>.html (每方案一页) + docs/schemes/index.html
"""
import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "schemes" / "schemes.json"
OUT = ROOT / "docs" / "schemes"

BADGE_COLORS = {
    "已提交": "green",
    "保留": "blue",
    "核心候选": "green",
    "已实测": "blue",
    "开发中": "amber",
    "已废弃": "red",
    "历史": "amber",
    "已实现": "blue",
    "已降级": "amber",
    "调研": "purple",
    "验证中": "amber",
    "未进入": "amber",
}

CAT_GROUPS = [
    ("全景适配 · 三平铺", "real"),
    ("全景适配 · ERP 环绕", "real"),
    ("通用 SOT · 自回归", "real"),
    ("通用 SOT · LoRA 微调", "real"),
    ("域微调 · 训练", "real"),
    ("多模型融合", "real"),
    ("多模型路由 · 创新实验", "real"),
    ("自适应系统 · Phase 2 创新", "real"),
    ("系统级外挂 · 可靠性门控", "real"),
    ("经典方法 · 几何框架", "real"),
    ("经典方法 · 几何框架 + 学习跟踪器", "real"),
    ("全帧跟踪 · 历史方案", "real"),
    ("全帧跟踪 · 轻量 ONNX", "real"),
    ("全景专项 · 记忆增强", "real"),
    ("调研候选 · 未实跑", "research"),
    ("调研候选 · SAM 生态", "research"),
    ("调研对照 · 文献标杆", "research"),
    ("调研对照 · 文献证据", "research"),
    ("调研借鉴 · 未开源", "research"),
    ("工程加速 · 部署优化", "research"),
]


def esc(s):
    return html.escape(str(s), quote=True)


def badge_for(status):
    for key, color in BADGE_COLORS.items():
        if key in status:
            return f'<span class="badge {color}">{esc(status)}</span>'
    return f'<span class="badge">{esc(status)}</span>'


def group_for(cat):
    for name, grp in CAT_GROUPS:
        if name in cat:
            return grp
    return "real"


def sidenav(schemes, active_id):
    parts = ['<div class="sidenav"><h4>📚 方案目录</h4>']
    groups = [("real", "真实跑过"), ("research", "调研候选")]
    for grp, title in groups:
        parts.append(f'<div class="grp">{title}</div>')
        for s in schemes:
            if group_for(s["category"]) != grp:
                continue
            cls = " active" if s["id"] == active_id else ""
            parts.append(
                f'<a class="{cls.strip() or ""}" href="{s["id"]}.html">{esc(s["name"])}</a>'
            )
    parts.append('</div>')
    return "\n".join(parts)


def page_shell(title, active_id, schemes, body):
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)} · GRT-360 技术方案图谱</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="topbar">
  <a class="logo" href="index.html"><span class="dot"></span>GRT-360 技术方案图谱</a>
  <span class="crumb">影石全景视频智能跟踪赛道</span>
  <span class="spacer"></span>
  <span class="badge-top">20 方案 · 2026-08-26</span>
</header>
<div class="layout">
{sidenav(schemes, active_id)}
<main class="main">
{body}
</main>
</div>
<footer class="footer">
  GRT-360 全景跟踪 · 数据源: docs/schemes/schemes.json（2026-08-26）· 由生成器渲染，改数据后运行 scripts/gen_schemes_html.py 刷新
</footer>
</body>
</html>
"""


def scheme_page(s, schemes):
    metrics = "\n".join(
        f'<div class="metric"><div class="m-name">{esc(m["name"])}</div>'
        f'<div class="m-value">{esc(m["value"])}</div>'
        f'<div class="m-scope">{esc(m.get("scope", ""))}</div></div>'
        for m in s["metrics"]
    )
    pros = "\n".join(f"<li>{esc(p)}</li>" for p in s["pros"])
    cons = "\n".join(f"<li>{esc(c)}</li>" for c in s["cons"])
    files = "\n".join(
        f'<div class="file-item"><span class="tag">📄</span>{esc(f)}</div>' for f in s["files"]
    )
    tech_details = esc(s.get("tech_details", ""))
    why_works = esc(s.get("why_works", ""))
    body = f"""
<section class="scheme-header">
  <div class="meta-line">
    {badge_for(s['status'])}
    <span class="badge">{esc(s['category'])}</span>
    <span class="badge blue">⏱ {esc(s['period'])}</span>
  </div>
  <h1>{esc(s['name'])}</h1>
  <div class="summary">{esc(s['summary'])}</div>
</section>

<div class="card">
  <h2><span class="ico">📊</span>关键指标</h2>
  <div class="metrics">{metrics}</div>
</div>

<div class="card">
  <h2><span class="ico">🏗️</span>架构设计</h2>
  <div class="arch">{esc(s['architecture'])}</div>
</div>

<div class="card">
  <h2><span class="ico">🔬</span>技术细节</h2>
  <div class="arch">{tech_details}</div>
</div>

<div class="card">
  <h2><span class="ico">🧠</span>机理分析：为什么在长处场景有效</h2>
  <div class="arch">{why_works}</div>
</div>

<div class="card">
  <h2><span class="ico">⚖️</span>优缺点</h2>
  <div class="two-col">
    <div class="pros"><h3>优点</h3><ul>{pros}</ul></div>
    <div class="cons"><h3>缺点</h3><ul>{cons}</ul></div>
  </div>
</div>

<div class="card">
  <h2><span class="ico">🎯</span>适用场景</h2>
  <div class="arch">{esc(s['scenarios'])}</div>
</div>

<div class="card">
  <h2><span class="ico">🗂️</span>相关文件</h2>
  <div class="file-list">{files}</div>
</div>

<div class="card">
  <h2><span class="ico">📌</span>结论</h2>
  <div class="decision"><span class="label">决策 →</span>{esc(s['decision'])}</div>
</div>
"""
    return page_shell(s["name"], s["id"], schemes, body)


def index_page(schemes):
    rows = []
    for s in schemes:
        rows.append(
            f"<tr><td><a href='{s['id']}.html'>{esc(s['name'])}</a></td>"
            f"<td>{esc(s['category'])}</td>"
            f"<td>{badge_for(s['status'])}</td>"
            f"<td class='num'>{'; '.join(m['value'] for m in s['metrics'][:2])}</td>"
            f"<td style='color:var(--text-dim);font-size:12px;'>{esc(s['period'])}</td></tr>"
        )
    cards = []
    for s in schemes:
        cards.append(
            f"""<a class="index-card" href="{s['id']}.html">
  <div class="meta-line">{badge_for(s['status'])}</div>
  <h3>{esc(s['name'])}</h3>
  <div class="cat">{esc(s['category'])} · {esc(s['period'])}</div>
  <div class="desc">{esc(s['summary'])}</div>
  <div class="foot">
    <span class="badge blue">{'; '.join(m['value'] for m in s['metrics'][:2])}</span>
  </div>
</a>"""
        )
    body = f"""
<section class="index-hero">
  <h1>GRT-360 全景跟踪技术方案图谱</h1>
  <p>20 个技术方案（14 个真实跑过 + 6 个调研候选）的完整剖析：架构设计、优缺点、适用场景与关键指标。
  数据来自 2026-08-26 拉取的远端赛马汇总与 8/21 四路技术调研。</p>
  <div class="legend">
    <span class="item"><span class="badge green">已提交/核心候选</span></span>
    <span class="item"><span class="badge blue">保留/已实测</span></span>
    <span class="item"><span class="badge amber">开发中/降级</span></span>
    <span class="item"><span class="badge red">已废弃</span></span>
    <span class="item"><span class="badge purple">调研候选</span></span>
  </div>
</section>

<div class="card">
  <h2><span class="ico">📑</span>全方案总览</h2>
  <div style="overflow-x:auto;">
  <table class="overview-table">
    <thead><tr><th>方案</th><th>类别</th><th>状态</th><th>关键指标</th><th>阶段</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  </div>
</div>

<h2 style="margin:8px 4px 4px;font-size:17px;">方案卡片</h2>
<div class="index-grid">{''.join(cards)}</div>
"""
    return page_shell("技术方案图谱", "", schemes, body)


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    schemes = data["schemes"]
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for s in schemes:
        (OUT / f"{s['id']}.html").write_text(scheme_page(s, schemes), encoding="utf-8")
        n += 1
    (OUT / "index.html").write_text(index_page(schemes), encoding="utf-8")
    print(f"生成完成: {n} 个方案页面 + index.html，输出到 {OUT}")


if __name__ == "__main__":
    main()
