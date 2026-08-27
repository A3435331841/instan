#!/usr/bin/env python3
"""生成全量130序列逐序列技术方案对比表。"""
import csv

INPUT = 'reports/grt360_scene_comparison_20260826/failure_matrix.csv'
OUTPUT = 'reports/full_130_comparison.md'

methods = [
    ('baseline', 'ODTrack'),
    ('sutrack_t224', 'SUTrack-T224'),
    ('sutrack_b224', 'SUTrack-B224*'),
    ('uetrack', 'UETrack'),
    ('lorat', 'LoRAT'),
    ('ft_v4_ep1', 'ft-v4-ep1'),
    ('ft_v4_ep4', 'ft-v4-ep4'),
    ('ft_ep6', 'ft-ep6'),
    ('ft_ep7', 'ft-ep7'),
]

with open(INPUT, encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))

rows_sorted = sorted(rows, key=lambda r: float(r.get('baseline_auc', 0)))

with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('# 全量130序列逐序列技术方案对比\n\n')
    f.write('> **B224\\*** 列仅有30/130条数据（medium30子集）。B224全量130 AUC=0.6113（单独评测）。\n')
    f.write('> ODTrack全量130 AUC=0.5882（官方基线），本表中0.5813为冻结参考版本。\n\n')

    # 覆盖率表
    f.write('## 方法覆盖率\n\n')
    f.write('| 方法 | 覆盖 | 平均AUC | 平均SR |\n')
    f.write('|---|---:|---:|---:|\n')
    for prefix, name in methods:
        aucs = [float(r[f'{prefix}_auc']) for r in rows
                if r.get(f'{prefix}_auc', 'nan') not in ('nan', '', 'None')]
        srs = [float(r[f'{prefix}_sr']) for r in rows
               if r.get(f'{prefix}_sr', 'nan') not in ('nan', '', 'None')]
        cov = f'{len(aucs)}/130'
        avg_auc = f'{sum(aucs)/len(aucs):.4f}' if aucs else '-'
        avg_sr = f'{sum(srs)/len(srs):.4f}' if srs else '-'
        f.write(f'| {name} | {cov} | {avg_auc} | {avg_sr} |\n')

    # 逐序列表
    f.write('\n## 逐序列AUC对比（按ODTrack AUC升序，最难在前）\n\n')
    f.write('| # | 序列 | 域 | 帧 | 场景 | ODTrack | T224 | B224* | UETrack | LoRAT | ft-v4-ep1 | ft-v4-ep4 | ft-ep6 | ft-ep7 | 最优方案 |\n')
    f.write('|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n')

    for i, r in enumerate(rows_sorted, 1):
        seq = r['sequence'].replace('train_real/', 'R/').replace('train_sim/', 'S/')
        domain = 'real' if 'real' in r['domain'] else 'sim'
        nframes = r['n_frames']
        tags = r.get('scene_tags', '')[:25]

        cols = []
        for prefix, _ in methods:
            auc = r.get(f'{prefix}_auc', 'nan')
            if auc in ('nan', '', 'None'):
                cols.append('-')
            else:
                cols.append(f'{float(auc):.3f}')

        best = r.get('best_method', '')
        best_auc = r.get('best_auc', 'nan')
        if best_auc not in ('nan', '', 'None'):
            best_str = f'{best}({float(best_auc):.3f})'
        else:
            best_str = '-'

        f.write(f'| {i} | {seq} | {domain} | {nframes} | {tags} | '
                f'{" | ".join(cols)} | {best_str} |\n')

print(f'Saved {len(rows)} sequences to {OUTPUT}')
