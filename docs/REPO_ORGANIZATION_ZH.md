# 项目仓库整理建议（2026-08-14）

> 目的：梳理 D:\instan 仓库结构，区分「代码/交付物」与「临时产物/可归档数据」，
> 让队友 clone 后能快速上手，也避免磁盘被中间文件占满。

## 一、当前结构概览

```
D:\instan
├── pano360/          # ★ git 仓库根（= GitHub A3435331841/instan）
│   ├── docker/       #   镜像构建（odtrack/uetrack，含 cu128 版）
│   ├── integrations/ #   跟踪器适配（arena_protocol.py 官方协议入口）
│   ├── panotrack/    #   核心算法包（几何/跟踪器/pipeline）
│   ├── tests/        #   测试
│   ├── docs/         #   文档（ARENA_PROTOCOL_TEST_ZH 等）
│   ├── data360/      #   数据集（360VOT 解压帧 + zips 归档）
│   ├── artifacts/    #   上游源码/权重快照（git 忽略）
│   ├── models/       #   ONNX 模型（git 忽略）
│   ├── runs/reports/ #   评测记录/报告
│   └── tools_local/  #   本地工具（git 忽略）
├── deliverables/     # ★ 提交交付物（SUBMISSION_2026-08-10 正式包）
├── 交付物_2026-08-14/ # ★ 镜像+权重打包（给队友）
├── downloads/        #   平台下载（官方 demo 等）
├── project/          #   早期项目章程
├── external/         #   外部参考代码
├── 比赛策略_*.md     #   策略文档（v1.1）
└── (根目录临时文件已清理)
```

## 二、建议保留（核心）

| 位置 | 内容 | 原因 |
|---|---|---|
| pano360/ | 全部代码 + 构建文件 | git 仓库主体，队友 clone 即得 |
| deliverables/SUBMISSION_2026-08-10/ | 正式提交包（PPT/文档/镜像） | 比赛交付记录 |
| 交付物_2026-08-14/ | UETrack/ODTrack cu128 镜像 + 权重 tar | 给队友的二进制交付 |
| 比赛策略_影石全景跟踪赛道.md | 策略 v1.1 | 团队共识文档 |

## 三、建议清理/归档

### 已清理（本轮完成）

- 根目录 _tmp_*.py / _remote_doc_api.json（对比脚本）
- pano360/seq_0008.tgz、seq_0116.tgz 移入 data360/zips/ 归档

### 待确认（需要你/队友决定）

| 项 | 建议 | 风险 |
|---|---|---|
| smoke_dataset/、smoke_dataset2/、smoke_result/ | 删（协议测试数据，可重建） | 低 |
| deliverables/GRT360_2026-08-09/rendered*、ppt_tmp/ | 删（PPT 中间渲染） | 低（正式 PPT 已存 02_答辩PPT） |
| 根目录 airsim_index.html、modlens_test.png 等 | 确认无用后删 | 需人工确认 |
| deliverables/GRT360_2026-08-09/ 整体 | 归档或删除 | 中（含早期结果） |
| pano360/data360/ 解压帧图（59.7GB） | 保留 zips，删解压帧 | 中（本地评测需重新解压） |

## 四、仓库整洁度评估

- 代码部分（pano360）：结构清晰，docker/integrations/panotrack 分区合理，不需要大改。
- 根目录：混有交付物、策略文档、早期材料、临时文件，建议按清单清理。
- git 仓库：pano360/.git 是唯一有效仓库；根目录不在 git 管理内（策略文档未入库）。

## 五、建议的下一步（可选）

1. 把比赛策略_影石全景跟踪赛道.md 复制到 pano360/docs/ 并提交（队友 clone 可见）。
2. 清理确认项后，pano360/data360 可瘦身到只剩 zips（释放约 60GB）。
3. 根目录保留 README 说明各目录用途。

*生成时间：2026-08-14*
