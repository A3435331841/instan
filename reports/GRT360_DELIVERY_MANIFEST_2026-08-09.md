# GRT-360 交付清单（2026-08-09）

## 版本与范围

- Git 分支：`agent/panotrack-v2`
- GitHub 草稿 PR：<https://github.com/A3435331841/instan/pull/1>
- 结果发布提交：`694c176`（后续仅更新文档锚点）
- 严格评测范围：360VOT `0001–0120`，120 条序列。
- 已完成的数值架构：UETrack、UETrack ERP-wrap、LightFC ONNX、ODTrack Base + ERP 三平铺适配。
- LoRAT 官方 `base.bin` 已取得并校验，但尚未接入严格 GRT-360 帧级适配器，未列入数值排名。

## 全量赛马摘要

| 架构 | AUC | 双 IoU AUC | SR | 双 IoU SR | FPS | 条数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ODTrack Base + ERP 三平铺适配 | 0.5792135073 | 0.5819064920 | 0.6531941586 | 0.6562441036 | 8.9945 | 120 |
| UETrack ERP-wrap | 0.5142648726 | 0.5162523388 | 0.5776136689 | 0.5797524149 | 57.1606 | 120 |
| UETrack 基线 | 0.4167971998 | 0.4238449196 | 0.4524688818 | 0.4605407583 | 63.7707 | 120 |
| LightFC ONNX | 0.3115977321 | 0.3127627102 | 0.3298604941 | 0.3310282900 | 9.7875 | 120 |

## 本地文件

| 交付物 | 路径 | SHA-256 |
| --- | --- | --- |
| 中文最终报告 | `D:/instan/pano360/reports/STAGE_RESULTS_2026-08-09.md` | `8c9056b0998f40ba1ae1a0ec425fb1508ab5b34e48f61eb37e1397c2d252f1e4` |
| 四架构汇总 JSON | `D:/instan/pano360/reports/results/architecture_bakeoff_0001_0120.json` | `f01b0fa771ff01301ddd4c6e62c4f28a26d3385b4da2535393e5b9a70d274b2a` |
| UETrack JSON | `D:/instan/pano360/reports/results/erpwrap_ablation_0001_0120_bakeoff.json` | `d49046a7a0395c48aae451f584e8f729244f0ead6e286dc29f00055d9dc0a333` |
| UETrack CSV | `D:/instan/pano360/reports/results/erpwrap_ablation_0001_0120_scores.csv` | `b5e99afd9ce2322e7d9b3104e79dd5850ca8a76842e06618e18a972d41fa3953` |
| LightFC JSON | `D:/instan/pano360/reports/results/lightfc_120_score/bakeoff.json` | `0a1d8b00a6aec53d611c5ea1597abb69975cf2ff3af07d5a01ce2a3d78c406d5` |
| LightFC CSV | `D:/instan/pano360/reports/results/lightfc_120_score/scores.csv` | `8030e7132961c87b2f317d6255e491d3ea92dcae1146e6b8910fa465d11b6684` |
| ODTrack JSON | `D:/instan/pano360/reports/results/odtrack_120_score/bakeoff.json` | `ec09cbfd3582c2bf798a19aef9932756711a2f501d8bda51c62dcfd1c3c0956e` |
| ODTrack CSV | `D:/instan/pano360/reports/results/odtrack_120_score/scores.csv` | `208191a09eb6ef9e0a8e4f611719d5c0934d26c89e705b02dd2e254adaaabd4b` |
| LightFC 原始归档 | `D:/instan/pano360/reports/results/lightfc_results_0001_0120.tar.zst` | `0c608c0ea28177d6d75aaca0cfb8097f3d327bf9afc97166b8cad8a705445aa7` |
| ODTrack 原始归档 | `D:/instan/pano360/reports/results/odtrack_results_0001_0120.tar.zst` | `15a2f551dea0043466a24c0620601d321d3bc3dff63301239aa6765c8e32a1a9` |
| 中文汇报 PPT | `D:/instan/pano360/reports/GRT360_Research_Handoff_2026-08-09.pptx` | 已有交付物 |

原始结果归档不放入 Git；评分 JSON/CSV、汇总 JSON、中文报告和清单会提交 GitHub。

## 服务器文件

- 代码：`/data/projects/instan_grt360`
- 数据：`/data/projects/instan/data360`
- 本轮运行根目录：`/data/projects/instan/runs/grt360_20260809/`
- LightFC 评分：`/data/projects/instan/runs/grt360_20260809/lightfc_120_score/`
- ODTrack 评分：`/data/projects/instan/runs/grt360_20260809/odtrack_120_score/`
- ODTrack 合并结果：`/data/projects/instan/runs/grt360_20260809/odtrack_120/`
- 收尾标记：`/data/projects/instan/runs/grt360_20260809/FINALIZED`
- LightFC 120 条结果目录：`/data/projects/instan/runs/grt360_20260809/lightfc_120/`
- ODTrack 两个 60 条分片：`odtrack_120_gpu0/`、`odtrack_120_gpu1/`

## 复现锚点

- UETrack 上游提交：`fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack 权重 SHA-256：`1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 缓存 SHA-256：`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`
- ODTrack 官方权重 SHA-256：`2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e`
- LoRAT `base.bin` SHA-256：`150edc6635c7615a82d7fd50d95d84f8e47a47c9217e8fd5b3dd326589aac23e`

## 验证边界

- UETrack 两个版本各 120 条；LightFC 120 条；ODTrack GPU0/GPU1 各 60 条，合并后 120 条。
- 四组评分 JSON 的 `n_sequences` 均为 120，逐序列 CSV 均为 120 行。
- 预测框与 GT 行数严格匹配；首帧只初始化；普通 IoU 与双 IoU 均已计算。
- LightFC 0069 的异常裁剪问题已修复并从该序列重跑；最终 120 条无 OOM 退出。
- 密码、token 和私有缓存未进入 Git 跟踪文件。
