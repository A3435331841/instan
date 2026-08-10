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
| 中文最终报告 | `D:/instan/pano360/reports/STAGE_RESULTS_2026-08-09.md` | `ce5d38da1ffcd0786408e27255f85ec20f33a263db4c2eb2e21647d077802e5f` |
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

> **哈希口径说明（2026-08-10 复核）**：上表 SHA-256 按 **Git 内 LF 规范内容**
> 计算。Windows 检出会做 CRLF 转换，直接对磁盘文件 `sha256sum` 会得到不同
> 哈希（这是换行符差异，不是内容差异）；复核方式：
> `git show HEAD:<路径> | sha256sum`，或磁盘文件去 `\r\n` 后计算。
> 2026-08-10 已复核：5 个本地文件全部与上表一致（数值 AUC/SR 逐位相同）。

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
-
## 融合架构筛选证据

- 融合器：`D:/instan/pano360/scripts/fuse_external_results.py`（SHA-256 `545e6bcd85728c0bca6a80381f69d6805d799a7f05ed725d9271deeab5058fe3`）
- 120 条周期路由结果：`D:/instan/pano360/reports/results/fusion_m5_score/bakeoff.json`（`9f8baf6591b1f5bbed9417b8e0edf75aa50d316b046862bebea7e484f1d59d27`）和 `scores.csv`（`e4dfebe92d4b2615470b004aeb61de9c4e074bc245c1a3a11087bb51a055e6d7`）；AUC `0.5792135073`、SR `0.6531941586`。
- 10 条 ODTrack IoU-head 置信度试跑：`D:/instan/pano360/reports/results/fusion_conf_10_score/`；AUC `0.5344530`、SR `0.5952625`。
- 单元测试：`D:/instan/pano360/tests/test_external_fusion.py`（SHA-256 `6a2ebc7b912d16bc09da657de21109fc1b591d342ef8a94a6705ee7ea19c82fc`）。
-
## 融合器速度证据

- 速度报告：`D:/instan/pano360/reports/results/fusion_m5_runtime.json`。
- 融合路由器本身：112657 个非首帧预测，3 次耗时 `8.04/7.53/8.09 s`，中位约 `14012 FPS`。
- 真实端到端：两张 GPU 并行等待 ODTrack 与 UETrack，约 `8.99 FPS`；串行约 `7.77 FPS`。该融合器没有超过 UETrack 的 57.16 FPS。
-
## 快路径校正试验

- UETrack 置信度试验：`D:/instan/pano360/reports/results/uetrack_confidence_pilot_0001_0010.json`。
- 结论：低阈值速度较快但精度明显下降；高阈值接近 ODTrack 时速度优势消失，暂不作为最终提交架构。

## 2026-08-10 补充：ODTrack 精度版提交镜像

- 镜像：`grt360-odtrack:2026-08-10`（GPU，pytorch 2.3.1 / numpy 2.2.6）；
- 导出：`D:/instan/pano360/artifacts/grt360-odtrack_2026-08-10_linux-amd64.tar`（约 4.3 GB）；
- 构建文件：`docker/odtrack/Dockerfile`、`docker/odtrack/requirements.txt`、
  `docker/odtrack/build_odtrack_image.sh`（构建上下文由
  `artifacts/server_snapshot/upstream/odtrack` + 权重临时组装，权重不入 Git）；
- 提交入口：`integrations/odtrack/file_protocol.py`（文件协议，与 UETrack 镜像同接口）；
- 权重 SHA-256：`2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e`（与上表一致）；
- 验证：`--help` ✅；容器内权重哈希 ✅；CPU 结构冒烟 3 帧 ✅；
  GPU 全量 120 序列复测待服务器执行（本机无 NVIDIA runtime）。
- git 提交：`19972a5`（agent/panotrack-v2）。
