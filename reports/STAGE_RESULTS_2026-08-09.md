# GRT-360 完整赛马研究报告（2026-08-09）

## 结论先说

已完成 0001–0120 全部 120 条序列的多架构严格评测。当前纳入数值排名的四组结果如下：

| 架构/版本 | 序列数 | 普通 AUC | 双 IoU AUC | SR@0.5 | 双 IoU SR | 观测 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ODTrack Base + ERP 三平铺适配 | 120 | **0.5792** | **0.5819** | **0.6532** | **0.6562** | 8.99 |
| UETrack ERP-wrap | 120 | 0.5143 | 0.5163 | 0.5776 | 0.5798 | **57.16** |
| UETrack 基线 | 120 | 0.4168 | 0.4238 | 0.4525 | 0.4605 | 63.77 |
| LightFC ONNX | 120 | 0.3116 | 0.3128 | 0.3299 | 0.3310 | 9.79 |

因此，按精度指标 ODTrack 是本轮全量赛马第一；按速度 UETrack 基线最快。ERP-wrap 相对 UETrack 基线提升普通 AUC `+0.0975`、SR `+0.1251`。LightFC 已完成全量评测，但精度低于另外三组。

LoRAT 官方 `base.bin` 已取得并校验，但官方 TrackIt/VOT 流程还没有直接接入本项目严格 360VOT 帧级协议的适配器。为避免把不一致的普通基准结果混入排名，LoRAT 只作为“权重已就绪、尚未纳入严格数值表”记录。

## 一、统一评测协议

- 数据集：GRT-360 / 360VOT 测试集，序列 `0001–0120`，共 120 条。
- 首帧只用于初始化，不计入精度；预测框与 GT 行数必须严格一致。
- 计算普通 IoU 以及 ERP 水平环绕后的双 IoU（水平位移 `-W/0/+W`）。
- AUC 使用 21 个成功阈值，SR 为 IoU 阈值 0.5 下的成功率；序列等权宏平均。
- 所有全量结果均由同一 `score_external_results.py` 严格评分器生成。

## 二、各架构全量结果

### 1. UETrack

ERP-wrap 将水平方向的黑色填充替换为 ERP 环形采样，并保留跨 seam 目标框范围；垂直方向仍使用普通填充。两种版本均完成 120 条全量评测：

- ERP-wrap：AUC `0.5142648726`，双 IoU AUC `0.5162523388`，SR `0.5776136689`，双 IoU SR `0.5797524149`，FPS `57.1606`。
- 基线：AUC `0.4167971998`，双 IoU AUC `0.4238449196`，SR `0.4524688818`，双 IoU SR `0.4605407583`，FPS `63.7707`。

### 2. LightFC ONNX

使用真实 LightFC backbone/tracking ONNX，CUDAExecutionProvider，完整 120 条、严格 1.0 分辨率评测。全量平均：AUC `0.3115977321`，双 IoU AUC `0.3127627102`，SR `0.3298604941`，双 IoU SR `0.3310282900`，FPS `9.7875`。

为处理 4K ERP 长序列中的异常失锁框，加入了每条序列独立进程和 `max_crop_size=2048` 硬上限；这样 0069 等序列不会再把主机内存推到 OOM，且 120 条结果均已落盘。

### 3. ODTrack

使用官方 Base 全数据 300 epoch 权重，并加入 ERP 三平铺适配：将每帧水平复制为三块，初始框放入中间块，输出框再映射回原 ERP 坐标。两张 RTX 3090 各跑 60 条，最终合并为 120 条严格结果。

全量平均：AUC `0.5792135073`，双 IoU AUC `0.5819064920`，SR `0.6531941586`，双 IoU SR `0.6562441036`，FPS `8.9945`。GPU0/GPU1 各 60 条，预测行数与 GT 全部匹配。

### 4. LoRAT

官方权重 `base.bin` 已下载、上传服务器并完成 SHA-256 校验（`150edc6635c7615a82d7fd50d95d84f8e47a47c9217e8fd5b3dd326589aac23e`）。当前缺少与本项目严格协议一致的直接帧级适配器及依赖闭环，因此没有把普通 TrackIt/VOT 结果冒充 GRT-360 严格成绩。

## 三、工程与运行记录

- 服务器：`root@153.0.134.134:12409`，两张 RTX 3090 24GB。
- LightFC 使用 GPU0/GPU1 分片队列；每条序列独立 Python 进程，完成即写入 `metrics.json` 和 `results.txt`。
- 0069 曾暴露异常预测框导致裁剪数组膨胀的问题；已终止异常进程、加入裁剪上限后从 0069 重跑，内存恢复正常，未影响其他序列。
- ODTrack 使用同样的两 GPU 分片策略，并在两个分片全部结束后进行统一严格评分。
- 关键修复已推送 GitHub：`2484144`（ODTrack 适配与运行保护）、`2cc7ddd`（异常裁剪保护）、`d12def4`（配置显式记录）。

## 四、结果与证据保存位置

本地（Windows）：

- 中文报告：`D:/instan/pano360/reports/STAGE_RESULTS_2026-08-09.md`
- 四架构汇总证据：`D:/instan/pano360/reports/results/architecture_bakeoff_0001_0120.json`
- UETrack 全量 JSON/CSV：`D:/instan/pano360/reports/results/erpwrap_ablation_0001_0120_bakeoff.json`、`erpwrap_ablation_0001_0120_scores.csv`
- LightFC 全量 JSON/CSV：`D:/instan/pano360/reports/results/lightfc_120_score/bakeoff.json`、`scores.csv`
- ODTrack 全量 JSON/CSV：`D:/instan/pano360/reports/results/odtrack_120_score/bakeoff.json`、`scores.csv`
- LightFC 原始结果归档：`D:/instan/pano360/reports/results/lightfc_results_0001_0120.tar.zst`
- ODTrack 原始结果归档：`D:/instan/pano360/reports/results/odtrack_results_0001_0120.tar.zst`
- 中文汇报 PPT：`D:/instan/pano360/reports/GRT360_Research_Handoff_2026-08-09.pptx`

服务器：

- 代码：`/data/projects/instan_grt360`
- 数据：`/data/projects/instan/data360`
- 运行根目录：`/data/projects/instan/runs/grt360_20260809/`
- LightFC 严格评分：`/data/projects/instan/runs/grt360_20260809/lightfc_120_score/`
- ODTrack 严格评分：`/data/projects/instan/runs/grt360_20260809/odtrack_120_score/`
- ODTrack 合并结果：`/data/projects/instan/runs/grt360_20260809/odtrack_120/`
- 自动收尾标记：`/data/projects/instan/runs/grt360_20260809/FINALIZED`

GitHub：

- 分支：`agent/panotrack-v2`
- 草稿 PR：<https://github.com/A3435331841/instan/pull/1>
- 结果发布提交：`694c176`（后续仅更新文档锚点）

## 五、完整性检查

- UETrack：基线 120 条 + ERP-wrap 120 条，预测/GT 行数全部匹配。
- LightFC：120/120 条，CSV 120 行，JSON `n_sequences=120`。
- ODTrack：GPU0 60/60 + GPU1 60/60，合并后 120/120 条，CSV 120 行，JSON `n_sequences=120`。
- 所有原始结果归档和评分 JSON/CSV 均已传回本地；原始归档不放入 Git，评分证据和中文报告会提交 GitHub。
- 密码、token 和私有缓存未进入 Git 跟踪文件。
