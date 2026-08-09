# GRT-360 最终研究报告（2026-08-09）

120 条序列的成对评测已经全部完成。本报告记录统一评测协议、最终结果、
验证情况和交付物位置。

## 一、版本与复现锚点

- 主仓库分支：`agent/panotrack-v2`
- UETrack 上游提交：`fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack 权重 SHA-256：
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 缓存 SHA-256：
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

## 二、统一三序列初筛

评测协议：严格检查预测框与 GT 行数；第一帧只用于初始化、不计入精度；
同时计算普通 IoU 与 ERP 双 IoU；使用 21 点 AUC、SR@0.5，并按序列等权
进行宏平均。

| 跟踪器 | 序列数 | 普通 AUC | SR@0.5 | 原生 FPS |
| --- | ---: | ---: | ---: | ---: |
| UETrack 基线 | 3 | 0.5247 | 0.5953 | 61.40 |
| LightFC ONNX | 3 | 0.3364 | 0.3687 | 6.78 |

因此后续实验选用 UETrack 作为主干。LoRAT 和 ODTrack 没有可验证的固定权重，
本项目没有编造它们的成绩。

## 三、120 序列最终评测

ERP-wrap 改进将水平方向的黑色填充替换为 ERP 环形采样，同时保留跨 seam
目标框的范围；垂直方向仍使用普通填充。两种版本使用完全相同的严格 scorer
在 0001–0120 全部 120 条序列上评测。

| 版本 | 序列数 | 普通 AUC | 双 IoU AUC | SR@0.5 | 双 IoU SR | 观测 FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UETrack ERP-wrap | 120 | 0.5143 | 0.5163 | 0.5776 | 0.5798 | 57.16 |
| UETrack 基线 | 120 | 0.4168 | 0.4238 | 0.4525 | 0.4605 | 63.77 |

最终 ERP-wrap 相比基线提升：普通 AUC `+0.0975`，SR@0.5 `+0.1251`。
原始逐序列 CSV 保留了所有提升和回归，没有删除任何序列。

此前 39 序列阶段的结果方向一致：ERP-wrap AUC/SR 为 `0.4988/0.5578`，
基线为 `0.4060/0.4397`。

FPS 测量期间两张 GPU、JPEG 解码和数据同步同时运行，因此速度差异只作为
观测值，不作为严格的单因素开销结论。

## 四、离线交付

- LightFC 轻量镜像已通过五帧 `--network none` 断网运行检查。
- UETrack 镜像提供通用文件协议：`--frames --init --out --timing`，并固定
  CUDA/PyTorch 基础镜像，内置 UETrack 权重和 CLIP 缓存。
- 最终 `linux/amd64` 镜像 ID：
  `sha256:21508ea8959c0dda8b96747a670d06a68d897aa3a949e0f1c4e146a6adf0368a`
- 基础镜像清单 ID：
  `sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a`
- 镜像归档大小：`5,991,662,592` 字节；SHA-256：
  `1919ba75a90a07a54e7def09234a0ea492dee22629685a262ca2e1892cc50c54`
- 断网导入检查确认 PyTorch `2.3.1`、CUDA `12.1`，且权重与 CLIP 哈希一致。
- 本地 WSL Docker 没有可用 NVIDIA 适配器，因此 GPU 检查在服务器两张 RTX 3090
  上完成：5/5 结果行、5/5 计时行，第一帧框数值保持一致。

## 五、验证结果

- 原有及新增核心测试模块共 14 个，已在本地和服务器通过。
- CUDA 检查发现并修复 LoRA/MoE 设备返回问题，GPU 往返检查 21 项全部通过。
- 外部 scorer 回归检查：3/3 通过。
- UETrack 环形裁剪检查：4/4 通过。
- 文件协议辅助检查：服务器 3/3 通过。
- 文件协议 GPU 冒烟：5/5 帧通过。
- 几何融合回归检查：3/3 通过。
- 最终严格 scorer 完成 240 行结果，即基线 120 条加 ERP-wrap 120 条；预测与 GT
  行数全部严格匹配。

## 六、成果保存位置

- 最终 JSON：`reports/results/erpwrap_ablation_0001_0120_bakeoff.json`
- 最终 CSV：`reports/results/erpwrap_ablation_0001_0120_scores.csv`
- 39 序列阶段结果仍保留在 `reports/results/*0039*`。
- 原始结果归档：
  `D:/instan/deliverables/GRT360_2026-08-09/results/uetrack_results_0001_0120.tar.zst`
- 研究汇报 PPT：`reports/GRT360_Research_Handoff_2026-08-09.pptx`
- 交付清单：`reports/GRT360_DELIVERY_MANIFEST_2026-08-09.md`
- 服务器原始结果：`/data/projects/instan_check/uetrack_output/test/tracking_results/uetrack/`
- 服务器运行日志和 scorer 输出：`/data/projects/instan/runs/grt360_20260809/`
- 服务器干净代码目录：`/data/projects/instan_grt360`
- 服务器改动前备份：`/data/backups/instan_code_before_grt360_20260809_043053.tgz`
- GitHub PR：<https://github.com/A3435331841/instan/pull/1>
