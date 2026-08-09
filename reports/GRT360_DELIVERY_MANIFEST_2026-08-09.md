# GRT-360 交付清单（2026-08-09）

## 版本锚点

- Git 分支：`agent/panotrack-v2`
- GitHub 草稿 PR：https://github.com/A3435331841/instan/pull/1
- 最终成对评测：120/120 条序列，基线与 ERP-wrap 均已完成
- UETrack 上游提交：`fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack 权重 SHA-256：
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 缓存 SHA-256：
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

## 本地交付物

| 交付物 | 位置 | SHA-256 / 状态 |
| --- | --- | --- |
| 最终研究汇报 PPT | `D:/instan/deliverables/GRT360_2026-08-09/GRT360_Research_Handoff_2026-08-09.pptx` | `18022a707201a25c0cdbba7e2221930a27e1c6f5b0940c093618a15f5fb5de75` |
| 离线 UETrack 镜像 | `D:/instan/deliverables/GRT360_2026-08-09/docker/grt360-uetrack_2026-08-09_linux-amd64.tar` | `1919ba75a90a07a54e7def09234a0ea492dee22629685a262ca2e1892cc50c54` |
| 镜像清单 ID | `grt360-uetrack:2026-08-09` | `sha256:21508ea8959c0dda8b96747a670d06a68d897aa3a949e0f1c4e146a6adf0368a` |
| 最终原始结果归档 | `D:/instan/deliverables/GRT360_2026-08-09/results/uetrack_results_0001_0120.tar.zst` | `b255106a2e3711612a9e2aec86d8ba5ec45c3971049f847c6f520a2d4f7810e8` |

Docker 镜像约 5.99 GB，原始结果归档也保存在本地交付目录；二者均不放入 Git。

## Git 中的最终证据

- `reports/STAGE_RESULTS_2026-08-09.md`：中文最终研究报告。
- `reports/results/erpwrap_ablation_0001_0120_bakeoff.json`：严格协议、宏平均和
  全部 240 行结果。SHA-256：
  `d49046a7a0395c48aae451f584e8f729244f0ead6e286dc29f00055d9dc0a333`。
- `reports/results/erpwrap_ablation_0001_0120_scores.csv`：最终逐序列表格。
  SHA-256：`b5e99afd9ce2322e7d9b3104e79dd5850ca8a76842e06618e18a972d41fa3953`。
- `reports/GRT360_Research_Handoff_2026-08-09.pptx`：中文汇报 PPT。
- `reports/results/*0039*`：保留的 39 序列历史阶段证据。

## 验证边界

- 本地容器已通过 `--network none` 断网导入和哈希检查。
- 本地 WSL Docker 没有可用 NVIDIA 适配器；相同 UETrack 代码和权重已在服务器
  两张 RTX 3090 上完成 5 帧 GPU 文件协议冒烟。
- 服务器数据同步日志已出现 `ALL_SEQUENCE_SHARDS_COMPLETE`。
- 两个成对评测队列均已出现 `QUEUE_DONE`；基线和 ERP-wrap 各有 120 条输出。
- 最终严格 scorer 完成 240 行，预测与 GT 行数全部匹配。
- 密码、token 和私有缓存均未进入 Git 跟踪文件。
