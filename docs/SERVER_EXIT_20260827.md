# GRT-360 服务器撤离记录（2026-08-27）

## 目的

服务器运行环境即将到期。本记录描述如何从本地归档和 GitHub 恢复 GRT-360，
不包含任何密码、token 或私钥。

## 资产分层

- GitHub 普通仓库：源代码、配置、测试、Docker定义、评测脚本、实验摘要和恢复说明。
- GitHub Release `grt360-server-exit-20260827`：精选小于2GiB的基线/候选权重。
- 本地 `D:\instan\grt360_storage`：完整checkpoint、逐帧结果、远端工作区、wheels和日志。
- 原始官方数据：本地已有，使用 `OFFICIAL_TRAIN_ROOT` 指向本地数据根目录。

## 服务器侧最终实验状态

| 项目 | 结果/状态 |
|---|---|
| ODTrack官方全130 | AUC 0.5813 / SR 0.6853，既有精度基线 |
| SUTRACK-T224全130 | AUC 0.5598 / SR 0.6573 / E2E FPS 36.7 |
| SUTRACK-B224全130 | AUC 0.6113 / SR 0.7223 / E2E FPS 27.8 |
| v5代表集最佳ep3 | AUC 0.5767 / SR 0.7075 / E2E FPS 31.6 |
| 当前最终门槛 | 尚未达到 AUC>0.8、SR>0.8、FPS>30 |

## 本地归档根目录

```text
D:\instan\grt360_storage\experiments\server_exit_20260827\
├── checkpoints/
├── weights/
├── runs/
├── remote_workspace/
├── upstream_sources/
├── environment/
├── server_control/
├── transfer_manifest.json
└── SHA256SUMS.csv
```

`transfer_manifest.json`和`SHA256SUMS.csv`以实际同步结果为准；它们不是手工估计清单。

## 恢复顺序

1. 克隆本仓库并阅读 `artifacts_manifest/RESTORE.md`。
2. 安装 `requirements.txt`，必要时使用本地 `environment/wheels`。
3. 下载GitHub Release中的精选权重，或直接挂载本地完整归档。
4. 设置 `OFFICIAL_TRAIN_ROOT`、`GRT360_STORAGE_ROOT`等本地路径变量。
5. 先运行协议/几何/评测单元测试，再运行一条代表序列。
6. 读取 `failure_matrix.csv`和各summary，不把partial结果当作全量结论。

## 安全边界

- 不把SSH凭据写入Git、Release或迁移清单。
- 不把训练集、完整Docker tar、逐帧结果和全部checkpoint提交到普通Git仓库。
- 不自动删除本地重复文件；删除前需独立确认。
- Docker镜像只做离线恢复/干跑，比赛仓库推送需要显式授权。
