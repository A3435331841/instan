# GRT-360 服务器撤离记录（2026-08-27）

## 目的

服务器运行环境即将到期。本记录描述如何从本地归档和 GitHub 恢复 GRT-360，
不包含任何密码、token 或私钥。

## 资产分层

- GitHub 普通仓库：源代码、配置、测试、Docker定义、评测脚本、实验摘要和恢复说明。
- GitHub Release `grt360-server-exit-20260827`：原计划上传精选小于2GiB的基线/候选权重；当前 GitHub PAT 的 Release API 权限返回 403，因此未伪造 Release，完整资产已保存在本地归档，代码已推送到 `origin/main`。
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

## 本地归档根目录（已完成）

```text
D:\instan\grt360_storage\experiments\server_exit_20260827\
├── checkpoints/
├── weights/
├── runs/
├── remote_workspace/
├── upstream_sources/
├── environment/
├── server_control/
├── server_exit_manifest.json
├── MIGRATION_COMPLETE.json
├── SHA256SUMS.csv
├── remote_SHA256SUMS
└── failed_partials_20260827/       # 仅保留失败传输残片
```

`server_exit_manifest.json`记录 36 个逐文件校验资产和 7 个逐树 tar 校验资产；远端审计库存为 16,005 项、哈希 16,002 项。`traindata`、`finetune` 和 venv 按计划排除，并在审计清单中保留大小与原因。

## 恢复顺序

1. 克隆本仓库并阅读 `artifacts_manifest/RESTORE.md`。
2. 安装 `requirements.txt`，必要时使用本地 `environment/wheels`。
3. 若 Release 权限已补齐，下载 `grt360-server-exit-20260827`；否则直接挂载本地完整归档。
4. 设置 `OFFICIAL_TRAIN_ROOT`、`GRT360_STORAGE_ROOT`等本地路径变量。
5. 先运行协议/几何/评测单元测试，再运行一条代表序列。
6. 读取 `failure_matrix.csv`和各summary，不把partial结果当作全量结论。

## 安全边界

- 不把SSH凭据写入Git、Release或迁移清单。
- 不把训练集、完整Docker tar、逐帧结果和全部checkpoint提交到普通Git仓库。
- 不自动删除本地重复文件；删除前需独立确认。
- Docker镜像只做离线恢复/干跑，比赛仓库推送需要显式授权。

## 本地整理验收

- 旧入口（`初赛数据`、`deliverables`、`external`、仓库 `artifacts` 等）均为 Junction，源目录与新目录抽样计数/字节数一致。
- 本地整理日志：`D:\instan\grt360_storage\manifests\LOCAL_REORGANIZATION_LOG_20260827.csv`。
- 路径映射与整理前统计：同目录下 `LOCAL_PATH_MAP_20260827.csv`、`LOCAL_SOURCE_INVENTORY_20260827.csv`。
- 本次操作没有删除远端或本地文件；失败残片仅移入隔离目录。
