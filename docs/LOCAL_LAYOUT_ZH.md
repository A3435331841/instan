# GRT-360 本地目录布局

## 代码仓库

`D:\instan\pano360` 是唯一Git仓库，包含算法、适配器、配置、测试、Docker定义、
文档和小型报告。`artifacts/`、`runs/`、`models/`、`tools_local/`等大文件目录由
`.gitignore`隔离，不进入普通Git提交。

## 大文件与实验存储

`D:\instan\grt360_storage` 保存完整数据和实验资产：

```text
grt360_storage/
├── datasets/official_train/       # 官方训练数据的本地唯一副本
├── datasets/360vot_legacy/        # 旧测试包/压缩包
├── checkpoints/                   # baselines、finetunes、experts
├── experiments/                   # server_exit和历史runs
├── upstream_sources/              # 上游源码快照
├── environment/wheels/            # 离线Python依赖
├── docker_images/                 # 镜像/构建上下文
└── manifests/                     # 清单、SHA256、恢复记录
```

## 交付物与临时目录

- `D:\instan\grt360_deliverables\current`：当前交付物。
- `D:\instan\grt360_deliverables\legacy_*`：历史交付包和参考文档。
- `D:\instan\grt360_scratch`：smoke、渲染、下载、graphify和临时包。
- 根目录 `tools_local` 是兼容 Junction，实际快照位于
  `grt360_storage\experiments\local_legacy_202608\tools_local`；根目录其它历史入口也保留 Junction。
- `grt360_storage\manifests` 保存 `LOCAL_PATH_MAP_20260827.csv`、源目录统计、迁移日志和重复文件报告。

本轮整理只移动文件和建立兼容Junction，不递归删除。重复文件先输出报告，待单独确认后再处理。
