# v5 恢复说明

1. 从 GitHub 克隆 `pano360`，确认提交 `a7ac8bc`。
2. 从本地交接根目录复制或解包 `GRT360_FINAL_ORT_CUDA128`；先验证包内 `SHA256SUMS`。
3. 将任意 Arena 数据集只读挂载到 `/mnt/dataset`，结果目录挂载到 `/mnt/result`。
4. 按 `docs/BUILD_ARENA_CUDA128.md` 构建 ORT CUDA 镜像；如果只需源码/训练，使用 `GRT360_CONTINUE_TRAINING`。
5. 权重和 checkpoint 不在 GitHub，也不需要复制进 Git 工作树；它们通过包内 `models/`、`checkpoints/` 和清单路径传入。

ODTrack 训练 checkpoint 含 `net`、optimizer moments 等训练状态，部署只需 `net`。运行 `scripts/extract_inference_weights.py --input ... --output ...` 生成独立 net-only 文件；原文件保持不变。

恢复演练应记录：源码提交、包 SHA256、模型图/权重 SHA256、Torch/ORT/CUDA 版本、序列数量、输出行数、AUC/SR/FPS 和 P95 延迟。未经确认不要向任何比赛仓库 push。
