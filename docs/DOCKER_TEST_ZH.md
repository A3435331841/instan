# 本地 Docker 构建与验证记录

更新时间：2026-08-10

## 轻量最终离线镜像

镜像：`grt360-final:2026-08-10`

- 构建文件：`docker/Dockerfile.offline`
- 构建方式：`docker build --network=none --platform linux/amd64 ...`
- 构建结果：成功；镜像约 675 MB；
- 导出文件：`artifacts/grt360-final-2026-08-10.tar`，约 182 MB；
- `--help`：通过；
- 代码、中文报告和配置存在性检查：通过；
- 251 帧 CPU/协议冒烟运行：通过，输出结果文件正常生成。

## 高速 UETrack GPU 离线镜像

镜像：`grt360-uetrack:2026-08-09`

- 本地镜像：存在，约 16.8 GB；
- 导出文件：`artifacts/grt360-uetrack-2026-08-09.tar`，约 5.99 GB；
- `--help` 和无网络容器启动：通过；
- 镜像内包含固定的 UETrack 源码、权重和 CLIP 缓存；
- 本机真实 GPU 推理：未通过，原因不是镜像代码，而是当前 Windows/WSL 报错：
  `nvidia-container-cli: initialization error: WSL environment detected but no adapters were found`；
- 不带 GPU 运行也不能代替 GPU 测试，因为 UETrack 会调用 `.cuda()`，随后报
  `Found no NVIDIA driver on your system`。

因此，高速镜像已构建并通过容器级离线自检，但本机暂时没有可用 NVIDIA 运行时，
不能在这台电脑上给出真实 57.16 FPS 的 GPU 复测。57.16 FPS 来自服务器上的两套
UETrack ERP-wrap 全量评测记录。

## 旧镜像清理状态

旧镜像没有被删除，避免误删仍被最终离线镜像复用的基础层。当前相关镜像包括：

- `grt360-final:2026-08-10`：最终轻量离线镜像；
- `grt360-uetrack:2026-08-09`：高速 GPU 离线镜像；
- `panotrack:latest`：轻量离线镜像使用的本地基础镜像；
- `panotrack:grt360-stage1`：旧阶段镜像，暂未删除。

另外还有与本项目无关的本地镜像。若要清理旧阶段镜像，应单独确认具体 tag 后再删除，
不能直接清空 Docker 镜像缓存。
