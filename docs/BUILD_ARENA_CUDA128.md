# 构建 Arena CUDA 12.8 镜像

交接包已经把 Docker 构建上下文和大模型分开组织好：`src/` 是 Git 源码快照，`models/` 是本地权重/图。构建只在本地执行，脚本不会执行 `docker push`。

## ONNX Runtime（主交付）

```powershell
cd D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 `
  -Backend ort -Context . -Tag grt360-v5-ort:cu128
docker run --rm --gpus device=0 `
  -v D:\data360\official_split:/mnt/dataset:ro `
  -v D:\runs\arena_v5:/mnt/result `
  grt360-v5-ort:cu128
```

默认基础镜像是 `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`，可用 `-BaseImage` 替换。镜像入口默认 `v5_final`，也支持 `geometry_v1` 和 `geometry_v4` 做回归比较。

## PyTorch CUDA（参考/回退）

```powershell
cd D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_TORCH_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 `
  -Backend torch -Context . -Tag grt360-sutrack-torch:cu128
docker run --rm --gpus device=0 `
  -e GRT360_TORCH_PROFILE=b224_erp `
  -v D:\data360\official_split:/mnt/dataset:ro `
  -v D:\runs\torch_b224:/mnt/result `
  grt360-sutrack-torch:cu128
```

该镜像运行上游 SUTrack B224/T224 三平铺路径，作为 CUDA 后端基准、导出图排错和低风险回退；它不冒充 ORT v5 多专家路由的逐帧等价实现。

## 5090 对比

两个镜像必须挂载同一数据目录、同一 GPU、同一序列子集。使用 `scripts/benchmark_cuda_backends.py` 生成 `benchmark.json` 和两份完整日志；不要使用并发多 GPU 数字作为最终 FPS。最终比赛提交动作另行确认。
