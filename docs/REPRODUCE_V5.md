# GRT-360 v5 复现说明

当前锁定源码标签为 `delivery-v20260829`。最终候选是 B224 主干、T224 快路径、ODTrack 几何专家和 v5 ODTrack 稀疏专家组成的因果路由；路由只读取首帧 BFoV 与推理时质量/几何信号，不读取 GT、序列名或离线结果表。

## 资产位置

GitHub 仓库只保存源码、配置、文档和小型结果清单。权重、ONNX/OpenVINO 图和完整实验历史位于本地交接包：

`D:\\instan\\grt360_deliverables\\team_v5_20260829\\`

优先使用 `GRT360_FINAL_ORT_CUDA128`；PyTorch 包是同数据同 GPU 的后端参考和保守回退。每个包的 `SHA256SUMS` 必须先验证。

## ORT v5

```powershell
python scripts/run_profile.py v5_final `
  --dataset D:\data360\official_split `
  --result D:\runs\v5 `
  --model-root D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\models `
  --backend ort_cuda --print-only
```

在 CUDA 镜像中入口是 `integrations/final/arena_protocol_v5.py`。`--force-cpu --max-frames 2` 仅用于结构性冒烟，不作为性能成绩。

## 评测口径

使用仓库中的 `scripts/eval_official.py` 和同一 BFoV/球面 IoU 评分器；速度验收必须记录解码、预处理、推理、结果写盘在内的端到端 FPS，并同时保存 mean、weighted、P50/P95 延迟。valid35 只锁定验证，不用于重新调阈值。

## 已锁定参考

v5 full130：AUC `0.7007805295`，SR `0.8535501637`，mean e2e `38.7409 FPS`，weighted e2e `36.2231 FPS`，最大 P95 `67.38 ms`。valid35：AUC `0.6944711096`，SR `0.8410733642`，weighted e2e `35.2652 FPS`。

这些数字是本地验证参考；换机器后必须重新测量，不能把 3090/OpenVINO 数字直接当作 5090 CUDA 成绩。
