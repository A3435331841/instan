# ODTrack OpenVINO 专家验证（2026-08-28）

## 图导出与一致性

已把 ODTrack baseline checkpoint 导出为显式 `track_query` 状态的 ONNX 图，并转换为 OpenVINO：

- `D:\instan\grt360_scratch\openvino\odtrack_first.xml`：首个跟踪步，单模板、无 query；
- `D:\instan\grt360_scratch\openvino\odtrack_state.xml`：稳态跟踪步，单模板 + `track_query` 输入/输出。

CPU PyTorch 与 OpenVINO 首步输出的 `score_map/size_map/offset_map` 最大绝对误差分别约为 `6e-8/8e-7/5e-6`，图数值链路通过。此前发现并修复了搜索裁剪 resize-factor 漏除导致的大框塌缩问题。

## GPU 结果

使用三平铺 ERP、ODTrack 384 搜索 / 192 模板及显式 query 状态：

| 序列 | 帧数 | AUC | SR | 端到端 FPS |
|---|---:|---:|---:|---:|
| `train_sim/seq_0071` | 900 | 0.6087 | 0.8265 | 31.65 |
| `train_sim/seq_0024` | 450 | 0.6433 | 0.8864 | 32.56 |
| `train_sim/seq_0082` | 450 | 0.2068 | 0.1425 | 28.78 |
| `train_real/seq_0031` | 450 | 0.5114 | 0.7671 | 21.07 |

ODTrack 在 `sim/0071`、`sim/0024` 这类极区/小目标上有明确潜力，但在 `sim/0082` 和大真实序列上速度或精度未过单序列门。因此当前保留为待门控低频专家，不替换 B224 主干，也未把它硬接入全量路由。

产物：`scripts/export_odtrack_onnx.py`、`scripts/run_odtrack_openvino_sequence.py`，实验目录 `D:\instan\grt360_scratch\odtrack_openvino_gpu_20260828`。

