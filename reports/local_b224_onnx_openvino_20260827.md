# B224 ONNX / OpenVINO 本地验证（2026-08-27）

## 转换结果

- 模型：`SUTRACK_b224_ep0180.pth.tar`，视觉 per-frame encoder/decoder 图。
- ONNX：`D:\instan\grt360_scratch\onnx\sutrack_b224_frame.onnx`，356,494,457 bytes，1,532 nodes，`onnx.checker` 通过。
- OpenVINO IR：`D:\instan\grt360_scratch\openvino\sutrack_b224.xml/.bin`，转换耗时约 1.82 s。
- 输入固定为两个 `[1,6,112,112]` 模板、两个 `[1,4]` 模板框标注和一个 `[1,6,224,224]` 搜索图；模板记忆和ERP框状态仍由Python维护。

## 设备探测

| 后端 | 结果 |
|---|---|
| ONNX Runtime CPU | 可运行 |
| ONNX Runtime DirectML | 可运行，识别 `DmlExecutionProvider` |
| OpenVINO CPU | 可编译 |
| OpenVINO GPU（Arc 130T） | 可编译、可运行 |
| OpenVINO NPU（Intel AI Boost） | 不可用；`Core.available_devices` 只有 `CPU,GPU`，未暴露 NPU plugin/驱动 |

## 速度与数值

### 单图稳态推理（20次，随机输入）

| 后端 | P50 | P95 | 等效FPS |
|---|---:|---:|---:|
| ORT DirectML / Arc | 46.99 ms | 53.07 ms | 21.0 |
| OpenVINO GPU / Arc | 15.54 ms | 19.28 ms | 62.86 |
| OpenVINO CPU | 157.38 ms | 163.27 ms | 6.32 |

OpenVINO GPU 与 ORT CPU 输出的最大绝对差为 score 0.00030、size 0.00154、offset 0.01114，满足框回归的数值冒烟要求。

### 真实序列端到端

序列为 `train_sim/seq_0011`，450帧，1440×720；包含真实视频解码、ERP三平铺、模板更新、框后处理和评分。

| 指标 | PyTorch CPU | OpenVINO GPU / Arc |
|---|---:|---:|
| 序列耗时 | 108.4 s | 13.84 s（不含首次编译） |
| tracker FPS | 4.19 | 35.34 |
| 端到端 FPS | 4.15 | **32.71** |
| P95端到端延迟 | 271.1 ms | **33.69 ms** |
| AUC | 0.2604 | 0.2695 |
| SR | 0.2940 | 0.3051 |

OpenVINO GPU 的行为接近已归档 GPU B224 结果（该序列 AUC 0.2641、SR 0.2984、E2E 29.87 FPS），差异来自后端数值和运行状态；不是使用GT或离线结果查表。

## 结论与边界

1. **ONNX/OpenVINO 转换成功，Arc 130T 路径可用于小序列实测，当前单序列端到端已超过30 FPS。**
2. DirectML 也能运行，但明显慢于 OpenVINO GPU；后续优先使用 OpenVINO GPU。
3. NPU 硬件虽被 Windows 识别为 Intel AI Boost，但当前 OpenVINO 不暴露 NPU，暂不能把 B224 或门控模块放到 NPU；需要匹配的 Intel NPU runtime/驱动后再测。
4. 这只是一个 450 帧序列和固定 B224 图的验证，不能直接替代全量130条验收；下一步应在 `sim/0071 + sim/0012` 上接入同一 OpenVINO tracker，再测试OD式改量。
