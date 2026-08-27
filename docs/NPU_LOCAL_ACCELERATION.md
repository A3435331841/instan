# GRT-360 本地 NPU 利用说明

## 当前探测结论（2026-08-27）

本机确实有可用的 NPU 硬件，但 OpenVINO 当前没有把它枚举成可执行设备：

| 项目 | 当前值 |
|---|---|
| PnP 设备 | `Intel(R) AI Boost`, `OK`, `CM_PROB_NONE` |
| 已装驱动 | `32.0.100.3104`（2024-10-25） |
| OpenVINO | `2026.3.1` |
| OpenVINO 设备 | `CPU`, `GPU`；没有 `NPU` |
| NPU 后端列表 | `AVAILABLE_DEVICES=[]` |
| NPU 插件/编译器 | DLL 已随 OpenVINO 安装，编译器版本 `524290` |
| B224 图 | 5 个输入全部静态（template 112、search 224），满足 NPU 静态形状要求 |

探针先用默认 `AUTO_DETECT` 做了只读编译尝试，失败发生在 NPU 后端/平台识别阶段；随后根据
本机 PCI `DEV_7D1D` 对应的 NPU 3720 显式指定 `NPU_PLATFORM=3720`，B224 主图离线编译成功
（约 16.1 秒），template128 高质量图也编译成功（约 17.7 秒）。这证明当前图可以被 NPU
编译器接受；尚未证明能在硬件上执行，因为 Level-Zero 后端仍未枚举。报告分别在
`D:\instan\grt360_scratch\npu_probe_20260827.json`、
`D:\instan\grt360_scratch\npu_probe_offline3720_20260827.json` 和
`D:\instan\grt360_scratch\npu_probe_high_offline3720_20260827.json`。

Windows 注册表显示本机版本为 `25H2/build 26200`。某些工具仍把内核报告为
`Windows 10`，因此探针按内核 build 判断 Windows 11 兼容性，不直接使用 `ProductName` 字符串。

## 已实现的工具链

- `scripts/probe_openvino_npu.py`：只读检查 PnP、驱动、OpenVINO 插件、设备枚举、模型静态形状，并可显式尝试编译。
- `scripts/autonomous_precision_controller.py`：每次自主实验快照都会记录 NPU 是否枚举、插件属性和后端列表。
- B224 单序列/批量 runner 的 `--device` 已支持 `NPU`，并新增 `--cache-dir`，避免每次启动重复编译。

## 驱动恢复后验证步骤

不自动安装驱动或重启机器。用户确认后，先从 [Intel NPU Driver - Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
安装与 `Core Ultra 5 225H` 和 Windows 11 25H2 对应的 NPU 驱动，然后重启；再运行：

```powershell
$env:PYTHONPATH='D:\instan\grt360_scratch\intel_runtime_probe_20260827\Lib\site-packages'
& 'D:\HermesData\hermes-agent\venv\Scripts\python.exe' `
  'D:\instan\pano360\scripts\probe_openvino_npu.py' `
  --model 'D:\instan\grt360_scratch\openvino\sutrack_b224.xml' `
  --output 'D:\instan\grt360_scratch\npu_probe_after_driver.json' `
  --cache-dir 'D:\instan\grt360_scratch\openvino_cache\npu_b224' `
  --npu-platform 3720 `
  --force-compile
```

只有报告中的 `available_devices` 出现 `NPU` 且 `compile_status=success`，才进入真实序列测试：

```powershell
& 'D:\HermesData\hermes-agent\venv\Scripts\python.exe' `
  'D:\instan\pano360\scripts\run_sutrack_b224_openvino_sequence.py' `
  --xml 'D:\instan\grt360_scratch\openvino\sutrack_b224.xml' `
  --high-xml 'D:\instan\grt360_scratch\openvino\sutrack_b224_s224_t128.xml' `
  --device NPU --cache-dir 'D:\instan\grt360_scratch\openvino_cache\npu_b224' `
  --motion-adaptive --data 'D:\instan\grt360_storage\datasets\official_train\train' `
  --seq train_sim/seq_0072 --out 'D:\instan\grt360_scratch\npu_smoke_seq0072'
```

先用一个 seam/polar 困难序列和一个正常负对照，记录编译时间、端到端 FPS、P50/P95
延迟和 AUC/SR；随后才考虑 6--10 条场景簇。速度验收仍按完整 runner 口径，不把纯 NPU
推理数字冒充比赛 FPS。

## NPU 在最终方案中的定位

## 2026-08-28 实测决策

驱动更新后 NPU 已能枚举和执行，但不适合作为 B224 主干。相同 60 帧输入的结果如下：

| 序列/设备 | AUC | SR | 端到端 FPS | P50 延迟 |
|---|---:|---:|---:|---:|
| `train_sim/seq_0072` GPU | 0.7272 | 0.9661 | 36.79 | 27.11 ms |
| `train_sim/seq_0072` NPU（DRIVER） | 0.1800 | 0.0000 | 14.78 | 67.75 ms |
| `train_sim/seq_0072` NPU（PLUGIN） | 0.1469 | 0.0000 | 14.06 | 68.73 ms |
| `train_real/seq_0001` GPU | 0.9056 | 1.0000 | 38.59 | 25.92 ms |
| `train_real/seq_0001` NPU（DRIVER） | 0.1743 | 0.0000 | 17.07 | 56.93 ms |

逐张量对比确认输入完全相同，但 NPU 的 `size_map` 最大绝对差约 `0.662`、
`offset_map` 最大绝对差约 `3.81`；切换内置 PLUGIN 编译器、显式 FP16 输入和
`HETERO:NPU,CPU` 均未恢复数值一致性。因此当前验收策略明确为：**CPU+GPU 主线，NPU
不进入 tracker 主路径**。原始对比报告在
`D:\instan\grt360_scratch\device_compare_cpu_npu_seq0072_f1.json`，诊断脚本为
`scripts/compare_openvino_device_outputs.py`。

CPU+GPU 的资源分工固定为：Arc GPU 独占 B224/几何专家推理，CPU 负责视频解码、数据审计、
失败矩阵、OOF 路由器和指标统计；不再用 NPU 数字参与 full130 或 FPS 晋级。NPU 仅保留为
将来固定形状小型质量/几何子图的独立候选，必须先通过逐张量数值一致性和 30 FPS 预算。

1. 若 B224 全图在 NPU 编译并达到可接受延迟：把 NPU 作为低功耗备用主干，与 Arc GPU
   做同图对照；最终仍以单设备端到端 FPS 和精度共同决定。
2. 若 B224 因算子或编译器限制无法落 NPU：Arc GPU 继续跑主跟踪器，NPU 只承载固定形状的
   轻量质量校准、响应熵/分歧 MLP、几何风险分类或低频身份描述子。这样不改变 B224 的
   主路径精度，却能把门控和验证的 CPU 开销挪走。
3. NPU 只接收推理时信号；不允许按序列名称、GT 或离线结果查表。所有 NPU 候选仍需通过
   valid35 锁定验证，再进入 full130 和端到端速度复测。

当前没有执行驱动安装、系统升级、重启或 Docker push；GPU 全量评测和自主 watch 也未被
NPU 探测打断。

OpenVINO 的设备调用、缓存和静态形状限制以其 [NPU Device 文档](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
为准。
