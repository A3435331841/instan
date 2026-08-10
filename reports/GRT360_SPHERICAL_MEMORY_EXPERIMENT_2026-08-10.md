# GRT360-Spherical-Memory 底层架构实验

## 实验目的

前一版 `GRT360-Causal-DTP-ERP` 是对 ODTrack、UETrack 和 LightFC 输出框做因果路由，适合验证几何先验，但仍属于结果级融合。本实验改为单一可训练跟踪器，目标是在网络内部解决 360° ERP 的几何不连续和遮挡记忆问题。

## 架构

1. **ERP 环形卷积**：卷积水平边界使用 circular padding，垂直边界使用 replicate padding，避免目标穿过左右边界时产生人工边缘。
2. **球面位置编码**：每个 patch 额外输入 `sin/cos(lon)` 与 `sin/cos(lat)`，同时保留中心经度、中心纬度和水平/垂直 FoV。
3. **共享轻量编码器**：5×5 环形卷积 + 两个 depthwise-separable block，模板和搜索区域共享权重。
4. **模板—搜索相关**：以模板特征作为 grouped correlation kernel，通过 soft-argmax 得到局部球面位移，不调用其它跟踪器的框。
5. **因果记忆门控**：当前特征只有在置信度高于 0.45 时才以 0.08 的动量写入记忆，遮挡或错误匹配时冻结模板。
6. **几何感知输出头**：输出归一化 `(dx, dy, log_w, log_h)` 和可见性置信度；训练损失为 Smooth-L1 位移损失 + 0.25×可见性 BCE。

## 已完成的工程验证

- `panotrack/models/spherical_memory.py`：模型、球面编码、环形卷积和记忆门控。
- `panotrack/trackers/spherical_memory.py`：接入 `BaseTracker`，可由 `PanoTracker` 调用。
- `panotrack/pipeline/pipeline.py`：新增可选 `set_geometry(BFoV)` 钩子，几何信息在初始化和每帧更新时进入模型。
- `scripts/train_spherical_memory.py`：合成位移训练入口；不读取其它跟踪器结果。
- `configs/grt360_spherical_memory.json`：正式实验配置模板。

## 合成过拟合结果

在 CPU 上使用 4 个样本、40 步训练，局部位移误差从 `1.4886` 降到 `0.1410`，总损失从 `0.2756` 降到 `0.0026`，说明相关头和梯度链路确实能学习位移，而不是只完成前向冒烟。该权重仅用于工程验证，不代表 360VOT 最终成绩。

随后用本地 `data360` 的 1 个序列、8 个相邻帧对、0.1 缩放做了真实图像训练冒烟：1 个 epoch 的平均损失为 `0.00784`，训练权重写入 `artifacts/spherical_memory_real_smoke.pt`；加载该权重后，`PanoTracker` 的初始化、几何注入和单帧更新均已跑通。这个规模只验证数据接口和反向传播，不作为正式 AUC/SR/FPS 结论。

## 下一轮真实实验

真实训练应从 `data360` 的 GT 相邻帧构造 `(template, search, target_delta)`，按序列划分训练/验证，至少做以下消融：

- 去掉球面坐标编码；
- 将 circular padding 改为 zero padding；
- 关闭因果记忆更新；
- 仅训练相关位移，不训练尺度和可见性头。

只有完成真实 360VOT 验证集和 GPU FPS 测量后，才把它与 `UETrack ERP-wrap`、`GRT360-Causal-DTP-ERP` 的 AUC/SR/FPS 放在同一张最终表中。
