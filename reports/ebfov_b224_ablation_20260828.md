# B224 eBFoV 分支消融记录（2026-08-28）

## 实验目的

验证 360VOTS 论文中的大 FoV eBFoV 球面采样是否能直接提升当前 SUTrack-B224
主力，而不把论文的 VOS 结果误当成 BBox 结果。

## 实现修正

- `--projection-mode {erp,auto,ebfov}`，默认 `erp`，不改变旧基准。
- `auto` 在搜索窗口 FoV 超过 90° 时使用球面等角采样，否则使用 gnomonic/ERP 旧路径。
- 使用 `panotrack.geometry` 的 `tangent_remap` 与 `local_bbox_to_erp`；状态在 projected
  模式下保存首帧 BFoV，并从局部预测直接更新 BFoV，避免从超大 ERP 框反算出约 359° 的错误视场。
- 用 OpenCV `remap` 和 256 项量化 LRU 缓存替代逐帧纯 NumPy remap，保证速度测量不被原型
  重采样实现主导。

## 单序列门结果

使用与 B224 基准相同的高模板、接缝、极区、小目标、fallback 和 scale-freeze 配置，
只增加 `--projection-mode auto`，单 GPU OpenVINO：

| 序列 | 基准 AUC | eBFoV AUC | 基准 SR | eBFoV SR | 基准端到端 FPS | eBFoV 端到端 FPS |
|---|---:|---:|---:|---:|---:|---:|
| `train_real/seq_0041` | 0.11095 | 0.09535 | 0.01138 | 0.01138 | 28.03 | 24.64 |

该序列有 3781 帧；eBFoV 分支的 `batch_wall_seconds=153.98`，低于 30 FPS 门槛，且
AUC 下降约 0.0156，因此立即停止其余簇运行，不晋级主线。早期未经 BFoV 状态修正的
结果不纳入结论。

## 结论与后续定位

论文的 eBFoV 表示本身是正确的，但未微调的 B224 ERP 三平铺/回归头对球面重映射后的
输入分布不稳，直接替换主路径既没有精度收益又损失速度。后续若重新启用，只允许作为
质量门控后的低频候选，并必须与 ERP 主结果做一致性/anchor 验证；不允许全帧双路运行。

当前正式主线继续使用 `--projection-mode erp`。实验产物：

- `D:\instan\grt360_scratch\ebfov_cluster_20260828_state\train_real_seq_0041`
- `D:\instan\grt360_scratch\ebfov_cluster_20260828_cache\train_real_seq_0041`

