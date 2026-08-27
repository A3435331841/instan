# XMem-360 方向本地可行性记录（2026-08-28）

## 结论

已下载并校验官方 XMem checkpoint（SHA256：`27776291D2F0639B4E6B372A67651579B51180AA4D8F8F89BBFF2DCC09EBF6`），并完成 CPU 适配器。适配器使用协议首帧 BFoV 生成种子 mask，调用 XMem 长短期记忆传播，再将 mask 转回 seam-aware ERP 框；推理过程不使用后续 GT。

## Smoke 结果

- `train_sim/seq_0027`，输入缩放至 `480x240`，30 帧：平均 158.85 ms/帧，约 6.30 FPS；BBox 评分 AUC 约 0.443、SR 约 0.621（仅30帧诊断，不是全量成绩）。
- `train_sim/seq_0027`，输入缩放至 `240x120`，3 帧：约 12.72 FPS，说明 CPU 可执行但仍明显慢于 B224/T224 GPU 路径。

## 决策

XMem 的记忆思想值得保留，尤其适合长期消失/重现；但原始 XMem 不是 360VOTS 专用模型，直接用矩形种子 mask 在当前 ERP 数据上不足以作为主干。当前不把它接入每帧路径，只保留为后续低频重捕获专家候选；如要继续，需要 360 mask 训练或至少球面/接缝增强后再做簇级赛马。

产物：`scripts/xmem360_bbox_smoke.py`，实验目录 `D:\instan\grt360_scratch\xmem_smoke_20260828`。

