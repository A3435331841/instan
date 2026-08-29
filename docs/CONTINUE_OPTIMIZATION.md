# 队友继续优化指导

## 当前基线与边界

不要直接把大 checkpoint 提交 Git，也不要在 valid35 上调阈值。以 `configs/repro/v5_final.json`、`results/failure_matrix_130.csv` 和包内清单为唯一输入契约。每个实验建立新目录，保留 `experiment.json`、`config.json`、`summary.json`、`summary.csv`、`trace.jsonl` 和 `failure_notes.md`，禁止覆盖 v5 结果。

当前 v5 已超过 AUC 0.7 的阶段目标，但不是“最终门槛完成”：full130 AUC 0.7008、SR 0.8536、weighted e2e 36.22 FPS；valid35 AUC 0.6945、SR 0.8411。下一个目标是让 AUC 在 valid35 和 full130 同时稳定提高，速度只要保持端到端 >30 FPS 即可。

## 推荐迭代顺序

1. 先复现 ORT v5 full130 的结果和延迟分布；若差异超过 AUC 0.01 或 FPS 15%，先查数据、图版本和设备，不改算法。
2. 用 `failure_matrix_130.csv` 按极区、接缝、小目标、大目标、快速运动、尺度突变、消失重现分桶。每次只选一个场景簇和一个正常负对照。
3. 优先改几何表示和状态机：大 FoV 使用 eBFoV/球面分支；接缝使用圆周索引和 dual-IoU；长时失锁使用 anchor + `NORMAL/SUSPECT/LOST/VERIFY`，冻结可疑模板。
4. 再做轻量存在信号/质量校准器。标签只能来自 train95 的序列级 OOF，输入只能是响应峰值、margin、熵、anchor 相似度、运动残差、模型分歧、纬度/FoV 等推理时信号。
5. 只有场景簇平均 AUC 提升至少 0.05、胜率至少 60%、至少 3 条独占救援且无单条回退超过 0.10，才进入 train95 OOF；通过 OOF 后再跑锁定 valid35，最后才跑 full130。

## 速度纪律

ORT v5 是主路线；PyTorch 包只用于比较。5090 上测量单 GPU 串行端到端速度，分别保存纯推理和含解码/预处理/写盘 FPS。慢专家使用 token bucket；出现专家超时、显存不足或无效框时回退 B224，并把原因写进 trace。单条序列下降可以接受，但全量 weighted e2e 必须保持 >30 FPS。

## 不要做的事

- 不按序列名称、已知 GT 场景标签或离线结果查表路由。
- 不把 LoRA ep5 或任一失败的微调 checkpoint 直接替换主干；先做 OOF。
- 不把训练 checkpoint 当作部署权重；先用 `extract_inference_weights.py` 单独导出 net-only 文件。
- 不删除低分实验、原始 checkpoint 或服务器撤离归档。
- 未经用户确认不执行比赛仓库 `docker push`。
