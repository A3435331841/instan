# GRT360-Causal-DTP-ERP 候选架构阶段性验证

更新时间：2026-08-10

## 结论

本轮已经把候选架构做成了可运行的预测级原型，并完成 360VOT 全部 120 条序列的严格评分。当前结果是**守住 ODTrack 教师基线，但尚未超过它**：

| 版本 | 序列数 | AUC | SR | 说明 |
|---|---:|---:|---:|---|
| ODTrack ERP 三平铺 | 120/120 | 0.5792135073 | 0.6531941586 | 当前精度基线 |
| GRT360-Causal-DTP-ERP（保守路由） | 120/120 | 0.5792135073 | 0.6531941586 | 默认不损伤教师 |
| GRT360-Causal-DTP-ERP（激进路由试验） | 120/120 | 0.4660 | 0.5076 | 仅预测后处理，出现明显退化 |

保守版本的含义是：现阶段没有训练好的可靠性头和学生模型，因此只有在非常大的可靠性差距下才允许 UETrack/LightFC 接管。这样可以避免把未经训练的路由器误当成创新模型。

## 已实现模块

代码位置：

- `panotrack/geometry/causal_dtp.py`
- `scripts/fuse_causal_dtp_erp.py`
- `tests/test_causal_dtp.py`

当前原型包含：

1. ERP 经度的圆周常速度先验，处理跨 `0/360°` 接缝的位移；
2. 基于创新量、尺度变化、专家分歧和极区/接缝风险的因果可靠性估计；
3. ODTrack 教师、UETrack 学生、LightFC scout 的带滞回切换；
4. 学生接管时向教师做小幅圆周融合，避免框中心跳变；
5. 每帧输出 `expert_ids.txt` 和 `reliability.txt`，便于后续训练可靠性头。

## 速度测量

在本地 112657 个非首帧上，纯路由计算耗时约 0.106 秒，约 **15153 FPS**。这只是路由层速度，不等同端到端模型 FPS；端到端速度仍由 ODTrack/UETrack 的 GPU 推理时间决定。按当前教师路径，端到端上限仍约为 ODTrack 的 8.99 FPS。

## 为什么还没有超过 ODTrack

这轮输入的是已经生成好的三个 tracker 框，没有训练：

- 没有真正的 DTPTrack Temporal Reliability Calibrator；
- 没有 LoRATv2 的 frame-wise causal attention 和 KV Cache；
- 没有 ETCTrack 的可学习 Token 压缩；
- 没有 360 数据上的教师-学生蒸馏；
- 没有把球面特征直接送进 backbone。

因此，当前结果只能证明“候选路由层可运行且不会破坏教师基线”，不能证明创新模型已经成立。

## 下一步

下一步应在服务器 GPU 上进行真正的模型级改造：

1. 在 UETrack 学生分支加入 causal KV-cache；
2. 用 ODTrack 生成 teacher logits/框级软标签，训练可靠性头；
3. 将接缝、极区和球面位置编码加入输入特征；
4. 加入 ETCTrack 式历史模板 Token 压缩；
5. 先用未参与调参的留出序列做消融，再重跑 120 序列正式结果。

## 结果文件

- 严格评分：`reports/results/grt360_causal_dtp_erp_120_score/bakeoff.json`
- 预测输出：`runs/candidate_outputs_final/`
- 路由器代码：`panotrack/geometry/causal_dtp.py`
- 运行脚本：`scripts/fuse_causal_dtp_erp.py`

