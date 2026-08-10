# GRT360 稀疏教师调用实验

更新时间：2026-08-10

## 实验目的

保留 ODTrack 精度版和 UETrack ERP-wrap 高速版，在不改变两个基线的前提下，
测试“UETrack 每帧运行、ODTrack 每 K 帧做一次因果校正”的第一版创新路线。
该实验只使用预测框，不读取真值做路由决策；真值只用于最终评分。

## 结果

| 方案 | ODTrack 调用比例 | AUC | SR | 串行 FPS 估计 | 结论 |
|---|---:|---:|---:|---:|---|
| UETrack ERP-wrap | 100% UE | 0.5143 | 0.5776 | 57.16 | 高速基线 |
| 稀疏教师 K=2 | 50% OD + 50% UE | **0.5472** | **0.6175** | **15.54** | 当前第一轮最好折中 |
| 稀疏教师 K=5 | 20% OD + 80% UE | 0.4981 | 0.5554 | 27.60 | 校正衰减不足，精度下降 |
| 稀疏教师 K=10 | 10% OD + 90% UE | 0.4909 | 0.5451 | 37.23 | 误差累积明显 |
| 稀疏教师 K=20 | 5% OD + 95% UE | 0.5009 | 0.5598 | 45.09 | 接近高速但精度不够 |
| ODTrack ERP 三平铺 | 100% OD | 0.5792 | 0.6532 | 8.99 | 精度基线 |

FPS 估计使用：

```text
1 / (OD比例 / 8.9945 + UE比例 / 57.1606)
```

这是串行上限估计，实际速度还会受到 GPU 调度、数据搬运和 Docker I/O 影响。

## 当前判断

K=2 已经证明“少调用教师可以同时提高 UETrack 精度”，但仍没有达到 ODTrack，
而且速度只到约 15.54 FPS。K≥5 时，学生误差在两次教师校正之间累积，说明仅靠
固定间隔不能解决长期漂移。

## 下一轮创新

1. 用 UETrack 的 `best_score`、运动创新量、接缝/极区风险训练可靠性门控；
2. 将固定 K 改为事件触发：低可靠性时立即调用 ODTrack，否则保持 UETrack；
3. 用 ODTrack 输出的框和响应分数蒸馏 UETrack 学生；
4. 在 UETrack 中加入 causal KV-cache，避免每帧重新处理完整历史模板；
5. 在未参与阈值选择的留出序列上重新验证。

## 代码和结果

- 实验入口：`scripts/fuse_sparse_teacher.py`
- K=2 评分：`reports/results/sparse_teacher_k2_score/bakeoff.json`
- K=5 评分：`reports/results/sparse_teacher_k5_score/bakeoff.json`
- K=10 评分：`reports/results/sparse_teacher_k10_score/bakeoff.json`
- K=20 评分：`reports/results/sparse_teacher_k20_score/bakeoff.json`

