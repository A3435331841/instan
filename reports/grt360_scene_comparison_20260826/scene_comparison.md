# GRT-360 场景—方案对比

说明：ODTrack 和 SUTRACK-T224 的覆盖为全量时可直接比较；其他方案目前多为medium子集，表中括号是该场景的可比序列数，不能外推为全量成绩。

| 场景 | 序列数 | OD AUC/SR | T224 ΔAUC (n) | 最强已测专家 ΔAUC (n) | 建议技术 |
|---|---:|---:|---:|---:|---|
| hard | 30 | 0.217/0.167 | +0.038 (30) | sutrack_b224 +0.085 (13) | 逐帧失锁归因；先做重捕获/身份验证，再做主干训练 |
| polar | 12 | 0.391/0.472 | -0.061 (12) | sutrack_b224 +0.138 (6) | 切平面投影、极区旋转增强、球面状态 |
| seam | 49 | 0.472/0.539 | +0.024 (49) | lorat +0.062 (11) | 圆周裁剪/三平铺、dual-IoU、接缝一致性 |
| small | 28 | 0.609/0.724 | -0.065 (28) | ft_v4_ep1 +0.025 (4) | 高分辨率搜索、短时特征聚合、小目标增强 |
| large | 7 | 0.315/0.257 | -0.042 (7) | ft_ep6 -0.063 (4) | 宽ROI+紧ROI双分支、动态搜索尺度 |
| fast | 25 | 0.650/0.790 | -0.058 (25) | ft_v4_ep4 +0.021 (7) | S²速度先验、多假设搜索中心 |
| scale | 70 | 0.615/0.738 | -0.028 (70) | lorat +0.032 (19) | log-FoV滤波、尺度增强、动态搜索窗 |
| absent | 12 | 0.532/0.600 | +0.001 (12) | ft_ep6 +0.014 (4) | 冻结记忆、全局重捕获、re-init验证 |
| drift | 34 | 0.243/0.195 | +0.038 (34) | sutrack_b224 +0.063 (15) | anchor身份校验、低置信门控、模板污染阻断 |

## ODTrack最困难的30条

| 序列 | 场景 | OD AUC | T224 AUC | 当前最优已测方案 | 增益 |
|---|---|---:|---:|---|---:|
| train_real/seq_0016 | polar;seam;hard;drift | 0.060 | 0.389 | ft_v4_ep1 | +0.370 |
| train_sim/seq_0018 | small;scale;hard;drift | 0.068 | 0.058 | baseline | +0.000 |
| train_sim/seq_0025 | seam;hard;drift | 0.071 | 0.133 | sutrack_t224 | +0.062 |
| train_sim/seq_0046 | polar;hard;drift | 0.072 | 0.077 | sutrack_b224 | +0.441 |
| train_real/seq_0010 | seam;scale;hard;drift | 0.077 | 0.073 | baseline | +0.000 |
| train_real/seq_0041 | large;seam;hard;drift | 0.077 | 0.083 | sutrack_t224 | +0.006 |
| train_sim/seq_0011 | small;polar;seam;hard;drift | 0.085 | 0.240 | sutrack_b224 | +0.179 |
| train_sim/seq_0075 | hard;drift | 0.090 | 0.105 | sutrack_t224 | +0.015 |
| train_real/seq_0027 | large;seam;fast;hard;drift | 0.094 | 0.107 | sutrack_t224 | +0.013 |
| train_real/seq_0037 | seam;scale;hard;drift | 0.101 | 0.102 | sutrack_t224 | +0.001 |
| train_sim/seq_0012 | seam;hard;drift | 0.115 | 0.242 | sutrack_t224 | +0.127 |
| train_sim/seq_0066 | hard;drift | 0.198 | 0.137 | baseline | +0.000 |
| train_sim/seq_0009 | fast;scale;hard;drift | 0.201 | 0.183 | baseline | +0.000 |
| train_real/seq_0043 | seam;scale;hard;drift | 0.201 | 0.146 | sutrack_b224 | +0.141 |
| train_real/seq_0026 | seam;scale;absent;hard;drift | 0.202 | 0.548 | lorat | +0.346 |
| train_real/seq_0042 | large;seam;hard;drift | 0.248 | 0.185 | baseline | +0.000 |
| train_real/seq_0015 | large;seam;absent;hard;drift | 0.252 | 0.242 | baseline | +0.000 |
| train_sim/seq_0083 | seam;hard;drift | 0.259 | 0.534 | sutrack_t224 | +0.274 |
| train_sim/seq_0044 | small;polar;scale;hard;drift | 0.279 | 0.264 | ft_v4_ep1 | +0.038 |
| train_real/seq_0032 | scale;absent;hard;drift | 0.289 | 0.385 | sutrack_t224 | +0.096 |
| train_sim/seq_0068 | seam;hard;drift | 0.306 | 0.573 | sutrack_t224 | +0.268 |
| train_sim/seq_0064 | hard;drift | 0.306 | 0.276 | baseline | +0.000 |
| train_real/seq_0004 | scale;hard;drift | 0.318 | 0.460 | lorat | +0.428 |
| train_real/seq_0033 | large;hard;drift | 0.319 | 0.333 | ft_ep6 | +0.034 |
| train_real/seq_0013 | small;scale;absent;hard;drift | 0.330 | 0.336 | uetrack | +0.011 |
| train_sim/seq_0059 | polar;fast;scale;hard;drift | 0.331 | 0.201 | ft_ep7 | +0.176 |
| train_sim/seq_0053 | scale;hard;drift | 0.382 | 0.098 | baseline | +0.000 |
| train_sim/seq_0052 | small;polar;hard;drift | 0.387 | 0.345 | ft_v4_ep1 | +0.037 |
| train_sim/seq_0010 | seam;hard;drift | 0.387 | 0.321 | baseline | +0.000 |
| train_sim/seq_0013 | small;seam;hard;drift | 0.395 | 0.457 | sutrack_t224 | +0.062 |
