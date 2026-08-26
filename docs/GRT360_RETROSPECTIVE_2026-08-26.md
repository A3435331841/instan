# GRT-360 全景跟踪项目复盘（2026-08-26）

> 本文档汇总项目全貌、关键结果、技术演进与当前状态，作为技术复盘/答辩材料。
> 数据来源：交接文档（8/25、8/26）、远端 medium/representative 汇总（拉取于 2026-08-26）、graphify 知识图谱。
> 凭据一律不入库（见安全渠道）。

---

## 一、项目背景

- **赛道**：影石全景视频智能跟踪赛道 —— 360° ERP（等距柱状投影）全景视频实时单目标跟踪
- **队伍**：武汉大学 4 人学生队
- **比赛截止**：2026-08-25（已截止，初赛提交定格）
- **提交方式**：docker push 镜像到 Arena 平台触发评测（BFoV 接口）
- **评分口径**：OTB 协议 + 360VOTS 无偏球面 IoU，SR@0.5 + AUC
- **评测环境**：RTX 5090（Blackwell sm_120）+ CUDA 12.8 + torch 2.7，断网运行
- **提交目标**：AUC > 0.80、SR > 0.80、FPS > 30（未达成，见 §八）

## 二、技术路线演进

| 阶段 | 时间 | 内容 | 结果 |
|------|------|------|------|
| Stage 1 | ~7/25 | BFoV 几何模块、NCC 跟踪器、合成数据、基础 pipeline | ✅ 完成 |
| Stage 2 | ~8/3 | 集成 VitTrack、全局重检测 v2、自适应 patch、状态阻尼 | ✅ 完成；发现 BFoV 框架漂移 |
| Stage 3 | ~8/9 | 120 序列全量赛马、初赛提交（ODTrack 精度版镜像） | ✅ ODTrack 冠军（AUC 0.5792） |
| 决赛冲刺 | 8/16~8/21 | 官方 130 序列基线、微调、SUTrack/LoRAT/UETrack 赛马 | ✅ 基线完成，提交窗口已过 |
| Phase 2 | 8/25~8/26 | 失败审计、无泄漏训练清单、自适应球面状态机、风险策略 | 🔄 进行中（技术研究，不再提交） |

**关键发现**：
1. BFoV 框架的恒定角速度状态预测第 7-8 帧完全漂移 → 催生 Direct ERP 方案
2. LightFC 代表序列 AUC 0.618 看着很美，全量一跑只有 0.31 → 坚持全量评测
3. 单模型远达不到 0.8 目标，oracle 上限仅 ~0.58 → 需要真实技术增益

## 三、官方 130 序列基线（决赛锚点）

| 分项 | n | AUC | SR | FPS |
|---|---:|---:|---:|---:|
| 全部 | 130 | 0.5882 | 0.6939 | 29.9 |
| valid 留出 | 35 | 0.5613 | 0.6568 | — |
| train 训练 | 95 | 0.5981 | 0.7076 | — |
| real | 47 | 0.5622 | 0.6567 | — |
| sim | 83 | 0.6029 | 0.7149 | — |

冻结参考（Phase 2 文档）：AUC 0.5813 / SR 0.6853 / tracker FPS 27.4（`/data/runs/_all130_gpu0/odtrack_20260826_021411`）。

## 四、决赛期 medium 30 赛马（2026-08-26 拉取）

完整 30/30 结果（FPS 为并发观测值，提交前需 solo 复测）：

| 方法 | AUC | SR | FPS | real AUC | sim AUC | 判断 |
|---|---:|---:|---:|---:|---:|---|
| sutrack_b224 | 0.4919 | 0.5416 | 26.0 | 0.4435 | 0.5551 | 精度最稳核心 |
| ft_v4_ep4 | 0.4817 | 0.5279 | 14.0 | 0.4477 | 0.5263 | watchlist |
| lorat | 0.4725 | 0.5424 | 8.7 | 0.4656 | 0.4815 | 真实场景专家 |
| ft_v4_ep1 | 0.4701 | 0.5170 | 14.9 | 0.4303 | 0.5223 | 序列专家 |
| ft_ep6 | 0.4623 | 0.5035 | 14.1 | 0.4402 | 0.4911 | watchlist |
| sutrack_t224 | 0.4614 | 0.5070 | 39.5 | 0.4783 | 0.4393 | 快路径核心 |
| odtrack_t1 | 0.4599 | 0.5130 | 14.2 | 0.4334 | 0.4946 | 轻量 real 候选 |
| ft_ep5 | 0.4585 | 0.5018 | 13.7 | 0.4165 | 0.5135 | watchlist |
| ft_v4_ep5 | 0.4585 | 0.4988 | 15.4 | 0.4171 | 0.5126 | 序列专家 |
| uetrack | 0.4465 | 0.4815 | 38.5 | 0.4627 | 0.4254 | 快路径/回退 |
| lightfc | 0.3075 | 0.3135 | 17.1 | 0.3824 | 0.2097 | 弱，仅 watchlist |

per-sequence oracle（30/30 覆盖）：AUC 0.5779 / SR 0.6617 / FPS 16.0。
**没有任何完整方案满足提交指标，oracle 上限离 0.8 差距巨大。**

## 五、representative 9 潜力矩阵（组件价值筛选）

| 定位 | 方法 | AUC | SR | FPS |
|---|---|---:|---:|---:|
| 精度基线 | lorat | 0.6021 | 0.7275 | 10.2 |
| 精度核心 | sutrack_b224 | 0.5919 | 0.6899 | 27.6 |
| 快路径核心 | sutrack_t224 | 0.5632 | 0.6579 | 40.5 |
| sim/hard 专家 | ft_v4_ep4 / ep1 | 0.5479 / 0.5396 | 0.6627 / 0.6535 | 16.7 / 19.8 |
| 速度基线 peer | ft_ep6/7/8、odtrack_t1、uetrack | 0.52~0.54 | 0.60~0.66 | 26~52 |

drop：ft_ep1/2/3/4、ft_v4_ep2/3/6、odtrack_recapture（±ft_ep7）、direct_erp（AUC 0.26）。

**结论**：lorat / sutrack_b224 / sutrack_t224 是三条明确主线（精度专家 / 稳定核心 / 快路径），recapture 与 direct_erp 已确认无效。

## 六、融合尝试（component diagnostics）

| 融合方案 | done | AUC | SR | 定位 |
|---|---:|---:|---:|---|
| fusion_5way | 30/30 | 0.522 | 0.593 | 当前最优融合 |
| fusion_4way | 30/30 | 0.522 | 0.594 | watchlist |
| fusion_3way | 30/30 | 0.510 | 0.578 | sequence_expert |
| fusion_6way_v4 | 13/30 | 0.502 | 0.587 | partial |
| fusion_8way | 27/30 | 0.490 | 0.550 | partial |
| oracle 上限 | 30/30 | 0.582 | 0.670 | 逐序列最优 |

**关键结论**：简单融合（投票/oracle）只比单模型好一点点（0.49→0.52），**远达不到 0.8**；必须靠真实技术增益（自适应路由/状态机），而不是组合已知方法。

## 七、Phase 2 实施（2026-08-26，技术研究向）

已实现并同步远端（本地已入库，commit `6729c80`）：

| 组件 | 路径 | 作用 |
|------|------|------|
| 失败审计 | `panotrack/evaluation/failure_matrix.py` | 130 序列失败属性、丢失段、组件增量、场景聚类 |
| 训练清单 | `panotrack/data/training_manifest.py` | train95 无泄漏清单，real/sim 平衡采样，5 折 OOF |
| 自适应球面 | `panotrack/pipeline/adaptive_spherical.py` | NORMAL/SUSPECT/LOST/VERIFY 状态机 + 球面运动证据 + 专家预算 |
| 风险策略 | `panotrack/pipeline/risk_policy.py` | 风险策略 |
| 评测扩展 | `scripts/eval_official.py` | adaptive_spherical 后端 + trace.jsonl / latency |
| 融合/路由 | `scripts/eval_fusion.py`、`train_counterfactual_router.py` | 融合与反事实路由训练 |

**首个 micro gate（train_sim/seq_0046 前 80 帧）**：

| 方法 | AUC | SR | tracker FPS | E2E FPS |
|---|---:|---:|---:|---:|
| ODTrack | 0.2037 | 0.0633 | 21.7 | 21.0 |
| SUTRACK-T224 | 0.3623 | 0.5570 | 37.3 | 35.7 |
| adaptive T224+B224 v1 | 0.3623 | 0.5570 | 22.6 | 21.8 |

v1 路由器未晋升（复现 T224 结果却多付 20% B224 探测开销）；已改为**状态化专家片段**（accepted experts 按状态片段运行而非孤立探测），全序列验证排队中。

**远端进行中**：GPU0 = failure-balanced ODTrack v5 训练（checkpoint watcher 就位）；GPU1 = SUTRACK-T224/B224 valid35 bake-off（已完成并扩展至 all130）。审计时点：GPU0 100% 利用率、v5 训练进程健康、无 OOM/Traceback。

## 八、Gate 状态（2026-08-26 拉取）

```
NO_SUBMISSION_CANDIDATE: 无完整方案满足 AUC>0.80, SR>0.80, FPS>30.0
REPRESENTATIVE_GATE=OK: representative 探针已全部补齐（9/9 × 14 方法）
FUSION_GATE=HOLD: medium 尚有不完整项（ft_ep7/8、fusion_6way 系列）
SPEED_GATE=OK_FOR_CURRENT_LOAD: FPS 未受并发评测混淆
```

待补齐：ft_ep7 28/30、ft_ep8 27/30、fusion_5way_v1v4 19/30、fusion_6way_best 19/30、fusion_6way_ep7 28/30、fusion_6way_v4 13/30、fusion_6way_v4ep1 19/30、fusion_8way 27/30。

## 九、经验教训（答辩可用）

1. **小样本数字不能当成绩**：LightFC 代表序列 0.618 → 全量 0.31；此后所有实验坚持全量。
2. **框架假设要验证**：BFoV 恒定角速度状态预测第 7-8 帧漂移，Direct ERP 反而全面胜出。
3. **提交配额珍贵**：推送即占用（含失败），本地断网全链路自测通过才 push。
4. **单模型有上限**：130 序列上最强单模型 ~0.59，oracle ~0.58，融合 ~0.52 —— 0.8 目标需要架构级创新而非调参。
5. **评测口径纪律**：valid 35 永不训练；所有对比同评分器同序列集；FPS 注明卡型与并发条件。

## 十、架构全景（graphify 知识图谱，2026-08-26 构建）

- 规模：**12,015 节点 / 26,685 边 / 714 社区**（语料 1,767 文件，覆盖 pano360、external/LoRAT、deliverables、初赛数据）
- 输出：`D:/instan/graphify-out/`（graph.html 交互图谱 / GRAPH_REPORT.md / graph.json）
- 核心抽象：TrainData / TrackerEvalData / TrackingDataset（数据管道）、SequenceEvaluationResult_SOT（评测体系）
- **关键洞察**：pano360 的评测脚本（eval_official.py）深度耦合 LoRAT 的 `LoRAT_DINOv2` 模型与 LoRA 权重挂载工具（FullFrameAdapter/GtEchoTracker/MockTracker 均依赖）——external/LoRAT 不是纯参考，而是实际依赖面。
- TrainData 是 LoRAT 训练生态的中央协议（87 条直接边跨 6 社区），你的 Arena 冒烟脚本通过 siamese transform 插件间接接入整个 LoRAT 数据管道。

## 十一、下一步建议（技术研究向，不提交）

1. **等 v5 训练 + SUTRACK all130 完成**（GPU0/GPU1 均在跑，勿打断）；
2. **验证 adaptive stateful-episode 全序列 gate**（v1 未过，v2 排队中）——这是当前唯一有架构级增量的方向；
3. **补齐 fusion 6way 系列 medium**（FUSION_GATE 解除条件）；
4. **solo FPS 复测** sutrack_b224（26.0，唯一精度核心，看是否过 30）；
5. 复盘材料持续滚动：本文件 + graphify 图谱 + 远端汇总表即答辩素材。

---

## 十二、增量更新（2026-08-26 深夜审计）

### 🔥 重大进展：SUTrack-B224 全量 130 序列首超 ODTrack 基线

远端 GPU1 的 SUTrack bake-off 已推进至 all130 并完成：

| 方案 | n | AUC | SR | FPS | 对比 ODTrack 官方基线（0.5882） |
|---|---:|---:|---:|---:|---|
| **SUTrack-B224** | 130 | **0.6113** | **0.7223** | 28.9 | **+2.3 点（首超）** |
| SUTrack-T224 | 130 | 0.5598 | 0.6573 | 39.0 | -2.8 点 |
| SUTrack-B224 valid35 | 35 | 0.5657 | 0.6594 | 31.0 | 留出集参考 |
| SUTrack-T224 valid35 | 35 | 0.5300 | 0.6174 | 45.6 | 留出集参考 |

**B224 分项**：real 47 条 AUC 0.5764 / sim 83 条 0.6311——sim 域（0.6311）贡献主要增益。

**逐序列对比（B224 vs ODTrack 基线，130 条全对齐）**：
- **修复 36 条（+0.05 以上）**：`seq_0012` +0.562（0.115→0.677）、`seq_0046` +0.441（0.072→0.513）、`seq_0032` +0.418、`seq_0068` +0.383——**包括 ODTrack 曾经的失锁序列**（seq_0046 从 0.07 救到 0.51）
- **退步 16 条（-0.05 以上）**：`seq_0082` -0.118、`seq_0038` -0.111、`seq_0044` -0.081——B224 的 weak 点与 ODTrack 互补

**结论**：SUTrack-B224 成为**当前最强单模型**（AUC 0.6113 全面超 ODTrack 0.5882），且与 ODTrack 呈现清晰互补（36 修复 vs 16 退步）——这为融合/路由提供了比 medium 30 更强的证据。FPS 28.9 接近 30 阈值，需 solo 复测确认。

### 其他远端动态（2026-08-26 23:49 审计）

- **v5 微调**：checkpoint 已出 **ep0001-ep0006**（各 ~1.1GB）；ep5 的 representative9 快评进行中（3/9：seq_0002 0.6938 / seq_0003 0.8363 / seq_0004 0.3197）；ep6 快评排队
- **odtrack_geo（tangent 切平面新方案）**：调研路线落地，正在跑 medium30（13/30）——部分序列强（seq_0003 0.8764 / seq_0047 0.7331），但 seq_0016 0.0658 / seq_0046 0.0579 弱，等待完整结果
- **medium 补齐**：ft_ep7 已达 30/30、fusion_5way 30/30；仍待 ft_ep8 29/30、fusion_6way_v4 13/30、fusion_8way 27/30
- **router 系列实验**（router_oof / router_grid / adaptive_s_episode_probe）已在跑，对应 adaptive_spherical v2 的 episode 方向
- GPU0/GPU1 均在健康运行，无 OOM/Traceback

