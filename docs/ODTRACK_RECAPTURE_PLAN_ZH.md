# GRT-360 决赛实施方案：ODTrack + 可靠性门控 + 球面重捕获

版本：v1.0 | 日期：2026-08-10 | 状态：待评审
定位：**初赛提交数据冻结不动**；本方案只针对决赛阶段（8/11–8/14）的精度提升。

> 这篇文档是给队友看的，先说人话：初赛提交版 AUC 0.5792 排进决赛问题不大，
> 但 120 条里有 10 条是"烂尾"的——ODTrack 一旦跟丢就再也没有任何机制
> 把它拉回来，比如 0047 那条 2334 帧的序列，整条只对了 39 帧。
> 决赛前我们想做的事说白了就一句：**给 ODTrack 装一个"判丢 + 全图找回"的外挂**。
> 组件其实早就躺在仓库里了（全局重检测、可靠性门控、模板记忆都是之前
> 做 PanoTracker 时留下的），只是当初没接到 ODTrack 这条全帧路线上。
> 下面的方案就是把它们接起来，接完在服务器上全量验证一把。

---

## 1. 背景与诊断

### 1.1 现状成绩（初赛提交版，冻结）

| 版本 | AUC | SR | FPS |
|---|---:|---:|---:|
| ODTrack ERP 三平铺（提交） | 0.5792 | 0.6532 | 8.99 |
| UETrack ERP-wrap | 0.5143 | 0.5776 | 57.16 |

### 1.2 失败序列的共同模式

ODTrack 120 序列中 **10 个序列 AUC < 0.15**：0004、0020、0041、0045、0047、0072、0094、0098、0100、0106、0117。
其中 0041/0047/0020/0054/0060/0066/0085 等为 2400 帧超长序列。

诊断结论（有数据支撑）：

1. **失锁后不可自愈**：ODTrack 无判丢与重捕获机制，一旦时序记忆被污染就一路错到底。
   例：0047（2334 帧）SR=0.0167——只有 39 帧成功，其余全部跟错对象。
2. **时序记忆污染是主因**：ODTrack 的 dense temporal token（MEMORY_THRESHOLD=1000）
   在长时漂移下会持续吸收错误外观，形成正反馈（handoff §34 已记录同类问题）。
3. **这些序列拖累均值**：10 个失锁序列均值约 0.10；若救回一半到 0.35，
   宏平均 AUC 提升约 **+0.025~0.04**（120 等权）。

### 1.3 目标

- 主目标：把失锁序列的"丢失后找回"能力补上，AUC 0.5792 → 0.60+（目标 +0.03）。
- 硬约束：不改变 ODTrack 上游代码；不伤害正常序列成绩；可 feature-flag 回退。
- 心理预期：这是外挂方案，不是重训模型——能救回一半失锁序列就是成功，
  别指望 0.60+ 之后还能再涨（真要再涨得等官方训练集做域微调，那是另一个活）。

---

## 2. 总体架构：外挂式 wrapper（不动上游）

新模块 `integrations/odtrack/recapture.py`，实现 `OdtrackRecaptureTracker`，
对齐 `BaseTracker` 契约（`init(image, bbox)` / `update(image) -> {'bbox','score',...}`），
内部持有四个已有组件：

```
OdtrackRecaptureTracker
├── ODTrack（上游 tracker，逐帧推理，GPU）
├── ReliabilityGate      ← panotrack/pipeline/memory.py（判丢）
├── TemplateMemory       ← panotrack/pipeline/memory.py（anchor + 短/长记忆）
└── SphericalMultiViewRedetector ← panotrack/pipeline/redetect_v3.py（重捕获）
```

数据流（状态机）：

```
                    ┌────────────────────────────────────────┐
   ERP 帧 ──► ODTrack track ──► 框 + last_pred_iou ──► ReliabilityGate
                    │                                        │
                    │                                  高可靠（R ≥ τ_ok）
                    │                                        ▼
                    │                            输出框 + TemplateMemory.add()
                    │                                （门控写入，anchor 永不被覆盖）
                    │
                    │ 低可靠（R < τ_ok）→ 连续 N 帧计数
                    ▼
                LOST 状态
                    │
                    ▼
        冻结模板更新；每 K 帧执行一次：
        redetect_v3.search(frame, erp_downscale=2)
                    │
          命中且通过 VERIFY（anchor 相似 + 运动合理 + 分数阈值）
                    ▼
        用候选框重新 initialize ODTrack（清空时序记忆）
                    ▼
        观察 FOLLOW 帧确认不立即失锁 → NORMAL
          未通过 VERIFY → 继续 LOST（宁可丢，不可锁错）
```

**为什么外挂而不改上游**：
- ODTrack 是 2022 年代码（py3.8/visdom/torch._six），改源码会增加不可控风险；
- 重捕获是"系统级"能力（判丢 + 全图搜索），不属于模型本身；
- wrapper 可在 `file_protocol.py` 与评测脚本中一键替换，便于 A/B。

---

## 3. 判丢策略（ReliabilityGate 信号与阈值）

### 3.1 信号来源（全部为推理时可得，无真值）

| 分量 | 来源 | 说明 |
|---|---|---|
| C_visual | `tracker.last_pred_iou`（IoU-head 响应图峰值） | 已实现于 odtrack_360vot_conf.py；**离线标定阶段权重为 0**（全量 confidence 需 GPU 重跑，本地只有 10 条证据不足）；服务器跑完 Step 1 全量后可加入公式并重标 |
| C_anchor | 当前预测框 crop 与 anchor 模板的 NCC 相似度 | `TemplateMemory.anchor_similarity` 已有 |
| C_motion | 预测框中心与恒速外推预测的球面距离 | `SphericalState`/`causal_dtp` 已有先验，wrapper 内做轻量圆周恒速 |
| C_scale | 框面积 log 变化 vs 历史 EMA | 参考 `ReliabilityGate.c_scale` |
| geometry_risk | 极区 |lat|>55°、seam 距离 <12%W | `causal_dtp._geometry_risk` 已有公式 |

### 3.2 阈值标定（Step 1/2 的数据驱动）

- **禁止直接用 120 条全量调参**（项目纪律）。标定流程：
  1. 服务器重跑 120 条，输出每条 `confidence.txt`（`last_pred_iou`），
     同时落盘逐帧框（已有）；→ 全量置信度分布可得；
  2. 将 120 条按序**奇偶对半分**：60 条标定集 / 60 条验证集；
  3. 在标定集上扫描：`τ_ok`、连续低可靠帧数 N、重捕获间隔 K；
  4. 验证集上报告最终成绩；标定集成绩仅作参考，报告必须注明划分。
- 若奇偶划分不够稳（同源分布），可改用按失锁严重度分层抽样，但必须落盘划分文件。

### 3.3 滞回与防抖

- 连续 N 帧 R < τ_ok 才进入 LOST（N 默认 5，标定）；
- 找回后经 OBSERVE 观察期（observe_frames=3）才回 NORMAL，观察期再次失锁
  会直接回到 LOST（见 §4.3）——这就是恢复侧的防抖，代码里没有单独的
  "恢复后 M 帧不重触发"计数器；
- 单帧 R 抖动只影响模板写入门控（TemplateMemory 已自带 min_quality 门槛），
  不触发全图搜索。

---

## 4. 重捕获策略（redetect_v3 + VERIFY + REINIT）

### 4.1 搜索触发与预算

- 仅 LOST 状态触发；**每 K=5 帧搜一次**（非每帧），避免 CPU NCC 拖慢端到端 FPS；
- `search(frame, erp_downscale=2)`：12 视角 × 3 尺度 × 多模板池，全 numpy/CPU；
  单次搜索在 4K 帧上预算 < 2s（降采样后），K=5 时平均开销可接受；
- 模板池 = `TemplateMemory.get_bank()`（anchor + 高可靠短/长记忆，去冗余）。

### 4.2 VERIFY（防误锁，最重要的一关）

候选框 `(x,y,w,h), score` 必须同时满足：

1. **分数门槛**：NCC score ≥ min_score（默认 0.45，标定）；
2. **anchor 一致性**：候选框 crop 与 anchor 的 NCC ≥ 0.5（锚点强校验，防锁错相似目标）；
3. **运动合理性**：候选中心与失锁前最后位置/恒速外推的球面距离 ≤ 合理上限
   （默认 ≤ 90°，防瞬时全图乱跳）；
4. **双模板一致**：anchor 与至少一个 dynamic 模板在候选位置的响应都高
   （可选，默认关闭，作为 ablation 项）。

任一不满足 → 视为未命中，**保持 LOST**（false recovery 比继续 lost 更糟，handoff §45）。

### 4.3 REINIT（防记忆污染的关键）

- 用候选框**重新 `tracker.initialize`**（三平铺帧 + 候选框移到中间副本），
  而不是 resume：彻底清空被污染的 dense temporal memory；
- 重建后置 `follow_count=0`，连续 3 帧 R ≥ τ_ok 才回 NORMAL；
- 若 3 帧内再次失锁 → 回到 LOST，且该候选模板被标记（同一位置不重复尝试）。

---

## 5. 实现步骤（每步有独立验证门槛）

| 步骤 | 内容 | 验证门槛 | 环境 |
|---|---|---|---|
| **Step 1** | 服务器重跑 120 条，输出 `confidence.txt`（last_pred_iou）全量落盘；绘制置信度与逐帧 IoU 的相关性 | 相关系数 > 0.3；失锁段置信度显著低于正常段（可视化 5 个失锁序列确认） | 服务器 GPU |
| **Step 2** | `recapture.py` 骨架 + ReliabilityGate 接线；60 条标定集上扫 τ_ok/N/K | 标定集上判丢召回率（真实失锁帧被标出）≥ 0.8，误报率 ≤ 0.2 | 服务器 |
| **Step 3** | 重捕获链路（redetect_v3 + VERIFY + REINIT）；5 个失锁序列（0047/0100/0041/0094/0117）逐条调试 | 每条找回 ≥ 1 次且后续 ≥ 50 帧不再次失锁；误锁 = 0 | 本地 + 服务器 |
| **Step 4** | 60 条验证集严格评分（`score_external_results.py` 同协议）；与 ODTrack 基线并表 | 验证集 AUC ≥ 基线 +0.02；正常序列（标定集上 AUC>0.5 的）无回退 | 服务器 |
| **Step 5**（可选） | ablation：关 anchor 校验 / 关模板记忆 / 换 K | 每个开关的独立贡献可解释 | 服务器 |

## 6. 评测与验收（与初赛同协议）

- 数据集/协议/评分器与初赛完全一致：120 条、首帧不计、普通+dual 双口径、
  行数严格对齐、宏平均；
- 输出必须包含：`bakeoff.json` + `scores.csv` + 失败归因表（每条失败序列：
  失锁时刻、找回次数、误锁次数、最终失败原因分类）；
- **诚实边界**：若验证集提升 < +0.01 或出现正常序列回退，判定不通过，
  回退到纯 ODTrack（feature flag 关闭），不得为了"有提升"而调参后重报。

## 7. 风险与回退

| 风险 | 缓解 |
|---|---|
| false recovery（锁错对象） | anchor 强校验 + 宁可继续 LOST；VERIFY 全门槛 |
| 重捕获 CPU 开销拖慢 FPS | 降采样 2x + 每 K=5 帧一次；K 可调大 |
| 在测试集上调参的过拟合 | 60/60 留出划分并落盘；报告标注 |
| ODTrack 与 wrapper 状态交互异常 | 只通过 initialize/track 公共接口；不动上游代码 |
| 时间不足 | 每步有独立门槛，Step 2/3 失败即止损，回退基线 |

## 8. 时间线（决赛 8/15 前）

| 时间 | 事项 |
|---|---|
| 8/11 | Step 1（全量置信度采集 + 相关性分析） |
| 8/12 | Step 2（判丢门控 + 阈值标定） |
| 8/13 | Step 3（重捕获链路调试，5 条失锁序列） |
| 8/14 | Step 4（验证集严格评分）+ Step 5 ablation + 决策（提升则更新答辩材料，否则维持初赛版） |

## 10. Step 1 执行记录（2026-08-10，已完成）

### 10.1 服务器不可用，路径调整

服务器 `153.0.134.134:12409` 已无法连接（Connection refused，归档清单记载为
"到期前"最后核对，服务器大概率已停机）。因此"服务器重跑 120 条输出
confidence.txt"不可行；全量 `last_pred_iou` 需要 GPU 推理（本地无 NVIDIA
runtime，CPU 全量约 100+ 小时不现实）。

**调整后的路径**（不改变方案目标）：

1. ✅ 用**已有 10 条**（`runs/odtrack_conf_10/0001-0010`）的
   `confidence.txt` 验证 `last_pred_iou` 与失锁的相关性（本节已完成）；
2. 🔄 Step 2 的判丢门控信号改为**离线可计算代理信号**（不依赖模型内部值，
   可在 120 条已有 results.txt + 数据帧上直接计算）：
   - C_anchor：预测框 crop 与首帧 GT crop 的 NCC（逐帧可离线算）；
   - C_motion：预测框中心与圆周恒速外推的偏差；
   - C_scale：框面积 log 变化；
   - geometry_risk：极区/seam 距离。
   `last_pred_iou` 仅在 10 条上参与权重标定（证据有限，作辅助信号）。

### 10.2 last_pred_iou 相关性实测（10 条，0001-0010）

分析脚本 `scripts/analyze_odtrack_confidence.py`，结果
`reports/results/odtrack_confidence_analysis_10.json`：

| 指标 | 值 | 判读 |
|---|---|---|
| 帧级 Pearson（conf vs IoU）均值 | **+0.391**（范围 -0.09~+0.72） | 正相关、中等强度 |
| 正常帧置信度均值 | 0.684 | — |
| 失锁帧（IoU<0.5）置信度均值 | 0.550 | 失锁时置信度显著更低 |
| 判丢区分度（1 - ROC-AUC）均值 | **0.747** | 低置信→失锁方向正确、区分中等偏强 |
| 反例 | 0004：Pearson -0.087，失锁帧置信度反而更高 | **单信号不可靠** |

### 10.3 Step 1 结论

- ✅ `last_pred_iou` 方向正确、可作 C_visual 分量（区分度 ≈0.75）；
- ⚠️ 存在反例（0004 等自信错跟）——**必须多信号融合**，与方案 §3.1 一致；
- 🔄 判丢门控主体改用离线代理信号（anchor/运动/尺度），120 条全量可评估；
- ✅ 置信度分析脚本与结果已入库（`f40f65b`）。

## 9. 不做什么（边界声明）

- 不训练/微调模型（官方训练集未开放前不做域自适应；开放后另行立项）；
- 不改初赛提交镜像、评分数据与交付包；
- 不做多模型端到端融合（Causal-DTP 已证明结果级路由增益有限，留作叙事）；
- 若 Step 1 显示 last_pred_iou 与失锁相关性弱，先换 C_visual 信号
  （响应熵 / top1-top2 margin），不硬凑阈值。
