# 服务器交接手册：ODTrack + 可靠性门控 + 球面重捕获（2026-08-10）

> 交付给持有 GPU 服务器的队友执行。本地已完成：方案设计、判丢信号验证、
> 重捕获 wrapper 代码与 CPU 冒烟、离线 60/60 门控评估。服务器上只需按
> 本手册跑完全量评测并回报结果。

## 1. 背景

初赛提交版（ODTrack ERP 三平铺，AUC 0.5792）有 10 个失锁序列（AUC<0.15），
共性：**失锁后无重捕获能力，时序记忆被污染后一路错到底**。
决赛方案：给 ODTrack 外挂"可靠性门控 + 球面多视角重捕获"，
目标 AUC 0.5792 → 0.60+。初赛提交数据不动。

## 2. 代码与数据清单（均在 Git `main` = `a774509` 及之后提交）

| 项 | 路径 | 说明 |
|---|---|---|
| 重捕获 wrapper | `integrations/odtrack/recapture.py` | 状态机：NORMAL→LOST→(SEARCH+VERIFY)→OBSERVE |
| 服务器评测入口 | `scripts/eval_odtrack_recapture.py` | 全量 120 条 OPE + 评分（本手册的主入口） |
| 判丢信号计算器 | `scripts/compute_odtrack_signals.py` | 离线信号（anchor NCC/运动/尺度/几何） |
| 门控评估器 | `scripts/score_offline_gate.py` | 60/60 留出标定 + 召回/误报评估 |
| 置信度采集 | `scripts/odtrack_360vot_conf.py` | 重跑 120 条输出 `confidence.txt`（last_pred_iou） |
| CPU 冒烟测试 | `tests/test_odtrack_recapture.py` | 状态机 3 项测试（本地已全绿） |
| 实施方案 | `docs/ODTRACK_RECAPTURE_PLAN_ZH.md` | 完整设计、阈值、风险 |
| ODTrack 上游 | 服务器 `/data/projects/instan_check/odtrack`（或本地 `artifacts/server_snapshot/upstream/odtrack`） | 源码+权重 |
| 数据 | 服务器 `/data/projects/instan/data360` | 120 序列 4K ERP |
| 基线结果 | `runs/odtrack_results_120/`（本地解压，评分见 `reports/results/odtrack_120_score/`） | 对照组 |

## 3. 服务器执行步骤（按顺序，每步有门槛）

### Step 1：全量置信度采集（约 2-3 小时，两张 GPU 分片）

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/odtrack_360vot_conf.py \
  --odtrack-root /data/projects/instan_check/odtrack \
  --data /data/projects/instan/data360 \
  --checkpoint /data/projects/instan_check/odtrack/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar \
  --config baseline --seqs 0001-0060 --gpu 0 --downscale 1.0 --out runs/odtrack_conf_gpu0
# GPU1 跑 0061-0120（参考 scripts/launch_odtrack_after.sh 的分片方式）
```

**门槛**：120 条全部产出 `confidence.txt`；抽查 3 个失锁序列
（0047/0100/0094）确认失锁段置信度显著低于正常段
（可本地用 `scripts/analyze_odtrack_confidence.py` 复核）。

### Step 2：判丢门控 60/60 标定（CPU，分钟级）

```bash
python scripts/compute_odtrack_signals.py --data data360 \
  --result-root runs/odtrack_conf_gpu0 --seqs all \
  --out runs/odtrack_signals_120 --with-ncc   # NCC 版约 2-4 小时
python scripts/score_offline_gate.py --data data360 \
  --result-root runs/odtrack_conf_gpu0 --signal-root runs/odtrack_signals_120 \
  --out reports/results/odtrack_offline_gate_60_60_v2.json \
  --w-motion 1.0 --w-scale 1.0 --w-ncc 2.0 --w-geom 1.0
```

**门槛**（对照本地 v1 结果：验证集召回 0.695 / 误报 0.284）：
- 加入 NCC 后验证集误报率应显著下降（目标 < 0.15），召回不降；
- 输出 JSON 里的 `gate.threshold / run_len` 即为最终判丢参数。

### Step 3：全量重捕获评测（约 3-5 小时，GPU）

```bash
python scripts/eval_odtrack_recapture.py \
  --odtrack-root /data/projects/instan_check/odtrack \
  --data /data/projects/instan/data360 \
  --checkpoint /data/projects/instan_check/odtrack/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar \
  --config baseline --seqs all --gpu 0 \
  --run-len 5 --search-interval 5 --observe-frames 3 \
  --anchor-min-sim 0.5 --recapture-min-score 0.45 --motion-max-deg 90.0 \
  --out runs/odtrack_recapture_120
```

**门槛**：
- 与基线（`reports/results/odtrack_120_score/`）同协议对比：普通 AUC/SR、双口径、宏平均；
- 失锁序列（0047/0100/0041/0094/0117 等）应有 recovered 事件且其后跟踪不再立即失锁；
- **正常序列（基线 AUC>0.5 的约 90 条）不允许回退超过 0.01**。

### Step 4：报告（回传给队长）

- `summary.csv` + `bakeoff.json`（用 `scripts/score_external_results.py` 统一评分）；
- 失败归因表：每条失锁序列的 `recapture_stats.json`（lost 帧数 / recovered 次数）；
- 明确标注：**阈值是否在验证集上做过二次调整**（禁止——60/60 划分一旦确定不得重标）。

## 4. 纪律与止损

- **留出划分**：Step 2 标定只用奇数序列，验证只用偶数序列（划分已落盘在
  输出 JSON 的 `split` 字段）；任何调参后重报验证集成绩 = 无效；
- **止损**：验证集 AUC 提升 < +0.01 或正常序列回退 > 0.01 →
  判定方案不通过，回退初赛版（纯 ODTrack），**不要为了"有提升"硬调参**；
- **初赛数据**：提交镜像、`reports/results/odtrack_120_score/`、交付包
  `deliverables/SUBMISSION_2026-08-10/` 全部冻结，服务器实验不得改写。

## 5. 本地已完成的验证（供对照）

| 验证 | 结果 |
|---|---|
| last_pred_iou 与失锁相关性（10 条） | Pearson +0.391；判丢区分度 0.747；方向正确 |
| 离线判丢门控（纯框信号，120 条 60/60） | 验证集失锁段召回 0.695，误报 0.284（NCC 应进一步降误报） |
| recapture 状态机 CPU 冒烟（3 项测试） | 全绿：正常→lost→recovered→ok；目标消失持续 lost；远跳 VERIFY 拒绝 |
| 重捕获链路 | redetect_v3 在合成帧命中移动目标（score 0.55），VERIFY+reinit 生效 |

## 6. 产出时限

- 8/14 前完成 Step 1-3 并回报 Step 4 数据；
- 8/14 晚由队长决策：提升则更新决赛答辩材料（PPT 增补"可靠性门控+重捕获"一页），
  否则维持初赛版答辩。

## 7. 决赛若采用重捕获版：提交入口改造（决策通过后再做）

初赛镜像（`grt360-odtrack:2026-08-10`，入口 `integrations/odtrack/file_protocol.py`）
已冻结，**决赛若采用重捕获版，需在镜像重建时同步改造**：

1. `file_protocol.py` 增加 `--recapture` 开关：构造 `OdtrackRecaptureTracker`
   替代裸 `ODTrack`（判丢/重捕获参数用 Step 2 标定的阈值）；
2. `docker/odtrack/Dockerfile` 需新增 COPY `integrations/odtrack/recapture.py`
   与 `panotrack/pipeline/memory.py`、`redetect_v3.py`（重捕获依赖）；
3. 重建镜像 + CPU 冒烟（`--force-cpu` 3 帧）+ 服务器 GPU 全量复测后，
   更新 `deliverables/SUBMISSION_2026-08-10/` 与交付清单；
4. 注意：重捕获的 redetect_v3 是 numpy/PIL 实现（无 torch 依赖），
   镜像依赖面不变，体积增加可忽略。
