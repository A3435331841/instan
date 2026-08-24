# 决赛冲刺执行计划 v3（定稿，2026-08-24）

> 本版为**当前唯一有效版本**，整合 v1/v2 及 8/22~8/24 全部决策与实测结论。
> 前置文档：技术调研 `docs/TECH_RESEARCH_ZH_2026-08-21.md`；recapture 详细设计 `docs/ODTRACK_RECAPTURE_PLAN_ZH.md`。
> 时间轴：D0 = 服务器 SSH 到手当天；决赛日期未知（平台注册已锁，需队长用队伍账号查询）。

---

## 一、当前状态快照（2026-08-24）

### 已就绪（本地）

| 资产 | 位置 | 状态 |
|---|---|---|
| 官方训练集 | `D:\instan\初赛数据\train\`（已解压 24GB）+ 原 zip | **130/130 校验通过**（帧数=GT 行数，0 损坏）；sim 83 条（多为 450/900 帧）+ real 47 条（~1600 帧）；1440×720；BFoV 逐帧 GT |
| **train/valid 划分** | `pano360/data360/official_split/`（split.json + 两个 seqlist） | **train 95**（32 real + 63 sim）/ **valid 35**（15 real + 20 sim），seed 20260824 确定性划分；**valid 绝不参与训练** |
| 跟踪器权重 | `artifacts/server_snapshot/weights/`（ODTrack ep0300 / SUTrack ep0300 / UETrack / iTPN）+ `external/LoRAT`（源码+权重） | 全部本地就绪，SHA 已记录 |
| 初赛提交镜像 | `交付物_2026-08-14/grt360-odtrack_2026-08-14_cu128.tar`（4.7GB） | docker load 即用 |
| 协议/评测代码 | `integrations/odtrack/arena_protocol.py`（BFoV 协议内核）、`scripts/`（评分器/下载器） | 与初赛一致 |
| 本地 360VOT 序列 | `data360/zips/` 共 8 条 | 仅冒烟用 |

### 等待用户提供

1. **服务器 SSH**（host/port/账号/密码或密钥）+ 卡型确认 → D0 启动条件
2. **HF token**（read，`xuyzshaun/360VOTS` 已点同意）→ 可选线，仅用于 120 序列补充下载（**直接在服务器上跑**，本地不碰）

### 已定关键决策（不再讨论）

- **评测基准 = 官方 130 条**（与隐藏测试集同分布、同分辨率 1440×720）；360VOT 120 条降级为可选交叉验证；
- **先赛马、后训练**：微调预算只投给赛马冠军；
- 划分纪律：valid 35 条永不训练；所有对比同评分器同序列集；
- 每日提交配额 3/累计 10（v1 已废 1 次），本地全绿才 push。

---

## 二、D0 Runbook（SSH 到手后按序执行，全部命令可复制）

> 目标：当晚拿到**官方分布基线 AUC/FPS**。预估：传输 0.5~2h（看带宽）+ 环境 0.5h + 基线评测 ~1h。

### 2.1 体检（5 分钟）

```bash
nvidia-smi                                    # 卡型/驱动；驱动 <570 则见 §2.3 注
df -h / /data 2>/dev/null                     # 磁盘（需 ~120GB：数据58+镜像14+权重4+训练缓存）
docker --version && docker ps                 # docker 可用性
python3 --version
```

### 2.2 传输（后台并行，最大件先行）

```bash
# 本机执行（Windows Git Bash）；rsync 不可用则 scp，均带断点续传
rsync -avP "/d/instan/初赛数据/ys_panotracking_train.zip"  user@host:/data/pano/   # 24.7GB 最先
rsync -avP "/d/instan/pano360/artifacts/server_snapshot/weights/" user@host:/data/pano/weights/  # 3.9GB
# 代码走 git（official_split 随仓库走）；若 GitHub 不通则 rsync 整仓（排除 data360/ runs/）
```

### 2.3 环境（传输期间并行）

```bash
# 统一 torch 2.7.0+cu128：与评测机(5090)零漂移；cu128 支持 sm_70+，3090/4090/A100 均可跑
# 注：cu128 需驱动 ≥570；若租的机器驱动旧且无法升级 → 开发环境退 cu121，最终镜像仍按 cu128 构建
docker load -i /data/pano/grt360-odtrack_2026-08-14-cu128.tar     # 初赛镜像，作对照与复用底座
python3 -m venv ~/venv && pip install torch==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 2.4 数据就位

```bash
cd /data/pano && 7z x ys_panotracking_train.zip        # 或 unzip -q；产出 train/train_{real,sim}/
python scripts/validate_official_data.py               # 复核 130/130（本地已过，服务器抽查 quick 模式即可）
python scripts/make_official_split.py --root /data/pano/train   # 产出与本地完全一致的划分（seed 固定）
# HF token 到手后（可选线，tmux 挂后台，不占 GPU）：
HF_TOKEN=hf_xxx HF_ENDPOINT=https://hf-mirror.com nohup python scripts/download_all_360vot.py > dl360vot.log 2>&1 &
```

### 2.5 首个 GPU 任务：ODTrack 官方基线（D0 核心）

- 用**官方评测 runner**（P0-A 产出，见 §五）对 130 条全量推理：BFoV init → 三平铺 ODTrack → BFoV 输出 → 评分（像素 dual IoU + 球面 IoU 双口径）；
- 产出：**基线 AUC/SR/FPS 总表 + sim/real 分项 + 逐序列 CSV**（对齐 `score_external_results.py` 格式）；
- 验收：跑完 130 条退出码 0；FPS 为 1440×720 实测口径。此表 = 后续一切对比的锚点。

### 2.6 D0 晚（GPU 空闲窗口）：赛马适配器冒烟

SUTrack / LoRAT 各跑 1 条序列（valid 里挑），确认 import/权重/输出格式无误，为 D1 赛马扫清环境障碍。

---

## 三、D1：主干赛马（零训练，一天出冠军）

**候选与看点**：

| 候选 | 权重 | 看点 |
|---|---|---|
| ODTrack ep0300（基线） | 本地 | §2.5 已出分 |
| SUTrack-B/L | 本地 ep0300（源码当天 clone `chenxin-dlut/SUTrack`；变体缺则 HF 补） | LaSOT 74.4/75.2，自回归同源适配成本低 |
| LoRAT-B378/L378 | `external/LoRAT` 全套 | L 精度 75.1；**B378 285FPS@4090 若不掉分则速度问题一并解决** |
| SAMURAI-L（可选） | 需下 SAM2.1-L | 长时/遮挡最强（LaSOT_ext 61.0）；mask→BFoV 用 360VOT 工具包；跑不顺则降级为失锁分支 |

**纪律**：同一 runner、同一评分器、先 valid 35 条筛序（~10 分钟/候选）、前三名上全量 130 条；记录 AUC/SR/FPS/显存四元组。**决策门**：全量最高者为冠军；无人超 ODTrack+0.005 → 维持 ODTrack（SAMURAI 转失锁验证分支）。

## 四、D2~D5：冠军主干上的三大工程

### D2：recapture 外挂（目标 +0.02~0.04）

按 `ODTRACK_RECAPTURE_PLAN_ZH.md` Step1-4，替换项：
- 全局重检测 = **redetect_v4 多切平面扫描**（8 朝向 tangent + 极区 2 面；依据 360VOT Issue #11：整幅 ERP 全局重识别必败）；
- 判丢信号增加冠军 tracker 置信度 + （若引入）SAM2 occlusion head；
- 标定/验证 = **official-valid 内再对半分**（17/18），绝不碰训练 95 条；
- 失锁调参序列：从 valid 里挑 AUC 最低的 5 条（等价于初赛时 0047/0100 那批的角色）。
- 决策门：验证半区 ≥ 同半区纯冠军基线 **+0.02** → 采用；否则 feature-flag 关闭。

### D2~D3：官方数据微调（目标 +0.03~0.06）

- 数据：train 95 条（`seqlist_official_train.txt`）→ 三平铺样本流；随机经度 roll 必开；AirSim360-Human 增广**暂缓**（官方 sim 83 条已是仿真域，先看纯官方数据效果）；
- 训练：冠军权重热启动，lr 原 1/10，AMP；1440×720 下预计**数小时/轮**；逐 epoch 存 ckpt；
- GPU1 快评通道：每 ckpt 在 valid 35 条评分（~10 分钟），top-2 上全量 130；
- 决策门：全量 ≥ 冠军基线 +0.005 → 换权重；与 recapture 叠加回归不回退（≥ 单项最优 -0.005）→ 合并。

### D3~D4：速度（压力已因 1440×720 大减，按需裁剪）

先测冠军裸跑 FPS（1440×720 口径）：≥25 FPS 则本阶段只做零风险项（NVDEC 硬解 + 流水线），否则按序全做：环绕 pad 裁剪 → BF16 → torch.compile →（备选）TRT FP16。每步独立 A/B，掉点 >0.005 即回退。**FPS 全部注明卡型**，不冒充 5090 数字。

### D5：集成、提交、答辩

1. 版本 = 冠军(+微调)(+recapture)(+速度项)，全量 130 回归 + （若已到位）360VOT 120 条独立确认；
2. cu128 镜像重建 → `--network none` 断网全链路自测（720 分钟时限模拟、退出码 0、行数=帧数）；
3. 平台确认队伍 UID/剩余配额 → push（model:vN）；
4. 答辩素材（D3 起滚动更新）：官方分布成绩、赛马对比表、瓶颈归因、recapture=360VOT 点名的 open problem。

---

## 五、本地开发任务（现在 → 服务器到手前，按优先级）

| # | 任务 | 产出 | 验收（本地 CPU） |
|---|---|---|---|
| **P0-A** | **官方评测 runner**（D0 核心依赖，最高优先）：mp4 解码 → BFoV→ERP init → 三平铺推理 → ERP→BFoV 输出 → dual IoU + 球面 IoU(MC) 评分 → summary.csv；tracker 后端可插拔（D1 赛马直接换后端） | `scripts/eval_official.py` | 本地 2 条序列 CPU 冒烟（ODTrack CPU 慢，截 20 帧验证格式与评分正确性即可） |
| P0-B | SUTrack / LoRAT 适配器（BaseTracker 契约 + 三平铺输入） | `integrations/sutrack/`、`integrations/lorat/` | import 自检 + 1 条序列前向（CPU 可跑则跑） |
| P0-C | 微调数据管道：video+BFoV → 三平铺训练样本流（随机经度 roll） | `training/` | dry-run 1 step |
| P0-D | redetect_v4 多切平面重检测 | `panotrack/pipeline/redetect_v4.py` | 合成跨界序列找回测试 |
| P0-E | （token 到手后）服务器下载 120 序列的启动命令与断点续传核对 | - | - |

> P0-A 完成前不动 P0-B 以下任务；服务器到手即冻结本地开发转 D0。

---

## 六、决策门总表（一页速查）

| 时点 | 门 | 通过 | 不通过 |
|---|---|---|---|
| D0 | 数据校验 130/130 + runner 跑通 | 基线表落盘 | 修 runner/环境，不进 D1 |
| D1 | 赛马冠军 | 全量最高 | 无人超 +0.005 → ODTrack 留任 |
| D2 | recapture 验证半区 | ≥ +0.02 | flag 关闭 |
| D3 | 微调 ckpt | 全量 ≥ 基线+0.005 | 保留原权重 |
| D3-4 | 速度每步 | 掉点 ≤0.005 | 回退该步 |
| D5 | 提交 | 断网全绿 + 配额确认 | 不 push |

## 七、风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| 服务器驱动 <570（cu128 装不上） | 中 | 开发退 cu121；镜像构建仍在 Docker（cu128）内完成 |
| 传输带宽低（24.7GB 传几小时） | 中 | 最大件先行 + rsync 断点续传；传输期间并行装环境/写 P0 |
| SUTrack 源码环境老（pytracking 系） | 中 | 跑不通即弃，LoRAT 码基新风险低；候选不贪多 |
| 决赛时间突然公布且很近 | 中 | 砍 D3 微调（保赛马+recapture+零风险提速）；D0/D1 是不可砍底线 |
| 120 序列始终拿不到 | 低影响 | 全流程不依赖它；仅答辩少一个对比口径 |
| 平台侧故障（8/22 曾见 DB 未就绪） | 低 | 提交窗口尽早用，不压 deadline |

---

## 八、变更记录

- v1（8/22）：初版，ODTrack 增量改进路线
- v2（8/22）：调研成果落地--赛马提前、微调改训冠军、SAMURAI 纳入候选
- v3（8/24，本版）：训练集实测落地（130 条/1440×720/BFoV，校验通过）；官方数据取代 360VOT 成主基准；划分文件入库；120 序列改为服务器后台可选下载；D0 Runbook 细化到命令级
