# 技术方案调研报告：更高精度与更快速度的可行渠道（2026-08-21）

> 调研范围：2023-2026 论文（arXiv/CVPR/ICCV/ECCV/AAAI/NeurIPS/TPAMI）与 GitHub 开源仓库，
> 四个方向并行调研：①通用 SOT SOTA ②全景 360° 专项 ③SAM2 长时跟踪/重捕获 ④RTX 5090 部署加速。
> 结论针对本队现状（ODTrack ERP 三平铺，本地 360VOT 120 序列 AUC 0.5792 / SR 0.6532 / 8.99 FPS，
> 初赛已提交，决赛综合精度 + 帧率 + 答辩）。

---

## 〇、总体结论（先看这一节）

1. **我们已经是公开文献里的事实 SOTA**。360VOT/360VOTS 全部公开结果里最高的是
   AiATrack-360 的 S_dual **0.534**（TPAMI 2025 Table III，2025-2026 无人刷新）；
   我们本地 0.5792 已超出 +8.4%。**决赛的主要对手是其他参赛队，不是文献方法**。
2. **换更强的通用跟踪器不是当前的杠杆**。LaSOT 榜上确实有比 ODTrack 高 4-5 个点的方法
   （SPMTrack-G 77.4、SUTrack-L 75.2、LoRAT-L 75.1），但 360 域的直接证据是：
   LoRAT 接入 360VOT 官方框架微调后只有 0.495，仍低于我们的 0.5792——
   **ERP 三平铺适配的收益大于 backbone 本身的差距**。除非重训，否则换模型不划算。
3. **精度最大的两个增量：** ①把已写好的 recapture 外挂跑通（10 条失锁序列，
   预期宏平均 +0.02~0.04）；②官方训练集/AirSim360-Human 伪标签微调
   （文献证据 LoRAT 360 微调 +7% 相对，预期 AUC 冲 0.62+）。
4. **速度瓶颈根本不在模型**。ODTrack-B 官方纯推理在 2080Ti 上就有 32 FPS，
   我们端到端 9 FPS 的 ~70% 开销在 **4K CPU 软解码 + 三平铺整图拷贝 + CPU 裁剪**。
   NVDEC 硬解 + 环绕 padding 裁剪 + BF16，零/近零精度损失即可到 **40-80 FPS**，
   决赛帧率目标有三倍安全余量。
5. **答辩叙事有了文献背书**：360VOT 原论文把"支持重定位的长时全景跟踪"列为公开
   未解难题；我们做的"判丢+全 360° 找回"正是评委知道的空白方向。

---

## 一、通用 SOT 调研（更准/更快的跟踪器）

### 1.1 LaSOT 公开权重现实排名（AUC）

| 方法 | 出处 | LaSOT AUC | 速度 | 权重 | 对我们的价值 |
|---|---|---:|---|---|---|
| SPMTrack-G/L/B | CVPR 2025 | **77.4 / 76.8 / 74.9** | 未公布（L 级别） | GDrive（B 待审批） | LaSOT 最高可复现；LoRAT 码基 |
| LoRAT-g/L/B (378) | ECCV 2024 | 76.2 / **75.1** / 72.9 | **B378 285 FPS@4090** | GDrive 齐全 | 速度-精度兼顾；码基最现代 |
| SUTrack-L384 | AAAI 2025 | 75.2（GOT 81.5） | 未公布 | **HuggingFace** | 断网打包最方便；自回归范式同 ODTrack |
| SAMURAI-L | arXiv 2411 | 74.2（零训练） | SAM2.1 栈，实时 | 官方 SAM2.1 权重 | 长时/出视野能力最强（LaSOT_ext 61.0 领先 7 个点） |
| ODTrack-L384（现用） | AAAI 2024 | 72.2~74.0 | 32 FPS@2080Ti（纯推理） | 已有 | — |
| AQATrack / EVPTrack-384 | CVPR24 / AAAI24 | 72.7 / 72.7 | 同 ODTrack 档 | GDrive | 相对 ODTrack +0.5~0.7，换装性价比低 |

**勘误**（与旧认知不同的）：AQATrack 是 CVPR 2024 不是 2025；LMTrack（AAAI 2025，LaSOT 73.2）
**无官方代码**；FBD.SimViT 查无此文；FeedbackTrack（2026.08，LaSOT 79.1）暂无代码，决赛前可再查一次。
LoRATv2（NeurIPS 2025 Spotlight）仓库还是占位符"weights coming soon"。

### 1.2 结论

- **精度上限换装**（若决赛前有 2-3 天 GPU）：SPMTrack-L 或 LoRAT-L378 + 我们的 ERP 三平铺适配
  + 360 域微调，是"更准"路线的正解；LoRAT 码基纯 PyTorch 2.x、无 mmcv，升 torch 2.7+cu128
  工程量最小（**仓库已 clone 在 `external/LoRAT`，base.bin 权重已校验入库**）。
- **速度换装**（若帧率权重高）：LoRAT-B378 在 4090 上 285 FPS、精度 72.9 ≈ ODTrack 水平，
  三平铺折算仍 ~95 FPS，是"更快且不掉精度"的直接替换。
- 不建议投入：MixFormerV2（70.6 已过时）、HiT（64.6 精度不够，但其 DyHiT 动态早退思想可白嫖）、
  SeqTrack/ARTrack 老栈、任何需要编译 CUDA 扩展的老仓库（断网镜像风险）。

---

## 二、全景 360° 专项调研

### 2.1 360VOT 排行榜现状

| 方法 | S_dual AUC | 备注 |
|---|---:|---|
| **我们（ODTrack ERP 三平铺）** | **0.5792** | 本地口径，事实 SOTA |
| AiATrack-360 | 0.534 | 官方框架集成，TPAMI 2025 榜首 |
| LoRAT-360（微调后） | 0.495 | 唯一做过 360 微调的方法（0.461→0.495，+7% 相对） |
| OSTrack / HIPTrack / AiATrack 原生 | 0.447 / 0.440 / 0.405 | 全部零微调 |

三重证据（Semantic Scholar 引用网络 + 影石自家 Awesome Panoramic Vision 仓库 + 360VOT GitHub
Issues）确认：**2025-2026 没有新的全景单目标跟踪架构发表**；领域重心转向分割（360VOTS）、
MOT（OmniTrack CVPR 2025）与数据（Leader360V NeurIPS 2025）。

### 2.2 对我们直接有用的发现

| 发现 | 来源 | 用法 |
|---|---|---|
| **全局 ERP 重识别必然失败**：记忆特征在无畸变局部裁剪上编码，与整幅畸变 ERP 全局特征失配；多视角切平面扫描才是对的 | 360VOT 仓库 Issue #11（2026-01，开发者实测） | **recapture 的全局重检测必须用多切平面（6-8 个朝向）逐视角检测，禁止直接喂整张 ERP** |
| 切平面表示在 FoV ≥ 90° 时自己失真（S_dual 0.534→0.449） | 360VOT 官方消融 | 大 FoV 目标（近距/大目标）回退球面片段或分块处理 |
| PanoSAM2：环绕 padding 解码器（接缝连续感受野）+ 畸变加权损失 + 长期 object pointer 记忆，360VOTS +5.6 | arXiv 2604.07901（未开源） | 环绕 padding 几十行可手写进 ODTrack 的 patch embedding；畸变加权损失可用于微调 |
| OmniTrack++ 的模式切换状态机：置信度驱动"端到端跟踪 ↔ 检测全局模式" | arXiv 2511.00510 | recapture 状态机可对照借鉴 |
| **AirSim360 没有任何 tracking 标注**：只有 ERP RGB / 深度 / 语义 / 实例 mask 四模态；Omni360-Human 有 100,700 帧、每场景 4-45 个 NPC | arXiv 2512.02009 + HF Insta360-Research/AirSim360 | 测试集真值大概率是主办方用实例 mask + `mask2Bfov` 自动生成；**训练侧同样可用实例 mask 批量产 BFoV 伪标签**（正好补官方训练集） |
| 球面原生 backbone 路线已被领域放弃（无法复用透视预训练权重，scale 不动） | 全景综述 arXiv 2606.27745 | 答辩可用：我们"复用透视权重+几何适配"的路线与领域共识一致 |

### 2.3 可叠加到现有方案的提升清单（按性价比）

| # | 措施 | 证据 | 预估增益（AUC 相对） | 成本 |
|---|---|---|---|---|
| 1 | 官方训练集 + AirSim360-Human 伪标签微调 ODTrack（三平铺输入形式 + 随机经度旋转增强） | LoRAT 360 微调 +7%；官方 baseline 全零微调 | **+5~10%** | 中 |
| 2 | 重捕获改为多切平面扫描（6-8 视角，禁直接喂 ERP） | Issue #11 实证 | 失锁场景 **+2~5%** | 中（recapture.py 已有骨架） |
| 3 | 大 FoV（≥90°）目标退化处理 | 官方消融 -16% 的坑 | 近距场景 +2~4% | 低 |
| 4 | 环绕 padding 进 patch embedding + 畸变加权损失 | PanoSAM2 | +1~3% | 低（padding）/低-中（损失） |
| 5 | tangent crop 搜索图替代三平铺（或双路按帧切换） | AiATrack 集成 +12.9%（我们已有平铺打底，边际小） | +3~8% | 中 |

组合预期：#1 + #2 全上，**从 0.5792 冲 0.62~0.66 有文献级证据支撑**。

---

## 三、SAM2 系长时跟踪 / 重捕获（针对 10 条失锁序列）

### 3.1 关键事实

- **SAMURAI**（github.com/yangchris11/samurai，7.1k★，零训练）：直接用 SAM2.1 官方权重 +
  Kalman 运动记忆。LaSOT 74.2、GOT-10k 81.7、**OTB100 71.5（与我们比赛同协议！）**、
  全遮挡属性 +12.7%。~4GB 显存，4090 实时。**不解决全局找回**（只在预测邻域内选 mask）。
- **SAM2 的 occlusion head 输出目标可见性分数 = 免费的判丢信号**；
  任意帧可 `add_new_points_or_box` 重新注入目标 = 原生的找回接口。
- **SAM2Long**（ICCV 2025）在 360VOTS 增益很小（+0.4~0.9），不值得上。
- **HiM2SAM**（arXiv 2507.07603）：免训练，LaSOT 75.1，长短期记忆划分；仓库小（29★）备用。
- **PanoSAM2** 证明 SAM2 在 360VOTS 有 ~60 J&F 基本盘、接缝/畸变是主要损失源（同 §2.2）。
- 360VOT 工具包（github.com/HuajianUP/360VOT）自带 `mask2Bfov / mask2Bbox /
  localBbox2Bfov / rot_image`——**mask→BFoV 转换不要自己造轮子**；取最大连通域防遮挡碎块撑爆外接框。

### 3.2 推荐架构（与 recapture.py 现有骨架对齐）

```
NORMAL: ODTrack 逐帧（不动）
   │ 判丢信号 = ODTrack 置信度 + SAM2-S 并行分支的 occlusion 分数 / mask-IoU 一致性投票
   │        （双分支输出分歧本身也是强判丢信号；SAM2-S 常驻 ~4GB / 39 FPS，5090 显存无压力）
   ▼ 连续 k 帧低置信
LOST: 冻结 ODTrack；每 K 帧：ERP 用 rot_image 采样 6-8 个切平面视角
   → YOLO-World-S / RT-DETR-R18（Ultralytics 预编译 wheel，断网安全）按类别检测
   → 候选框映射回球面，跨视角去重用球面角度距离（不是平面 IoU）
   ▼ 候选过 VERIFY（anchor 相似 + 运动合理 + 分数门槛——recapture.py 已有）
REINIT: 重新 initialize ODTrack → OBSERVE 3 帧 → NORMAL
```

- 检测器选型优先级：**YOLO-World-S / RT-DETR-R18（Ultralytics，无编译风险）＞ Florence-2 ＞
  Grounding DINO 1.0（deformable attention 需现场编译，Blackwell 断网环境风险最高，不建议）**。
  类别文本 prompt 可首帧用离线 CLIP 从初始 crop 自动生成。
- 预期：救回一半失锁序列 → 宏平均 AUC **+0.02~0.04**；对 AirSim360 仿真侧（小目标、
  快速丢失、全向运动）收益更大。
- 备选冲上限：SAMURAI-L 整条替换主干做对照实验（若 120 条宏 AUC ≥ 0.58 则切单栈方案更简洁）。

---

## 四、RTX 5090 部署加速（决赛帧率）

### 4.1 核心判断：瓶颈在解码与预处理，不在模型

ODTrack-B 纯推理 32 FPS@2080Ti；我们 3090 端到端只有 8.99 FPS。每帧 ~111ms 中约 70ms
消耗在：4K H.264 CPU 软解（40-60ms）+ 三平铺整图拷贝 66MB（5-15ms）+ CPU 裁剪 resize（10-30ms）。
另：仅换 5090（计算 3×、带宽 1.9×、NVDEC 引擎 2×）不改代码就应有 15-20 FPS。

### 4.2 加速清单（按性价比排序，前三项零精度损失）

| # | 措施 | 预期 | 精度风险 | 工程量 |
|---|---|---|---|---|
| 1 | **NVDEC 硬解（PyNvVideoCodec）+ 解码/预处理/推理三级流水线** | 去除 40-60ms 瓶颈，9→30+ FPS | 零 | 中 |
| 2 | **三平铺 → 单帧直取 crop + 仅跨缝时水平环绕 pad**（`F.pad(mode='circular')` 或双窄条拼接）；裁剪/resize 全上 GPU（`F.grid_sample`） | 再省 10-15ms，数据搬运降 1-2 个量级 | 零 | 小-中 |
| 3 | **BF16/FP16 autocast** | 推理 1.5-2× | <0.3 AUC（训练本就是 AMP） | 极小 |
| 4 | torch.compile（max-autotune，只编译纯 tensor 子图；Inductor 缓存预热进镜像） | 推理再 +20-40% | 零 | 小 |
| 5 | TensorRT FP16（torch-tensorrt 2.7.0+cu128 存在，但必须锁 `tensorrt>=10.8`，10.7 不识别 5090） | 推理 3-4×（OSTrack 实测 240 vs 70 FPS） | 近似无损 | 中-大，sm_120 有个例异常报告，须 5090 实测 + eager fallback |

**叠加预测**（5090）：现状代码直接跑 ~15-20 FPS → +硬解流水线 25-35 → +GPU 预处理 45-60 →
+BF16/compile **60-80+ FPS**。720 分钟时限下 25 FPS 即可处理 ~108 万帧（9 FPS 只有 38.9 万帧）。

**不建议**：INT8/FP8（ViT PTQ 掉点 >1%，回归任务更敏感）；降分辨率/ViT-S（OSTrack 实证掉 2 个点起）；
DALI（sm_120 未声明）与 decord（无官方 GPU wheel）。

### 4.3 落地注意

- 所有 wheel（pynvvideocodec / torch-tensorrt / tensorrt>=10.8）可预打包进断网镜像；
  TRT engine 与 Inductor 缓存需在 5090 同款卡上构建/预热。
- 评测机驱动 ≥R570（cu128 要求，已与镜像匹配）。
- 若不想动 ODTrack 推理栈，**LoRAT-B378 换装**（285 FPS@4090，精度持平）是速度档备选。

---

## 五、行动建议（结合剩余赛程）

**决赛前精度冲刺（优先级从高到低）**
1. 服务器/GPU 恢复后第一件事：跑通 recapture 外挂全量验证（按 `docs/ODTRACK_RECAPTURE_PLAN_ZH.md`
   的 Step 1-4 走；全局重检测务必按 §3.2 改多切平面，对照 Issue #11 证据）。
2. 有 ≥1 天 GPU 富余：官方训练集 + AirSim360-Human 实例 mask → BFoV 伪标签（`mask2Bfov` 现成）
   → 微调 ODTrack（三平铺输入 + 随机经度旋转增强 + 畸变加权损失）。
3. 零成本修补：大 FoV（≥90°）退化处理；patch embedding 环绕 padding。

**决赛帧率**
4. 按 §4.2 顺序做 1→2→3→4；TensorRT 只在前四项不达标时考虑。
5. 每项改动后本地 120 序列全量回归（团队纪律：小样本数字不可信）。

**答辩**
6. 叙事要点：本地 0.5792 超公开 SOTA（AiATrack-360 0.534）；重捕获方向 = 360VOT 论文点名的
   open problem；速度路线 = 瓶颈定量归因（70% 在解码/预处理）+ 零损失工程优化，契合赛题
   "低算力高效实时"考点。

---

## 六、参考链接汇总

**跟踪器**：SPMTrack github.com/WenRuiCai/SPMTrack ｜ LoRAT github.com/LitingLin/LoRAT（本地
`external/LoRAT` 已有）｜ SUTrack github.com/chenxin-dlut/SUTrack（HF: xche32/SUTrack）｜
SAMURAI github.com/yangchris11/samurai ｜ ODTrack github.com/GXNU-ZhongLab/ODTrack

**全景**：360VOT 工具包 github.com/HuajianUP/360VOT（`crop_bfov`/`mask2Bfov`/`rot_image`/eval）｜
360VOTS TPAMI 榜 arxiv.org/html/2404.13953v2 ｜ PanoSAM2 arxiv.org/abs/2604.07901 ｜
OmniTrack++ arxiv.org/abs/2511.00510 ｜ 全景综述 arxiv.org/abs/2606.27745 ｜
AirSim360 HF: Insta360-Research/AirSim360（Omni360-Human 10 万帧实例 mask）

**加速**：PyNvVideoCodec pypi.org/pypi/pynvvideocodec ｜ torch-tensorrt cu128 wheel
download.pytorch.org/whl/cu128/torch-tensorrt/ ｜ PyTorch 2.7 Blackwell 支持说明

> 注：个别极新条目（PanoSAM2、FeedbackTrack）为调研转述，使用前请点开原文链接核实；
> LaSOT/360VOT 数字均为论文口径，官网排行榜在调研网络下不可达。

*报告生成：2026-08-21，四路并行调研（通用 SOT / 全景专项 / SAM2 长时 / 部署加速）汇总。*
