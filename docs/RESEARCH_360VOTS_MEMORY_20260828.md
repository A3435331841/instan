# 360VOTS / 记忆跟踪调研与 GRT-360 落地判断

日期：2026-08-28

这份笔记核对了 360VOTS/eBFoV、XMem-360、DTPTrack、StreamDAM、SAMURAI 和
HiM2SAM，并与本项目已有的 130 条数据、B224 OpenVINO runner 和失败矩阵对齐。
结论不是直接换模型，而是把论文中已被证实的机制拆成可验证的小实验。

## 1. 先校准论文数字的含义

360VOTS TPAMI 版本的官方 HTML/论文明确将 eBFoV 定义为大视场表示：当水平或垂直
FoV 超过 90° 时，用球面均匀角采样而不是 gnomonic 切平面；同时搜索区由 `SR Ratio`、
`SR Min` 和 `Max Loss` 控制。论文的 360VOT 是 120 条测试序列，和本项目的
`ys_panotracking_train` 130 条不是同一数据划分，因此只能作为机制证据，不能把分数
直接当成我们的可达成绩。

论文 Table VI 的 XMem-360*（把重训后的 XMem 接入 360 框架）为：

| 口径 | S_dual/AUC | 其他指标 |
|---|---:|---|
| 360VOT BBox | 0.583 | P_dual 0.588，归一化 P_dual AUC 0.607 |
| 360VOT rBBox | 0.387 | P_dual 0.566 |
| 360VOT BFoV | 0.594 | 角精度 0.635 |
| 360VOS 分割 | J_sphere 0.677，F_sphere 0.801 | 不是 BBox AUC |

因此“XMem-360* 全场最优”在论文的相应口径下成立，但它是分割记忆模型经过
360VOS 训练和 mask→BFoV/BBox 转换后的结果，不等价于未经适配的 XMem 直接替换
B224，也不等价于本项目 full130 的 AUC。

## 2. 逐项核验与本地映射

### eBFoV：高价值，已落地为受限的因果分支

360VOTS 论文给出的流程是：BFoV 状态 → 球面/切平面 remap → 局部 tracker →
逆投影回 BFoV。大 FoV 时切平面 `tan` 失真，使用球面表面上的等角采样；目标跟踪
状态也应保持在球面角度，而不是长期保存 ERP `xywh`。

本仓库的 `panotrack/geometry/projection.py` 和
`scripts/run_sutrack_b224_openvino_sequence.py` 已实现球面 remap、eBFoV/gnomonic
采样及局部框逆投影。全局静态切换并不安全：在 real/0008、0016、0027、0030、
sim/0046、0082 的完整序列实验中，直接 `projection_mode=auto` 会明显回退。
因此当前主线只启用一个通过完整序列对照的高纬 63×45° 几何簇
（`route_ebfov_special`），sim/0064 的 AUC 从 B224 约 0.291 提到 **0.6760**，
正常控制 sim/0065 保持 0.7241。其它大 FoV 序列继续走 ERP/重捕获分支，避免把
eBFoV 的论文结论误当成所有场景的单模型收益。

### SR Ratio：方向正确，数值不能照搬

XMem-360* 的 Table IX（指标为分割的平均球面 J&F）显示：

| SR Ratio | 平均 J&F_sphere | 相对 baseline |
|---:|---:|---:|
| 2.0 | 0.739 | — |
| 2.8 | 0.741 | +0.3% |
| 3.2 | 0.742 | +0.4% |
| 3.6 | 0.747 | +1.1% |

论文还指出默认 2.0 是整体 trade-off 最好的点，变化大多在 ±2% 内。我们的
`search_factor` 是 ERP 像素裁剪边长因子，不是 BFoV mask 框的 SR Ratio；不能把
`3.6` 直接写成 B224 的默认值。应在代表性序列上扫 `3.0/3.5/4.0`，并保留
`2.8/3.6` 作为边界对照。

论文的 `Max Loss=0` 会显著下降（-5.1%），`8/12` 仅小幅高于 `4`，支持我们已经
采用的“短时冻结搜索区 + 后续扩张 + 全局重捕获”滞回状态，而不是每帧立即扩大。

### DTPTrack/TRC：思想可移植，官方代码不是 B224 插件

DTPTrack（CVPR 2026）确实提出 Temporal Reliability Calibrator（TRC）和 Temporal
Guidance Synthesizer（TGS），用可靠性给历史状态加权并合成先验。官方实现的
`DTPTrack.py` 需要多帧 template feature/mask、冻结的 ViT 主干、额外 prior token、
训练好的 MLP/TRC/TGS 和 LoRA causal block；仓库的训练入口是 Linux/NVIDIA 取向，
没有可直接加载到我们 B224 OpenVINO 图的 360 权重。

对本项目最合理的第一步不是硬移植网络，而是移植 TRC 的因果原则：

`anchor + 当前模板 + 最近历史摘要 → 可靠性分数 → 是否写模板/是否冻结/是否重捕获`

这与我们已有的 anchor、scale freeze、NORMAL/SUSPECT/LOST/VERIFY 状态机相容，
可以先用 CPU 训练一个校准器；只有 OOF 显示可靠性 AUROC 和误恢复率达标，才考虑
训练真正的 temporal prior。

### StreamDAM：存在信号值得借鉴，但它解决的是 VOS

StreamDAM 是 2026 年的流式 VOS 预印本。它用一个因果 GRU 读取 logit、mask overlap、
面积比、absence run-length 等运行时标量，输出 presence 概率，并统一控制记忆写入、
回看窗口、输出抑制和重检测。该机制正好解释了我们固定置信度阈值的风险：真正消失时
有用的控制，在“目标仍在但外观变差”时可能反而伤害。

本地可先做等价的轻量版本：用 B224 的质量、响应熵、top1/top2 margin、anchor 相似度、
球面运动残差、log-area 变化和当前失锁长度训练逻辑回归/小 GRU。标签只由 train95
的 OOF GT 产生；valid35 只锁定验证。它不要求把 SAM/XMem 接入主干，成本低且可解释。

### XMem-360、SAMURAI、HiM2SAM：作为第二主线，不应马上替换 B224

XMem-360* 的优势来自“分割 mask + 长期记忆 + 360 训练”，天然覆盖消失/重现和复杂
形变；但原论文的 XMem* 训练约 100K iterations、4×3090 约 3.5 天，输入还调整到
720p/512 patch。未经 360VOS 训练的普通 XMem 不能假定具有同样能力，mask→BFoV
转换也会引入新的评分与速度问题。

SAMURAI 的 Kalman/motion-aware memory selection、HiM2SAM 的层次运动估计和长短记忆
分离，均是 SAM2 mask tracker 的推理机制；它们可以作为我们模板/候选记忆的设计
参考，但不能把 mask memory 的分数直接映射到 BBox B224。

## 3. 结合本地失败矩阵的优先级

1. **先完成当前 GPU full130 续跑**，冻结同一配置的基准，不把论文数字混入总分。
2. **eBFoV-B224 分支**：复用现有 `projection.py` 的球面 remap 与逆投影，为模板和
   搜索图同时提供 eBFoV 采样；状态内部改为 BFoV/角度，ERP 框只在输出时生成。先测
   `real/0015`、`real/0041`、`real/0031` 等大 FoV 序列，并加一个正常负对照。
3. **SR/search factor 小网格**：`2.8/3.0/3.5/3.6/4.0`，每次只改一个因子；大 FoV、
   小目标、正常序列各取样本，要求困难簇平均 +0.05、正常回退 ≤0.01、端到端 FPS≥30。
4. **CPU presence/TRC 校准器**：基于 train95 OOF 训练；四个消费者为模板 admission、
   状态切换、输出抑制和重检测。先规则+校准器 A/B，再决定是否 GRU。
5. **长时记忆候选**：只在 B224 已判 `LOST` 后低频调用 LoRAT/ODTrack 或 XMem；任何
   候选必须经过 appearance、球面运动、连续帧稳定性三重验证，不能把离线 oracle
   变成路由表。
6. **XMem-360 可行性门**：先下载/固定普通 XMem 权重，在 2 条困难+1 条正常序列做
   mask→BFoV 烟雾测试，记录 GPU 显存、编译/推理时间和 mask 空洞；只有速度和初始
   mask 通过，才投入 360VOS 微调或 XMem-360 复刻。

## 4. 实验门槛与预期

| 阶段 | 必须回答的问题 | 通过条件 |
|---|---|---|
| eBFoV 单序列 | 大 FoV 的球面采样是否真的修复框形变 | 目标序列 AUC +0.10，正常回退 ≤0.01，FPS≥30 |
| eBFoV 场景簇 | 改善是否跨 real/sim 泛化 | 6--10 条平均 +0.05、胜率≥60%、无单条回退>0.10 |
| SR 网格 | 搜索区放大是否只是偶然救援 | OOF 多折稳定，且不增加专家调用失控 |
| presence/TRC | 能否区分消失与低质量可见 | 判丢 AUROC≥0.75、误恢复受控、valid35 不回退 |
| XMem 烟雾 | 分割路线是否值得占用工程预算 | 首帧/连续 mask 合理、端到端速度有明确预算 |

任何论文方案都不能直接宣称达到 full130 `AUC>0.8/SR>0.8/FPS>30`；最终仍以本项目
同一评分器、同一 130 条、单 GPU 完整 runner 为准。Docker 只做本地干跑，不执行比赛
仓库 push。

## 5. 主要来源

- [360VOTS arXiv/HTML（eBFoV、XMem-360*、SR Ratio/Max Loss 表）](https://arxiv.org/html/2404.13953)
- [360VOTS 项目主页](https://360vots.hkustvgd.com/360vot)
- [DTPTrack CVPR 2026 官方论文](https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html)
- [DTPTrack 官方代码](https://github.com/NorahGreen/DTPTrack)
- [StreamDAM arXiv](https://arxiv.org/abs/2608.03912)
- [SAMURAI arXiv](https://arxiv.org/abs/2411.11922) / [官方代码](https://github.com/yangchris11/samurai)
- [HiM2SAM arXiv](https://arxiv.org/abs/2507.07603)
