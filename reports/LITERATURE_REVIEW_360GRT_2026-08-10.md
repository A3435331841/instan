# GRT-360 近期论文调研与完成度核对（2026-08-10）

## 结论先说

当前项目已经完成了“统一协议下的四架构赛马”，但还没有完成原计划中的完整科研闭环。下一阶段最值得做的不是继续平均已有预测框，而是把近两年的三个方向合并：可靠时序记忆、因果/缓存式高效时序建模、360 几何与接缝感知。

## 近期关键工作

### 1. 360VOT / 360VOTS：问题与数据边界

360VOT 提供 120 条序列、约 11.3 万帧 4K ERP 数据，核心难点是投影畸变、拼接接缝和长时漂移。360VOTS 在此基础上扩展到 290 条 360VOS 序列，并提供普通/旋转框、BFoV/rBFoV 与分割标注。来源：

- https://arxiv.org/abs/2307.14630
- https://360vots.hkustvgd.com/

### 2. DTPTrack（CVPR 2026）：可靠时序记忆

DTPTrack 用 Temporal Reliability Calibrator 给历史状态打可靠性分数，再用 Temporal Guidance Synthesizer 合成紧凑的动态时序先验，目标是防止错误历史帧污染记忆。论文还把模块接入 ODTrack 和 LoRAT，说明它适合作为现有 tracker 的外挂模块，而不是重写整个 backbone。

来源：https://openaccess.thecvf.com/content/CVPR2026/html/Huang_Drift-Resilient_Temporal_Priors_for_Visual_Tracking_CVPR_2026_paper.html

### 3. LoRATv2（NeurIPS 2025）：因果时序 + KV 缓存

LoRATv2 用 frame-wise causal attention 避免多帧全量二次复杂度，并用 KV cache 重用历史 embedding；同时采用 Stream-Specific LoRA Adapters 和渐进式多帧训练。这个方向最直接对应我们的 FPS 目标。

来源：https://papers.nips.cc/paper_files/paper/2025/hash/ad7e42e7b1f638e991d822724969be45-Abstract-Conference.html

### 4. ETCTrack（CVPR 2026）：模板 token 压缩

ETCTrack 学习压缩历史模板 token，再做层次化交互；论文报告模板 token 减少 60%、MAC 减少 21.4%，精度下降约 0.4%。这比简单减少历史帧更可靠，适合放到 UETrack/ODTrack 的历史模板分支。

来源：https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_An_Efficient_Token_Compression_Framework_for_Visual_Object_Tracking_CVPR_2026_paper.pdf

### 5. UETrack（CVPR 2026）：MoE + 目标感知蒸馏

UETrack 已经验证了 Token-Pooling MoE 与 Target-aware Adaptive Distillation 对速度/精度折中的价值。它是我们的快速学生模型基础，但当前 ERP-wrap 结果仍明显低于 ODTrack，说明还需要更强的可靠记忆和 360 几何约束。

来源：https://openaccess.thecvf.com/content/CVPR2026/papers/Kang_UETrack_A_Unified_and_Efficient_Framework_for_Single_Object_Tracking_CVPR_2026_paper.pdf

### 6. SphereUFormer（CVPR 2025）：球面局部注意力

SphereUFormer 证明了 spherical local self-attention 和球面定向模块可以直接处理 360 畸变。它不是现成的目标跟踪器，但其球面局部窗口/位置编码可以借鉴到接缝和极区建模中。

来源：https://openaccess.thecvf.com/content/CVPR2025/html/Benny_SphereUFormer_A_U-Shaped_Transformer_for_Spherical_360_Perception_CVPR_2025_paper.html

### 7. PanoSAM2（2026 预印本）：接缝一致解码 + 长短记忆

PanoSAM2 针对 360VOS 引入 seam-consistent receptive field、畸变引导损失和 long-short memory。虽然任务是分割，不应直接当作 bbox 结果，但“接缝一致 decoder + 长短记忆”的设计可转移到我们的重捕获/模板记忆模块。

来源：https://arxiv.org/abs/2604.07901

## 对 GRT-360 的建议架构

建议命名为 **GRT360-Causal-DTP-ERP**：

1. UETrack/LoRATv2 风格的轻量因果时序主干，历史 token 使用 KV cache；
2. ETCTrack 风格的历史模板 token 压缩；
3. DTPTrack 风格的可靠性校准和动态时序先验；
4. ERP 接缝一致采样、周期 longitude 编码和极区畸变权重；
5. ODTrack 作为训练教师或低频校正教师，不在每帧都运行；
6. 训练目标加入普通 bbox IoU、可靠性、接缝一致和时序稳定损失。

## 当前完成度

### 已完成

- UETrack 基线：120/120；
- UETrack ERP-wrap：120/120；
- LightFC ONNX：120/120；
- ODTrack ERP 三平铺：120/120；
- 统一普通 IoU / 双 IoU / AUC / SR / FPS 评分和归档；
- 融合器后处理 120 条筛选；
- UETrack 置信度触发 ODTrack 的 10 条试跑。

### 尚未完成

- LoRAT 严格 GRT-360 帧级适配与 120 条评分；
- DTPTrack、LoRATv2、ETCTrack 的实际代码接入；
- 360VOT 训练集上的新模型训练/蒸馏；
- 真正端到端的因果 KV-cache + 轻量学生模型 FPS 验证；
- 最终 Docker 离线提交镜像的全链路验收；
- 新架构在未参与调参的验证/测试划分上的严格消融。

因此，**360GRT 的基础赛马已经测完，但“研究创新版”还没有测完**。当前 ODTrack 是精度冠军，UETrack ERP-wrap 是速度/精度折中基线；还不能把已有结果称为最终创新模型。
