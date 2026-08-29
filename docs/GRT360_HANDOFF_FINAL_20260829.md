# GRT-360 最终交接文档（2026-08-29）

> 本文是当前项目从接手、复现、构建到继续优化的完整操作手册。
> 不包含 SSH 密码、token、私钥或比赛仓库地址。

## 0. 一页结论

### 0.1 当前交付

- 主路线：v5_final，B224 主干 + T224 快路径 + ODTrack 几何专家 + v5 ODTrack 稀疏专家。
- 首选后端：ONNX Runtime CUDA 12.8（简称 ORT），承载已锁定的因果几何路由。
- PyTorch CUDA 12.8 是 B224/T224 的参考和保守回退后端，用来检查 CUDA、原始权重和导出图；它不是 ORT v5 多专家路由的逐帧等价实现。
- full130 参考：AUC 0.7007805295，SR 0.8535501637，mean e2e 38.7408915256 FPS，weighted e2e 36.2230629038 FPS。
- valid35 参考：AUC 0.6944711096，SR 0.8410733642，weighted e2e 35.2651779622 FPS。
- full130 相对 B224 基线提高 AUC 0.0457618174、SR 0.0698051246；没有单条回退超过 0.10，也没有数据链路问题。
- full130 已达到当前阶段目标 AUC>0.7、SR>0.8、端到端 FPS>30；valid35 AUC 仍低于 0.7，不能写成所有集合都过线。
- GitHub 只放源码、第三方源码、配置、文档和小型清单；数据、权重、checkpoint、ONNX/OpenVINO 图和完整 run 谱系留在本地包。
- 本次交接没有删除原始资产，没有执行比赛仓库 docker push，也没有公开上传大文件。

### 0.2 接手人的最短路径

1. 克隆仓库，切换到 delivery-v20260829-r2。
2. 取得 D:\instan\grt360_deliverables\team_v5_20260829，先执行包校验。
3. 用 ORT 包跑 smoke_dataset 一条序列、两帧 CPU 冒烟，确认图和协议。
4. 在带 5090 的机器构建 ORT CUDA 镜像，以同一数据、同一 GPU 重测端到端 FPS。
5. 若继续研发，先复现 v5，再按第 11 节的单序列、场景簇、OOF、valid35、full130 逐级实验。

## 1. 版本、来源和责任边界

### 1.1 Git 版本

| 项目 | 值 |
|---|---|
| GitHub | https://github.com/A3435331841/instan |
| 分支 | main |
| 交付标签 | delivery-v20260829-r2 |
| 交付提交 | 1d029fb6ef0cca5ef0b4e448e22e2357a7579b48 |
| v5 full130 实验提交 | a7ac8bc005757283552d6aa925dd881758be4e90 |
| 服务器撤离标签 | grt360-server-exit-20260827 |

a7ac8bc 是 full130 v5 实际运行时的算法提交。之后的交付提交增加了 ORT 输出端口兼容、构建脚本、第三方源码和文档，不改变 v5 路由规则。交付标签是队友复现和恢复时的固定入口；full130 旧结果不会因打包而重新计算。

### 1.2 第三方源码

| 目录 | 上游 | 快照 |
|---|---|---|
| third_party/sutrack/ | chenxin-dlut/SUTrack | d65052d1ba3fcf55010e1fb3665ee6616c139a2c |
| third_party/lorat/ | LitingLin/LoRAT | 服务器撤离时的源码快照，无恢复的 git commit 元数据 |
| third_party/uetrack/ | kangben258/UETrack | fd13b0eaf16d51536008295f3b27807c69eaad50 |

SUTrack 带原始 LICENSE.txt。恢复的 LoRAT/UETrack 副本没有独立 license 文件；继续向团队外传播前必须核对上游许可证和 NOTICE。第三方源码随 GitHub 提交，但权重和数据不提交。

### 1.3 服务器撤离状态

服务器闭环标记：

D:\instan\grt360_storage\experiments\server_exit_20260827\MIGRATION_COMPLETE.json

服务器端没有执行删除。远端文件清单、SHA256、系统信息、pip freeze、源码、checkpoint、运行结果均在服务器撤离归档和本地四个交接包中。服务器不再是唯一数据源，也不应在冻结版本上重新启动训练 supervisor。

## 2. 统一数据和评分口径

### 2.1 数据集

- 官方 full130：train_real 47 条 + train_sim 83 条。
- valid35：锁定验证子集，只用于最后验证，不用于反复调阈值。
- train95：训练、OOF 和门控标定使用；不得把 valid35 混入样本生成。
- 本地官方数据目录：D:\instan\grt360_storage\datasets\official_train。
- D:\instan\初赛数据 是 official_train 的兼容 Junction；不要复制出第二份大数据。

### 2.2 输入和输出

输入：

~~~text
/mnt/dataset/
└── <sequence>/
    ├── video.mp4
    └── init.txt          # clon,clat,fov_h,fov_v，单位为度
~~~

输出：

~~~text
/mnt/result/
└── <sequence>.txt       # 每帧一行 clon,clat,fov_h,fov_v
~~~

入口优先读取 seqlist.txt，否则扫描含 video.mp4 的子目录。每条序列输出行数必须等于解码帧数，首行是初始化 BFoV。路由器不读取 GT，不按序列名查表，也不读取离线预测表。

### 2.3 评分和速度

- 使用 scripts/eval_official.py 与仓库球面/dual-IoU 评分链路。
- OPE 首帧用于初始化，不计入统计。
- SR 是成功率；AUC 是 21 个 IoU 阈值上的 success 曲线均值。
- 速度验收必须包括视频解码、预处理、模型推理和结果写盘。
- 纯模型 FPS 只用于诊断；比赛最终采用端到端 weighted FPS。
- 必须同时保存 mean、weighted、P50、P95 延迟和慢专家调用率。
- GPU 评测必须单 GPU 串行；并发评测 FPS 不能作为最终成绩。

## 3. 指标、证据和解释

### 3.1 版本矩阵

| 版本 | 路线 | 范围 | AUC | SR | weighted e2e FPS | 解释 |
|---|---|---:|---:|---:|---:|---|
| ODTrack preliminary | ERP 三平铺 | full130 | 0.5813 | 0.6853 | 27.4 | 初赛精度方案 |
| SUTrack-B224 | ERP 三平铺单模型 | full130 | 0.6550187120 | 0.7837450391 | 31.6694 | 决赛单模型基线 |
| geometry v1 | B/T + 几何 + 稀疏 OD | full130 | 0.6883183444 | 0.8351255462 | 36.5242 | 第一版完成路由 |
| geometry v4 | B/T + 窄恢复 | full130 | 0.6932785824 | 0.8421712258 | 36.9349 | 历史重建版本 |
| v5_final | B/T + OD base/v5 因果路由 | full130 | 0.7007805295 | 0.8535501637 | 36.2231 | 当前主交付 |
| v5_final | 同上 | valid35 | 0.6944711096 | 0.8410733642 | 35.2652 | 锁定验证 |

### 3.2 v5 与 B224

| 指标 | B224 | v5_final | 变化 |
|---|---:|---:|---:|
| full130 AUC | 0.6550187120 | 0.7007805295 | +0.0457618174 |
| full130 SR | 0.7837450391 | 0.8535501637 | +0.0698051246 |
| full130 weighted e2e FPS | 31.6694 | 36.2231 | +4.5537 |
| 最大单序列 P95 | — | 67.3799 ms | 5090 需复测 |
| valid35 AUC | 0.6588665426 | 0.6944711096 | +0.0356045670 |
| valid35 SR | 0.7873848569 | 0.8410733642 | +0.0536885072 |
| valid35 weighted FPS | — | 35.2652 | >30 |

full130 有 38 条候选胜出、17 条独占救援；valid35 有 10 条胜出、4 条独占救援。v5 summary 的 regressions_over_0_10 为 0，data_issue_count 为 0。min_e2e_fps 17.2960 是个别慢专家序列，不是全量 weighted FPS。

### 3.3 场景聚合

场景标签可重叠，以下条数不能相加为 130：

| 场景 | 条数 | 基线 AUC | v5 AUC | AUC 增益 | 基线 SR | v5 SR | v5 weighted FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 130 | 0.6550 | 0.7008 | +0.0458 | 0.7837 | 0.8536 | 36.2231 |
| absent | 6 | 0.5094 | 0.5378 | +0.0284 | 0.5682 | 0.6075 | 34.4873 |
| eBFoV | 9 | 0.3893 | 0.5381 | +0.1489 | 0.3506 | 0.5487 | 40.1591 |
| large_fov | 24 | 0.5436 | 0.6208 | +0.0772 | 0.6285 | 0.7340 | 37.7256 |
| polar | 22 | 0.6090 | 0.6602 | +0.0513 | 0.7444 | 0.8311 | 27.2624 |
| scale | 71 | 0.6583 | 0.7043 | +0.0460 | 0.7965 | 0.8637 | 34.5028 |
| small | 43 | 0.6922 | 0.7292 | +0.0370 | 0.8367 | 0.8929 | 34.5613 |
| routed_t224 | 2 | 0.7874 | 0.7874 | 0.0000 | 0.9777 | 0.9777 | 60.0146 |
| routed_b224 | 51 | 0.6911 | 0.6913 | +0.0002 | 0.8426 | 0.8428 | 36.1737 |

最明确的收益来自 eBFoV、large_fov、polar 和尺度簇；absent 只有 6 条且收益较小，不能仅靠固定阈值解决长期消失。

### 3.4 专家使用频次

来自 failure_matrix_130.csv 的实际选中方法计数：

| 方法 | 条数 | 作用 |
|---|---:|---|
| sutrack_b224 | 50 | 默认主干 |
| sutrack_b224_fixed | 34 | 几何条件下固定搜索 |
| sutrack_b224_bare_probe | 15 | 低风险早期 probe |
| odtrack_tangent_direct | 6 | 已验证的直接 tangent OD |
| b224_od_recovery | 5 | B224 + 窄重捕获 |
| odtrack_v5_tangent_direct | 3 | v5 训练 OD 专家 |
| sutrack_b224_factor2_probe | 3 | factor-2 试探 |
| sutrack_b224_noswitch 系列 | 5 | no-switch / high-lat |
| sutrack_t224 | 2 | 快路径 |
| 其他窄门 | 7 | scale-freeze、dynamic-polar、eBFoV、ERP 等 |

普通序列仍以 B224 为主。提升并不是简单地把慢专家扩大到所有序列；下一阶段要同时改善 B224 的普遍能力和困难簇尾部。

### 3.5 典型高增益和低分序列

高增益示例：

| 序列 | 原因 | 基线 AUC | v5 AUC | 增益 | 当前方法 |
|---|---|---:|---:|---:|---|
| train_sim/seq_0075 | 高纬中等视场 | 0.1183 | 0.6857 | +0.5674 | OD tangent |
| train_sim/seq_0009 | 紧凑视场 | 0.1797 | 0.7162 | +0.5364 | OD tangent |
| train_real/seq_0037 | 长 eBFoV 漂移 | 0.1184 | 0.6098 | +0.4915 | 窄恢复 |
| train_real/seq_0041 | 接缝/大 FoV | 0.1109 | 0.5683 | +0.4574 | 窄恢复 |
| train_sim/seq_0025 | 高纬中等视场 | 0.2693 | 0.6778 | +0.4085 | 窄恢复 |
| train_sim/seq_0064 | eBFoV | 0.2907 | 0.6760 | +0.3853 | eBFoV |
| train_sim/seq_0024 | 极区小目标 | 0.2721 | 0.6433 | +0.3712 | OD ERP |
| train_sim/seq_0044 | 中大视场 | 0.2805 | 0.6463 | +0.3658 | OD tangent |
| train_sim/seq_0046 | 大尺度/快速运动 | 0.3407 | 0.6840 | +0.3432 | v5 OD |
| train_real/seq_0016 | 高纬尺度 | 0.3307 | 0.6212 | +0.2906 | v5 OD |

仍拖后腿的重点：

| 序列 | 基线 AUC | v5 AUC | SR | 当前路线 | 诊断方向 |
|---|---:|---:|---:|---|---|
| train_sim/seq_0018 | 0.2787 | 0.2787 | 0.2985 | B224 no-switch | 小目标/纹理不足 |
| train_real/seq_0013 | 0.3362 | 0.3362 | 0.3484 | B224 fixed | 普通路径识别失败 |
| train_sim/seq_0082 | 0.3662 | 0.3816 | 0.3915 | dynamic polar | 动态投影不足 |
| train_real/seq_0015 | 0.2187 | 0.3889 | 0.3364 | OD recovery | 大视场/接缝仍失锁 |
| train_real/seq_0033 | 0.2548 | 0.3954 | 0.4184 | constant BFoV | 运动/尺度状态不足 |
| train_real/seq_0042 | 0.2093 | 0.4597 | 0.2756 | hemisphere recovery | 重捕获身份不稳 |
| train_real/seq_0030 | 0.4751 | 0.4751 | 0.2258 | B224 | 相似目标/高置信误跟 |
| train_sim/seq_0045 | 0.4789 | 0.4789 | 0.4934 | B224 | 几何条件未覆盖 |
| train_real/seq_0010 | 0.4955 | 0.4955 | 0.5918 | B224 | 中段漂移/尺度记忆 |
| train_real/seq_0027 | 0.5002 | 0.5002 | 0.6187 | B224 | 快速运动/大目标搜索 |

完整逐序矩阵在继续训练包的 results/failure_matrix_130.csv；它只能离线诊断，不能变成线上查表。

## 4. v5 代码和架构

### 4.1 代码入口

| 功能 | 文件 |
|---|---|
| ORT Arena 入口 | integrations/final/arena_protocol_v5.py |
| PyTorch Arena 参考 | integrations/final/arena_protocol_v5_torch.py |
| ORT/OpenVINO 适配器 | integrations/final/ort_adapter.py |
| 复现 profile | configs/repro/ |
| Docker 构建定义 | docker/final/ |
| profile 命令生成 | scripts/run_profile.py |
| 双后端测速 | scripts/benchmark_cuda_backends.py |
| 本地镜像构建 | scripts/build_image.ps1 |
| 交付包生成 | scripts/prepare_team_delivery.py |
| 交付校验 | scripts/check_delivery.py |
| checkpoint 精简 | scripts/extract_inference_weights.py |

### 4.2 数据流

~~~text
Arena ERP 帧 + init.txt
        |
        v
首帧 BFoV、纬度、FoV、接缝风险
        |
        +-- 普通路径：B224 几何路由
        |     +-- fixed/adaptive/no-switch/probe
        |     +-- 已验证窄门才使用 T224
        |
        +-- 直接专家：ODTrack base 或 v5 OD tangent
        |
        +-- 窄恢复：B224 + 稀疏 OD 重捕获
        |
        v
BFoV、质量、状态、专家名、触发原因和延迟 trace
~~~

ORT 适配器将 ONNX Runtime session 暴露成现有 OpenVINO tracker kernel 所需的 inputs、outputs 和可调用模型对象，并允许输出以端口对象或字符串索引。

### 4.3 路由规则原则

允许使用的信号：

- 初始化 BFoV 的 fov_h、fov_v、lat；
- 主干响应峰值、margin、响应熵、anchor 相似度；
- 主干与专家的框 IoU/角距离；
- 球面角速度、尺度历史、接缝距离、状态持续时间。

已验证代表性规则：

1. 20<=fov_h<30、25<=fov_v<35、|lat|>=80：v5 OD tangent。
2. fov_h<6、fov_v<6、|lat|>=85：v5 OD 极小极区专家。
3. 40<=fov_h<50、80<=fov_v<100、|lat|<45：v5 OD 大尺度/运动专家。
4. 100<=fov_h<=120、150<=fov_v<160、|lat|<=35：窄 eBFoV 重捕获。
5. 70<=fov_h<76、135<=fov_v<145、|lat|<=35：窄接缝/消失恢复。
6. fov_h>=175、fov_v>=175、|lat|<2：半球近赤道窄恢复。
7. 其他情况默认留在 B224 几何路径。

完整规则见本地 route_policy.json。不得加入 sequence_name、ground_truth 或 offline_result_lookup 条件。

## 5. 本地目录和交付包

### 5.1 总布局

~~~text
D:\instan\
├── pano360\                         # GitHub 唯一源码仓库
├── grt360_storage\                  # 数据、权重、服务器撤离归档
│   ├── datasets\official_train\
│   ├── checkpoints\
│   ├── experiments\server_exit_*\
│   ├── upstream_sources\
│   └── manifests\
├── grt360_deliverables\             # 队友正式交付包
│   └── team_v5_20260829\
├── grt360_scratch\                  # smoke、实验、导出图和临时 staging
├── 初赛数据\                         # official_train 兼容 Junction
├── deliverables\ / external\        # 历史兼容 Junction
├── .agents\ .workbuddy\ .zcode\     # 工具状态，保持原位
└── README_LOCAL_LAYOUT.md
~~~

仓库内 artifacts、runs、data360 大内容是本地缓存或 Junction，不进入 Git。用户留下的 data360/official_split/README.md 保持未跟踪，不会被加入源码快照。

### 5.2 四个正式包

根目录：

D:\instan\grt360_deliverables\team_v5_20260829\

| 包 | 逻辑大小 | 文件数 | 用途 |
|---|---:|---:|---|
| GRT360_FINAL_ORT_CUDA128 | 2,339,733,932 bytes | 2,268 | 主交付；七张 ONNX 图、源码、结果摘要 |
| GRT360_FINAL_TORCH_CUDA128 | 2,360,024,359 bytes | 2,444 | PyTorch B/T 参考和 SUTrack 源码 |
| GRT360_CONTINUE_TRAINING | 5,197,843,306 bytes | 4,412 | v5 ep6、LoRA ep5、B/T/OD 权重、上游源码和训练资料 |
| GRT360_HISTORY_ARCHIVE | 44,359,698,428 bytes | 11,508 | 完整 checkpoint、运行结果、图、远端源码和 provenance |

这些是逻辑文件大小。大资产在同一 D 盘时通常是 NTFS hardlink，不是额外的第二份物理数据。真正制作 portable tar 时才会产生第二份空间。

每个包都有 README.md、asset_manifest.json、SHA256SUMS、PACK_TO_TAR.ps1 和 src 源码快照。根目录还有 LOCAL_PACKAGES_INDEX.json、delivery_check.json 和 MIGRATION_COMPLETE.json。

### 5.3 给队友的最小交付

用于构建主镜像：

1. GitHub main 或 delivery-v20260829-r2；
2. GRT360_FINAL_ORT_CUDA128；
3. 官方数据目录；
4. 5090、NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。

用于继续训练：

1. 上述最小交付；
2. GRT360_CONTINUE_TRAINING；
3. 只有追溯历史时才复制 GRT360_HISTORY_ARCHIVE。

不需要把 41GB 可重建微调数据或整个服务器 venv 放进比赛镜像。

## 6. Checkpoint、权重和导出图

### 6.1 训练文件为什么大

v5 ODTrack ep6 训练文件约 1.11GB：

- net 状态约 371MB；
- optimizer 状态约 741MB，包含 Adam/动量等训练历史；
- 另有 scheduler、epoch 和配置元数据。

训练恢复需要 optimizer；部署不需要。训练 checkpoint 不能直接进入 Git 或比赛镜像。

### 6.2 部署资产

| 资产 | 约大小 | 用途 |
|---|---:|---|
| SUTRACK_b224_ep0180.pth.tar | 1.29GB | PyTorch B224 参考 |
| SUTRACK_t224_ep0180.pth.tar | 1.05GB | PyTorch T224 参考 |
| sutrack_b224_frame.onnx | 356MB | ORT B224 |
| sutrack_t224_s224_t112.onnx | 111MB | ORT T224 |
| ODTrack base/v5 ONNX | 各约 374MB | ORT 首帧/稳态专家 |

ONNX 图只保留推理时所需算子和输入输出；它不是把完整训练工程塞进镜像。

### 6.3 导出 net-only

~~~powershell
Set-Location D:\instan\pano360
python scripts\extract_inference_weights.py --input D:\instan\grt360_deliverables\team_v5_20260829\GRT360_CONTINUE_TRAINING\checkpoints\ODTrack_ep0006.pth.tar --output D:\instan\grt360_scratch\ODTrack_spherical_v5_ep0006_inference.pth
~~~

命令写新文件，不修改原 checkpoint。新文件必须重新计算 SHA256，并在配置中明确使用。

## 7. 源码和包复现

### 7.1 克隆固定版本

~~~powershell
git clone https://github.com/A3435331841/instan.git D:\work\pano360
Set-Location D:\work\pano360
git checkout delivery-v20260829-r2
git rev-parse HEAD
~~~

应得到提交 1d029fb6ef0cca5ef0b4e448e22e2357a7579b48。

### 7.2 四包校验

~~~powershell
Set-Location D:\instan\pano360
python scripts\check_delivery.py --repo D:\instan\pano360 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_TORCH_CUDA128 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_CONTINUE_TRAINING --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_HISTORY_ARCHIVE
~~~

期望：ok=true；四个包 checksum_failures=[]、secret_findings=[]；Git tracked_over_50MiB=[]。

### 7.3 ORT CPU 冒烟

~~~powershell
$src = 'D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\src'
$models = 'D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\models'
$out = 'D:\instan\grt360_scratch\handoff_ort_cpu_smoke'
$env:PYTHONPATH = $src
python "$src\integrations\final\arena_protocol_v5.py" --dataset D:\instan\grt360_scratch\smoke_dataset --result $out --model-root $models --force-cpu --seqs seq_0001 --max-frames 2 --trace-dir "$out\trace"
Remove-Item Env:PYTHONPATH
~~~

期望：每帧一行有限 BFoV；没有 FAILED、NaN、无效框或缺少输出。CPU FPS 不作为比赛成绩。

### 7.4 复现 profile

~~~powershell
python scripts\run_profile.py v5_final --dataset D:\instan\grt360_storage\datasets\official_train --result D:\instan\grt360_scratch\reproduce_v5 --model-root D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\models --backend ort_cuda --print-only
~~~

去掉 print-only 才执行；脚本不会下载权重、读取 GT 或覆盖旧结果。

## 8. CUDA 12.8 镜像

### 8.1 ORT 主交付

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 -Backend ort -Context . -Tag grt360-v5-ort:cu128
~~~

默认基础镜像是 nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04。容器默认 GRT360_PROFILE=v5_final、GRT360_MODEL_ROOT=/opt/models。

运行：

~~~powershell
docker run --rm --gpus device=0 -v D:\instan\grt360_storage\datasets\official_train:/mnt/dataset:ro -v D:\instan\grt360_scratch\arena_v5_result:/mnt/result grt360-v5-ort:cu128
~~~

### 8.2 PyTorch CUDA 参考

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_TORCH_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 -Backend torch -Context . -Tag grt360-sutrack-torch:cu128
docker run --rm --gpus device=0 -e GRT360_TORCH_PROFILE=b224_erp -v D:\instan\grt360_storage\datasets\official_train:/mnt/dataset:ro -v D:\instan\grt360_scratch\torch_b224_result:/mnt/result grt360-sutrack-torch:cu128
~~~

GRT360_TORCH_PROFILE 可改为 t224_erp。该镜像用于 CUDA backend sanity check、导出图排错和回退，不用于宣称 v5 指标。

### 8.3 5090 双后端测速

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128
python src\scripts\benchmark_cuda_backends.py --dataset D:\instan\grt360_storage\datasets\official_train --out D:\instan\grt360_scratch\cuda_backend_benchmark_5090 --ort-image grt360-v5-ort:cu128 --torch-image grt360-sutrack-torch:cu128 --gpu 0
~~~

脚本只调用 docker run，记录命令、日志和 benchmark.json；不构建、不 push。两个镜像必须使用同一序列列表、同一 GPU、单 GPU 串行。

## 9. 协议、trace 和异常

### 9.1 Arena 协议

每条序列依次：

1. 解码第一帧；
2. 读取 init.txt；
3. 将 BFoV 转 ERP 框；
4. 初始化 B/T/OD 图和状态；
5. 逐帧输出 BFoV；
6. 写入 result/<sequence>.txt。

无效 init、无法解码、无效框、NaN、模型缺失和专家异常必须显式写 stderr 并返回非零；不能拿旧结果补空行。

### 9.2 trace

开启 trace-dir 后，每条序列写 JSON trace。最少字段：

~~~json
{
  "frame_index": 123,
  "target_bfov": [0.0, 0.0, 20.0, 25.0],
  "quality": 0.83,
  "status": "normal",
  "expert_used": "sutrack_b224",
  "route_reasons": []
}
~~~

后续扩展应加入 response_entropy、anchor_similarity、latency_ms、expert_call_reason 和 fallback_reason。禁止加入 GT 标签或按序列名选择的字段。

### 9.3 状态和记忆

~~~text
NORMAL -> SUSPECT -> LOST -> VERIFY -> NORMAL
~~~

- NORMAL：质量和尺度稳定时才更新动态模板；
- SUSPECT：冻结动态模板和时序记忆；
- LOST：停止错误框更新，低频做宽搜索/球面重捕获；
- VERIFY：同时检查 anchor、运动合理性和连续帧稳定性；
- 第一帧 anchor 永久保留，不能被疑似错误目标覆盖。

## 10. 已知限制

1. valid35 AUC 为 0.6945；full130 过线不代表 valid35 也过线。
2. 5090 端到端 FPS 尚未在本地机器测得；36.22 是历史 weighted e2e 参考。
3. 本机 ORT 只暴露 CPU provider；CPU 冒烟不是 CUDA/NPU 成绩。
4. 本机没有实际构建 Docker；5090 机器按本文构建。
5. PyTorch 镜像不是 v5 等价实现，提交复现必须用 ORT。
6. geometry_v4 是基于历史策略的重建 profile，不是独立不可变源码 tag。
7. failure matrix 只用于研发诊断，不能变成线上查表。
8. 官方数据不在 GitHub；队友必须另外取得并校验数据。
9. LoRAT/UETrack 源码快照没有恢复 license 文件，外部再分发前需复核许可。
10. 慢专家可能导致个别序列 FPS<30，但全量 weighted e2e 应继续保持>30。

## 11. 队友继续优化路线

### 11.1 每轮必须落盘

~~~text
grt360_scratch/experiments/<experiment_id>/
├── experiment.json
├── config.json
├── summary.json
├── summary.csv
├── trace.jsonl
└── failure_notes.md
~~~

experiment.json 至少写入源码 commit、profile、数据版本、权重 SHA256、设备、CUDA/ORT/Torch 版本、父实验、开始/结束时间和是否通过晋级门槛。禁止覆盖 v5 目录。

### 11.2 固定诊断顺序

1. 数据链路：解码帧数、GT 行数、预测行数、NaN、越界和评分一致性；
2. 几何：纬度、接缝距离、FoV、球面/ERP 投影；
3. 尺度：宽高跳变、log-area 方差、搜索窗截断；
4. 置信度：峰值、margin、熵、anchor、主干/专家分歧；
5. 失锁：开始帧、最长失锁段、重捕获是否真的换回目标；
6. 速度：解码、预处理、推理、专家、写盘延迟拆分。

先排除链路和几何错误，再讨论训练或融合。

### 11.3 几何专项

- 极区：对 |lat|>60 逐帧动态判断，超过 ±75 度优先切平面，结果映回球面；复测 seq_0016、seq_0082 和正常负对照。
- eBFoV：FoV>90 度使用球面边界或切平面 remap，不只增加搜索因子；重点复测 seq_0015、seq_0041、seq_0042、seq_0037。
- 接缝：内部状态使用球面中心和角 FOV；接缝附近采用圆周索引双副本和 dual-IoU。
- 小目标：原始分辨率裁剪、连续特征聚合、压缩/模糊/下采样增强；重点 seq_0018、seq_0011、seq_0044。
- 大目标：紧致外观 ROI + 宽范围边界 ROI，防止目标占满搜索图。
- 快速运动/尺度突变：球面角速度和 log-FOV 滤波，三个预测中心候选，尺度独立滤波。
- 消失重现：冻结模板，球面粗到细重捕获，anchor + 运动 + 连续帧验证。

### 11.4 轻量学习专项

本地资源有限时只训练 CPU 可执行的质量校准器、存在判别器、专家门控器或小型 adapter。train95 生成序列级 OOF 标签：

- main_failure_next_15_frames；
- expert_advantage_over_main > 0.05；
- false_recovery。

门控输入只能是推理时特征。只有 OOF 失锁 AUROC>=0.75、专家选择精度>=0.70，才替换解释性规则。

### 11.5 晋级门槛

| 层级 | 要求 |
|---|---|
| 单序列 | 困难 AUC +0.10；正常负对照回退 <=0.01；端到端 FPS >=30；无新异常 |
| 场景簇 6-10 条 | 平均 AUC +0.05；胜率 >=60%；>=3 条独占救援；无单条回退 >0.10 |
| train95 OOF 5 折 | 总体 AUC/SR 各 +0.03；多数折同方向；无序列名路由 |
| valid35 | 锁定后只验证，不反复调阈值；无明显回退 |
| full130 | 只对通过上面门槛的候选运行并完整归档 |

连续三次同类数据/状态机错误，标记路线 blocked，保存现场后换路线。

### 11.6 速度预算

- 33.3ms/frame 是 30 FPS 总预算；
- 主干目标 <=25ms，专家预留约 8ms；
- 慢 OD 和 LoRAT 使用 token bucket；
- 5090 测量单 GPU 串行端到端；
- 同时保存 mean、weighted、P50、P95、专家调用率；
- 任意速度优化不能牺牲输出行数和协议完整性。

## 12. Git、数据和安全纪律

### 12.1 GitHub

- 普通 Git 不提交数据集、checkpoint、ONNX/OpenVINO 图和大 run。
- 提交前执行 python scripts/check_delivery.py --repo D:\instan\pano360。
- 最大 Git 跟踪文件必须小于 50MiB。
- 不提交 .env、密码、token、私钥、SSH 配置。
- 修改入口、profile 或协议时同步更新文档和清单。
- 大文件只通过本地包或经批准的内部存储传递。

### 12.2 数据

- 官方原始数据保留单一副本；
- train95 用于训练和 OOF；
- valid35 锁定；
- full130 只做候选最终评测；
- failure matrix 不得进入线上路由。

### 12.3 不可逆动作

以下动作需要另外确认：

- 删除重复权重、checkpoint 或历史 run；
- 覆盖正式结果；
- 向比赛仓库执行 docker push；
- 把本地大包上传公开平台；
- 移动 .agents、.workbuddy、.zcode 工具状态。

## 13. 故障恢复

### 13.1 包校验失败

1. 停止构建和评测，不覆盖旧包；
2. 查看 asset_manifest.json 的 source、大小和 SHA256；
3. 比较 grt360_storage 原文件；
4. 如果 hardlink 源文件变化，生成新的交付目录；
5. 记录新包、原因和新校验结果。

### 13.2 ORT CUDA provider 不可用

容器中检查：

~~~bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
nvidia-smi
~~~

没有 CUDAExecutionProvider 时，只能用 force-cpu 做结构 smoke，不能报告 FPS。检查 onnxruntime-gpu、CUDA 12.8/cuDNN、NVIDIA Container Toolkit 和 docker run 的 gpus 参数。

### 13.3 输出行数错误

- 检查 video.mp4 是否完整解码；
- 检查 init.txt 是否恰好四个有限值；
- 查看 stderr 的失败序列；
- 不用上一轮 result 补行；
- 修复后写入新的结果目录并保留 trace。

### 13.4 训练恢复

训练恢复必须保持 optimizer、scheduler、epoch 和 profile 一致；部署只使用 net-only 或 ONNX。恢复训练前复制 experiment.json，不能直接修改历史实验目录。

## 14. 最终验收清单

### 源码和资产

- [ ] GitHub main 与 delivery-v20260829-r2 可访问；
- [ ] third_party 源码和 provenance 在仓库；
- [ ] Git 无 >50MiB 文件和凭据；
- [ ] 四个包 SHA256 通过；
- [ ] 服务器撤离目录未删除；
- [ ] MIGRATION_COMPLETE.json complete=true；
- [ ] delivery_check.json ok=true。

### 运行和协议

- [ ] ORT CPU 两帧 smoke 输出合法 BFoV；
- [ ] ORT CUDA 镜像在 5090 启动；
- [ ] 130 条序列结果齐全，输出行数匹配；
- [ ] 无 NaN、无越界、无静默失败；
- [ ] trace 能解释路由和失败原因；
- [ ] 5090 单 GPU weighted e2e FPS >30；
- [ ] 使用固定数据版本和同一评分器。

### 算法继续研发

- [ ] 新实验有唯一目录和完整 manifest；
- [ ] 先做数据/几何/尺度/置信度/失锁诊断；
- [ ] 单序列通过后才扩场景簇；
- [ ] 场景簇通过后才做 OOF；
- [ ] valid35 不反复调阈值；
- [ ] full130 连同速度、回退和 trace 归档；
- [ ] 阶段性成绩不冒充最终完成。

## 15. 关键文件索引

### GitHub 仓库

~~~text
README.md
CONTRACTS.md
configs/repro/v5_final.json
integrations/final/arena_protocol_v5.py
integrations/final/arena_protocol_v5_torch.py
integrations/final/ort_adapter.py
scripts/run_profile.py
scripts/benchmark_cuda_backends.py
scripts/build_image.ps1
scripts/check_delivery.py
scripts/prepare_team_delivery.py
scripts/extract_inference_weights.py
docker/final/Dockerfile.ort-cu128
docker/final/Dockerfile.torch-cu128
docs/REPRODUCE_V5.md
docs/BUILD_ARENA_CUDA128.md
docs/CONTINUE_OPTIMIZATION.md
docs/VERSION_MATRIX.md
docs/GRT360_HANDOFF_FINAL_20260829.md
artifacts_manifest/FINAL_DELIVERY_ASSETS.json
artifacts_manifest/RESTORE_V5.md
third_party/README.md
~~~

### 本地结果和迁移证据

~~~text
D:\instan\grt360_scratch\geometry_recovery_v5_full130_20260829\summary.json
D:\instan\grt360_scratch\geometry_recovery_v5_full130_20260829\summary.csv
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\full130_summary.json
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\valid35_summary.json
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\failure_matrix_130.csv
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\scenario_summary.csv
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\route_policy.json
D:\instan\grt360_scratch\geometry_recovery_v5_artifacts_20260829\latency_summary.json
D:\instan\grt360_deliverables\team_v5_20260829\LOCAL_PACKAGES_INDEX.json
D:\instan\grt360_deliverables\team_v5_20260829\delivery_check.json
D:\instan\grt360_deliverables\team_v5_20260829\MIGRATION_COMPLETE.json
~~~

如果本文与旧交接文档冲突，以当前 Git 标签、交付包 manifest 和上面列出的 v5 JSON 为准；旧文档仅保留历史背景，不用于重新选择提交方案。

