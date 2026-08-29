# GRT-360 360° ERP 单目标跟踪

当前主交付是 v5_final：B224 主干、T224 快路径、ODTrack 几何专家和 v5 ODTrack 稀疏专家组成的因果路由。首选运行后端是 ONNX Runtime CUDA 12.8；PyTorch CUDA 12.8 保留为 B224/T224 参考和回退路径。

## 当前状态

| 项目 | 当前结果 |
|---|---:|
| full130 AUC | 0.7007805295 |
| full130 SR | 0.8535501637 |
| full130 mean e2e FPS | 38.7408915256 |
| full130 weighted e2e FPS | 36.2230629038 |
| valid35 AUC | 0.6944711096 |
| valid35 SR | 0.8410733642 |
| valid35 weighted e2e FPS | 35.2651779622 |
| full130 单条回退超过 0.10 | 0 |
| full130 数据链路异常 | 0 |

full130 已达到当前阶段目标 AUC>0.7、SR>0.8、端到端 FPS>30。valid35 是锁定验证集，AUC 0.6945，不应被写成同样超过 0.7。5090 端到端速度仍需在实际 CUDA 容器中复测；上表是已归档本地参考。

## 先看这些文档

- [最终详细交接文档](docs/GRT360_HANDOFF_FINAL_20260829.md)：版本、指标、资产、复现、构建、故障恢复和继续优化的完整手册。
- [v5 复现说明](docs/REPRODUCE_V5.md)：profile、数据口径和结果解释。
- [CUDA 12.8 构建说明](docs/BUILD_ARENA_CUDA128.md)：ORT 主交付和 PyTorch 参考镜像。
- [继续优化指导](docs/CONTINUE_OPTIMIZATION.md)：失败场景、OOF、晋级门槛和速度预算。
- [版本指标矩阵](docs/VERSION_MATRIX.md)：ODTrack、B224、geometry v1/v4、v5_final 对比。
- [最终资产清单](artifacts_manifest/FINAL_DELIVERY_ASSETS.json)：GitHub 与本地大文件边界。
- [恢复说明](artifacts_manifest/RESTORE_V5.md)：从源码和本地交接包恢复。
- [接口契约](CONTRACTS.md)：模块接口和跨界框约定。

## GitHub 与本地大文件策略

GitHub 仓库：

- 地址：https://github.com/A3435331841/instan
- 主分支：main
- 交付源码标签：delivery-v20260829-r2
- 保存：项目源码、第三方源码、配置、Dockerfile、脚本、文档和小型 JSON/CSV 清单。
- 不保存：数据集、PyTorch checkpoint、ONNX/OpenVINO 图、完整 run trace 和大缓存。
- 不执行比赛仓库 docker push。

本地正式交接根目录：

D:\instan\grt360_deliverables\team_v5_20260829

其中包含：

| 包 | 用途 |
|---|---|
| GRT360_FINAL_ORT_CUDA128 | 主交付：v5 ORT CUDA 图、源码和结果摘要 |
| GRT360_FINAL_TORCH_CUDA128 | PyTorch B224/T224 CUDA 参考和 SUTrack 源码 |
| GRT360_CONTINUE_TRAINING | v5 ep6、LoRA ep5、B/T/OD 权重、上游源码和训练资料 |
| GRT360_HISTORY_ARCHIVE | 完整 checkpoint、运行结果、导出图、服务器源码和 provenance |

四个包都有 asset_manifest.json、SHA256SUMS、README.md 和 PACK_TO_TAR.ps1。大文件通常是指向 grt360_storage 的 NTFS hardlink，因此包的逻辑大小不等于新增物理占用。根目录 delivery_check.json 已完成四包逐文件校验。

## 快速复现

### 1. 克隆固定源码

~~~powershell
git clone https://github.com/A3435331841/instan.git D:\work\pano360
Set-Location D:\work\pano360
git checkout delivery-v20260829-r2
git rev-parse HEAD
~~~

### 2. 校验本地交接包

~~~powershell
Set-Location D:\instan\pano360
python scripts\check_delivery.py --repo D:\instan\pano360 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_TORCH_CUDA128 --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_CONTINUE_TRAINING --package D:\instan\grt360_deliverables\team_v5_20260829\GRT360_HISTORY_ARCHIVE
~~~

期望输出：ok=true；四个包 checksum_failures=[]、secret_findings=[]；Git tracked_over_50MiB=[]。

### 3. ORT CPU 结构冒烟

~~~powershell
$src = 'D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\src'
$models = 'D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128\models'
$out = 'D:\instan\grt360_scratch\readme_ort_cpu_smoke'
$env:PYTHONPATH = $src
python "$src\integrations\final\arena_protocol_v5.py" --dataset D:\instan\grt360_scratch\smoke_dataset --result $out --model-root $models --force-cpu --seqs seq_0001 --max-frames 2 --trace-dir "$out\trace"
Remove-Item Env:PYTHONPATH
~~~

这一步只证明图、协议和输出格式可用；CPU FPS 不是比赛成绩。

## CUDA 12.8 镜像

### ORT 主交付

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 -Backend ort -Context . -Tag grt360-v5-ort:cu128
docker run --rm --gpus device=0 -v D:\instan\grt360_storage\datasets\official_train:/mnt/dataset:ro -v D:\instan\grt360_scratch\arena_v5_result:/mnt/result grt360-v5-ort:cu128
~~~

### PyTorch 参考

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_TORCH_CUDA128
powershell -ExecutionPolicy Bypass -File src\scripts\build_image.ps1 -Backend torch -Context . -Tag grt360-sutrack-torch:cu128
docker run --rm --gpus device=0 -e GRT360_TORCH_PROFILE=b224_erp -v D:\instan\grt360_storage\datasets\official_train:/mnt/dataset:ro -v D:\instan\grt360_scratch\torch_b224_result:/mnt/result grt360-sutrack-torch:cu128
~~~

PyTorch 镜像可以把 GRT360_TORCH_PROFILE 改为 t224_erp。它是原始 SUTrack 三平铺参考，不是 v5 ORT 多专家路由的逐帧等价版本。

### 5090 双后端测速

~~~powershell
Set-Location D:\instan\grt360_deliverables\team_v5_20260829\GRT360_FINAL_ORT_CUDA128
python src\scripts\benchmark_cuda_backends.py --dataset D:\instan\grt360_storage\datasets\official_train --out D:\instan\grt360_scratch\cuda_backend_benchmark_5090 --ort-image grt360-v5-ort:cu128 --torch-image grt360-sutrack-torch:cu128 --gpu 0
~~~

必须使用同一数据、同一序列列表、同一 GPU、单 GPU 串行；最终速度包括解码、预处理、推理和写盘。脚本只记录 docker run，不会 push。

## 当前算法结构

~~~text
ERP 帧 + init.txt
    |
    +-- 首帧 BFoV / 纬度 / FoV / 接缝分析
    |
    +-- B224 几何主路径
    |      +-- fixed / adaptive / no-switch / probe
    |      +-- 已验证窄门才使用 T224
    |
    +-- ODTrack base / v5 tangent 直接专家
    |
    +-- B224 + 稀疏 OD 窄恢复
    |
    +-- BFoV 输出 + quality/status/expert/route trace
~~~

路由只能使用初始化 BFoV 和推理时信号：响应峰值、margin、熵、anchor 相似度、框 IoU/角距离、角速度、尺度历史、接缝距离和状态持续时间。禁止 sequence name、GT 和 offline result lookup。

当前最有效的技术方向是 eBFoV/大视场、极区几何、尺度冻结和窄重捕获。完整场景矩阵和低分序列见最终交接文档；不能把逐序列诊断表转成线上查表。

## 当前已知问题

- valid35 AUC 仍为 0.6945，需要继续观察泛化。
- absent 长时消失场景收益小于 eBFoV/large_fov，固定阈值不是最终解。
- 极区场景聚合 weighted FPS 较低，需在 5090 重新测慢专家预算。
- 本机只暴露 ORT CPU provider，未伪造 CUDA/NPU 结果。
- 本机没有实际构建 Docker；CUDA Dockerfile 和 build context 已结构化检查，由 5090 机器执行。
- LoRAT/UETrack 恢复源码没有独立 license 文件，团队外传播前需要重新核对上游许可。

## 队友继续优化

每个实验必须新建唯一目录并保存：

~~~text
grt360_scratch/experiments/<experiment_id>/
├── experiment.json
├── config.json
├── summary.json
├── summary.csv
├── trace.jsonl
└── failure_notes.md
~~~

固定顺序：

1. 先查解码、预测行数、NaN、越界和评分链路；
2. 再查接缝、纬度、FoV 和投影；
3. 再查尺度跳变、搜索窗截断和模板冻结；
4. 再查置信度、anchor、模型分歧和长时失锁；
5. 最后拆分解码、预处理、推理、专家和写盘延迟；
6. 单序列通过后扩场景簇，场景簇通过后再做 train95 OOF；
7. valid35 锁定后不调阈值，最后才跑 full130。

门槛：

| 阶段 | 通过条件 |
|---|---|
| 单序列 | 困难 AUC +0.10；正常负对照回退 <=0.01；FPS >=30 |
| 场景簇 | 平均 AUC +0.05；胜率 >=60%；>=3 条独占救援；无单条回退 >0.10 |
| train95 OOF | AUC/SR 各 +0.03；多数折稳定；无序列名路由 |
| valid35 | 仅锁定验证，无明显回退 |
| full130 | 通过前面所有门槛后才运行并归档 |

慢专家使用 token bucket；个别序列降速可以接受，但全量 weighted e2e FPS 必须保持 >30。

## 测试和安全

当前仓库已完成：

- 34 个单元测试通过；
- 相关 Python 文件编译通过；
- Git 大文件扫描通过；
- 凭据扫描通过；
- ORT CPU 两帧 smoke 通过；
- 四个本地交接包逐文件 SHA256 通过。

提交前执行：

~~~powershell
python scripts\check_delivery.py --repo D:\instan\pano360
python -m unittest discover -s tests -p "test_*.py"
~~~

不得做：

- 不把数据、权重、checkpoint、ONNX/OpenVINO 图加入 Git；
- 不把训练 checkpoint 当部署权重；
- 不按序列名称或 GT 路由；
- 不覆盖历史实验目录；
- 不删除原始数据或 checkpoint；
- 未经确认不执行任何比赛仓库 docker push。

## 目录索引

~~~text
pano360/
├── panotrack/                 # 基础几何、跟踪器、评测和 pipeline
├── integrations/final/        # v5 ORT 和 PyTorch Arena 入口
├── configs/repro/             # v5、geometry v1/v4、B224 profile
├── docker/final/              # CUDA 12.8 Dockerfile
├── scripts/                   # 评测、导出、交付、校验和测速
├── third_party/               # SUTrack、LoRAT、UETrack 源码快照
├── docs/                      # 交接、复现、构建、研究和历史文档
├── artifacts_manifest/        # 小型资产和恢复清单
├── models/                    # 本地模型目录，Git 忽略
├── data360/                   # 本地数据目录，Git 忽略大内容
└── runs/                      # 本地运行输出，Git 忽略
~~~

旧的初赛、LightFC、Direct ERP 和早期 120 序列说明仍保留在历史文档中；当前交付、指标和恢复操作以本 README、最终交接文档、profile JSON、本地包 manifest 和 v5 结果 JSON 为准。
