# panotrack —— 影石全景视频智能跟踪赛道 · 360° ERP 实时单目标跟踪原型

> **交付状态（2026-08-29）**：源码已冻结在 `delivery-v20260829-r2`，当前主交付为 **v5_final ORT CUDA** 几何路由。
> full130 AUC 0.7008 / SR 0.8536 / weighted e2e 36.22 FPS；valid35 AUC 0.6945 / SR 0.8411。
> GitHub 只保存源码、配置、文档和小型清单；权重、图和历史结果位于本地 `grt360_deliverables\team_v5_20260829` 交接包。
> 不执行未经确认的比赛仓库 docker push。完整构建与恢复步骤见 `docs/REPRODUCE_V5.md`、`docs/BUILD_ARENA_CUDA128.md`。
> LightFC 曾为推荐方案（代表序列 AUC 0.618），全量评测后仅 0.31，已降级为历史方案。
> 详见 `docs/ARENA_PROTOCOL_TEST_ZH.md` 与 `deliverables/SUBMISSION_2026-08-10/05_官方联系/ARENA平台提交指南_2026-08-14.md`。

武汉大学 4 人学生队参赛作品。面向 360° ERP（等距柱状投影）全景视频的实时单目标跟踪：
**球面状态层预测 → tangent 局部切图 → 轻量跟踪器 → 置信度判丢 → 逐级扩大 FoV 重试 → 全局重检测**，
最终以 Docker 镜像形式断网自包含提交。

运行环境：Python 3.12，轻量线生产仅依赖 **numpy / Pillow / scipy + onnxruntime**（禁止 cv2、torch、yaml、pytest；torch 仅本地 lightfc_cpu 验证用）。
ODTrack/UETrack 提交镜像走独立依赖面（torch/cv2/timm，见 `integrations/` 与 `docs/FINAL_DELIVERY_ZH.md`）。

---

## 项目进展

| 阶段 | 状态 | 核心成果 |
|------|------|----------|
| Stage 1 | ✅ 完成 | BFoV 几何模块、NCC 跟踪器、合成数据生成、基础 pipeline |
| Stage 2 | ✅ 完成 | 集成 VitTrack、全局重检测 v2、自适应 patch_size、状态阻尼 |
| Stage 2 验证 | ✅ 完成 | 发现 BFoV 框架漂移问题；Direct ERP 方案 AUC 0.26 / FPS 155 |
| Stage 3 | ✅ 完成 | 120 序列全量赛马（ODTrack 0.5792 精度冠军）、初赛提交（ODTrack 精度版镜像）、决赛重捕获方案代码就绪 |

**关键发现**：BFoV 框架的恒定角速度状态预测会累积误差，到第 7-8 帧时完全漂移。
Direct ERP 方案（绕过 BFoV）在精度和速度上均显著优于传统框架；
进一步接入 LightFC 后，3 个代表序列（0008/0036/0116，各 150 帧）全帧平均
**AUC 0.618 / SR@0.5 0.749**（先进水平），CPU 实时约 10 FPS
（纯模型推理 38 FPS，评测瓶颈在 4K JPEG 解码）。

**Stage 3 全量赛马（2026-08-09）**：120 序列严格评测后排名为
ODTrack ERP 三平铺（AUC 0.5792 / 8.99 FPS）> UETrack ERP-wrap
（0.5143 / 57.16 FPS）> UETrack 基线（0.4168）> LightFC ONNX（0.3116）。
初赛按精度排名，最终提交 ODTrack 精度版（详见 `reports/STAGE_RESULTS_2026-08-09.md`）。
教训：LightFC 代表序列 0.618 看着很美，全量一跑只有 0.31——小样本数字
真的不能当最终成绩，这也是后来所有实验都坚持跑全量的原因。

---

## 队友快速上手指南

### 环境准备

1. **克隆仓库**
   ```bash
   git clone <repo_url> && cd pano360
   ```

2. **安装依赖**（Python 3.12）
   ```bash
   pip install -r requirements.txt
   ```

3. **下载模型**（约 100MB）
   - 从 [VitTrack 官方仓库](https://github.com/VDIGPKU/VitTrack) 下载 `object_tracking_vittrack_2023sep.onnx`
   - 放到 `models/` 目录（该目录已加入 `.gitignore`，不会入库）

4. **下载代表序列**（约 4GB，用于本地验证）
   ```bash
   python scripts/download_360vot.py --seqs 001,004,036,046,054,058 --extract
   ```

### 验证环境

```bash
# 运行全部测试
python tests/test_geometry.py
python tests/test_trackers.py
python tests/test_metrics.py
python tests/test_synth.py
python tests/test_pipeline.py

# 跑端到端 demo（合成数据）
python demo/run_demo.py
```

### 当前优先级

| 优先级 | 任务 | 负责人 |
|--------|------|--------|
| P0 | 对接官方 120 序列评测协议 | 待定 |
| P0 | Docker 离线部署验证 | 待定 |
| P1 | 小目标增强（多尺度/检测器 fallback） | 待定 |
| P1 | LightFC 接入评估 | 待定 |

### 模块分工

| 模块 | 路径 | 负责人 | 状态 |
|------|------|--------|------|
| 球面几何 / BFoV | `panotrack/geometry/` | 几何工程师 | ✅ 完成 |
| 局部跟踪器 | `panotrack/trackers/` | 跟踪器工程师 | ✅ 完成 |
| 评测 / 数据 | `panotrack/evaluation/` + `panotrack/data/` | 评测工程师 | ✅ 完成 |
| 集成 pipeline | `panotrack/pipeline/` + `panotrack/io/` | 集成工程师 | ✅ 完成 |
| Docker / CLI | `docker/` + `panotrack/cli.py` | 工程化工程师 | ✅ 完成 |

### 重要约定

- **接口契约**：所有模块必须遵守 `CONTRACTS.md`，不要跨模块修改接口
- **注释规范**：中文 docstring + 英文标识符
- **提交前检查**：`python tests/test_*.py` 全绿，`runs/` 不提交
- **大文件**：ONNX 模型、数据集不进 Git，单独共享；`models/` 已由 `.gitignore` 忽略、
  当前未入库，ONNX 模型需自行准备或团队共享

---

## 目录结构

```
pano360/
├── CONTRACTS.md              # 全项目接口契约（所有模块必须逐字遵守）
├── README.md                 # 本文档
├── requirements.txt          # 运行依赖（numpy / Pillow / scipy 最低版本）
├── .dockerignore             # Docker 构建上下文裁剪
├── configs/
│   └── default.json          # PanoTracker 默认配置（键即 PanoTracker config 键）
├── docker/
│   ├── Dockerfile            # 赛事提交镜像（python:3.12-slim，断网自包含）
│   └── entrypoint.sh         # 容器辅助入口（file / trax 两种模式，联调用）
├── panotrack/
│   ├── geometry/             # 模块 A：球面几何 / BFoV / tangent 投影
│   ├── trackers/             # 模块 B：BaseTracker / NCC / 工厂（LightFC 预留）
│   ├── evaluation/           # 模块 C：IoU / dual IoU / SR / AUC / OPE runner
│   ├── data/                 # 模块 C：合成数据生成 / 序列读取 / 可视化 / 360VOT 加载
│   ├── pipeline/             # 模块 E：PanoTracker / 球面状态 / 重检测（集成阶段）
│   ├── io/                   # 模块 D：官方评测 I/O 适配层
│   │   ├── file_protocol.py  #   图像序列文件协议
│   │   └── trax_protocol.py  #   trax 风格行协议（占位，官方协议公布后替换）
│   └── cli.py                # 模块 D：命令行入口 python -m panotrack.cli
├── tests/                    # 纯 assert 测试脚本（python tests/test_xxx.py）
├── demo/
│   └── run_demo.py           # 端到端 demo（集成阶段提供）
├── scripts/                  # 360VOT 数据下载与本地批量评测
│   ├── download_360vot.py    #   HF gated 仓库下载（HF_TOKEN，默认 hf-mirror 镜像）
│   └── eval_360vot.py        #   批量 OPE 评测（普通/dual 双口径，summary.csv）
├── data360/                  # 360VOT 本地数据目录（下载脚本产出，勿入库）
└── runs/                     # 运行输出（results.txt / metrics.json / demo.gif）
```

---

## 快速开始

### 1. 克隆仓库与安装依赖

```bash
git clone <repo_url> && cd pano360
pip install -r requirements.txt
```

### 2. 下载代表序列（用于本地验证）

```bash
# 下载 6 个代表序列（共约 4GB）到 data360/
python scripts/download_360vot.py --seqs 001,004,036,046,054,058 --extract
```

> 完整 120 序列测试集（58GB）需先从 [官方主页](https://360vots.hkustvgd.com/) 申请下载权限，解压到 `data360/official/` 后再评测。

### 3. 生成合成数据（模块 C）

```bash
python -c "from panotrack.data.synth import generate_sequence; \
generate_sequence('runs/equator', n_frames=60, scenario='equator', seed=0)"
```

可选场景：`equator`（赤道正常运动）、`crossing`（跨越右边界回绕）、
`pole`（极区拉伸）、`occlusion`（中段遮挡）。
输出：`runs/<scenario>/frames/%06d.png` 与 `runs/<scenario>/gt.txt`
（每行 `x,y,w,h`，跨界约定见契约）。

### 3. 跑端到端 demo（集成阶段提供）

```bash
python demo/run_demo.py
```

在 4 个合成场景上端到端跑通，结果输出到 `runs/<scenario>/`：
`results.txt`、`metrics.json`、`demo.gif`。
验收指标：equator / crossing 的 SR@0.5 ≥ 0.9（双口径），pole ≥ 0.7，
occlusion 丢失后 10 帧内找回。

### 4. 文件协议 CLI（模块 D，对标官方评测形态）

```bash
python -m panotrack.cli \
  --frames runs/equator/frames \
  --init   runs/equator/gt.txt \
  --out    runs/equator/results.txt \
  --config configs/default.json \
  --visualize runs/equator/vis
```

- `--init` 文件首行 `x,y,w,h`（gt.txt 首行即首帧标注，可直接复用）。
- 结果逐帧追加写入 `--out`：首行为初始化框，之后每帧一行 `x,y,w,h`
  （保留 2 位小数），与输入帧逐一对齐。
- 统计与调试日志输出到 **stderr**，stdout 保持干净（协议卫生）。
- `--visualize` 可选，把跟踪框画到各帧保存（跨界框自动拆两段绘制）。

### 5. 离线评测（模块 C）

```python
import numpy as np
from panotrack.evaluation.metrics import ope_evaluate
from panotrack.data.io import load_sequence

frames, gt = load_sequence('runs/equator')
pred = np.loadtxt('runs/equator/results.txt', delimiter=',')
H, W = frames[0].shape[:2]
print(ope_evaluate(pred, gt, W))
# {'sr', 'auc', 'sr_dual', 'auc_dual', 'ious', 'ious_dual'}
```

### 6. 运行测试

```bash
python tests/test_cli.py          # 模块 D（本目录）
python tests/test_geometry.py     # 模块 A（由对应工程师提供）
python tests/test_trackers.py     # 模块 B
python tests/test_metrics.py      # 模块 C
python tests/test_synth.py        # 模块 C
python tests/test_pipeline.py     # 模块 E
python tests/test_vot360.py       # 360VOT 加载器与批量评测脚本
```

---

## 评测指标说明

| 指标 | 含义 |
| --- | --- |
| IoU | 预测框与 GT 的交并比（不跨界口径，`iou_xywh`） |
| dual IoU | 360VOT 口径：预测框水平平移 ±W 后与 GT 的 IoU 取最大，处理跨界回绕（`dual_iou`） |
| SR@0.5 | Success Rate：IoU ≥ 0.5 的帧占比（`success_rate`） |
| AUC | 阈值 0~1 步长 0.05（21 点）的 SR 均值（`auc`） |
| OPE | One-Pass Evaluation：首帧用 GT 初始化，逐帧跟踪，**首帧不计入统计**（`ope_evaluate`） |

`ope_evaluate(pred, gt, width)` 返回普通与 dual 两套口径的 SR / AUC，
跨界场景以 **双口径（dual）为准**。

---

## 360VOT 本地评测

测试集（120 序列，4K ERP 3840×1920，标注 `BBox=[x1 y1 w h]` 逐帧 txt）
托管在 HuggingFace gated 仓库
[xuyzshaun/360VOTS](https://huggingface.co/datasets/xuyzshaun/360VOTS)
（`360VOT-test/001.zip` 每序列一个 zip）。**权限申请通过后，
一条命令下载、一条命令评测即可跑通。**

### 1. 申请访问权限（一次性）

1. 注册/登录 HuggingFace：https://huggingface.co/join
2. 打开仓库页 https://huggingface.co/datasets/xuyzshaun/360VOTS ，
   点击 **Request access** 提交申请，等待通过（邮件/页面确认）。
3. 创建 Read 权限令牌：https://huggingface.co/settings/tokens
4. 把令牌写入环境变量（**禁止写进代码**）：
   - Windows 当前会话：`set HF_TOKEN=hf_xxxxxxxx`
   - Windows 永久：`setx HF_TOKEN hf_xxxxxxxx`（重开终端生效）
   - Linux/macOS：`export HF_TOKEN=hf_xxxxxxxx`
5. 安装下载依赖：`pip install huggingface_hub`
   （未安装时脚本会提示并以退出码 2 退出；该依赖仅供下载脚本使用，
   跟踪/评测代码仍只依赖 numpy/Pillow/scipy）

下载脚本默认使用国内镜像 `HF_ENDPOINT=https://hf-mirror.com`
（已自行设置该环境变量时尊重原值）。

### 2. 下载并解压序列

```bash
# 默认下载 5 个代表序列（001~005）到 data360/，解压并删除 zip
python scripts/download_360vot.py --extract

# 指定序列与输出目录
python scripts/download_360vot.py --seqs 001,002,003 --out data360 --extract
```

zip 解压后若多套一层目录（如 `data360/360VOT-test/001/...`）无需手工整理，
`panotrack.data.vot360.find_sequences` 会递归发现序列目录。

### 3. 批量评测

```bash
# 全部序列：0.5 倍缩放提速（帧与 GT 同步缩放，IoU 尺度不变）
python scripts/eval_360vot.py --data data360 --seqs all --downscale 0.5 --out runs/360vot

# 调试：指定序列 + 截断帧数
python scripts/eval_360vot.py --seqs 001,003 --max-frames 100
```

注意：4K 原始分辨率（--downscale 1.0）内存与耗时都很大，日常使用建议 0.5。

### 4. 结果解读

`--out` 目录下的产物：

| 文件 | 内容 |
| --- | --- |
| `<seq>/results.txt` | 逐帧跟踪框 `x,y,w,h`（2 位小数，首帧为初始化 GT 框） |
| `<seq>/metrics.json` | 单序列指标：`sr` / `sr_dual` / `auc` / `auc_dual` / `fps` 与丢失/找回统计 |
| `summary.csv` | 全部序列汇总（sequence,n_frames,sr,sr_dual,auc,auc_dual,fps + MEAN 行） |

终端 stdout 同步打印汇总表（含 MEAN 行），过程日志一律走 stderr。
跨界场景以 **dual 口径（SR_dual / AUC_dual）为准**（360VOT 官方口径：
预测框水平平移 ±W 后与 GT 的 IoU 取最大）。

---

## Docker 构建与运行

### 三份离线交付镜像（2026-08-10）

| 镜像 | 内容 | 定位 |
| --- | --- | --- |
| `grt360-final:2026-08-10` | panotrack + LightFC/VitTrack ONNX（CPU，约 675 MB） | 轻量离线协议验收 |
| `grt360-uetrack:2026-08-09` | UETrack ERP-wrap GPU 镜像（约 16.8 GB） | 高速版提交（AUC 0.5143 / 57.16 FPS） |
| `grt360-odtrack:2026-08-10` | ODTrack ERP 三平铺 GPU 镜像（约 13.4 GB） | **精度版提交（AUC 0.5792 / SR 0.6532 / 8.99 FPS）** |

ODTrack 精度版镜像入口 `integrations/odtrack/file_protocol.py`（与 UETrack 镜像
同一文件协议：`--frames` 帧目录 + `--init` 初始框 -> `--out` results.txt），
构建脚本 `docker/odtrack/build_odtrack_image.sh`（从 `artifacts/server_snapshot/`
组装构建上下文）。构建、验证与运行细节见 `docs/FINAL_DELIVERY_ZH.md` 与
`docs/DOCKER_TEST_ZH.md`。

镜像基于 `python:3.12-slim`，**断网自包含**：构建期一次性安装依赖并
COPY 源码，运行期零网络访问。在仓库根目录执行：

```bash
# 构建（赛事要求 linux/amd64）
docker build --platform linux/amd64 -f docker/Dockerfile -t panotrack:latest .

# 运行：文件协议（挂载数据目录）
docker run --rm --platform linux/amd64 \
  -v /path/to/data:/data panotrack:latest \
  --frames /data/frames --init /data/init.txt --out /data/results.txt \
  --config /app/configs/default.json

# 用 LightFC(推荐,先进水平)评测:
#   configs/lightfc.json -> tracker=lightfc_onnx(双子图 ONNX,CPU 实时)
docker run --rm --platform linux/amd64 \
  -v /path/to/data:/data panotrack:latest \
  --frames /data/frames --init /data/init.txt --out /data/results.txt \
  --config /app/configs/lightfc.json
```

镜像正式 `ENTRYPOINT` 为 `python -m panotrack.cli`。辅助脚本
`/entrypoint.sh` 提供两种联调模式（覆盖 entrypoint 使用）：

```bash
# 文件协议（等价默认入口）
docker run --rm --entrypoint /entrypoint.sh panotrack:latest file \
  --frames /data/frames --init /data/init.txt --out /data/results.txt

# trax 风格行协议（占位实现，stdin/stdout 联调）
docker run --rm -i --entrypoint /entrypoint.sh panotrack:latest trax < cmds.txt
```

---

## 接口契约摘要（详见 CONTRACTS.md）

- **图像**：`np.ndarray`，uint8，`(H, W, 3)`，RGB；ERP 全景图 W = 2×H。
- **ERP 框**：`(x, y, w, h)` 浮点像素；跨界约定 `x ∈ [0, W)`，
  跨右边界时 `x + w > W`（超出部分回绕到左边缘，模 W）；y 方向不回绕。
- **角度**：一律为度。经度 lon ∈ (-180, 180]，纬度 lat ∈ [-90, 90]，lat=+90 为北极。
- **PanoTracker**（模块 E，集成阶段实现，接口已冻结）：
  `__init__(config: dict | None)`；`init(frame, bbox)`；
  `update(frame) -> {'bbox', 'score', 'status': 'ok'|'lost'|'recovered', 'fov'}`。
  缺省配置即 `configs/default.json`。
- **BaseTracker**（局部跟踪器接口）：`init(image, bbox)` /
  `update(image) -> {'bbox', 'score', 'psr', 'apce'}`，输入为局部透视图。

---

## LightFC 接入指南（历史方案：代表序列先进水平，全量评测后已降级）

LightFC 已作为**全帧跟踪器**接入（同 Direct ERP 思路，不走 BFoV 切图），
代表序列全帧评测平均 **AUC 0.618 / SR@0.5 0.749 / CPU 约 10 FPS**（先进水平）。

### 生产部署路径（onnxruntime，推荐）

1. **模型文件**（不入 Git，需自行准备，放 `models/`）：
   - `models/lightfc_backbone.onnx`（模板特征子图，2.2MB）
   - `models/lightfc_tracking.onnx`（跟踪子图，12.7MB）
   - 生成方式：本地有 torch 时运行 `tools_local/export_lightfc_onnx.py`
     （需先按下方"本地验证路径"获取权重）；或从队友处拷贝两个 ONNX。
2. **配置**：`configs/lightfc.json`（`tracker: "lightfc_onnx"` + 两个模型路径）。
3. **一键评测**：`python scripts/eval_360vot.py --config configs/lightfc.json
   --data data360 --seqs 0036,0116 --downscale 0.5`
   （`eval_360vot.py` 对 lightfc/direct_erp 类全帧跟踪器走直连 OPE 路径，
   不再经 BFoV 切图与全局重检测）。
4. **CLI / Docker**：`python -m panotrack.cli --config configs/lightfc.json`；
   Dockerfile 已 COPY `models/`，镜像断网自包含。

### 本地验证路径（torch CPU）

1. 克隆 LightFC 官方仓库到 `tools_local/lightfc`（已 .gitignore，不入库）：
   `git clone --depth 1 https://github.com/LiYunfengLYF/LightFC.git tools_local/lightfc`
2. 下载预训练权重 `outputs_lightfc.zip`（Google Drive 链接见官方 README），
   解压出 `lightfc_ep0400.pth.tar` 放到 `output/checkpoints/.../`。
3. 依赖：`pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu`
   + `torchvision==0.17.2` + `numpy<2` + `timm`（lightfc_cpu 路径用）。
4. 使用 `panotrack/trackers/lightfc_cpu.py`（`tracker: "lightfc_cpu"`）。

### 说明

- 两个封装均对齐 `BaseTracker` 契约（init/update + 状态代理字段），
  输入为 **ERP 全帧**，搜索区裁剪水平回绕处理 360° 跨界。
- 全帧跟踪器在 `eval_360vot.py` 中走 `_eval_full_frame` 直连路径；
  若接入 PanoTracker（BFoV 框架）则语义不匹配，不建议混用。

---

## 直接 ERP 跟踪方案（历史 / 本地验证方案）

在 Stage 2 验证中发现，BFoV 框架的状态预测漂移是比小目标稀释更严重的瓶颈，
因此曾提出 **Direct ERP Tracker**：直接在全帧 ERP 上运行 VitTrack，绕过 BFoV 切图。

> **当前状态**：此为历史 / 本地验证方案，Stage 3 主推方案已切换为 **LightFC**（见上方
> 「LightFC 接入指南」）。`direct_erp` 与新 `vittrack_onnx` 已统一为 real onnxruntime
> 推理（复用 VitTrackONNX，无 cv2 依赖）；旧 `cv2.TrackerVit` 仅作为本地验证 fallback，
> 生产路径不依赖 cv2。

### 性能对比

| 对比项 | BFoV + VitTrack | Direct ERP + VitTrack |
|--------|-----------------|----------------------|
| Mean AUC | 0.25 | **0.26** |
| FPS | ~8 | **155** |
| 漂移问题 | 严重 | 无 |
| 实现复杂度 | 高 | 低 |

### 为什么 Direct ERP 更优？

1. **消除漂移**：BFoV 框架的恒定角速度模型会累积误差，到第 7-8 帧时预测位置严重偏离，导致完全丢失。Direct ERP 直接在全帧上跟踪，无状态预测漂移。
2. **保持完整上下文**：VitTrack 能在完整 ERP 全景图上搜索，不会被小切图限制上下文。
3. **速度提升 20 倍**：无需切图、状态预测、坐标转换等开销，FPS 从 ~8 提升到 ~155。
4. **实现简单**：仅需处理 360° 边界穿越（x 坐标回绕），代码量远小于 BFoV 框架。

### 使用方法

```python
from panotrack.trackers.factory import create_tracker

# 方式 1：通过工厂创建（推荐）
tracker = create_tracker('direct_erp', model_path='models/object_tracking_vittrack_2023sep.onnx')

# 方式 2：直接实例化
from panotrack.trackers.direct_erp import DirectERPTracker
tracker = DirectERPTracker(model_path='models/object_tracking_vittrack_2023sep.onnx')

# 使用方式与 BaseTracker 一致
tracker.init(erp_frame, bbox)      # bbox 为 ERP 坐标 (x,y,w,h)
result = tracker.update(erp_frame) # 自动处理 360° 边界穿越
```

### 边界穿越处理

DirectERPTracker 在 `update()` 中自动处理 360° 边界穿越：
- 当 tracker 输出的 x < 0 时，自动加上 `erp_w`
- 当 x >= erp_w 时，自动减去 `erp_w`
- 最终 x 被钳制到 `[0, erp_w - w)` 范围内

### 预处理与多尺度策略

| 策略 | Mean AUC | FPS | 结论 |
|------|----------|-----|------|
| Raw ERP | 0.263 | 159 | ✅ **最佳** |
| Highpass σ=3 | 0.082 | 8 | ❌ 严重损害 |
| Highpass σ=5 | 0.094 | 7 | ❌ 严重损害 |
| Single scale | 0.263 | 172 | ✅ **最佳** |
| 0.5x pyramid | 0.261 | 93 | ❌ 略降 |

**结论**：
- **Raw ERP 帧** 最适合 VitTrack，highpass 滤波会移除关键低频信息
- **多尺度金字塔** 无益，VitTrack 内置 `sr_ratio` 已处理多尺度搜索

### 配置示例

```json
{
  "tracker": "direct_erp",
  "model_path": "models/object_tracking_vittrack_2023sep.onnx"
}
```

### 实验结论

- **Raw ERP 帧** 最适合 VitTrack，highpass 滤波会移除关键低频信息，导致 AUC 从 0.26 降到 0.08
- **多尺度金字塔** 无益，VitTrack 内置 `sr_ratio` 已处理多尺度搜索
- **边界穿越** 已通过 x 坐标回绕自动处理

---

## 8 月官方评测对接清单

### I/O 适配层位置

| 适配层 | 文件 | 状态 |
| --- | --- | --- |
| 图像序列文件协议 | `panotrack/io/file_protocol.py` | 已实现（frames 目录 + init 文件 → results 文件） |
| trax 风格行协议 | `panotrack/io/trax_protocol.py` | **占位实现**，官方协议公布后替换（行协议骨架已就位） |
| 命令行入口 | `panotrack/cli.py`（`python -m panotrack.cli`） | 已实现 |
| 容器入口 | `docker/Dockerfile` ENTRYPOINT + `docker/entrypoint.sh` | 已实现 |

官方协议公布后：仅需替换/新增 `panotrack/io/` 下适配层，核心
`PanoTracker` 与算法栈零改动。

### 待确认事项（拿到官方评测手册后逐条核对）

1. **框格式**：是否确为 `x,y,w,h` 浮点（还是 `x1,y1,x2,y2` 或整数像素）？
   精度要求（当前输出保留 2 位小数）？
2. **跨界约定**：官方 GT 与提交结果是否允许 `x + w > W`（回绕表示），
   还是要求拆成两段 / 取模到 `[0, W)`？dual IoU 口径是否一致？
3. **输入形态：图像序列还是视频流？** 若为 mp4 视频流，解码方案待定——
   当前依赖白名单（numpy/Pillow/scipy）不含解码器，需官方提供解码后帧，
   或评审引入解码依赖（同时复核 Docker 断网自包含）。
4. **初始化方式**：首帧 GT 初始化（OPE）还是外部触发初始化 / 丢失后重新 init？
5. **实时性约束**：帧率 / 单帧耗时上限、是否限制 CPU 线程数与内存。
6. **结果提交格式**：文件命名、分隔符（逗号/空格/Tab）、是否需要逐帧置信度与状态字段。
7. **协议细节**（若为 trax 类交互协议）：握手字段、超时、错误码、重启语义。

---

## 许可与协作

学生竞赛项目，内部仓库。各模块负责人以 CONTRACTS.md 为唯一接口准绳，
并行开发、集成阶段联调。问题对齐请优先核对契约原文。

---

## 如何贡献

1. **分支策略**：每人开 `feat/xxx` 或 `fix/xxx` 分支，PR 前确保 `tests/` 全绿
2. **接口变更**：修改 `CONTRACTS.md` 后必须同步更新所有相关模块，并在 PR 描述中说明影响范围
3. **大文件**：模型、数据集、运行产物禁止入库，放到对应目录后更新 `.gitignore`
4. **文档同步**：新增/修改功能必须同步更新 `README.md` 与 `CONTRACTS.md`

## 常见问题

**Q: 克隆后跑测试报 `ModuleNotFoundError`？**  
A: 确认已执行 `pip install -r requirements.txt`，且 Python 版本 ≥ 3.12。

**Q: `models/` 目录是空的？**  
A: VitTrack ONNX 模型需自行下载（见上方「下载模型」），放到 `models/` 后即可运行。

**Q: 官方数据集怎么获取？**  
A: 提交邮箱申请 → https://360vots.hkustvgd.com/ ，通过后下载到 `data360/official/`。

**Q: 为什么 Direct ERP 方案推荐，但代码里还有 BFoV 相关模块？**  
A: BFoV 几何模块仍是项目重要积累，且 `SphericalState`、`RemapCache` 等组件有复用价值。当前主推 Direct ERP，但 BFoV 代码保留供参考和可能的混合方案。

**Q: Docker 镜像为什么这么大？**  
A: 基础镜像 `python:3.12-slim` + onnxruntime 约 400MB，如需减小可改用 `python:3.12-alpine` 并调整依赖。

**Q: 如何调试特定序列？**  
A: 用 `tools_local/` 下的脚本（如 `debug_seq0036.py`）或在 `tests/` 里加临时断言。调试工具不提交 Git。

---

## 相关链接

- [360VOTS 官方主页](https://360vots.hkustvgd.com/)
- [360VOT GitHub](https://github.com/HuajianUP/360VOT)
- [VitTrack 官方实现](https://github.com/VDIGPKU/VitTrack)
- [HuggingFace 数据集](https://huggingface.co/datasets/xuyzshaun/360VOTS)

---

> **最后更新**：2026-08-10 | **当前阶段**：初赛已提交（ODTrack 精度版 0.5792）| **决赛方向**：ODTrack + 可靠性门控 + 球面重捕获（见 `docs/ODTRACK_RECAPTURE_PLAN_ZH.md`）
