# GRT-360 最终交付说明（中文）

更新时间：2026-08-10

## 两套最终代码

### 1. 精度冠军：ODTrack ERP 三平铺

入口：`integrations/odtrack/run_erp.py`（360VOT 序列评测）与
`integrations/odtrack/file_protocol.py`（官方提交文件协议，见下方 GPU 镜像一节）。

该入口保存了完整的 360VOT 三平铺适配逻辑。ODTrack 上游源码和权重没有复制进
GitHub，因为体积和授权不适合直接提交；运行时通过 `--odtrack-root` 与
`--checkpoint` 提供，并应在实验记录中保存提交号和 SHA-256。

120 条序列严格结果：AUC `0.5792135073`，SR `0.6531941586`，双 GPU 端到端约
`8.99 FPS`。

### 2. 综合均衡：GRT360-Causal-DTP-ERP

入口：`scripts/fuse_causal_dtp_erp.py`；核心模块：
`panotrack/geometry/causal_dtp.py`。

该版本使用 ODTrack 作为精度教师、UETrack ERP-wrap 作为恢复学生、LightFC 作为
低成本 scout，并加入 ERP 接缝圆周运动、因果可靠性和滞回切换。

当前保守版 120 条序列结果与 ODTrack 持平，目的是保证不损伤精度。真正的
KV-cache、Token 压缩和教师-学生训练仍属于下一阶段模型改造。

## 本地离线 Docker

主镜像 `docker/Dockerfile` 是无网络运行镜像，包含 `panotrack/`、`scripts/`、
`integrations/`、配置和中文报告。构建时使用本机已缓存的 Python 基础镜像和依赖层：

```bash
docker build --network=none --platform linux/amd64 \
  -f docker/Dockerfile -t grt360-final:2026-08-10 .
```

离线自检：

```bash
docker run --rm --network none grt360-final:2026-08-10 --help
docker run --rm --network none grt360-final:2026-08-10 \
  --frames /data/frames --init /data/init.txt \
  --out /data/results.txt --config /app/configs/default_v2.json
```

容器运行期不访问网络；数据和结果通过挂载目录提供。ODTrack/UETrack 的大模型
权重仍需按其各自授权放在宿主机或专用 GPU 镜像中，不能假装已经包含在轻量 CPU
离线镜像里。

本机还保留了已经构建好的 GPU 版 UETrack ERP-wrap 镜像，包含固定的上游源码、
权重和 CLIP 缓存：

- 镜像：`grt360-uetrack:2026-08-09`；
- 离线导出：`artifacts/grt360-uetrack-2026-08-09.tar`，约 5.99 GB；
- 权重 SHA-256：`1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`。

因此本地有三份镜像：轻量最终交付镜像用于仓库代码/协议验收，GPU UETrack
镜像用于实际均衡路线推理，GPU ODTrack 镜像用于实际精度路线推理。
三份镜像都不需要运行期联网。

## Arena 平台官方协议适配（2026-08-14）

平台 https://yjy-arena.insta360.cn/ 实测确认：提交方式为 docker push 到本队专属地址，接口为 BFoV（不是上传 tar，也不是 ERP xywh）。

- 新入口：`integrations/odtrack/arena_protocol.py`（无参自启动；
  读 `/mnt/dataset/<seq>/video.mp4` + `init.txt`（BFoV），写 `/mnt/result/<seq>.txt`（BFoV，行号=帧号，丢失帧 0,0,0,0）；
  接口规范与官方 demo.zip 一致。
- 镜像规范：linux/amd64、断网自包含、退出码 0、提交配额每日 3 / 累计 10。
- 验证：`tests/test_arena_protocol.py` 协议层全部通过；
  带实际 ODTrack 权重 CPU 冒烟通过（多序列、丢失帧、seqlist）。
- 详见 `docs/ARENA_PROTOCOL_TEST_ZH.md`。


## ODTrack 精度版 GPU 镜像（2026-08-10 新增）

精度提交版镜像，把上游源码与权重完整打进镜像，运行期零网络：

- 镜像：`grt360-odtrack:2026-08-10`；
- 构建文件：`docker/odtrack/Dockerfile` + `docker/odtrack/requirements.txt`；
- 构建脚本：`docker/odtrack/build_odtrack_image.sh`（从
  `artifacts/server_snapshot/` 组装临时构建上下文；国内环境无法访问 docker.io
  时用 `BASE_IMAGE=docker.m.daocloud.io/pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`）；
- 提交入口：`integrations/odtrack/file_protocol.py`（与 UETrack 镜像同一
  文件协议：`--frames` 帧目录 + `--init` 初始框 -> `--out` results.txt，
  逐帧 x,y,w,h 12 位小数，首行为初始框）；
- 离线导出：`artifacts/grt360-odtrack_2026-08-10_linux-amd64.tar`；
- 权重 SHA-256：`2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e`；
- 依赖：torch 2.3.1 / numpy 2.2.6 / opencv-headless / timm 0.5.4 等，
  基础镜像 `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`（Python 3.10）。

运行（正式评测需 GPU）：

```bash
docker run --rm --gpus all \
  -v /path/to/data:/data grt360-odtrack:2026-08-10 \
  --frames /data/frames --init /data/init.txt --out /data/results.txt
```

验证状态（2026-08-10）：

- ✅ `--help` 参数解析；
- ✅ 容器内权重 SHA-256 与上表一致；
- ✅ CPU 结构冒烟 3 帧全链路（init + 2 帧 track），输出框随目标移动正确
  （CPU 约 0.36 FPS，仅验证管线）；
- ⏳ GPU 全量 120 序列（8.99 FPS）需在服务器或有 NVIDIA 运行时的机器复测，
  本机 WSL 无 NVIDIA runtime 无法本地验证。

## 评分证据

- `reports/results/odtrack_120_score/bakeoff.json`
- `reports/results/grt360_causal_dtp_erp_120_score/bakeoff.json`
- `reports/GRT360_CANDIDATE_ARCHITECTURE_2026-08-10.md`
- `reports/STAGE_RESULTS_2026-08-09.md`

## 两套代码的实际参数

### ODTrack 精度版

- 上游配置：`baseline.yaml`；ViT-Base/patch16，stride=16；CE 层 `[3,6,9]`；
  CE keep ratio `[0.7,0.7,0.7]`；`CE_TEMPLATE_RANGE=CTR_POINT`；`ATTN_TYPE=concat`；
  Center head 通道数 256。
- 模板：`TEMPLATE_FACTOR=2.0`、`TEMPLATE_SIZE=192`、`TEMPLATE_NUMBER=3`。
- 搜索：`SEARCH_FACTOR=5.0`、`SEARCH_SIZE=384`。
- 测试检查点：`EPOCH=300`、`MEMORY_THRESHOLD=1000`。
- ERP 适配：每帧水平复制 3 次；首帧框放到中间副本；输出横坐标对原宽度取模。
- 训练配置记录：AdamW、batch=8、epoch=300、LR=1e-4、GIoU=2.0、L1=5.0、AMP=False。

### GRT360-Causal-DTP-ERP 均衡版

- 专家顺序固定为：`0=ODTrack`、`1=UETrack ERP-wrap`、`2=LightFC`。
- 路由默认值：`hold_frames=3`、`blend_alpha=0.18`、`teacher_margin=0.90`、
  `recovery_margin=0.20`、`reliability_decay=18.0`、`geometry_penalty=0.35`。
- 内部运动/可靠性参数：`velocity_alpha=0.35`、`scale_decay=3.0`、
  `agreement_decay=12.0`。
- ERP 风险：极区从绝对纬度 55° 开始加权；接缝按中心距小于约 12% 画面宽度计算风险；
  同时惩罚尺度突变、预测创新量和专家分歧。
- UETrack 底层上游参数：`fastitpnt_layer6`、stride=16、8 个 MoE expert、MoE layer `[5]`、
  `SEARCH_FACTOR=4.0`、`SEARCH_SIZE=224`、`TEMPLATE_FACTOR=2.0`、`TEMPLATE_SIZE=112`、
  `WINDOW=True`。
- 当前采用保守阈值，优先保证 ODTrack 精度；降低 `teacher_margin` 才会增加学生接管，
  但未经训练的激进版本已实测退化。
