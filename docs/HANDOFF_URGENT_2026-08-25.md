# GRT-360 完整交接文档（2026-08-25 紧急交接）

> **比赛截止：今天 2026-08-25！** 初赛提交 = docker push 镜像到平台即触发评测。
> 当前已有一个初赛镜像在平台（ODTrack 精度版 AUC 0.5792），微调版如果来得及可以重推。
> 本文档面向接手者（人或 AI），包含全部凭据、路径、命令、坑。

---

## 一、服务器信息

| 项 | 值 |
|---|---|
| SSH | `wuyou@153.0.134.134:12407` 密码见安全渠道（勿入库） |
| GPU | 2× RTX 3090 24GB，驱动 570.133（cu128 OK） |
| 系统 | Ubuntu 20.04，Python 3.8（系统），conda env grt = Python 3.12 |
| 数据盘 | `/data`（600G ext4 已挂载，415G 可用） |
| 内存 | 102GB |
| sudo | 免密 |

**连接方式**（本地 Windows）：
```bash
# 本地工具脚本（推荐）
cd D:\instan\pano360
python tools_local/srv.py cmd "<命令>"           # 执行远程命令
python tools_local/srv.py put <本地> <远程>       # 上传（断点续传）
# 注意：远程路径前加 MSYS_NO_PATHCONV=1（Git Bash 路径转换坑）
```
或直接 SSH 客户端连 `wuyou@153.0.134.134 -p 12407`。

---

## 二、当前状态快照

### ✅ 已完成

| 项 | 结果 |
|---|---|
| **ODTrack 官方 130 条基线** | **AUC 0.5882 / SR 0.6939 / 29.9 FPS**（1440×720） |
| 基线结果路径 | `/data/runs/baseline/odtrack_*/<block>/<seq>/metrics.json`（130 个） |
| 基线汇总脚本 | `/data/summarize_baseline.py`（`python /data/summarize_baseline.py` 即可重跑） |
| 微调数据 | `/data/finetune/official_got10k/`（150 子序列 / 178,806 帧 / list.txt 就绪） |
| ODTrack 权重 | `/data/weights/ODTrack_ep0300.pth.tar`（371MB，初赛同款） |
| ODTrack 源码 | `/data/odtrack_ws/odtrack/`（已打好补丁，详见下文） |
| 360VOT 测试集 | `/data/pano360/data360/` 122 个目录（后台下载中，不占 GPU） |
| 代码 | `/data/pano360/`（git 仓库，含 eval_official.py / make_official_split.py 等） |
| conda env | `/data/miniconda3/envs/grt/`（torch 2.7.0+cu128 双卡 OK，wandb/pycocotools/lmdb 已装） |

### 🔴 当前卡点：微调训练启动失败

**根因**：`lmdb` 模块已安装（2.3.0），但**最后一次启动的进程是在安装前启动的**，需要重启即可。

**修复命令**（在服务器上执行）：
```bash
pkill -9 -f run_training
sleep 1
cd /data/odtrack_ws/odtrack/lib/train
CUDA_VISIBLE_DEVICES=0 nohup /data/miniconda3/envs/grt/bin/python run_training.py --script train --config finetune_official --save_dir /data/training/checkpoints > /data/finetune_train.log 2>&1 < /dev/null &
sleep 60
tail -20 /data/finetune_train.log
```
如果还报错，看日志最后几行，逐个装缺失的包（已装：wandb/pycocotools/lmdb/tensorboardX/jpeg4py）。

### 微调配置（已就位）

- yaml：`/data/odtrack_ws/odtrack/experiments/odtrack/finetune_official.yaml`
  - PRETRAIN_FILE = `/data/weights/ODTrack_ep0300.pth.tar`（热启动）
  - LR = 0.00001（原 0.0001 的 1/10）
  - EPOCH = 20，LR_DROP_EPOCH = 15，SAMPLE_PER_EPOCH = 15000
- split 文件：`/data/odtrack_ws/odtrack/lib/train/data_specs/got10k_train_full_split.txt`（150 行，0~149）
- local.py：`/data/odtrack_ws/odtrack/lib/train/admin/local.py`（got10k_dir 指向微调数据）
- dataset/__init__.py：已注释掉 coco/imagenetvid/tracking_net 导入（只用 Got10k）
- base_functions.py：已改为 `from lib.train.dataset import Lasot, Got10k`
- loader.py：已去掉 `torch._six` 引用

---

## 三、比赛关键信息

| 项 | 值 |
|---|---|
| 平台 | https://yjy-arena.insta360.cn/ |
| 登录账号 | `xiaobai` / 密码见安全渠道（勿入库） |
| 比赛 ID | 1（compUID: `pekjqegykk`） |
| 比赛截止 | **2026-08-25（今天！）** |
| 提交方式 | docker push 镜像到平台即触发自动评测 |
| 推送地址 | `yjy-arena.insta360.cn/pekjqegykk/<teamUID>/model:vN` |
| teamUID | 需在平台「我的队伍」页确认（本地凭据只对 `ywd9xdx7rp` 有效） |
| 配额 | 每日 3 次 / 累计 10 次（v1 已废 1 次，剩约 9 次） |
| 评测环境 | RTX 5090（Blackwell sm_120）+ CUDA 12.8 + torch 2.7，断网运行 |
| 镜像要求 | linux/amd64、无参自启动、退出码 0、断网自包含、<20GB |
| 接口 | 输入 `/mnt/dataset/<seq>/video.mp4 + init.txt`（BFoV），输出 `/mnt/result/<seq>.txt`（BFoV） |
| 评分 | OTB 协议 + 360VOTS 无偏球面 IoU，SR@0.5 + AUC |

---

## 四、本地资产（Windows 机器）

| 路径 | 内容 |
|---|---|
| `D:\instan\初赛数据\` | 训练集 zip（24.7GB）+ 解压后 train/ |
| `D:\instan\pano360\` | git 仓库（代码 + 脚本 + 文档） |
| `D:\instan\pano360\tools_local\srv.py` | SSH 连接工具（凭据已写入） |
| `D:\instan\pano360\scripts\eval_official.py` | 官方数据评测 runner（已验证） |
| `D:\instan\pano360\scripts\make_official_split.py` | 95/35 划分（seed 20260824） |
| `D:\instan\pano360\scripts\validate_official_data.py` | 数据校验（130/130 通过） |
| `D:\instan\pano360\scripts\prepare_finetune_data.py` | 微调数据准备（已跑完） |
| `D:\instan\pano360\docs\EXECUTION_PLAN_ZH_2026-08-22.md` | 完整执行计划 v3 |
| `D:\instan\pano360\docs\TECH_RESEARCH_ZH_2026-08-21.md` | 四路调研报告 |
| `D:\instan\交付物_2026-08-14\` | cu128 镜像 tar + 权重 tar |
| `D:\instan\pano360\integrations\odtrack\arena_protocol.py` | **官方提交协议入口** |

---

## 五、下一步行动（按优先级，截止今天）

### 5.1 立即：启动微调（预计 2~4h 完成 20 epoch）

```bash
# 服务器上执行（修复 lmdb 后重启训练）
pkill -9 -f run_training; sleep 1
cd /data/odtrack_ws/odtrack/lib/train
CUDA_VISIBLE_DEVICES=0 nohup /data/miniconda3/envs/grt/bin/python run_training.py --script train --config finetune_official --save_dir /data/training/checkpoints > /data/finetune_train.log 2>&1 < /dev/null &
# 监控
tail -f /data/finetune_train.log
```

**如果还报错**，逐个排查：
1. `ModuleNotFoundError: No module named 'xxx'` → `pip install xxx`
2. `RuntimeError: YOU HAVE NOT SETUP YOUR local.py` → local.py 被覆盖了，重新写入（见上文内容）
3. CUDA OOM → batch_size 减半（yaml 里 `TRAIN.NUM_PER_GPU`）

### 5.2 微调完成后：验证 + 重建镜像 + 推送

**验证**（微调 checkpoint 在 `/data/training/checkpoints/` 下，取最新 ep）：
```bash
cd /data/pano360
# 用微调后的权重跑 valid 35 条（~20 分钟）
/data/miniconda3/envs/grt/bin/python scripts/eval_official.py --tracker odtrack \
  --data /data/traindata/train --split valid \
  --odtrack-workspace /data/odtrack_ws/odtrack \
  --odtrack-ckpt /data/training/checkpoints/finetune_official/ODTrack_ep0020.pth.tar \
  --gpu 0 --out /data/runs/ft_valid
# 对比 valid 基线（AUC 0.5613），提升 ≥ +0.005 才值得提交
```

**重建镜像**（如果微调有效）：
```bash
# 方法 A：直接在服务器上构建（需要装 docker，sudo 免密）
sudo apt-get install -y docker.io
# 或方法 B：把微调权重传回本地，用本地已有的构建脚本
# 本地已有构建脚本: D:\instan\pano360\docker\odtrack\build_odtrack_cu128.sh
# 镜像入口: integrations/odtrack/arena_protocol.py（已验证通过）
```

**推送**：
```bash
# 确认 teamUID（平台「我的队伍」页）
docker tag grt360-odtrack:fintuned yjy-arena.insta360.cn/pekjqegykk/<teamUID>/model:v2
docker push yjy-arena.insta360.cn/pekjqegykk/<teamUID>/model:v2
```

### 5.3 如果微调来不及/无效

**保底方案：不重推**。初赛已提交的 ODTrack 精度版（AUC 0.5792 on 360VOT）仍在平台评测队列。
官方数据基线 AUC 0.5882 证明方案没有问题，不折腾比折腾好。

---

## 六、已知坑（接手者必读）

1. **SSH 路径转换**：Git Bash 下远程路径 `/data/...` 会被转成本地 Windows 路径，必须加 `MSYS_NO_PATHCONV=1`。
2. **Python 路径**：`/d/instan/...` 是 Git Bash 路径，Windows Python 需要 `D:/instan/...`。
3. **ODTrack 上游代码是 Python 3.8 时代的**：`torch._six`、`visdom` 等需要补丁（已打好，在 `/data/odtrack_ws/odtrack/`）。
4. **训练依赖链**：wandb → pycocotools → lmdb → tensorboardX → jpeg4py 全部已装，如果环境重建需重装。
5. **评测 runner 的 OFFICIAL_TRAIN_ROOT**：默认指向 Windows 路径，服务器上需 `export OFFICIAL_TRAIN_ROOT=/data/traindata/train`。
6. **360VOT 测试集下载**：HF gated 仓库，token 见安全渠道（read 权限，勿入库），走 `HF_ENDPOINT=https://hf-mirror.com`。
7. **平台注册已锁**：无法新建账号，只能用 xiaobai。
8. **提交配额**：推送即占用（含失败），本地断网自测通过才 push。
9. **Blackwell 兼容**：评测机是 RTX 5090（sm_120），镜像必须 torch 2.7+cu128（本地 cu128 已构建过，脚本在 `docker/odtrack/build_odtrack_cu128.sh`）。
10. **官方数据有 11% 消失帧**（0,0,0,0）：评测 runner 已做掩码（跳过这些帧），微调数据已按消失段切分子序列。

---

## 七、基线详细数据（决策依据）

| 分项 | n | AUC | SR | 备注 |
|---|---:|---:|---:|---|
| 全部 | 130 | 0.5882 | 0.6939 | 29.9 FPS @1440×720 |
| valid 留出 | 35 | 0.5613 | 0.6568 | 微调对比基准 |
| train 训练 | 95 | 0.5981 | 0.7076 | 微调后应 > 0.62 |
| real | 47 | 0.5622 | 0.6567 | |
| sim | 83 | 0.6029 | 0.7149 | |

**烂尾 10 条**（全部 absent=0，即目标在场但跟丢）：
seq_0046(0.057) / seq_0025(0.069) / seq_0011(0.075) / seq_0010(0.077) / seq_0041(0.078) /
seq_0027(0.085) / seq_0075(0.087) / seq_0016(0.098) / seq_0037(0.100) / seq_0012(0.114)

**重捕获收益估算**：救回 5 条至 0.4 → 宏平均 +0.012；全救回 → +0.025。
**微调收益估算**：LoRAT 360 域微调文献 +7% 相对 → 我们可能 0.588 → 0.62。

---

## 八、文件清单（服务器 /data/）

```
/data/
├── traindata/train/           # 官方训练集解压（train_real 47 + train_sim 83）
├── finetune/official_got10k/  # 微调数据（150 子序列 GOT-10k 格式）
├── weights/ODTrack_ep0300.pth.tar
├── odtrack_ws/odtrack/        # ODTrack 上游源码（已打补丁）
├── pano360/                   # git 仓库（代码+脚本）
├── runs/baseline/             # 基线评测结果（130 个 metrics.json）
├── training/checkpoints/      # 微调 checkpoint 输出目录
├── miniconda3/envs/grt/       # Python 3.12 + torch 2.7.0+cu128
├── pano360/data360/           # 360VOT 测试集（122 个目录，下载中）
├── setup_server2.sh           # 环境安装脚本
├── prep_data.sh               # 数据准备接力脚本
├── start_finetune_v2.sh       # 微调启动脚本
├── restart_ft.sh              # 微调重启脚本
├── summarize_baseline.py      # 基线汇总
└── finetune_train.log         # 微调日志（tail -f 监控）
```

---

## 九、本地代码 git 状态

- 分支 `main`，最新 commit `d8d5364`
- 未推送 GitHub（本地 commit 而已），服务器上代码是 tar 上传的（非 git clone）
- 如需同步：本地 `git bundle create /tmp/repo.bundle main` → 上传 → 服务器 `git clone /tmp/repo.bundle`

---

*交接完成时间：2026-08-25 00:15。比赛截止今天，祝好运！*
