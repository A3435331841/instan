# GRT-360 GPU 全量验证与平台提交交接（2026-08-15）

> **给有 GPU 的队友（紧急，比赛 8/16 开始）：**
> 本地（无 NVIDIA GPU）已完成协议层测试、依赖检查、CPU 冒烟（全部通过，证据见 §2）。
> 需要你在 GPU 机器上完成最后两步：**① cu128 镜像 120 序列全量复测 → ② RTX 5090 (Blackwell) 实测 → ③ 通过后推送平台**。
> 全部命令可直接复制执行；完成后把 §4 的验证记录回传。

---

## 一、当前唯一可提交镜像（先读这一段）

| 项 | 值 |
|---|---|
| 镜像名 | `grt360-odtrack:2026-08-14-cu128`（Image ID `54643d42c62b`，13.1GB） |
| 为什么是它 | 评测机 GPU 为 **RTX 5090（Blackwell sm_120）**，官方明确要求 CUDA 12.8 + torch≥2.6；本镜像 torch `2.7.0+cu128` / CUDA `12.8` |
| 入口 | `arena_protocol.py`：无参自启动，读 `/mnt/dataset/<seq>/video.mp4 + init.txt`（BFoV），写 `/mnt/result/<seq>.txt`（BFoV，行号=帧号，丢失帧 `0,0,0,0`），退出码 0 |
| 权重 | ODTrack `ODTrack_ep0300.pth.tar`（SHA-256 `2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e`）已打入镜像 |
| ⚠️ | **旧 CUDA 12.1 镜像（`...-arena` / `...-minimal`）已删除**（不兼容 Blackwell），不要再找旧镜像 |

**平台硬性限制**：linux/amd64；断网运行（`--network none`）；提交配额**每日 3 次、累计 10 次（推送即占用，含失败）**——不要用真实评测当调试场。

---

## 二、本地已完成验证（2026-08-15 实测，可直接复现）

1. **协议层测试** ✅ `python tests/test_arena_protocol.py` → `ALL PROTOCOL TESTS PASSED`
   （mock tracker，不依赖权重/GPU；覆盖 BFoV 转换、多序列遍历、seqlist、丢失帧占位、输出格式）
2. **镜像内依赖** ✅ cv2 4.12.0 / numpy 2.2.6 / timm 0.5.4 / easydict / yacs / einops 全部可导入
3. **CPU 冒烟** ✅ 合成 2 条序列（`scripts/make_arena_smoke_dataset.py`，含一条跨 0/360° 接缝），
   `docker run --rm -v <dataset>:/mnt/dataset:ro -v <result>:/mnt/result <镜像> --force-cpu`
   结果：退出码 0，18 帧 / 52.8s（CPU 0.3 FPS），输出示例：

   ```
   seq_0001（目标右移）: clon -138.4 → -119.6 → -100.8 → ... → +30.6   （单调递增，方向正确）
   seq_0002（跨接缝）:   clon 171.0 → 127.2 → 82.8 → 38.6 → -5.9 → ... → -138.3  （穿缝回绕正确）
   ```

   > 说明：直接 `import ODTrack` 报 `No module named 'torch._six'` 是**正常现象**——
   > `arena_protocol.py` 入口会先打兼容补丁，必须从该入口跑。

---

## 三、需要你完成的事（按顺序执行）

### Step 1：获取镜像（二选一）

**方式 A（推荐）：本机 `docker save` 导出后传给你，你 `docker load`**
```bash
# 本机（无 GPU 那台）：
docker save grt360-odtrack:2026-08-14-cu128 -o grt360-odtrack-cu128.tar   # 约 4~6GB，U盘/网盘传
# GPU 机器：
docker load -i grt360-odtrack-cu128.tar
```

**方式 B：GPU 机器上重新构建**（构建依赖包已打包，见下方获取方式）
```bash
# ① clone 代码（main 已含构建脚本/协议代码）
git clone https://github.com/A3435331841/instan && cd instan
# ② 获取构建依赖包 grt360-odtrack-build-deps.tar.gz（330MB：ODTrack 源码+权重）
#    途径：GitHub Release（tag: grt360-odtrack-build-deps-v1，若已上传）
#         或 网盘/QQ（找队友要，文件在无 GPU 那台 D:\grt360-odtrack-build-deps.tar.gz）
tar -xzf grt360-odtrack-build-deps.tar.gz     # 解出 artifacts/server_snapshot/...
# ③ 构建（需联网拉基础镜像+pip；评测断网但镜像自包含）
bash docker/odtrack/build_odtrack_cu128.sh grt360-odtrack:2026-08-14-cu128
```

### Step 2：GPU 全量 120 序列复测（精度基线对账）

- **数据**：360VOT 测试集 120 序列（4K ERP，58GB）。来源：原服务器 `/data/projects/instan/data360`（服务器可能已停机，需从备份恢复或官网 https://360vots.hkustvgd.com/ 重新申请下载）。
- **格式转换**：360VOT 是帧目录 + `groundtruth.txt`（xywh），需转为 Arena 格式：
  - 每序列目录内生成 `video.mp4`（帧目录用 cv2.VideoWriter 编码，mp4v，顺序=帧号）
  - 生成 `init.txt`（首帧 GT 框 → BFoV：`clon,clat,fov_h,fov_v`，转换算法见 `panotrack/geometry/bfov.py` 的 `bfov_from_erp_bbox`）
  - 顶层可选 `seqlist.txt`（每行一个序列名，含 `video.mp4` 的子目录会被自动扫描）
- **跑评测**（GPU）：
```bash
docker run --rm --gpus all \
  -v <数据根>:/mnt/dataset:ro -v <输出根>:/mnt/result \
  grt360-odtrack:2026-08-14-cu128
```
- **验收门槛**：
  1. 120 条全部跑完，退出码 0，无序列 FAILED；
  2. 每条输出行数 = 帧数，每行 4 个数值；
  3. **精度对账**：用官方球面 IoU 口径评分，对照基线 **AUC 0.5792 / SR 0.6532**（360VOT 本地 dual-IoU 口径约 0.5819/0.6562）；若偏差 >±0.02 需排查；
  4. 记录端到端 FPS（基线 8.99，单卡口径）。

### Step 3：Blackwell 实测（最关键的一步）

```bash
nvidia-smi   # 确认 GPU = RTX 5090（sm_120）
# 先只跑 2~3 条序列（用 --seqs 参数限制）：
docker run --rm --gpus all -v <数据>:/mnt/dataset:ro -v <输出>:/mnt/result \
  grt360-odtrack:2026-08-14-cu128 --seqs seq_0001,seq_0002
```
- 无 `CUDA error: no kernel image is available for execution on the device` 类报错即为通过；
- **若失败**：截图/贴错误信息回群，**先不要占用提交配额**。

### Step 4：推送平台（全部验证通过后）

```bash
# ① 到平台「我的队伍」页确认【当前队伍 UID】（重要！）
#    本地 docker 凭据目前只对 teamUID=ywd9xdx7rp 有效；
#    旧文档里的 ugjpuufva2 已无权限（unauthorized），推了必失败并占配额。
docker login yjy-arena.insta360.cn

# ② 打 tag（<真实teamUID> 用上面确认的值；若 v1 已占用过配额，用 v2/v3）
docker tag grt360-odtrack:2026-08-14-cu128 \
  yjy-arena.insta360.cn/pekjqegykk/<真实teamUID>/model:v2

# ③ 推送即自动评测（配额：每日3 / 累计10）
docker push yjy-arena.insta360.cn/pekjqegykk/<真实teamUID>/model:v2

# ④ 评测完成后到平台「我的提交」/排行榜查看成绩
```

> 若第 1 次推送出现格式类问题，当天修复后用剩余配额重推；推送前本地断网全量自测。

---

## 四、验证记录回传模板（在群里按此格式回）

```
[GPU验证] 机器: <型号/显卡> 
nvidia-smi: <输出首行>
序列: 120/120 完成, 退出码 0, FAILED: <无或列表>
行数对齐: <全对/异常列表>
FPS: <单卡端到端>
精度: AUC=<?> SR=<?>（口径: <官方球面IoU / 360VOT dual>）
Blackwell: <通过/报错信息>
推送: <model:vN / 未推 / 结果>
```

---

## 五、相关文件索引（本次已提交 GitHub）

| 文件 | 说明 |
|---|---|
| `integrations/odtrack/arena_protocol.py` | 官方协议入口（BFoV 输入输出 + 三平铺内核 + 兼容补丁） |
| `docker/odtrack/Dockerfile.cu128` + `build_odtrack_cu128.sh` | Blackwell 版镜像构建（torch 2.7.0+cu128） |
| `docker/odtrack/Dockerfile.minimal` + `build_odtrack_minimal.sh` | 旧 12.1 精简版（已弃用，仅存档） |
| `tests/test_arena_protocol.py` | 协议层测试（本地可跑，无需 GPU） |
| `scripts/make_arena_smoke_dataset.py` | 合成冒烟数据集生成脚本（CPU/GPU 通用） |
| `docs/ARENA_PROTOCOL_TEST_ZH.md` | 协议适配与本地验证详细记录 |
| `integrations/odtrack/README_ARENA_PROTOCOL.md` | 协议入口使用说明 |
| `deliverables/SUBMISSION_2026-08-10/05_官方联系/ARENA平台提交指南_2026-08-14.md` | 平台完整提交指南（配额/镜像规范/格式） |

**不进入 Git 的大文件**：
- **构建依赖包** `grt360-odtrack-build-deps.tar.gz`（330MB，含 ODTrack 源码 + 权重）——
  已打包在无 GPU 那台 `D:\grt360-odtrack-build-deps.tar.gz`；上传途径：GitHub Release
  （tag `grt360-odtrack-build-deps-v1`，网页手动上传，GitHub 单文件上限 2GB）或网盘/QQ。
  解压到仓库根（保留 `artifacts/server_snapshot/` 结构）即可直接构建。
- **已构建镜像本身**（方式 A 直接 docker save 传输，约 4.4GB，无需源码/权重）。
- 权重 SHA-256：`2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e`（见 §1）

---

## 六、常见问题

- **为什么直接 import ODTrack 报 torch._six 缺失？** 正常，必须走 `arena_protocol.py` 入口（内置补丁）。
- **CPU 上能全量跑 120 条吗？** 不建议：CPU 约 0.3 FPS，120 条 11 万帧需 ~100 小时。
- **旧镜像还能用吗？** 已删除；且 CUDA 12.1 在 RTX 5090 上无 kernel，不能用于提交。
- **数据从哪来？** 原服务器 `/data/projects/instan/data360`（若已停机则从备份恢复或官网重新申请 360VOT）。
- **配额定心丸**：本地全量自测通过后再 push；前 3 次配额留足给格式试探。

*本交接基于 2026-08-15 本地实测与平台 8/14 实况；平台规则变动以「我的队伍」页与官方答疑为准。*
