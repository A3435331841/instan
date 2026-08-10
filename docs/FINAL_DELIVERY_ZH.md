# GRT-360 最终交付说明（中文）

更新时间：2026-08-10

## 两套最终代码

### 1. 精度冠军：ODTrack ERP 三平铺

入口：`integrations/odtrack/run_erp.py`。

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

## 评分证据

- `reports/results/odtrack_120_score/bakeoff.json`
- `reports/results/grt360_causal_dtp_erp_120_score/bakeoff.json`
- `reports/GRT360_CANDIDATE_ARCHITECTURE_2026-08-10.md`
- `reports/STAGE_RESULTS_2026-08-09.md`
