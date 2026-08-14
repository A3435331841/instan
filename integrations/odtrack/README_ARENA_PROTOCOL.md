# arena_protocol.py —— Arena 平台官方提交协议入口（BFoV 输出）

> 2026-08-14 按 https://yjy-arena.insta360.cn/ 平台实测接口规范实现。

## 为什么需要它

平台要求镜像按官方接口跑评测：

| 项 | 官方要求 | 本项目旧协议（file_protocol.py） |
|---|---|---|
| 输入 | `/mnt/dataset/<seq>/video.mp4` + `init.txt`（BFoV） | `--frames` 帧目录 + `--init` x,y,w,h |
| 输出 | `/mnt/result/<seq>.txt`（BFoV：`clon,clat,fov_h,fov_v`） | `--out` x,y,w,h |
| 启动 | 容器无参自启动，退出码 0 | 带 CLI 参数 |
| 丢失帧 | `0,0,0,0` 占位（行号=帧号） | 兜底框 |

`arena_protocol.py` 是官方协议的完整实现：读 video.mp4 逐帧 + init.txt BFoV，
内部用 ODTrack ERP 三平铺推理，输出 BFoV 到 `/mnt/result`。

## 用法

```bash
# 容器内（平台评测方式，无参）：
docker run --rm --gpus all \
  -v <测试集>:/mnt/dataset:ro -v <输出>:/mnt/result <镜像>

# 本地调试（可覆盖路径 / 强制 CPU / 限制帧数）：
python integrations/odtrack/arena_protocol.py \
  --dataset /path/to/dataset --result /path/to/result \
  --workspace /opt/odtrack \
  --checkpoint /opt/models/ODTrack_ep0300.pth.tar \
  --force-cpu --max-frames 10
```

## 环境变量

- `DATASET_DIR`：数据集根目录（默认 `/mnt/dataset`）
- `RESULT_DIR`：结果输出目录（默认 `/mnt/result`）

## 输入 / 输出格式（与官方 demo 完全一致）

输入：
```
/mnt/dataset/
├── seqlist.txt          # 可选：每行一个序列名
├── seq_0001/
│   ├── video.mp4        # ERP 全景视频（逐帧解码即时间顺序）
│   └── init.txt         # 第 1 帧初始 BFoV：clon,clat,fov_h,fov_v
└── ...
```

输出：
```
/mnt/result/
├── seq_0001.txt         # 每行：clon,clat,fov_h,fov_v（行号=帧号）
└── ...
```

- BFoV 四元组均为角度（度）：`clon`∈[-180,180)，`clat`∈[-90,90]，`fov_h/fov_v`∈(0,180)
- 目标丢失/不可见帧输出 `0,0,0,0` 占位，不跳过、不留空行
- 首行恒为 init.txt 的初始 BFoV

## 关键实现

1. **BFoV↔ERP 转换**：内联纯 numpy（与 `panotrack/geometry/bfov.py` 同算法），
   镜像无需携带 panotrack 包；支持跨界框与极点。
2. **推理内核**：与 `file_protocol.py` 一致——ODTrack 上游 tracker + ERP 帧水平三平铺，
   首帧框移到中间副本，预测框横坐标折回 [0, W)。
3. **丢失帧判定**：`--lost-iou-threshold`（默认 0 = 关闭）。开启时用
   `tracker.last_pred_iou`（ODTrack 的 IoU 头质量信号）低于阈值判丢失 → `0,0,0,0`。
4. **单序列失败不中断整体**：记录到 stderr，全部完成后如有失败返回退出码 1。

## 验证

- `python tests/test_arena_protocol.py`：协议层（mock tracker）全链路验证，不依赖 ODTrack 权重
- 带真实 ODTrack 权重的 CPU 冒烟：见 `docs/ARENA_PROTOCOL_TEST_ZH.md`
- 镜像构建后离线自检：`docker run --rm --network none <镜像> --help`

## 与旧协议的关系

`file_protocol.py`（`--frames/--init/--out`，x,y,w,h）保留不动，供本地/历史流程使用；
提交镜像的 ENTRYPOINT 已切换为 `arena_protocol.py`。
