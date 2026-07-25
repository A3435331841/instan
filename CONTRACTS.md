# panotrack 接口契约（所有模块必须严格遵守）

> 项目：影石全景视频智能跟踪赛道 —— 360° ERP 全景视频实时单目标跟踪原型
> 运行环境：Python 3.12，仅依赖 **numpy / Pillow / scipy**（禁止 cv2、torch、yaml、pytest）。
> 测试为纯 assert 脚本，用 `python tests/test_xxx.py` 直接运行。
> 所有代码 UTF-8；注释用简洁中文；标识符用英文。
> **不要修改任何 `__init__.py`**；只创建/修改本模块范围内文件。

## 通用约定

- 图像：`np.ndarray`，uint8，形状 `(H, W, 3)`，RGB 顺序。全景图均为 ERP（等距柱状投影），宽 W = 2×高 H。
- ERP 边界框：`(x, y, w, h)` 浮点，像素单位。**跨界约定**：`x ∈ [0, W)`；若框跨越右边界（经线），`x + w > W`，超出部分回绕到左边缘（模 W）。y 方向不回绕。
- 角度单位一律为度（degree）。经度 lon ∈ (-180, 180]，纬度 lat ∈ [-90, 90]。lat=+90 为北极（图像顶行）。
- 所有公共函数必须带 docstring（一句话功能 + 参数 + 返回）。

## 模块 A：panotrack.geometry（几何工程师负责）

```
panotrack/geometry/sphere.py
panotrack/geometry/bfov.py
panotrack/geometry/projection.py
```

### sphere.py
- `wrap_lon(lon) -> float|np.ndarray`：归一化到 (-180, 180]。
- `delta_lon(d) -> float|np.ndarray`：经度差归一化到 (-180, 180]（环绕差分）。
- `lonlat_to_unit(lon, lat) -> (x, y, z)`：球面经纬度转单位向量（y 轴指向北极）。
- `unit_to_lonlat(x, y, z) -> (lon, lat)`：单位向量转经纬度（自动 wrap）。

### bfov.py
- `@dataclass BFoV`：字段 `lon, lat, fov_h, fov_v, rotation=0.0`（均为度）；BFoV 表示以 (lon,lat) 为中心、水平/垂直视场角为 fov_h/fov_v 的球面窗口。
- `bfov_from_erp_bbox(x, y, w, h, erp_w, erp_h) -> BFoV`：由 ERP 框估算 BFoV（采样框边界点转球面，取中心与角跨度；跨界框先对 x 做模 W 展开）。
- `erp_bbox_from_bfov(bfov, erp_w, erp_h, samples=48) -> (x, y, w, h)`：BFoV 边界采样投影回 ERP，取最小面积轴对齐框，跨界时 `x+w` 可超过 W。

### projection.py
- `tangent_remap(bfov, out_w, out_h, erp_w, erp_h) -> (map_x, map_y)`：
  返回 float32 数组 `(out_h, out_w)`，每个输出像素给出 ERP 源坐标（map_x 已模 W 回绕，map_y clamp 到 [0, H-1]）。
  fov ≤ 90° 用切平面（gnomonic）投影；fov > 90° 用 eBFoV 球面均匀角采样。rotation 暂不实现（保留参数）。
- `remap_image(img, map_x, map_y) -> np.ndarray`：双线性重采样，支持 (H,W) 与 (H,W,3)；水平回绕、垂直 clamp；输出 dtype 与输入一致。
- `local_bbox_to_erp(lx, ly, lw, lh, map_x, map_y, erp_w, erp_h) -> (x, y, w, h)`：把局部（切图）坐标框经 map 逆投影回 ERP 最小面积框（采样边界 ≥16 点/边；跨界时 x+w 可超 W）。
- `class RemapCache`：`get_remap(bfov, out_w, out_h, erp_w, erp_h) -> (map_x, map_y)`，按量化键 (lon/2°, lat/2°, fov/2°) LRU 缓存（容量可配，默认 64）。

## 模块 B：panotrack.trackers（跟踪器工程师负责）

```
panotrack/trackers/base.py
panotrack/trackers/ncc.py
panotrack/trackers/factory.py
```

### base.py
- `class BaseTracker(ABC)`：
  - `init(image, bbox) -> None`：image 为 (H,W,3) uint8（**局部透视图，非全景**），bbox=(x,y,w,h) 局部像素框。
  - `update(image) -> dict`：返回 `{'bbox': (x,y,w,h), 'score': float ∈[0,1], 'psr': float, 'apce': float}`。

### ncc.py
- `class NCCTracker(BaseTracker)`：
  `__init__(self, context=1.0, scales=(0.98,1.0,1.02), lr=0.02, search_scale=2.0, template_size=127)`。
  纯 numpy FFT 归一化互相关模板匹配（SiamFC 思想经典版）：余弦窗预处理、多尺度搜索、门控模板更新（score 低于阈值不更新）。`score` 为峰值 NCC 相关度。
- 必须线程无关、无全局状态；大补丁下复杂度可控（FFT 实现）。

### factory.py
- `create_tracker(name='ncc', **kwargs) -> BaseTracker`；`name='lightfc'` 时抛出 `NotImplementedError` 并在消息中说明"待接入 LightFC 深度学习模型（需 torch/onnxruntime），接口已预留"。
- `name='direct_erp'` 时返回 `DirectERPTracker`，直接在全帧 ERP 上运行 VitTrack，绕过 BFoV 框架。

### direct_erp.py
- `class DirectERPTracker(BaseTracker)`：
  - `init(image, bbox)`：image 为 (H,W,3) uint8 ERP 全帧；bbox=(x,y,w,h) ERP 坐标。
  - `update(image) -> dict`：返回 `{'bbox': (x,y,w,h), 'score': float, 'psr': float, 'apce': float}`。
  - 自动处理 360° 边界穿越：x 坐标回绕到 [0, erp_w)。
  - 注意：输入图像应为原始 ERP 帧，**不需要** highpass 滤波。

## 模块 C：panotrack.evaluation + panotrack.data（评测与数据工程师负责）

```
panotrack/evaluation/metrics.py
panotrack/evaluation/runner.py
panotrack/data/synth.py
panotrack/data/io.py
panotrack/data/viz.py
```

### metrics.py
- `iou_xywh(b1, b2) -> float`：普通 IoU（不跨界）。
- `dual_iou(b1, b2, width) -> float`：360VOT dual IoU —— b1 水平平移 ±width 后与 b2 的 IoU 取最大（处理跨界）。
- `success_rate(ious, thr=0.5) -> float`
- `auc(ious) -> float`：阈值 0~1 步长 0.05（21 点）的 SR 均值。
- `ope_evaluate(pred, gt, width) -> dict`：pred/gt 为 (N,4)；返回 `{'sr','auc','sr_dual','auc_dual','ious','ious_dual'}`。首帧（初始化帧）不计入统计。

### runner.py
- `run_tracker_on_sequence(tracker, frames, gt) -> np.ndarray (N,4)`：OPE 协议 —— 用 gt[0] 初始化，逐帧 update；要求 tracker 有 `init(image, bbox)` / `update(image) -> dict 含 'bbox'`（PanoTracker 与本契约的 BaseTracker 都满足）。

### synth.py（自包含，不依赖 geometry 模块）
- `generate_sequence(out_dir, n_frames=60, w=1024, h=512, scenario=..., seed=0) -> str`：
  scenario ∈ `{'equator','crossing','pole','occlusion'}`。
  生成合成 ERP 序列：背景为经纬网格渐变 + 固定噪声纹理；目标为带独特纹理的矩形块，在 ERP 像素域按场景运动（crossing 需跨右边界回绕绘制；pole 场景目标接近顶行时按 1/cos(lat) 拉宽压扁模拟 ERP 拉伸；occlusion 中段 5~10 帧被遮挡块盖住）。
  输出：`out_dir/frames/%06d.png`、`out_dir/gt.txt`（每行 `x,y,w,h`，跨界约定同契约）；被完全遮挡帧 GT 与遮挡框一致（保持逐帧对齐）。
- `make_gallery(...)` 不需要。

### io.py
- `load_sequence(seq_dir) -> (frames: list[np.ndarray], gt: np.ndarray)`：读取 frames/*.png 与 gt.txt。

### viz.py
- `draw_bbox(img, bbox, color=(0,255,0), thickness=2) -> np.ndarray`：画框，跨界时自动拆成左右两段绘制；纯 numpy/PIL 实现，不修改原图。
- `save_gif(frames, out_path, fps=10) -> None`：PIL 保存 GIF。

## 模块 D：panotrack.io + 打包（工程化工程师负责）

```
panotrack/io/file_protocol.py
panotrack/io/trax_protocol.py
panotrack/cli.py
docker/Dockerfile
docker/entrypoint.sh
requirements.txt
.dockerignore
configs/default.json
README.md
```

### file_protocol.py
- `run_file_protocol(frames_dir, init_file, out_file, config=None) -> dict`：
  读取 `frames_dir` 下按文件名排序的图像（png/jpg），`init_file` 首行 `x,y,w,h`；
  创建 `PanoTracker(config)`（**from panotrack.pipeline.pipeline import PanoTracker**，契约见下）；
  逐帧把结果以 `x,y,w,h`（保留 2 位小数）追加写入 `out_file`；返回耗时与帧率统计。
  调试日志一律 `print(..., file=sys.stderr)`，stdout 保持干净。

### trax_protocol.py
- 最小可用的 trax 风格 stdin/stdout 适配（行协议，文档内注明为占位实现，8 月官方协议公布后替换）； stdout 只输出协议行。

### cli.py
- `python -m panotrack.cli --frames DIR --init init.txt --out results.txt [--config configs/default.json] [--visualize OUT_DIR]`

### 打包
- `docker/Dockerfile`：`python:3.12-slim`，安装 requirements，COPY panotrack，`ENTRYPOINT ["python","-m","panotrack.cli"]`；注释说明断网自包含原则与 `docker build --platform linux/amd64`。
- `requirements.txt`：`numpy`、`Pillow`、`scipy`（给最低版本）。
- `configs/default.json`（键即 PanoTracker config 键）：
  ```json
  {"tracker": "ncc", "patch_size": 255, "sr_ratio": 3.0, "sr_min_fov": 20.0,
   "lost_score": 0.45, "lost_psr": 7.0, "lost_apce": 0.0,
   "redetect_interval": 1, "max_lost_frames": 1000000, "template_size": 127,
   "search_scale": 2.0, "lr": 0.02}
  ```
- `README.md`：中文完整文档（项目简介、目录结构、快速开始：合成数据 → 跑 demo → 评测；Docker 构建运行；接口契约摘要；LightFC 接入指南；赛事对接清单）。

## 模块 E：panotrack.pipeline（集成阶段实现，接口先行冻结）

```
panotrack/pipeline/state.py
panotrack/pipeline/redetect.py
panotrack/pipeline/pipeline.py
```

- `class PanoTracker`：
  - `__init__(self, config: dict | None = None)`：缺省值同 configs/default.json。
  - `init(self, frame, bbox) -> None`：frame 为 ERP (H,W,3) uint8，bbox 遵循跨界约定。
  - `update(self, frame) -> dict`：`{'bbox': (x,y,w,h) 跨界约定, 'score': float, 'status': 'ok'|'lost'|'recovered', 'fov': (fov_h, fov_v)}`。
  - 内部流程：球面状态预测 → tangent 切图 → 局部 tracker 更新 → 逆投影回 ERP → 置信度判丢 → 逐级扩大 FoV 重试 → 超限后全局重检测。
- `class SphericalState`：`predict() -> BFoV`；`update(measured: BFoV) -> None`；恒定角速度 + 指数平滑 + 阻尼；属性 `bfov`。
- `class GlobalRedetector`：`__init__(self, get_template)`；`search(frame, erp_downscale=4) -> ((x,y,w,h), score) | None`。

## 端到端验收标准（集成阶段）

1. `python tests/test_geometry.py / test_trackers.py / test_metrics.py / test_synth.py` 全部通过。
2. `python demo/run_demo.py` 在 4 个合成场景（equator/crossing/pole/occlusion）上端到端跑通：
   - equator 与 crossing：SR@0.5 ≥ 0.9（双口径）；
   - pole：SR@0.5 ≥ 0.7；
   - occlusion：丢失后 10 帧内找回（统计 recovered 帧）。
3. 结果输出到 `runs/<scenario>/`：results.txt、metrics.json、demo.gif。
4. `python -m panotrack.cli` 文件协议跑通。
