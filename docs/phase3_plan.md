# 阶段 3 官方数据接入方案

## 1. 数据集概览

| 属性 | 详情 |
|------|------|
| **数据集名称** | 360VOTS (360VOT + 360VOS) |
| **官方主页** | https://360vots.hkustvgd.com/ |
| **GitHub** | https://github.com/HuajianUP/360VOT |
| **Hugging Face** | https://huggingface.co/datasets/xuyzshaun/360VOTS |
| **序列数量** | 120 (测试集) + 170 (训练集) |
| **分辨率** | 3840 × 1920 (4K ERP) |
| **帧数** | 最多 113K 帧 |
| **标注格式** | BBox, rBBox, BFoV, rBFoV |
| **新指标** | dual success rate, dual precision, angle precision |

## 2. 数据下载策略

### 2.1 优先下载测试集 (120 序列)

```bash
# 方式 1: Hugging Face datasets 库
pip install datasets
python -c "from datasets import load_dataset; load_dataset('xuyzshaun/360VOTS', split='test')"

# 方式 2: 直接下载 zip (适合离线环境)
# 访问 https://huggingface.co/datasets/xuyzshaun/360VOTS/tree/main/test
# 下载 0001.zip ~ 0120.zip
```

### 2.2 目录结构规划

```
D:/instan/pano360/
├── data360/
│   ├── official/                    # 官方完整数据集
│   │   ├── 0001/
│   │   ├── 0002/
│   │   ├── ...
│   │   └── 0120/
│   ├── train/                       # 训练集 (170 序列，后期使用)
│   └── test/                        # 测试集 (120 序列)
├── results/                         # 官方格式评测结果
│   └── our_method/
│       ├── 0001.txt
│       ├── ...
│       └── 0120.txt
└── tools_official/                  # 官方评测工具
    └── 360VOT/
```

## 3. 评测协议对齐

### 3.1 官方评测指标

| 指标 | 含义 | 我们的实现 |
|------|------|-----------|
| **AUC** | 阈值 0~1 步长 0.05 的 SR 均值 | ✅ `metrics.auc()` |
| **AUC_dual** | 考虑 ±width 跨界的 AUC | ✅ `metrics.dual_iou()` |
| **Precision** | 中心点距离阈值下的成功率 | ⚠️ 需验证 |
| **Precision_dual** | 考虑跨界的 Precision | ⚠️ 需验证 |
| **Angle Precision** | 球面角度精度 | ❌ 需实现 |

### 3.2 结果格式要求

官方评测要求 **BBox 格式**：`[x1 y1 w h]`，每行一个框，共 N 行（N = 序列帧数）。

我们的 `run_tracker_on_sequence` 已输出 `(N, 4)` 数组，可直接保存为 txt。

### 3.3 对齐步骤

1. **下载官方评测脚本**
   ```bash
   cd D:/instan/pano360/tools_official
   git clone https://github.com/HuajianUP/360VOT.git
   ```

2. **格式化我们的结果**
   ```python
   # 将 preds (N,4) 保存为官方格式
   np.savetxt(f"results/our_method/{seq_name:04d}.txt", preds, fmt="%.4f")
   ```

3. **运行官方评测**
   ```bash
   python 360VOT/scripts/eval_360VOT.py \
       -d D:/instan/pano360/data360/official \
       -b results/our_method \
       -v  # 打印详细指标
   ```

## 4. 核心问题：BFoV 框架 vs 直接 ERP 跟踪

### 4.1 当前性能对比 (50 frames, downscale=0.5)

| 方法 | Mean AUC | Mean AUC_dual | 问题 |
|------|----------|---------------|------|
| NCC baseline | 0.11 | 0.12 | 无学习，精度低 |
| BFoV + VitTrack (旧) | 0.25 | 0.26 | 漂移严重 |
| BFoV + VitTrack (修复后) | 0.13 | 0.14 | 漂移依然存在 |
| **直接 ERP + VitTrack** | **0.37** | **0.37** | ✅ 无漂移 |

### 4.2 结论

**BFoV 框架的漂移问题比小目标稀释更严重**：
- 状态预测累积误差 → 切图窗口偏离目标
- 局部 tracker 在错误切图中跟踪 → 输出错误 bbox
- 错误 bbox 污染 state → 下一帧预测更不准
- 恶性循环，到 frame 7-8 时完全丢失

### 4.3 建议方案

**方案 A：直接全帧 ERP 跟踪（推荐）**
- ✅ 避免 BFoV 框架漂移
- ✅ 保持 VitTrack 完整上下文
- ✅ 简单可靠，易于部署
- ❌ 360° 图像宽高比 2:1，VitTrack 的搜索区域可能受限
- ❌ 大 FoV 目标可能超出搜索范围

**方案 B：改进 BFoV 框架**
- 定期重置 tracker（每 N 帧重新 init）
- 使用更复杂的状态模型（Kalman filter）
- 增加全局重检测频率
- ❌ 复杂度高，效果不确定

## 5. 实施计划

### Phase 1: 数据接入 (本周)
1. [ ] 下载官方 120 序列测试集到 `data360/official/`
2. [ ] 克隆官方评测工具到 `tools_official/360VOT/`
3. [ ] 验证数据加载器兼容官方格式
4. [ ] 运行官方评测脚本，对齐 baseline

### Phase 2: 直接 ERP 跟踪优化 (下周)
1. [x] 实现 DirectERPTracker 类（替代 BFoV 框架）
2. [ ] 测试 120 序列全量数据
3. [x] 处理 360° 边界穿越问题（水平回绕）
4. [ ] 优化 FPS（当前 ~10-30 FPS，目标 >30 FPS）

### Phase 3: 小目标增强 (可选)
1. [ ] 实现多尺度测试（image pyramid）
2. [ ] 尝试 LightFC/LightTrack（更小目标友好）
3. [ ] 集成 YOLO 检测器做 fallback

## 6. Docker 离线部署

### 6.1 当前依赖

```dockerfile
FROM python:3.11-slim
RUN pip install opencv-python-headless numpy scipy pillow onnxruntime
# 生产环境用 vittrack_onnx.py (纯 numpy + onnxruntime)
```

### 6.2 离线运行要求

- 所有模型文件打包进镜像
- 数据集预下载到挂载卷
- 评测脚本内置（不依赖外部网络）

## 7. 下一步行动

1. **立即**：下载官方 120 序列测试集
2. **本周**：实现 DirectERPTracker 并完成全量评测
3. **下周**：根据评测结果决定是否需要 BFoV 框架改进
