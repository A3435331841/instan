# GRT-360 三条架构路线（中文）

更新时间：2026-08-10

## 必须保留的两个基线

### 精度基线：ODTrack ERP 三平铺

- AUC：`0.5792135073`
- SR：`0.6531941586`
- FPS：`8.9945`
- 用途：最终精度上限、教师标签、困难序列复核。
- 配置：`configs/grt360_best_accuracy.json`
- 入口：`integrations/odtrack/run_erp.py`

### 高速基线：UETrack ERP-wrap

- AUC：`0.5142648726`
- SR：`0.5776136689`
- FPS：`57.1606`
- 用途：实时运行基线、学生模型初始化、速度上限参考。
- 配置：`configs/grt360_fast_uetrack.json`
- 入口：`integrations/uetrack/run_erp.py --erp-wrap`

这两个版本不删除、不互相替代，所有新实验必须同时和它们比较。

## 创新路线：GRT360-Causal-DTP-ERP

当前候选版本是预测级原型，默认保守地依赖 ODTrack 教师，因此端到端速度约
8.99 FPS。真正的创新目标不是继续调后处理阈值，而是：

1. UETrack 学生每帧运行，保持接近 57 FPS；
2. 通过因果 KV-cache 保留历史状态；
3. 通过 DTP 可靠性头预测漂移风险；
4. 只有低可靠性或重捕获事件才调用 ODTrack；
5. 用 ODTrack 软标签和 360 接缝/极区增强训练学生模型。

验收标准：新版本必须同时报告普通 AUC、SR、端到端 FPS、教师调用比例和失败序列，
并且不能只用已有测试集调参后宣称最终提升。
