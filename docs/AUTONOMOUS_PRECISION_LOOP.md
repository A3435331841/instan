# GRT-360 本地自主精度闭环

`scripts/autonomous_precision_controller.py` 是本地实验控制入口。它把当前结果根目录当作只读证据，自动生成快照、设备清单、失败诊断、场景均值和下一项实验建议；带 `--apply` 时才会启动白名单中的 OpenVINO B224 批量评测。

## 安全边界

- 不删除本地或远端文件。
- 不执行比赛仓库 Docker push。
- 检测到已有 B224 batch/sequence 进程时拒绝并发启动。
- 只把 train95/代表性序列用于实验；valid35/full130 结果必须锁定后再验收。
- 运行时路由只能使用模型响应、尺度、运动和几何信号，不能使用序列名或 GT。

## 只读诊断

```powershell
$env:PYTHONPATH='D:\instan\grt360_scratch\intel_runtime_probe_20260827\Lib\site-packages'
python scripts/autonomous_precision_controller.py `
  --results D:\instan\grt360_scratch\b224_full_precision_v2_20260827 `
  --data D:\instan\grt360_storage\datasets\official_train\train `
  --xml D:\instan\grt360_scratch\openvino\sutrack_b224.xml `
  --high-xml D:\instan\grt360_scratch\openvino\sutrack_b224_s224_t128.xml `
  --out D:\instan\grt360_scratch\autonomous_precision
```

输出包括 `autonomous_run_manifest.json`、`scenario_summary.csv`、`failure_notes.md`、`next_experiment.json` 和 `promotion.json`。

## 启动下一轮

默认 `--apply-scope micro` 只跑当前最差序列和一个正常对照；`cluster` 跑相似标签的最多 8 条；`full` 扫描完整 130 条。已有 GPU 任务结束后，才允许启动：

```powershell
python scripts/autonomous_precision_controller.py `
  --results D:\instan\grt360_scratch\b224_full_precision_v2_20260827 `
  --data D:\instan\grt360_storage\datasets\official_train\train `
  --xml D:\instan\grt360_scratch\openvino\sutrack_b224.xml `
  --high-xml D:\instan\grt360_scratch\openvino\sutrack_b224_s224_t128.xml `
  --out D:\instan\grt360_scratch\autonomous_precision `
  --apply --apply-scope micro
```

每轮晋级门：单序列困难样本 AUC 至少 +0.10 且正常对照回退不超过 0.01；场景簇平均 +0.05、胜率至少 60% 且至少 3 条独占救援；最终 full130 同时满足 AUC>0.8、SR>0.8、端到端 FPS>30。

当前本地 OpenVINO 暴露 `CPU` 和 `GPU`；若后续运行时暴露 `NPU`，控制器会自动记录但不会伪造不可用设备的成绩。
