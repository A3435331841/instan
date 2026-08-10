# ODTrack ERP 三平铺适配

这是 GRT-360 当前精度冠军的可复现入口。适配器将每个 ERP 帧水平复制三次，
让跨越 `0/360°` 接缝的搜索窗口保持连续，输出时再将横坐标归一化回原始 ERP 宽度。

## 运行

```bash
python integrations/odtrack/run_erp.py \
  --odtrack-root /path/to/ODTrack \
  --data /data/data360 \
  --checkpoint /models/ODTrack.pth.tar \
  --out /results/odtrack_erp \
  --gpu 0
```

`--odtrack-root` 必须是已安装的官方 ODTrack 源码目录，`--checkpoint` 必须是
官方权重或经过授权的内部权重。本仓库只保存适配器和实验协议，不把上游源码、
大模型权重或授权不明确的文件提交到 GitHub。

## 已验证结果

- 360VOT：120/120 序列；
- 普通 AUC：0.5792135073；
- 普通 SR：0.6531941586；
- 两张 RTX 3090 并行端到端 FPS：约 8.99。

完整中文交付报告见 `docs/FINAL_DELIVERY_ZH.md`。
