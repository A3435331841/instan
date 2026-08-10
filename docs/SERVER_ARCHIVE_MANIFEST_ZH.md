# 服务器归档清单（2026-08-10）

本清单用于服务器 `153.0.134.134:12409` 到期前的最终核对。代码、配置和说明文档进入 GitHub；模型权重、赛马逐序列输出和离线镜像等大文件保存在本地 `artifacts/` 或 `runs/server_snapshot/`，不直接提交到 GitHub。

## 已在本地保存且与服务器校验一致

| 内容 | 服务器来源 | 本地位置 | SHA256 |
|---|---|---|---|
| LightFC 120 序列结果 | `/data/projects/instan/runs/grt360_20260809/lightfc_results_0001_0120.tar.zst` | `runs/server_snapshot/grt360_20260809/lightfc_results_0001_0120.tar.zst` | `0c608c0ea28177d6d75aaca0cfb8097f3d327bf9afc97166b8cad8a705445aa7` |
| ODTrack 120 序列结果 | `/data/projects/instan/runs/grt360_20260809/odtrack_results_0001_0120.tar.zst` | `runs/server_snapshot/grt360_20260809/odtrack_results_0001_0120.tar.zst` | `15a2f551dea0043466a24c0620601d321d3bc3dff63301239aa6765c8e32a1a9` |
| UETrack 120 序列结果 | `/data/projects/instan/runs/grt360_20260809/uetrack_results_0001_0120.tar.zst` | `runs/server_snapshot/grt360_20260809/uetrack_results_0001_0120.tar.zst` | `b255106a2e3711612a9e2aec86d8ba5ec45c3971049f847c6f520a2d4f7810e8` |
| ODTrack 300 轮权重 | `/data/projects/instan_check/odtrack/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar` | `artifacts/server_snapshot/weights/ODTrack_ep0300.pth.tar` | `2fba6ddeb826014ac0bb871623406d16c3a162afbf09accb49312b526c21068e` |
| UETrack small 权重 | `/data/projects/instan_check/uetrack_weights/uetrack_small.tar` | `artifacts/server_snapshot/weights/uetrack_small.tar` | `06eb9b583ec2389b6939b96f166270a31b53d3db210b161553ca5259a979574a` |
| UETrack tiny 权重 | `/data/projects/instan_check/uetrack_weights/uetrack_tiny.tar` | `artifacts/server_snapshot/weights/uetrack_tiny.tar` | `c76de29721ab6940a1f4f56e295fae62c5bf49d8153f2a937a54a49e5240f122` |
| UETrack SUTRACK 300 轮权重 | `/data/projects/instan_check/uetrack_weights/SUTRACK_ep0300.pth.tar` | `artifacts/server_snapshot/weights/SUTRACK_ep0300.pth.tar` | `944d34f3266771ac8f5d2f85a781ccc35c340f226f81872b57b34f4c0206b9e3` |
| FastITPN tiny 权重 | `/data/projects/instan_check/uetrack_weights/fast_itpn_tiny_1600e_1k.pt` | `artifacts/server_snapshot/code/fast_itpn_tiny_1600e_1k.pt` | `5f144a02627d08230e5e336f431a1a27fe6dbe94d5172400a30f5232ce84790a` |
| LoRAT checkpoint | `/data/projects/instan_check/base.bin` | `artifacts/server_snapshot/code/base.bin` | `150edc6635c7615a82d7fd50d95d84f8e47a47c9217e8fd5b3dd326589aac23e` |
| LoRAT 源码压缩包 | `/data/projects/instan_check/lorat_code.tgz` | `artifacts/server_snapshot/code/lorat_code.tgz` | `53c1081c09e7688629bb69865d902b4f8e3d02a73e3446e36fc27b4036cb909c` |

LoRAT 压缩包已解压到 `artifacts/server_snapshot/code/lorat_source/`，便于离线查阅。完整赛马目录还包含 1771 个服务器文件；本地快照目录中已覆盖服务器文件，另保留了本地先前生成的少量合并输出。

服务器上的完整上游源码也已保存到 `artifacts/server_snapshot/upstream/`：

- `odtrack/`：226 个文件，含上游代码和训练权重副本。
- `UETrack/`：254 个文件，含实验配置和模型代码。
- `lorat/`：1028 个文件，含 LoRAT 训练/评测源码。

## GitHub 中已有的内容

- `panotrack/` 核心代码、360VOT 数据协议、评测指标和所有融合器实现。
- `integrations/odtrack/`、`integrations/uetrack/` 适配器及上游版本记录。
- `scripts/`、`configs/`、`tests/`、中文报告和 Docker 构建文件。
- 120 序列汇总指标、架构对比、消融实验和最终交付说明（见 `reports/`、`docs/`）。

GitHub 不存放上述大权重、逐帧结果和 Docker tar；这些路径由 `.gitignore` 明确排除，避免仓库膨胀和超过 GitHub 单文件限制。

## 仍未搬运、且不建议在服务器到期前强行搬运的内容

- `/data/projects/instan/data360`：约 62GB 数据集。仓库只保留下载/校验脚本和 `data360/` 说明，不复制数据本体。
- 服务器运行环境缓存、Python `__pycache__`、临时日志和重复的中间文件。
- 上游源码不直接提交 GitHub：本地已保存完整快照，但其许可证和上游版本仍应以官方仓库为准，不将带训练产物的整目录塞入 GitHub。

## 本地复核命令

在 PowerShell 中执行：

```powershell
Get-FileHash artifacts/server_snapshot/weights/ODTrack_ep0300.pth.tar -Algorithm SHA256
Get-FileHash artifacts/server_snapshot/weights/SUTRACK_ep0300.pth.tar -Algorithm SHA256
Get-ChildItem runs/server_snapshot/grt360_20260809 -Recurse -File | Measure-Object Length -Sum
```

如需恢复实验，优先使用仓库中的 Docker/适配器，再从本清单对应的本地权重路径挂载模型；不需要服务器在线。
