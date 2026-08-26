# GRT-360 恢复说明

## 路径变量

```powershell
$env:GRT360_STORAGE_ROOT = 'D:\instan\grt360_storage'
$env:OFFICIAL_TRAIN_ROOT = 'D:\instan\grt360_storage\datasets\official_train'
$env:GRT360_CHECKPOINT_ROOT = 'D:\instan\grt360_storage\checkpoints'
$env:GRT360_EXPERIMENT_ROOT = 'D:\instan\grt360_storage\experiments'
```

如果仍使用旧目录布局，仓库原有默认路径也可以继续使用。整理脚本会保留兼容Junction，
直到路径验证完成。

## 完整性校验

在本地归档根目录执行：

```powershell
Import-Csv .\SHA256SUMS.csv | Where-Object { $_.type -eq 'file' -or -not $_.type } | Where-Object { $_.local } | ForEach-Object {
  $actual = (Get-FileHash -LiteralPath $_.local -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -ne $_.sha256) { throw "SHA256 mismatch: $($_.local)" }
}
```

`server_exit_manifest.json`中的`tree_archives`记录高文件数目录的tar包哈希；
`remote_SHA256SUMS`是远端归档树的逐文件哈希清单。若需要重新核对树目录，
先验证对应 `.tar`，再按 `tar_*_manifest.json`中的`file_count`检查展开结果。

## 最小恢复验证

```powershell
Set-Location D:\instan\pano360
python -m py_compile scripts\eval_official.py
python tests\test_geometry.py
python tests\test_s2_state.py
python tests\test_metrics.py
```

权重必须从归档或GitHub Release挂载，不得把二进制写入普通Git历史。
