# GRT-360 本地资产路径（复制到个人会话中执行，不要提交含个人路径的 .env）
$env:GRT360_STORAGE_ROOT = 'D:\instan\grt360_storage'
$env:OFFICIAL_TRAIN_ROOT = 'D:\instan\grt360_storage\datasets\official_train\train'
$env:GRT360_CHECKPOINT_ROOT = 'D:\instan\grt360_storage\checkpoints'
$env:GRT360_EXPERIMENT_ROOT = 'D:\instan\grt360_storage\experiments'

# 旧路径 Junction 仍可用；新脚本优先读取上面的变量。
