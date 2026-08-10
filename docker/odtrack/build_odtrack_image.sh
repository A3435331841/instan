#!/usr/bin/env bash
# 构建 GRT-360 ODTrack 精度版镜像（grt360-odtrack）。
#
# ODTrack 上游源码与权重存放在 artifacts/server_snapshot/（.dockerignore 排除），
# 因此本脚本组装临时构建上下文再构建，不改动全局 .dockerignore，构建后自动清理。
# 用法: bash tools_local/build_odtrack_image.sh [TAG]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:-grt360-odtrack:2026-08-10}"

SRC="$ROOT/artifacts/server_snapshot/upstream/odtrack"
WEIGHT="$ROOT/artifacts/server_snapshot/weights/ODTrack_ep0300.pth.tar"
for path in "$SRC" "$WEIGHT"; do
  if [ ! -e "$path" ]; then
    echo "missing build input: $path" >&2
    exit 2
  fi
done

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

mkdir -p "$CTX/integrations" "$CTX/scripts"
cp "$ROOT/docker/odtrack/Dockerfile" "$CTX/Dockerfile"
cp "$ROOT/docker/odtrack/requirements.txt" "$CTX/requirements.txt"
cp -r "$SRC" "$CTX/odtrack_src"
# 快照 output/ 内的权重副本与独立权重重复，删除之避免镜像体积翻倍
rm -f "$CTX/odtrack_src/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar"
cp "$WEIGHT" "$CTX/ODTrack_ep0300.pth.tar"
cp -r "$ROOT/integrations/odtrack" "$CTX/integrations/odtrack"
cp "$ROOT/scripts/odtrack_360vot.py" "$CTX/scripts/"
find "$CTX" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "build context: $CTX ($(du -sh "$CTX" | cut -f1))"
# 国内环境无法访问 docker.io 时，使用本地 daocloud 镜像缓存
BASE_IMAGE="${BASE_IMAGE:-pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime}"
docker build --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$TAG" "$CTX"
echo "BUILT $TAG"
