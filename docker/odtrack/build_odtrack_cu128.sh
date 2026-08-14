#!/usr/bin/env bash
# 构建 CUDA 12.8 版 GRT-360 ODTrack 精度片镜像（支持 RTX 5090 Blackwell）。
# 用法: bash docker/odtrack/build_odtrack_cu128.sh [TAG]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-grt360-odtrack:2026-08-14-cu128}"

SRC="$ROOT/artifacts/server_snapshot/upstream/odtrack"
WEIGHT="$ROOT/artifacts/server_snapshot/weights/ODTrack_ep0300.pth.tar"
for path in "$SRC" "$WEIGHT"; do
  if [ ! -e "$path" ]; then
    echo "missing build input: $path" >&2
    exit 2
  fi
done
if [ ! -f "$ROOT/integrations/odtrack/arena_protocol.py" ]; then
  echo "missing arena_protocol.py: $ROOT/integrations/odtrack/arena_protocol.py" >&2
  exit 2
fi

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

mkdir -p "$CTX/integrations" "$CTX/scripts"
cp "$ROOT/docker/odtrack/Dockerfile.cu128" "$CTX/Dockerfile"
cp "$ROOT/docker/odtrack/requirements.txt" "$CTX/requirements.txt"
cp -r "$SRC" "$CTX/odtrack_src"
rm -f "$CTX/odtrack_src/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar"
cp "$WEIGHT" "$CTX/ODTrack_ep0300.pth.tar"
cp -r "$ROOT/integrations/odtrack" "$CTX/integrations/odtrack"
cp "$ROOT/scripts/odtrack_360vot.py" "$CTX/scripts/"
find "$CTX" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "build context: $CTX ($(du -sh "$CTX" | cut -f1))"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.2-base-ubuntu22.04}"
docker build --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$TAG" "$CTX"
echo "BUILT $TAG"
