#!/usr/bin/env bash
# 构建 CUDA 12.8 版 UETrack 高速片镜像（支持 RTX 5090 Blackwell）。
# 用法: bash docker/uetrack/build_uetrack_cu128.sh [TAG]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-grt360-uetrack:2026-08-14-cu128}"

SRC="$ROOT/tools_local/uetrack_docker/external/UETrack"
WEIGHT="$ROOT/tools_local/uetrack_docker/models/uetrack_base.tar"
CLIP="$ROOT/tools_local/uetrack_docker/clip_cache/ViT-L-14.pt"
for path in "$SRC" "$WEIGHT" "$CLIP"; do
  if [ ! -e "$path" ]; then
    echo "missing build input: $path" >&2
    exit 2
  fi
done
if [ ! -f "$ROOT/integrations/uetrack/arena_protocol.py" ]; then
  echo "missing arena_protocol.py: $ROOT/integrations/uetrack/arena_protocol.py" >&2
  exit 2
fi

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

mkdir -p "$CTX/docker/uetrack" "$CTX/tools_local/uetrack_docker/external" "$CTX/tools_local/uetrack_docker/models" "$CTX/tools_local/uetrack_docker/clip_cache" "$CTX/integrations"
cp "$ROOT/docker/uetrack/Dockerfile.cu128" "$CTX/Dockerfile"
cp "$ROOT/docker/uetrack/requirements.txt" "$CTX/docker/uetrack/requirements.txt"
cp -r "$SRC" "$CTX/tools_local/uetrack_docker/external/UETrack"
cp "$WEIGHT" "$CTX/tools_local/uetrack_docker/models/uetrack_base.tar"
cp "$CLIP" "$CTX/tools_local/uetrack_docker/clip_cache/ViT-L-14.pt"
cp -r "$ROOT/integrations/uetrack" "$CTX/integrations/uetrack"
find "$CTX" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "build context: $CTX ($(du -sh "$CTX" | cut -f1))"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.8.2-base-ubuntu22.04}"
docker build --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$TAG" "$CTX"
echo "BUILT $TAG"
