#!/usr/bin/env bash
# 构建精简版 GRT-360 ODTrack 精度片镜像（grt360-odtrack-minimal，Arena 平台 BFoV 协议）。
#
# 与原 build_odtrack_image.sh 的差异：使用 Dockerfile.minimal
# （nvidia/cuda 精简基础镜像 + pip torch cu121，目标把 13.4GB 压到 ~7GB）。
# 上下文组装、权重去重、构建后清理逻辑与原脚本一致。
# 用法: bash docker/odtrack/build_odtrack_minimal.sh [TAG]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-grt360-odtrack-minimal:2026-08-14-arena}"

SRC="$ROOT/artifacts/server_snapshot/upstream/odtrack"
WEIGHT="$ROOT/artifacts/server_snapshot/weights/ODTrack_ep0300.pth.tar"
for path in "$SRC" "$WEIGHT"; do
  if [ ! -e "$path" ]; then
    echo "missing build input: $path" >&2
    exit 2
  fi
done

# 官方协议入口必须存在（arena_protocol.py 随 integrations/odtrack 一并 COPY）
if [ ! -f "$ROOT/integrations/odtrack/arena_protocol.py" ]; then
  echo "missing arena_protocol.py entry: $ROOT/integrations/odtrack/arena_protocol.py" >&2
  exit 2
fi

CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

mkdir -p "$CTX/integrations" "$CTX/scripts"
cp "$ROOT/docker/odtrack/Dockerfile.minimal" "$CTX/Dockerfile"
cp "$ROOT/docker/odtrack/requirements.txt" "$CTX/requirements.txt"
cp -r "$SRC" "$CTX/odtrack_src"
# 快照 output/ 内的权重副本与独立权重重复，删除之避免镜像体积翻倍
rm -f "$CTX/odtrack_src/output/checkpoints/train/odtrack/baseline/ODTrack_ep0300.pth.tar"
cp "$WEIGHT" "$CTX/ODTrack_ep0300.pth.tar"
cp -r "$ROOT/integrations/odtrack" "$CTX/integrations/odtrack"
cp "$ROOT/scripts/odtrack_360vot.py" "$CTX/scripts/"
find "$CTX" -name "__pycache__" -type d -prune -exec rm -rf {} +

echo "build context: $CTX ($(du -sh "$CTX" | cut -f1))"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04}"
docker build --platform linux/amd64 \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$TAG" "$CTX"
echo "BUILT $TAG"

# 离线自检：无参启动且断网（模拟平台评测方式）
echo "--- offline smoke (docker run --network none) ---"
docker run --rm --network none "$TAG" --help >/dev/null 2>&1 && \
  echo "--help OK" || echo "--help FAILED (entry may need GPU)"
