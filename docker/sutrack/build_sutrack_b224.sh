#!/bin/bash
# 构建 SUTrack-B224 cu128 提交镜像
# 用法: bash docker/sutrack/build_sutrack_b224.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="grt360-sutrack-b224"
TAG="2026-08-27_cu128"

echo "=== Building ${IMAGE_NAME}:${TAG} ==="
echo "Project root: ${PROJECT_ROOT}"

# 准备构建上下文
BUILD_CTX="${SCRIPT_DIR}/build_ctx"
rm -rf "${BUILD_CTX}"
mkdir -p "${BUILD_CTX}/integrations/sutrack"

# 复制 SUTrack 源码（排除 __pycache__ 和 .git）
echo "Copying SUTrack source..."
if [ -d "${PROJECT_ROOT}/artifacts/server_snapshot/upstream/SUTrack" ]; then
    rsync -a --exclude='__pycache__' --exclude='.git' \
        "${PROJECT_ROOT}/artifacts/server_snapshot/upstream/SUTrack/" \
        "${BUILD_CTX}/sutrack_src/"
elif [ -d "/data/sutrack_src_20260825/SUTrack" ]; then
    echo "Using remote SUTrack source (run on server)"
    rsync -a --exclude='__pycache__' --exclude='.git' \
        "/data/sutrack_src_20260825/SUTrack/" \
        "${BUILD_CTX}/sutrack_src/"
else
    echo "ERROR: SUTrack source not found!"
    echo "Expected: ${PROJECT_ROOT}/artifacts/server_snapshot/upstream/SUTrack/"
    exit 1
fi

# 复制权重
echo "Copying SUTrack-B224 checkpoint..."
if [ -f "${PROJECT_ROOT}/artifacts/hf/sutrack_b224/SUTRACK_ep0180.pth.tar" ]; then
    cp "${PROJECT_ROOT}/artifacts/hf/sutrack_b224/SUTRACK_ep0180.pth.tar" \
       "${BUILD_CTX}/SUTRACK_b224_ep0180.pth.tar"
elif [ -f "/data/weights/SUTRACK_b224_ep0180.pth.tar" ]; then
    echo "Using remote checkpoint (run on server)"
    cp "/data/weights/SUTRACK_b224_ep0180.pth.tar" \
       "${BUILD_CTX}/SUTRACK_b224_ep0180.pth.tar"
else
    echo "ERROR: SUTrack-B224 checkpoint not found!"
    exit 1
fi

# 复制协议入口
cp "${PROJECT_ROOT}/integrations/sutrack/arena_protocol_sutrack.py" \
   "${BUILD_CTX}/integrations/sutrack/"

# 复制 Dockerfile
cp "${SCRIPT_DIR}/Dockerfile.b224_cu128" "${BUILD_CTX}/Dockerfile"

# 构建
echo ""
echo "Building Docker image..."
docker build -t "${IMAGE_NAME}:${TAG}" "${BUILD_CTX}"

echo ""
echo "=== Build complete ==="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Size: $(docker image inspect "${IMAGE_NAME}:${TAG}" --format='{{.Size}}' | numfmt --to=iec 2>/dev/null || echo 'unknown')"
echo ""
echo "Test locally:"
echo "  docker run --rm -v /path/to/test:/mnt/dataset -v /tmp/result:/mnt/result ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with GPU:"
echo "  docker run --gpus 0 --rm -v /path/to/test:/mnt/dataset -v /tmp/result:/mnt/result ${IMAGE_NAME}:${TAG}"
