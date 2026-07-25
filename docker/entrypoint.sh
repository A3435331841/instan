#!/usr/bin/env bash
# panotrack 容器辅助入口脚本（联调/冒烟用）。
# 注意：镜像正式 ENTRYPOINT 为 python -m panotrack.cli（见 Dockerfile）。
# 本脚本用于手工协议联调，例如：
#   docker run --rm --entrypoint /entrypoint.sh panotrack:latest file \
#       --frames /data/frames --init /data/init.txt --out /data/results.txt
#   docker run --rm -i --entrypoint /entrypoint.sh panotrack:latest trax < cmds.txt
set -euo pipefail

MODE="${1:-file}"
if [ $# -gt 0 ]; then shift; fi

case "$MODE" in
  file)
    # 文件协议：等价于默认 ENTRYPOINT
    exec python -m panotrack.cli "$@"
    ;;
  trax)
    # trax 风格行协议（占位实现，8 月官方协议公布后替换）
    exec python -m panotrack.io.trax_protocol
    ;;
  *)
    echo "usage: entrypoint.sh [file|trax] [args...]" >&2
    exit 2
    ;;
esac
