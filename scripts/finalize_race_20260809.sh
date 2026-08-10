#!/usr/bin/env bash
set -u

# 历史服务器赛马收尾脚本（路径通过环境变量覆盖）。
# 该脚本只负责等待两个分片完成并调用统一评分器；不会删除任何结果。
BASE="${GRT360_RUNS:-/data/projects/instan/runs/grt360_20260809}"
CODE="${GRT360_CODE:-/data/projects/instan_grt360}"
PYTHON="${GRT360_PYTHON:-python}"
LOG="$BASE/finalize_race.log"
mkdir -p "$BASE"
exec > >(tee -a "$LOG") 2>&1

echo "[finalize] waiting for 120 LightFC metrics"
while :; do
  n=$(find "$BASE/lightfc_120" -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null | wc -l)
  echo "[finalize] LightFC metrics=$n/120"
  (( n >= 120 )) && break
  sleep 60
done

cd "$CODE"
"$PYTHON" scripts/score_external_results.py \
  --data "${GRT360_DATA:-/data/projects/instan/data360}" \
  --tracker LightFC="$BASE/lightfc_120" --seqs all --out "$BASE/lightfc_120_score"

for split in gpu0 gpu1; do
  out="$BASE/odtrack_120_$split"
  while pgrep -af "odtrack_360vot.py.*$out" >/dev/null 2>&1 || \
        pgrep -af "launch_odtrack_after.sh.*$out" >/dev/null 2>&1; do
    n=$(find "$out" -mindepth 2 -maxdepth 2 -name metrics.json 2>/dev/null | wc -l)
    echo "[finalize] $split ODTrack metrics=$n/60"
    sleep 60
  done
done

merged="$BASE/odtrack_120"
mkdir -p "$merged"
for split in gpu0 gpu1; do
  for d in "$BASE/odtrack_120_$split"/[0-9][0-9][0-9][0-9]; do
    [[ -d "$d" ]] || continue
    ln -sfn "$(realpath "$d")" "$merged/$(basename "$d")"
  done
done

"$PYTHON" scripts/score_external_results.py \
  --data "${GRT360_DATA:-/data/projects/instan/data360}" \
  --tracker ODTrack="$merged" --seqs all --out "$BASE/odtrack_120_score"

date -Is > "$BASE/FINALIZED"
echo "[finalize] done"
