#!/bin/bash
set -e
cd /data/projects/instan_check/UETrack
/opt/miniconda3/envs/uetrack/bin/python _uetrack_eval.py > /data/projects/instan_check/uetrack_eval.log 2>&1
echo "RUN DONE rc=$?"