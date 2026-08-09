# UETrack / GRT-360 integration

This directory pins the upstream UETrack checkout and checkpoint used by the
GRT-360 bake-off. `UPSTREAM.json` is the immutable anchor. The adapter supports
both normal `data360/0001/{image,label.json}` and legacy nested
`data360/0001/0001/{image,label.json}` extraction layouts.

Install into a clean checkout:

```bash
python integrations/uetrack/install_adapter.py \
  --workspace /data/projects/instan_check/UETrack \
  --output-root /data/projects/instan_check/uetrack_output \
  --checkpoint /data/projects/instan_check/uetrack_weights/uetrack_base.tar
```

Run selected sequences on one GPU, resuming completed outputs:

```bash
/opt/miniconda3/envs/uetrack/bin/python \
  /data/projects/instan/integrations/uetrack/run_erp.py \
  --workspace /data/projects/instan_check/UETrack \
  --data /data/projects/instan/data360 \
  --seqs 0001,0002,0003 --gpu 0 --skip-existing
```

Score the generated files with `scripts/score_external_results.py`. Accuracy
and native tracker timing remain separate; the scorer excludes the initialized
first frame and emits both ordinary and seam-aware dual IoU metrics.
