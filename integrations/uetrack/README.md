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

Run the seam-aware ablation into a separate result directory:

```bash
/opt/miniconda3/envs/uetrack/bin/python \
  /data/projects/instan/integrations/uetrack/run_erp.py \
  --workspace /data/projects/instan_check/UETrack \
  --data /data/projects/instan/data360 \
  --seqs 0001,0002,0003 --gpu 0 --erp-wrap --result-tag erpwrap \
  --skip-existing
```

Score the generated files with `scripts/score_external_results.py`. Accuracy
and native tracker timing remain separate; the scorer excludes the initialized
first frame and emits both ordinary and seam-aware dual IoU metrics.

The final GPU image exposes a generic file protocol. It uses ERP wrapping by
default and requires no network at runtime:

```bash
docker run --rm --gpus all --network none \
  -v /absolute/input:/data grt360-uetrack:2026-08-09 \
  --frames /data/frames --init /data/init.txt \
  --out /data/results.txt --timing /data/timing.txt
```

The first result row is copied exactly from `init.txt`; subsequent rows are the
tracker predictions. Pass `--no-erp-wrap` only when reproducing the upstream
zero-padding ablation.

Build prerequisites are staged outside Git under
`tools_local/uetrack_docker/`: the pinned upstream source checkout, the
checkpoint named `models/uetrack_base.tar`, and the CLIP cache named
`clip_cache/ViT-L-14.pt`. Build from the repository root:

```bash
docker build --platform linux/amd64 \
  -f docker/uetrack/Dockerfile \
  -t grt360-uetrack:2026-08-09 .
```

If Docker Hub is unavailable, pass a byte-equivalent mirror with
`--build-arg BASE_IMAGE=...`; the validated base manifest digest is recorded in
the stage report.
