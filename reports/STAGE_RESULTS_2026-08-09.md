# GRT-360 stage results — 2026-08-09

This is a reproducible checkpoint, not the final 120-sequence result. The full
dataset transfer and paired evaluation are still running.

## Immutable anchors

- Main repository branch: `agent/panotrack-v2`
- UETrack upstream commit: `fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack base checkpoint SHA-256:
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 cache SHA-256:
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

## Unified 3-sequence bake-off

Protocol: strict prediction/GT length, initialized first frame excluded,
ordinary and dual IoU, SR@0.5, 21-point AUC, equal-weight macro average.

| Tracker | Sequences | Ordinary AUC | SR@0.5 | Native FPS |
| --- | ---: | ---: | ---: | ---: |
| UETrack base | 3 | 0.5247 | 0.5953 | 61.40 |
| LightFC ONNX | 3 | 0.3364 | 0.3687 | 6.78 |

UETrack is the selected backbone. LoRAT and ODTrack were not assigned fabricated
scores: no usable pinned checkpoints were present in the handoff environment.

## Seam-aware UETrack ablation (0001–0039)

The enhancement replaces horizontal black padding with ERP circular sampling
and retains seam-crossing box extent. Vertical padding remains unchanged. It
is opt-in for ablation runs and is the default in the final file protocol.

| Variant | Sequences | Ordinary AUC | SR@0.5 | Observed FPS |
| --- | ---: | ---: | ---: | ---: |
| UETrack ERP wrap | 39 | 0.4988 | 0.5578 | 56.63 |
| UETrack base | 39 | 0.4060 | 0.4397 | 62.39 |
| Geometry soft fusion, best transition | 10 | 0.3535 | 0.3767 | not comparable |

Across 39 sequences, ERP wrap improves AUC by `+0.0928` and SR by `+0.1181`.
The large gains include `0001` (`0.5470→0.8806`), `0003`
(`0.2551→0.6413`), `0014` (`0.2616→0.7284`), `0017`
(`0.3748→0.6045`), `0018` (`0.4995→0.8145`), and `0035`
(`0.6005→0.7360`). Regressions are retained in
the raw per-sequence CSV rather than hidden. Box-level soft fusion remains a
negative ablation because independently evolving experts lose temporal
consistency.

FPS was measured while two GPUs, concurrent JPEG decoding, and dataset transfer
were active. Accuracy is comparable; the aggregate speed difference is not a
clean isolated overhead measurement.

## Offline delivery status

- The lightweight LightFC image passed `--network none` on five frames.
- The final UETrack image has a generic `--frames/--init/--out/--timing` file
  protocol, pinned CUDA/PyTorch base, embedded UETrack checkpoint, and embedded
  CLIP ViT-L/14 cache.
- The final `linux/amd64` image was built successfully. Image ID:
  `sha256:21508ea8959c0dda8b96747a670d06a68d897aa3a949e0f1c4e146a6adf0368a`.
- Base image manifest: `sha256:fc47f8018254e6df30f48c48f2db1c758d44de21a8c553de1a1c451a65baa70a`.
- Saved image archive: `5,991,662,592` bytes, SHA-256
  `1919ba75a90a07a54e7def09234a0ea492dee22629685a262ca2e1892cc50c54`.
- With `--network none`, container imports reported PyTorch `2.3.1`, CUDA
  `12.1`, and exact checkpoint/CLIP hashes.
- The local WSL Docker runtime has no visible NVIDIA adapter, so GPU inference
  was validated with the identical installed entrypoint on the two-RTX-3090
  server: 5/5 result rows, 5/5 timing rows, and preserved first-frame box.

## Validation

- 14 original/new core test modules passed locally and on the server.
- CUDA exposed and fixed a LoRA/MoE device-return bug; the GPU round trip now
  passes all 21 checks.
- External scorer regression checks: 3/3 passed.
- UETrack circular crop checks: 4/4 passed locally and on the server.
- UETrack file-protocol helper checks: 3/3 passed on the server.
- UETrack file-protocol GPU smoke: 5/5 frames passed on the server.
- Geometry fusion regression checks: 3/3 passed.

## Artifact locations

- Local raw stage results: `runs/grt360_20260809/`
- Tracked 39-sequence evidence: `reports/results/erpwrap_ablation_0001_0039_bakeoff.json`
- Research handoff deck: `reports/GRT360_Research_Handoff_2026-08-09.pptx`
- Delivery manifest: `reports/GRT360_DELIVERY_MANIFEST_2026-08-09.md`
- Server raw stage results: `/data/projects/instan/runs/grt360_20260809/`
- Server clean code checkout: `/data/projects/instan_grt360`
- Server pre-change backup:
  `/data/backups/instan_code_before_grt360_20260809_043053.tgz`
- GitHub draft PR: <https://github.com/A3435331841/instan/pull/1>
