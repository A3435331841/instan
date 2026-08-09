# GRT-360 stage results — 2026-08-09

This is a reproducible checkpoint, not the final 120-sequence result. The full
dataset transfer and paired evaluation are still running.

## Immutable anchors

- Main repository branch: `agent/panotrack-v2`
- Stage code commit: `eae4785`
- UETrack upstream commit: `fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack base checkpoint SHA-256:
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`

## Unified 3-sequence bake-off

Protocol: strict prediction/GT length, initialized first frame excluded,
ordinary and dual IoU, SR@0.5, 21-point AUC, equal-weight macro average.

| Tracker | Sequences | Ordinary AUC | SR@0.5 | Native FPS |
| --- | ---: | ---: | ---: | ---: |
| UETrack base | 3 | 0.5247 | 0.5953 | 61.40 |
| LightFC ONNX | 3 | 0.3364 | 0.3687 | 6.78 |

Current winner: UETrack. This conclusion is provisional until the full set is
complete.

## Seam-aware UETrack ablation (0001–0010)

The enhancement replaces horizontal black padding with ERP circular sampling
and retains seam-crossing box extent. Vertical padding remains unchanged. It
is opt-in and writes to an independent result directory.

| Variant | Sequences | Ordinary AUC | SR@0.5 | Observed FPS |
| --- | ---: | ---: | ---: | ---: |
| UETrack base | 10 | 0.3356 | 0.3561 | 56.32 |
| UETrack ERP wrap | 10 | 0.3930 | 0.4242 | 48.60 |
| Geometry soft fusion, best transition | 10 | 0.3535 | 0.3767 | not comparable |

ERP wrap improves AUC by `+0.0574` and SR by `+0.0681`. The largest gains are
on `0001` (AUC `0.5470→0.8806`) and `0003` (`0.2551→0.6413`). It regresses on
some low-seam sequences, so the full paired run remains necessary. Box-level
soft fusion is retained as a negative ablation because independently evolving
expert states lose temporal consistency.

FPS was measured while two GPUs, concurrent JPEG decoding, and dataset transfer
were active. A clean single-process speed run is required before the final
performance claim.

## Validation

- 14 original/new core test modules passed locally and on the server.
- CUDA exposed and fixed a LoRA/MoE device-return bug; the GPU round trip now
  passes all 21 checks.
- External scorer regression tests: 3/3 passed.
- UETrack circular crop regression tests: 4/4 passed locally and on server.
- Geometry fusion regression tests: 3/3 passed.
- Offline LightFC Docker smoke test passed with `--network none` and emitted the
  required five result rows.

## Artifact locations

- Local raw stage results: `runs/grt360_20260809/`
- Server raw stage results: `/data/projects/instan/runs/grt360_20260809/`
- Server clean code checkout: `/data/projects/instan_grt360`
- Server pre-change backup:
  `/data/backups/instan_code_before_grt360_20260809_043053.tgz`
- GitHub draft PR: <https://github.com/A3435331841/instan/pull/1>

