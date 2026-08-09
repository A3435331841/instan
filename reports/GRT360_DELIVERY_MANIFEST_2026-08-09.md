# GRT-360 delivery manifest — 2026-08-09

## Version anchors

- Branch: `agent/panotrack-v2`
- GitHub draft PR: https://github.com/A3435331841/instan/pull/1
- Final paired evaluation: 120/120 sequences, baseline + ERP-wrap
- UETrack upstream commit: `fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack checkpoint SHA-256:
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 cache SHA-256:
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

## Local deliverables

| Artifact | Location | SHA-256 / status |
| --- | --- | --- |
| Final research handoff PPTX | `D:/instan/deliverables/GRT360_2026-08-09/GRT360_Research_Handoff_2026-08-09.pptx` | `18022a707201a25c0cdbba7e2221930a27e1c6f5b0940c093618a15f5fb5de75` |
| Offline UETrack image tar | `D:/instan/deliverables/GRT360_2026-08-09/docker/grt360-uetrack_2026-08-09_linux-amd64.tar` | `1919ba75a90a07a54e7def09234a0ea492dee22629685a262ca2e1892cc50c54` |
| Image manifest ID | `grt360-uetrack:2026-08-09` | `sha256:21508ea8959c0dda8b96747a670d06a68d897aa3a949e0f1c4e146a6adf0368a` |
| Final raw result archive | `D:/instan/deliverables/GRT360_2026-08-09/results/uetrack_results_0001_0120.tar.zst` | `b255106a2e3711612a9e2aec86d8ba5ec45c3971049f847c6f520a2d4f7810e8` |

The Docker image archive is approximately 5.99 GB and is intentionally not
tracked in Git. The raw result archive is kept in the local deliverables
directory for reproducibility and is also not tracked in Git.

## Tracked evidence

- `reports/STAGE_RESULTS_2026-08-09.md` — final 120-sequence report.
- `reports/results/erpwrap_ablation_0001_0120_bakeoff.json` — strict protocol,
  macro summaries, and all 240 tracker/sequence rows. SHA-256:
  `d49046a7a0395c48aae451f584e8f729244f0ead6e286dc29f00055d9dc0a333`.
- `reports/results/erpwrap_ablation_0001_0120_scores.csv` — tabular final rows.
  SHA-256: `b5e99afd9ce2322e7d9b3104e79dd5850ca8a76842e06618e18a972d41fa3953`.
- `reports/GRT360_Research_Handoff_2026-08-09.pptx` — audience-facing final summary.
- Earlier 39-sequence evidence remains under `reports/results/*0039*` as a
  historical checkpoint.

## Validation boundary

- Local container passed `--network none` import/hash checks.
- The local WSL Docker runtime has no visible NVIDIA adapter. Identical
  installed UETrack code and checkpoint passed a 5-frame GPU file-protocol
  smoke test on the two-RTX-3090 server.
- Server data sync completed through `ALL_SEQUENCE_SHARDS_COMPLETE`.
- Both paired queues completed with `QUEUE_DONE`; baseline and ERP-wrap each
  contain 120 sequence outputs.
- Final strict scorer completed 240 rows with exact prediction/GT lengths.
- Secrets and private caches are excluded from tracked files.
