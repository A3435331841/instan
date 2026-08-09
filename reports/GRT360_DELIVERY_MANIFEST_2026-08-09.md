# GRT-360 delivery manifest — 2026-08-09

## Version anchors

- Branch: `agent/panotrack-v2`
- GitHub draft PR: https://github.com/A3435331841/instan/pull/1
- UETrack upstream commit: `fd13b0eaf16d51536008295f3b27807c69eaad50`
- UETrack checkpoint SHA-256:
  `1d34778a41c553e3a5e17829d33df4a644f7c948b054a64f46e02fa99558b901`
- CLIP ViT-L/14 cache SHA-256:
  `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`

## Local deliverables

| Artifact | Location | SHA-256 / status |
| --- | --- | --- |
| Research handoff PPTX | `D:/instan/deliverables/GRT360_2026-08-09/GRT360_Research_Handoff_2026-08-09.pptx` | `3a64fb1ab9a27f0c1b3e486ae4927b9743ba716a09939aa52eb516a483a72ec3` |
| Offline UETrack image tar | `D:/instan/deliverables/GRT360_2026-08-09/docker/grt360-uetrack_2026-08-09_linux-amd64.tar` | `1919ba75a90a07a54e7def09234a0ea492dee22629685a262ca2e1892cc50c54` |
| Image manifest ID | `grt360-uetrack:2026-08-09` | `sha256:21508ea8959c0dda8b96747a670d06a68d897aa3a949e0f1c4e146a6adf0368a` |

The Docker archive is approximately 5.99 GB and is intentionally not tracked
in Git. The repository contains the Dockerfile and the exact build inputs are
kept in the local handoff directory.

## Tracked evidence

- `reports/STAGE_RESULTS_2026-08-09.md` — current status and interpretation.
- `reports/results/erpwrap_ablation_0001_0029_bakeoff.json` — strict protocol,
  macro summaries, and per-sequence rows for the 29-sequence stage.
- `reports/results/erpwrap_ablation_0001_0029_scores.csv` — tabular stage rows.
- `reports/GRT360_Research_Handoff_2026-08-09.pptx` — audience-facing summary.

## Validation boundary

- Local container passed `--network none` import/hash checks.
- The local WSL Docker runtime has no visible NVIDIA adapter. Identical
  installed UETrack code and checkpoint passed a 5-frame GPU file-protocol
  smoke test on the two-RTX-3090 server.
- The 120-sequence paired queue is still running on the server; 29 sequences
  are frozen in the tracked stage report, not presented as a final 120 result.
