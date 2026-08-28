# B224 + sparse ODTrack tangent recovery (2026-08-28)

This is a causal recovery experiment.  B224 remains the normal-path tracker;
an OpenVINO ODTrack tangent graph is sampled every 15 frames and can be used
only after the B224 quality state enters `lost` (or on the transition into
`lost`).  The candidate must pass an anchor/motion verification gate before
B224 is re-initialized.  No ground truth, sequence name, or offline result is
used by the runtime wrapper.

## 450-frame tail probe

| sequence | B224 baseline AUC | sparse OD recovery AUC | B224+OD SR | endpoint FPS | OD selections |
|---|---:|---:|---:|---:|---:|
| `train_real/seq_0041` | 0.1109 | 0.2305 | 0.0713 | 30.61 | 0 |
| `train_real/seq_0042` | 0.2093 | 0.4696 | 0.5256 | 31.85 | 2 |
| `train_real/seq_0015` | 0.2187 | 0.3516 | 0.2561 | 30.15 | 1 |
| `train_real/seq_0037` | 0.1184 | 0.4271 | 0.2806 | 30.39 | 0 |
| `train_sim/seq_0075` | 0.1183 | 0.4483 | 0.5323 | 38.78 | 1 |

The 450-frame numbers are diagnostic only.  The sequence-length-dependent
long runs are stored under
`D:\\instan\\grt360_scratch\\b224_od_recovery_20260828\\long_eBFoV_loose`.
Their full-sequence results currently show `real/0041` AUC `0.1832` and
`real/0037` AUC `0.4883`, both above the B224 baseline, but the route has not
yet been merged into full130.

## Current interpretation

The expert is useful for the eBFoV/long-loss tail, but it is not yet a final
route: cadence, VERIFY thresholds, and per-sequence end-to-end latency must be
validated on train95 OOF and locked valid35 before promotion.  The normal path
must retain the v4 geometry router when this expert is integrated.
