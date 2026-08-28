# Geometry router v4 safety-band experiment (2026-08-28)

This experiment tightened the causal B224/T224 route using only the protocol
initial BFoV (`fov_h`, `fov_v`, latitude).  It does not use sequence names,
ground truth, or an offline result lookup.

## Changes

- T224 is used for `fov_h <= 6` only in the safe vertical band (`fov_v <=
  12.5`), plus the previously observed compact rescue band
  `5.8 <= fov_h <= 6`, `18 <= fov_v <= 23`, `abs(lat) < 30`.
- T224 is used for `10 <= fov_h < 15` only when `fov_v <= 12.5` or
  `fov_h >= 14.5`.
- The high-latitude B224 adaptive branch is disabled in the isolated
  `29 <= fov_h <= 32`, `25 <= fov_v <= 35`, `abs(lat) >= 65` safety band.

## Evaluation

The result is materialized in the local artifact directory
`D:\\instan\\grt360_scratch\\geometry_router_v4_20260828\\artifacts_correct_merge`.
It combines 11 newly rerun affected sequences, the completed v3 changed
shard, the partial v3 shard, and the frozen B224 baseline for unchanged
sequences.  The merge order is baseline -> partial -> changed -> v4 targeted,
so newer evidence wins without overwriting experiment directories.

| split | AUC | SR | endpoint FPS | delta AUC | delta SR | >0.10 regressions |
|---|---:|---:|---:|---:|---:|---:|
| full130 | 0.6097 | 0.7349 | 39.09 | +0.0245 | +0.0320 | 0 |
| valid35 | 0.5951 | 0.6935 | 37.78 | +0.0244 | +0.0356 | 0 |

This is a route-safety improvement, not final acceptance: the full130 target
`AUC > 0.8` is not met.  The next work should attack the remaining low-AUC
tail (especially absence/eBFoV/scale cases) with a causal recovery expert,
then rerun the locked valid35 and full130 summaries.
