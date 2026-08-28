# Hybrid geometry + sparse OD recovery evidence (2026-08-28)

This local diagnostic merges the locked v4 geometry-router outputs with
completed full-sequence sparse OD recovery runs for the most damaging broad-
FoV cases.  It is an evidence table, not a claim that every row was rerun
under the same executable in one batch: unchanged rows are explicitly reused
from the frozen B224/v4 artifacts, while newer per-sequence outputs override
them.

Artifact directory:

`D:\\instan\\grt360_scratch\\geometry_od_recovery_hybrid_20260828\\artifacts`

| split | AUC | SR | endpoint FPS | delta AUC | delta SR | >0.10 regressions |
|---|---:|---:|---:|---:|---:|---:|
| full130 hybrid | 0.6244 | 0.7547 | 39.40 | +0.0392 | +0.0518 | 0 |
| valid35 hybrid | 0.6092 | 0.7118 | 38.05 | +0.0384 | +0.0539 | 0 |

Full-sequence recovery gains include:

- `real/0041`: AUC `0.1109 -> 0.5683`, SR `0.0108 -> 0.6095`, FPS `32.40`;
- `real/0037`: AUC `0.1184 -> 0.6098`, SR `0.0541 -> 0.6954`, FPS `35.73`;
- `real/0042`: AUC `0.2093 -> 0.4597`, SR `0.0182 -> 0.2756`, FPS `31.60`;
- `real/0015`: AUC `0.2187 -> 0.3889`, SR `0.1005 -> 0.3364`, FPS `31.14`;
- `sim/0075`: AUC `0.1183 -> 0.6667`, SR `0.0178 -> 0.8530`, FPS `32.69`.

The hybrid is a meaningful tail breakthrough, but it is still far below the
strict full130 AUC `>0.8` requirement.  Before promotion, the same recovery
executable must be run through train95 OOF, locked valid35, and a single-GPU
full130 run with per-sequence P95 latency and no false recoveries.
