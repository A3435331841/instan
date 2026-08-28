# Hybrid v2: geometry routing plus high-latitude OD recovery

The v2 diagnostic adds the validated high-latitude broad-view recovery gate
to the earlier hybrid.  New evidence is included for `sim/0025` (full 450
frames) and `sim/0075` (full 450 frames); all other rows retain the explicit
v4 or frozen-baseline provenance in the merge inputs.

Artifact directory:

`D:\\instan\\grt360_scratch\\geometry_od_recovery_hybrid_v2_20260828\\artifacts`

| split | AUC | SR | endpoint FPS | delta AUC | delta SR | >0.10 regressions |
|---|---:|---:|---:|---:|---:|---:|
| full130 hybrid v2 | 0.6290 | 0.7616 | 39.43 | +0.0438 | +0.0588 | 0 |
| valid35 hybrid v2 | 0.6092 | 0.7118 | 38.05 | +0.0384 | +0.0539 | 0 |

The new high-latitude broad-view gate lifts `sim/0025` from AUC `0.0844` to
`0.6778` with SR `0.9376` at 32.13 FPS, while the `sim/0027` normal control
remains at AUC `0.8480`/SR `0.9733` and 60.07 FPS.  The full130 value is still
diagnostic and does not meet the strict AUC `>0.8` target; train95 OOF and a
single-executable full130 rerun remain required before promotion.
