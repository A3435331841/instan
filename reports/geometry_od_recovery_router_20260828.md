# Geometry router + sparse OD recovery integration (2026-08-28)

`scripts/run_geometry_routed_od_recovery.py` keeps the v4 B224/T224 geometry
router for ordinary views and selects the B224 + sparse ODTrack tangent
recovery wrapper for broad views (`fov_h >= 70°` or `fov_v >= 130°`) plus the
validated high-latitude medium-view band (`30° <= fov_h <= 50°`,
`65° <= fov_v <= 100°`, `abs(lat) >= 40°`) and a pending high-latitude broad
band (`60° <= fov_h < 80°`, `75° <= fov_v < 100°`, `abs(lat) >= 40°`).  The
subsequent OD calls are runtime-state gated, with normal cadence 30 and LOST
cadence 5.  No sequence name, GT, or offline score table is available to the
router.

Representative 450-frame integration check:

| sequence | branch | AUC | SR | endpoint FPS | OD selections |
|---|---|---:|---:|---:|---:|
| `train_real/seq_0042` | B224 + OD recovery | 0.5395 | 0.6013 | 30.59 | 2 |
| `train_sim/seq_0027` | v4 geometry B/T | 0.8480 | 0.9733 | 59.92 | 0 |

The ordinary compact route remains unchanged on the normal control.  The
eBFoV branch is promising but has not yet been promoted to the 130-sequence
candidate; full-sequence and locked-valid aggregation are still required.
