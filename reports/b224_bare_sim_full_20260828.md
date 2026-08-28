# Bare SUTrack-B224 sim full run (2026-08-28)

The report diagnosis was tested with the same local OpenVINO B224 graph while
turning off motion-adaptive switching, fallback crops, polar remapping,
template freeze, and other rescue mechanisms.  The run covers all 83
`train_sim` sequences at their complete lengths; it is not a max-450 probe.

| split | sequences | AUC | SR | endpoint FPS |
|---|---:|---:|---:|---:|
| train_sim bare B224 | 83 | 0.6345 | 0.7558 | 34.14 |

Representative results include `sim/0002=0.8179`, `sim/0023=0.8794`,
`sim/0033=0.8366`, `sim/0041=0.8795`, `sim/0057=0.8535`, while known
mechanism-sensitive tails remain (`sim/0018=0.0675`, `sim/0024=0.1536`,
`sim/0071=0.2281`, `sim/0075=0.1160`).

This establishes a stable sim-domain base for the early quality probe.  It is
not by itself a full130 submission candidate because several real-domain
sequences require the geometry and sparse recovery branches.
