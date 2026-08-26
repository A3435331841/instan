# GRT-360 Phase 2 implementation handoff — 2026-08-26

## Outcome

The difficulty-driven bake-off and the first online adaptive architecture are
implemented locally and synchronized to `/data/pano360`.  Duplicate ODTrack
baseline retries were retired after preserving their partial outputs; the
already-complete 130-sequence baseline remains the frozen precision reference.
No competition image was pushed.

Frozen ODTrack reference:

- AUC `0.5812714744`
- SR `0.6852958392`
- tracker FPS `27.4150`
- result: `/data/runs/_all130_gpu0/odtrack_20260826_021411`

## Implemented artifacts

- `panotrack/evaluation/failure_matrix.py` and
  `scripts/build_failure_matrix.py`: 130-sequence failure attributes, lost
  segments, component deltas, scene clusters and potential-preserving gates.
- `panotrack/data/training_manifest.py` and
  `scripts/build_spherical_training_manifest.py`: leakage-safe train95 manifest,
  balanced real/sim sampling mass, five sequence-level OOF folds and per-scene
  augmentation policy.
- `panotrack/pipeline/adaptive_spherical.py`: NORMAL/SUSPECT/LOST/VERIFY state
  machine, spherical motion evidence, anchor verification, sparse expert token
  budget, latency-adaptive call fraction and stateful expert episodes.
- `scripts/eval_official.py`: `adaptive_spherical` backend plus `trace.jsonl`,
  per-sequence/root `latency.json`, tracker FPS and decode-inclusive E2E FPS.
- Unit/smoke tests cover failure audit, training split leakage, seam/pole
  geometry, expert budget, state transitions, expert episodes and trace output.

Remote evidence:

- `/data/runs/failure_audit_v2_20260826`
- `/data/runs/spherical_training_manifest_20260826`
- train95 folds are exactly `19/19/19/19/19`; valid35 is excluded.
- retained components: SUTRACK-T224, SUTRACK-B224, UETrack, LoRAT,
  FT-v4-ep4 and ft_ep7.

## First micro gate

On the first 80 frames of `train_sim/seq_0046`, under the same runner:

| method | AUC | SR | tracker FPS | E2E FPS |
|---|---:|---:|---:|---:|
| ODTrack | 0.2037 | 0.0633 | 21.7 | 21.0 |
| SUTRACK-T224 | 0.3623 | 0.5570 | 37.3 | 35.7 |
| adaptive T224+B224 v1 | 0.3623 | 0.5570 | 22.6 | 21.8 |

The v1 router was therefore not promoted: it reproduced the T224 boxes while
paying for 20% reset-style B224 probes.  Trace evidence measured main P95 about
28 ms and reset-probe expert P95 about 58 ms.  A subsequent full-sequence
stateful-expert episode probe was worse (`AUC 0.0692 / SR 0.0336 / E2E 20.2
FPS`), so this current T224+B224 routing policy is explicitly rejected rather
than promoted to medium/full evaluation.

## Active remote work

GPU0:

- failure-balanced ODTrack v5 training with strict v4 checkpoint loading;
- config: `/data/odtrack/experiments/odtrack/finetune_spherical_v5.yaml`;
- manifest: `/data/runs/spherical_training_manifest_20260826/training_manifest.jsonl`;
- log: `/data/finetune_spherical_v5_20260826.log`; initial 50-step training
  IoU is `0.769`, confirming the v4 weights are now actually loaded;
- checkpoints: `/data/training_spherical_v5_20260826/checkpoints`.

GPU1:

- SUTRACK-T224 completed all130 at `AUC 0.5598 / SR 0.6573 / E2E 36.7 FPS`;
- SUTRACK-B224 valid35 is `AUC 0.5657 / SR 0.6594 / E2E 29.8 FPS`; its all130
  run is active because it is close to both the precision and speed boundaries;
- queue: `/data/grt360_phase2_gpu1_queue_20260826.sh`;
- results: `/data/runs/phase2_sutrack_20260826`.

After the GPU1 queue:

- run the full `seq_0046` adaptive stateful-expert episode gate;
- evaluate each stable v5 checkpoint on representative9;
- promote the best v5 checkpoint to medium30;
- watcher: `/data/grt360_v5_checkpoint_watcher_20260826.sh`.

## External training patch and rollback

ODTrack's sampler now reads `GRT360_TRAIN_MANIFEST`, maps prepared
`train_real_seq_XXXX_rYY` subsequences back to official sequence ids, and uses
the capped train95 sampling weights.  A smoke test confirmed non-uniform weights
for hard sequences.  The pre-change sampler is recoverable at:

`/data/odtrack/lib/train/data/sampler.py.bak_grt360_20260826`

Do not consume valid35 in training and do not run `docker push` without an
explicit user confirmation.
