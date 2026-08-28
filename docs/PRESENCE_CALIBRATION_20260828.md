# Causal presence/TRC calibration record

## Scope

The calibrator is an offline CPU component.  It converts the official BFoV
annotations to ERP boxes only while making train95 labels; the exported policy
uses causal tracker signals only.  It does not contain sequence names, GT, or
an offline result lookup and is not enabled as a submission router until the
paired-expert gate is passed.

Command used:

```text
python scripts/train_presence_calibrator.py \
  --results <immutable B224 trace root> \
  --train-list data360/official_split/seqlist_official_train.txt \
  --horizon 15 --failure-iou 0.30
```

## Evidence

The 2026-08-28 train95 run used 95 sequence-disjoint groups and 62,475 causal
rows.  The OOF AUROC for “any dual-IoU failure below 0.30 in the next 15
frames” was **0.88999**; fold AUROCs were 0.88269, 0.85060, 0.89203, 0.93291,
and 0.84837.  The F1 operating point selected a 0.35192 risk threshold with
precision 0.73716, recall 0.72929, and probe rate 0.25985.

The policy was replayed causally on a 450-frame tiny-target control.  A
diagnostic update-block-only run did not improve that control (AUC 0.0777 vs
0.0777 without the policy), and an opt-in hold-last-reliable run regressed to
0.0513.  Therefore the policy remains a diagnostic/TRC signal and the hold
consumer is not promoted.  A paired B224/ODTrack or B224/LoRAT OOF experiment
must show expert-selection gain and controlled false recovery before routing
is enabled.

Artifacts are written under `grt360_scratch/presence_calibrator_*` and include
`presence_policy.json`, `oof_predictions.csv`, `calibration_summary.json`, and
`sequence_summary.csv`.
