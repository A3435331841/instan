# Medium component diagnostics

Partial results are evidence for component potential, not final submission gates.
Submission target remains AUC>0.80, SR>0.80, FPS>30.0.

| method | role | done | AUC | SR | FPS | real AUC | sim AUC | wins | missing |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fusion_5way | watchlist | 30/30 | 0.522 | 0.593 | nan | 0.480 | 0.577 | 1 | 0 |
| fusion_5way_final | watchlist | 30/30 | 0.522 | 0.593 | nan | 0.480 | 0.577 | 0 | 0 |
| fusion_4way | watchlist | 30/30 | 0.522 | 0.594 | nan | 0.485 | 0.570 | 0 | 0 |
| fusion_v2_4way | watchlist | 30/30 | 0.512 | 0.574 | nan | 0.469 | 0.569 | 0 | 0 |
| fusion_3way | sequence_expert | 30/30 | 0.510 | 0.578 | nan | 0.478 | 0.553 | 2 | 0 |
| fusion_6way_v4 | partial | 13/30 | 0.502 | 0.587 | nan | 0.596 | 0.391 | 0 | 17 |
| fusion_2way | sequence_expert | 30/30 | 0.498 | 0.549 | nan | 0.453 | 0.556 | 2 | 0 |
| sutrack_b224 | sequence_expert | 30/30 | 0.492 | 0.542 | 26.0 | 0.444 | 0.555 | 4 | 0 |
| fusion_8way | sequence_expert+partial | 27/30 | 0.490 | 0.550 | nan | 0.484 | 0.499 | 2 | 3 |
| fusion_5way_v1v4 | partial | 19/30 | 0.489 | 0.543 | nan | 0.489 | 0.488 | 0 | 11 |
| fusion_6way_ep7 | sequence_expert+partial | 28/30 | 0.488 | 0.543 | nan | 0.474 | 0.509 | 3 | 2 |
| ft_v4_ep4 | watchlist | 30/30 | 0.482 | 0.528 | 14.0 | 0.448 | 0.526 | 1 | 0 |
| lorat | sequence_expert | 30/30 | 0.472 | 0.542 | 8.7 | 0.466 | 0.481 | 5 | 0 |
| ft_v4_ep1 | sequence_expert | 30/30 | 0.470 | 0.517 | 14.9 | 0.430 | 0.522 | 2 | 0 |
| fusion_6way_best | partial | 19/30 | 0.464 | 0.502 | nan | 0.469 | 0.454 | 1 | 11 |
| fusion_6way_v4ep1 | partial | 19/30 | 0.464 | 0.502 | nan | 0.469 | 0.454 | 0 | 11 |
| ft_ep6 | watchlist | 30/30 | 0.462 | 0.503 | 14.1 | 0.440 | 0.491 | 1 | 0 |
| sutrack_t224 | fast_main | 30/30 | 0.461 | 0.507 | 39.5 | 0.478 | 0.439 | 0 | 0 |
| odtrack_t1 | sequence_expert | 30/30 | 0.460 | 0.513 | 14.2 | 0.433 | 0.495 | 3 | 0 |
| ft_ep5 | watchlist | 30/30 | 0.459 | 0.502 | 13.7 | 0.416 | 0.513 | 0 | 0 |
| ft_v4_ep5 | sequence_expert | 30/30 | 0.458 | 0.499 | 15.4 | 0.417 | 0.513 | 2 | 0 |
| ft_ep7 | partial | 28/30 | 0.453 | 0.496 | 13.8 | 0.447 | 0.463 | 0 | 2 |
| uetrack | fast_main | 30/30 | 0.447 | 0.481 | 38.5 | 0.463 | 0.425 | 0 | 0 |
| ft_ep8 | partial | 27/30 | 0.440 | 0.486 | 13.4 | 0.428 | 0.457 | 1 | 3 |
| lightfc | watchlist | 30/30 | 0.308 | 0.314 | 17.1 | 0.382 | 0.210 | 0 | 0 |

Current per-sequence oracle over 30/30 sequences: AUC=0.582 SR=0.670 FPS=14.4

## Hard sequences under current oracle

| sequence | best method | AUC | SR | FPS |
|---|---|---:|---:|---:|
| train_real/seq_0042 | odtrack_t1 | 0.250 | 0.040 | 6.9 |
| train_real/seq_0015 | odtrack_t1 | 0.252 | 0.149 | 8.3 |
| train_sim/seq_0011 | sutrack_b224 | 0.264 | 0.298 | 28.6 |
| train_sim/seq_0044 | ft_v4_ep1 | 0.316 | 0.346 | 15.1 |
| train_real/seq_0043 | sutrack_b224 | 0.342 | 0.309 | 24.5 |
| train_real/seq_0013 | fusion_8way | 0.344 | 0.348 | nan |
| train_real/seq_0033 | ft_ep6 | 0.353 | 0.165 | 7.5 |
| train_real/seq_0016 | ft_v4_ep1 | 0.429 | 0.428 | 15.3 |
| train_sim/seq_0052 | fusion_6way_ep7 | 0.442 | 0.461 | nan |
| train_sim/seq_0045 | sutrack_b224 | 0.483 | 0.494 | 24.8 |

## Best two-component oracle pairs

| pair | common seqs | AUC | SR | FPS |
|---|---:|---:|---:|---:|
| fusion_3way + fusion_6way_v4 | 13 | 0.552 | 0.643 | nan |
| fusion_6way_v4 + lorat | 13 | 0.551 | 0.657 | 9.0 |
| fusion_4way + fusion_6way_v4 | 13 | 0.549 | 0.644 | nan |
| fusion_2way + lorat | 30 | 0.548 | 0.635 | 8.7 |
| lorat + sutrack_b224 | 30 | 0.547 | 0.633 | 20.1 |
| fusion_5way_v1v4 + fusion_6way_v4 | 13 | 0.544 | 0.644 | nan |
| ft_v4_ep1 + fusion_3way | 30 | 0.544 | 0.614 | 14.6 |
| ft_v4_ep1 + fusion_4way | 30 | 0.543 | 0.614 | 13.5 |
| fusion_2way + fusion_6way_v4 | 13 | 0.543 | 0.632 | nan |
| fusion_5way_final + lorat | 30 | 0.542 | 0.630 | 8.8 |

## Gate status

NO_SUBMISSION_CANDIDATE: no completed method currently satisfies AUC>0.80, SR>0.80, FPS>30.0.
REPRESENTATIVE_GATE=OK: no incomplete representative probes in the current matrix.
FUSION_GATE=HOLD: medium validation is still incomplete; do not start final fusion yet.
SPEED_GATE=OK_FOR_CURRENT_LOAD: FPS is not confounded by concurrent medium evaluations.

## Coverage still missing

ft_ep7:28/30, ft_ep8:27/30, fusion_5way_v1v4:19/30, fusion_6way_best:19/30, fusion_6way_ep7:28/30, fusion_6way_v4:13/30, fusion_6way_v4ep1:19/30, fusion_8way:27/30

