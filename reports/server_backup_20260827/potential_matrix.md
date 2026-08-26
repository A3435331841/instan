# Representative potential matrix (auto)

This table is a component-potential screen, not the final submission gate.

Fast baseline: uetrack AUC=0.5229 SR=0.6007 FPS=51.6
Precision baseline: lorat AUC=0.6021 SR=0.7275 FPS=10.2

| method | status | n | AUC | SR | FPS | real AUC | sim AUC | reason |
|---|---|---:|---:|---:|---:|---:|---:|---|
| lorat | core_keep | 9 | 0.6021 | 0.7275 | 10.2 | 0.6873 | 0.4315 | near/top precision; real expert; wins 4 seq |
| sutrack_b224 | core_keep | 9 | 0.5919 | 0.6899 | 27.6 | 0.6567 | 0.4624 | near/top precision; speed-baseline peer; sim/hard expert; wins 4 seq; stable usable |
| sutrack_t224 | core_keep | 9 | 0.5632 | 0.6579 | 40.5 | 0.6776 | 0.3346 | fast-main; speed-baseline peer; real expert; stable usable; lightweight real-scene candidate |
| ft_v4_ep4 | core_keep | 9 | 0.5479 | 0.6627 | 16.7 | 0.6007 | 0.4423 | sim/hard expert |
| ft_v4_ep1 | core_keep | 9 | 0.5396 | 0.6535 | 19.8 | 0.5913 | 0.4363 | sim/hard expert |
| ft_ep8 | core_keep | 9 | 0.5389 | 0.6586 | 27.1 | 0.5982 | 0.4203 | speed-baseline peer; stable usable |
| ft_ep7 | core_keep | 9 | 0.5365 | 0.6513 | 26.4 | 0.5938 | 0.4218 | speed-baseline peer; stable usable |
| ft_v4_ep5 | core_keep | 9 | 0.5339 | 0.6470 | 15.2 | 0.5922 | 0.4172 | wins 1 seq |
| ft_ep6 | core_keep | 9 | 0.5335 | 0.6474 | 26.7 | 0.5825 | 0.4354 | speed-baseline peer; sim/hard expert |
| odtrack_t1 | core_keep | 9 | 0.5274 | 0.6222 | 29.5 | 0.6258 | 0.3304 | speed-baseline peer; lightweight real-scene candidate |
| uetrack | core_keep | 9 | 0.5229 | 0.6007 | 51.6 | 0.6461 | 0.2764 | fast-main; speed-baseline peer; lightweight real-scene candidate |
| ft_ep5 | core_keep | 9 | 0.5226 | 0.6356 | 27.0 | 0.5760 | 0.4159 | speed-baseline peer |
| lightfc | watchlist | 9 | 0.4312 | 0.5112 | 28.8 | 0.5662 | 0.1613 | lightweight real-scene candidate |
| ft_v4_ep2 | drop | 9 | 0.5386 | 0.6516 | 15.0 | 0.5987 | 0.4183 | no clear component value yet |
| ft_v4_ep6 | drop | 9 | 0.5344 | 0.6481 | 14.1 | 0.5920 | 0.4191 | no clear component value yet |
| ft_v4_ep3 | drop | 9 | 0.5343 | 0.6500 | 18.2 | 0.5915 | 0.4200 | no clear component value yet |
| ft_ep3 | drop | 9 | 0.5268 | 0.6278 | 16.3 | 0.5817 | 0.4169 | no clear component value yet |
| ft_ep4 | drop | 9 | 0.5251 | 0.6353 | 20.9 | 0.5761 | 0.4231 | no clear component value yet |
| ft_ep2 | drop | 9 | 0.5123 | 0.5872 | 12.4 | 0.5612 | 0.4146 | no clear component value yet |
| ft_ep1 | drop | 9 | 0.4900 | 0.5440 | 13.2 | 0.5520 | 0.3660 | no clear component value yet |
| odtrack_recapture_ft_ep7 | drop | 9 | 0.4007 | 0.4634 | 23.0 | 0.5043 | 0.1937 | no clear component value yet |
| odtrack_recapture | drop | 9 | 0.3981 | 0.4692 | 23.7 | 0.5148 | 0.1646 | no clear component value yet |
| direct_erp | drop | 9 | 0.2646 | 0.2459 | 22.7 | 0.3352 | 0.1234 | no clear component value yet |

## Per-sequence best

| sequence | best method | AUC | SR | FPS |
|---|---|---:|---:|---:|
| train_real/seq_0002 | lorat | 0.7553 | 0.9277 | 10.1 |
| train_real/seq_0003 | lorat | 0.9165 | 0.9968 | 9.7 |
| train_real/seq_0004 | lorat | 0.7455 | 0.9979 | 10.2 |
| train_real/seq_0007 | sutrack_b224 | 0.6533 | 0.7739 | 28.3 |
| train_real/seq_0012 | ft_v4_ep5 | 0.6603 | 0.8700 | 13.9 |
| train_real/seq_0047 | sutrack_b224 | 0.8766 | 0.9896 | 27.0 |
| train_sim/seq_0011 | sutrack_b224 | 0.2641 | 0.2984 | 27.3 |
| train_sim/seq_0047 | sutrack_b224 | 0.8025 | 0.9659 | 27.5 |
| train_sim/seq_0082 | lorat | 0.4839 | 0.6107 | 9.7 |
