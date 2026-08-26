# Medium validation summary

Partial rows are progress signals, not final gates.

| method | done | AUC | SR | FPS | real AUC | sim AUC | worst AUC sequences |
|---|---:|---:|---:|---:|---:|---:|---|
| sutrack_b224 | 30/30 | 0.4919 | 0.5416 | 26.0 | 0.4435 | 0.5551 | train_real/seq_0022:0.154; train_sim/seq_0044:0.197; train_real/seq_0042:0.208; train_real/seq_0015:0.225; train_real/seq_0016:0.226 |
| ft_v4_ep4 | 30/30 | 0.4817 | 0.5279 | 14.0 | 0.4477 | 0.5263 | train_sim/seq_0046:0.062; train_sim/seq_0011:0.127; train_sim/seq_0044:0.156; train_real/seq_0043:0.169; train_real/seq_0016:0.207 |
| lorat | 30/30 | 0.4725 | 0.5424 | 8.7 | 0.4656 | 0.4815 | train_sim/seq_0052:0.050; train_sim/seq_0044:0.083; train_real/seq_0016:0.095; train_real/seq_0043:0.119; train_real/seq_0031:0.197 |
| ft_v4_ep1 | 30/30 | 0.4701 | 0.5170 | 14.9 | 0.4303 | 0.5223 | train_sim/seq_0046:0.058; train_sim/seq_0011:0.102; train_real/seq_0026:0.122; train_real/seq_0043:0.157; train_real/seq_0015:0.225 |
| ft_ep6 | 30/30 | 0.4623 | 0.5035 | 14.1 | 0.4402 | 0.4911 | train_sim/seq_0046:0.057; train_sim/seq_0011:0.107; train_sim/seq_0044:0.137; train_real/seq_0043:0.167; train_real/seq_0016:0.198 |
| sutrack_t224 | 30/30 | 0.4614 | 0.5070 | 39.5 | 0.4783 | 0.4393 | train_sim/seq_0046:0.077; train_real/seq_0043:0.146; train_real/seq_0042:0.185; train_sim/seq_0059:0.201; train_sim/seq_0011:0.240 |
| odtrack_t1 | 30/30 | 0.4599 | 0.5130 | 14.2 | 0.4334 | 0.4946 | train_sim/seq_0046:0.059; train_real/seq_0016:0.070; train_sim/seq_0011:0.087; train_real/seq_0026:0.127; train_real/seq_0043:0.201 |
| ft_ep5 | 30/30 | 0.4585 | 0.5018 | 13.7 | 0.4165 | 0.5135 | train_sim/seq_0046:0.057; train_sim/seq_0011:0.094; train_sim/seq_0044:0.142; train_real/seq_0026:0.146; train_real/seq_0043:0.168 |
| ft_v4_ep5 | 30/30 | 0.4585 | 0.4988 | 15.4 | 0.4171 | 0.5126 | train_sim/seq_0046:0.058; train_sim/seq_0011:0.103; train_real/seq_0026:0.111; train_real/seq_0043:0.152; train_sim/seq_0044:0.168 |
| uetrack | 30/30 | 0.4465 | 0.4815 | 38.5 | 0.4627 | 0.4254 | train_sim/seq_0046:0.088; train_sim/seq_0044:0.128; train_sim/seq_0011:0.166; train_real/seq_0043:0.183; train_real/seq_0015:0.202 |
| lightfc | 30/30 | 0.3075 | 0.3135 | 17.1 | 0.3824 | 0.2097 | train_sim/seq_0044:0.050; train_sim/seq_0052:0.058; train_sim/seq_0046:0.062; train_sim/seq_0011:0.063; train_real/seq_0022:0.097 |
| ft_ep7 | 28/30 | 0.4535 | 0.4962 | 13.8 | 0.4472 | 0.4631 | train_sim/seq_0046:0.064; train_sim/seq_0011:0.097; train_real/seq_0043:0.171; train_sim/seq_0044:0.186; train_real/seq_0016:0.216 |
| ft_ep8 | 27/30 | 0.4397 | 0.4864 | 13.4 | 0.4281 | 0.4566 | train_sim/seq_0046:0.073; train_sim/seq_0011:0.088; train_real/seq_0026:0.119; train_sim/seq_0044:0.142; train_real/seq_0043:0.173 |

Current oracle over 30/30 covered sequences: AUC=0.5779 SR=0.6617 FPS=16.0

## Current per-sequence oracle over medium results

| sequence | best method | AUC | SR | FPS |
|---|---|---:|---:|---:|
| train_real/seq_0002 | lorat | 0.7553 | 0.9277 | 9.3 |
| train_real/seq_0003 | lorat | 0.9165 | 0.9968 | 9.4 |
| train_real/seq_0004 | lorat | 0.7455 | 0.9979 | 9.0 |
| train_real/seq_0007 | sutrack_b224 | 0.6533 | 0.7739 | 27.8 |
| train_real/seq_0012 | ft_v4_ep5 | 0.6603 | 0.8700 | 11.4 |
| train_real/seq_0047 | sutrack_b224 | 0.8766 | 0.9896 | 28.6 |
| train_sim/seq_0011 | sutrack_b224 | 0.2641 | 0.2984 | 28.6 |
| train_sim/seq_0047 | sutrack_b224 | 0.8025 | 0.9659 | 28.8 |
| train_sim/seq_0082 | lorat | 0.4839 | 0.6107 | 8.7 |
| train_sim/seq_0052 | ft_ep5 | 0.4397 | 0.4487 | 21.1 |
| train_real/seq_0016 | ft_v4_ep1 | 0.4293 | 0.4276 | 15.3 |
| train_sim/seq_0044 | ft_v4_ep1 | 0.3164 | 0.3460 | 15.1 |
| train_sim/seq_0046 | sutrack_b224 | 0.5133 | 0.6415 | 27.8 |
| train_real/seq_0042 | odtrack_t1 | 0.2498 | 0.0398 | 6.9 |
| train_sim/seq_0015 | ft_ep8 | 0.7784 | 0.9555 | 14.4 |
| train_real/seq_0030 | lorat | 0.5138 | 0.6530 | 8.5 |
| train_real/seq_0043 | sutrack_b224 | 0.3418 | 0.3091 | 24.5 |
| train_real/seq_0013 | ft_ep8 | 0.3425 | 0.3484 | 14.6 |
| train_real/seq_0015 | odtrack_t1 | 0.2519 | 0.1492 | 8.3 |
| train_real/seq_0026 | lorat | 0.5485 | 0.6697 | 8.2 |
| train_real/seq_0031 | odtrack_t1 | 0.5082 | 0.5753 | 8.4 |
| train_sim/seq_0045 | sutrack_b224 | 0.4831 | 0.4942 | 24.8 |
| train_sim/seq_0055 | ft_v4_ep5 | 0.6793 | 0.9429 | 22.7 |
| train_sim/seq_0059 | ft_ep7 | 0.5073 | 0.6146 | 16.6 |
| train_sim/seq_0037 | ft_ep8 | 0.8024 | 0.9637 | 15.2 |
| train_real/seq_0022 | lorat | 0.7075 | 0.8709 | 7.8 |
| train_real/seq_0020 | ft_v4_ep5 | 0.7097 | 0.8693 | 22.3 |
| train_real/seq_0033 | ft_ep6 | 0.3533 | 0.1651 | 7.5 |
| train_sim/seq_0042 | ft_ep5 | 0.8567 | 0.9706 | 13.4 |
| train_sim/seq_0058 | ft_v4_ep4 | 0.8466 | 0.9650 | 15.4 |
