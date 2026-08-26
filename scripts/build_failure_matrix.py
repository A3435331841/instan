#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build GRT-360 sequence_failure_matrix and potential-preserving bake-off.

Example:
    python scripts/build_failure_matrix.py \
      --data /data/traindata/train \
      --baseline odtrack=/data/runs/all130/odtrack_... \
      --method sutrack_t224=/data/runs/medium/sutrack_t224 \
      --method sutrack_b224=/data/runs/medium/sutrack_b224 \
      --speed-baseline sutrack_t224 \
      --out /data/runs/failure_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.evaluation.failure_matrix import (  # noqa: E402
    build_failure_matrix,
    discover_sequence_records,
    summarize_bakeoff,
    write_failure_artifacts,
)


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path).expanduser()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="official dataset root")
    parser.add_argument("--baseline", required=True, type=parse_mapping,
                        help="precision baseline as NAME=PATH")
    parser.add_argument("--method", action="append", default=[], type=parse_mapping,
                        help="candidate method as NAME=PATH; repeatable")
    parser.add_argument("--speed-baseline", default=None,
                        help="method name used as the fast-main comparison")
    parser.add_argument("--sequences", default=None,
                        help="optional sequence-list text file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    baseline_name, baseline_path = args.baseline
    baseline = discover_sequence_records(baseline_path)
    if not baseline:
        parser.error(f"no sequence metrics found below baseline path: {baseline_path}")
    methods = {name: discover_sequence_records(path) for name, path in args.method}
    empty = [name for name, records in methods.items() if not records]
    if empty:
        parser.error(f"no sequence metrics found for: {', '.join(empty)}")
    sequences = None
    if args.sequences:
        sequences = [line.strip().replace("\\", "/") for line in
                     Path(args.sequences).read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    rows = build_failure_matrix(args.data, baseline, methods, sequences=sequences)
    bakeoff = summarize_bakeoff(
        rows, list(methods), baseline_name, speed_baseline_name=args.speed_baseline)
    paths = write_failure_artifacts(rows, bakeoff, args.out)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    print(f"sequences={len(rows)} baseline_auc={bakeoff['baseline_auc']:.4f}")
    for name, record in bakeoff["methods"].items():
        print(f"{name}: n={record['n']} auc={record['auc']:.4f} "
              f"delta={record['auc_delta']:+.4f} retain={record['retain']} "
              f"reasons={','.join(record['retain_reasons']) or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
