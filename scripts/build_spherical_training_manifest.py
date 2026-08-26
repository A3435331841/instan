#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the failure-balanced train95/5-fold spherical training manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from panotrack.data.training_manifest import (  # noqa: E402
    build_training_manifest,
    write_training_manifest,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-matrix", required=True,
                        help="failure_matrix.json from build_failure_matrix.py")
    parser.add_argument(
        "--train-split",
        default=str(PROJECT_ROOT / "data360" / "official_split" / "seqlist_official_train.txt"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    failure_rows = json.loads(Path(args.failure_matrix).read_text(encoding="utf-8"))
    train_sequences = [line.strip() for line in
                       Path(args.train_split).read_text(encoding="utf-8").splitlines()
                       if line.strip()]
    records = build_training_manifest(failure_rows, train_sequences, n_folds=args.folds)
    paths = write_training_manifest(records, args.out)
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    print(f"train_sequences={len(records)} folds={args.folds} valid35_included=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
