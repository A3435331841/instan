#!/usr/bin/env python3
"""PyTorch CUDA reference entrypoint for the GRT-360 final delivery.

This entrypoint intentionally keeps the original SUTrack PyTorch execution
path available beside the primary ONNX Runtime v5 route.  It is useful for
CUDA/backend benchmarking, debugging an exported graph, and a conservative
Arena fallback.  The validated v5 multi-expert policy is implemented by
``arena_protocol_v5.py`` on ONNX Runtime; this file must not be presented as
bit-for-bit equivalent to that policy.

It implements the same Arena BFoV file contract and supports the two source
checkpoints used in the race: B224 (accuracy reference) and T224 (speed
reference).  It never reads ground truth, sequence names as routing labels,
or offline result tables.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from integrations.sutrack.arena_protocol_sutrack import main as sutrack_main  # noqa: E402


PROFILES = {
    "b224_erp": {
        "checkpoint": "SUTRACK_b224_ep0180.pth.tar",
        "config": "sutrack_b224",
        "description": "SUTrack-B224 ERP three-tile accuracy reference",
    },
    "t224_erp": {
        "checkpoint": "SUTRACK_t224_ep0180.pth.tar",
        "config": "sutrack_t224",
        "description": "SUTrack-T224 ERP three-tile speed reference",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--result", default=None)
    parser.add_argument("--workspace", default=os.environ.get("SUTRACK_WORKSPACE", "/opt/sutrack"))
    parser.add_argument("--model-root", default=os.environ.get("GRT360_MODEL_ROOT", "/opt/models"))
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default=os.environ.get("GRT360_TORCH_PROFILE", "b224_erp"))
    parser.add_argument("--gpu", default=os.environ.get("GRT360_GPU", "0"))
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seqs", default=None)
    parser.add_argument("--lost-iou-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)

    spec = PROFILES[args.profile]
    checkpoint = Path(args.model_root) / spec["checkpoint"]
    if not checkpoint.is_file():
        print(f"[error] missing {args.profile} checkpoint: {checkpoint}", file=sys.stderr)
        return 2

    forwarded = [
        "--workspace", str(args.workspace),
        "--checkpoint", str(checkpoint),
        "--config", spec["config"],
        "--gpu", str(args.gpu),
        "--lost-iou-threshold", str(args.lost_iou_threshold),
    ]
    if args.dataset:
        forwarded += ["--dataset", str(args.dataset)]
    if args.result:
        forwarded += ["--result", str(args.result)]
    if args.max_frames is not None:
        forwarded += ["--max-frames", str(args.max_frames)]
    if args.seqs:
        forwarded += ["--seqs", str(args.seqs)]
    if args.force_cpu:
        forwarded.append("--force-cpu")
    print(f"[final_torch] profile={args.profile}; {spec['description']}", flush=True)
    return int(sutrack_main(forwarded))


if __name__ == "__main__":
    raise SystemExit(main())
