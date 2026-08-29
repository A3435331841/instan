#!/usr/bin/env python3
"""Run a declared GRT-360 reproduction profile without hidden local state.

This helper deliberately only materializes the command.  It never fetches a
weight, edits a profile, uses ground truth, or looks up a sequence-specific
result.  The profile JSON and the model manifest are the complete input
contract.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_profile(value: str) -> tuple[dict, Path]:
    candidate = Path(value)
    if not candidate.is_file():
        candidate = ROOT / "configs" / "repro" / f"{value}.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"profile not found: {value}")
    return json.loads(candidate.read_text(encoding="utf-8")), candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="profile name under configs/repro or JSON path")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--backend", choices=("ort_cuda", "torch_cuda"), default=None)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--seqs", default=None)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    profile, profile_path = _load_profile(args.profile)
    backend = args.backend or profile.get("backend")
    name = profile.get("name", profile_path.stem)
    common = ["--dataset", str(Path(args.dataset).resolve()), "--result", str(Path(args.result).resolve()),
              "--model-root", str(Path(args.model_root).resolve()), "--gpu", str(args.gpu)]
    if args.max_frames is not None:
        common += ["--max-frames", str(args.max_frames)]
    if args.seqs:
        common += ["--seqs", str(args.seqs)]

    if backend == "ort_cuda":
        if name not in {"v5_final", "geometry_v1", "geometry_v4"}:
            raise ValueError(f"{name} has no standalone ORT route entrypoint; use its declared PyTorch reference")
        command = [sys.executable, str(ROOT / "integrations" / "final" / "arena_protocol_v5.py"),
                   "--profile", name, *common]
    elif backend == "torch_cuda":
        torch_profile = "b224_erp" if name == "sutrack_b224" else "t224_erp"
        command = [sys.executable, str(ROOT / "integrations" / "final" / "arena_protocol_v5_torch.py"),
                   "--profile", torch_profile, *common]
    else:
        raise ValueError(f"unsupported profile backend: {backend}")

    print(json.dumps({"profile": name, "profile_file": str(profile_path), "command": command},
                     ensure_ascii=False, indent=2))
    if args.print_only:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
