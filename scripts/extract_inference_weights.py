#!/usr/bin/env python3
"""Strip optimizer and scheduler state from an ODTrack training checkpoint.

Training snapshots are large because they retain optimizer moments.  Arena
inference only needs the ``net`` state dictionary.  This tool is explicit and
non-destructive: it writes a distinct output file and reports source/output
keys; it never changes the original checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-key", default="net")
    args = parser.parse_args(argv)
    import torch

    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if target.exists():
        raise FileExistsError(target)
    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or args.state_key not in payload:
        raise KeyError(f"checkpoint does not contain state key {args.state_key!r}")
    state = payload[args.state_key]
    if not isinstance(state, dict):
        raise TypeError(f"{args.state_key!r} is not a state dictionary")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({args.state_key: state}, target)
    print({"input": str(source), "output": str(target),
           "source_keys": sorted(payload), "state_keys": len(state),
           "input_bytes": source.stat().st_size, "output_bytes": target.stat().st_size})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
