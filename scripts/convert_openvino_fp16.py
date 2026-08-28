#!/usr/bin/env python3
"""Convert OpenVINO IR graphs to FP16 copies without touching the originals."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", action="append", required=True,
                    help="source XML (repeatable; output is <stem>_fp16.xml)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    import openvino as ov

    core = ov.Core()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for raw in args.src:
        src = Path(raw).resolve()
        dst = out_dir / f"{src.stem}_fp16.xml"
        model = core.read_model(str(src))
        ov.save_model(model, str(dst), compress_to_fp16=True)
        print(dst, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
