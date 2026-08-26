#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a checksum manifest for the optional GitHub Release assets.

No files are copied or uploaded here.  The manifest is intentionally separate
from Git history so a user with ``contents:write`` permission can later run
``gh release create``/``gh release upload`` against the exact same bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path


ASSETS = (
    ("odtrack_official_ep0300.pth.tar", Path(r"D:\instan\pano360\artifacts\server_snapshot\weights\ODTrack_ep0300.pth.tar")),
    ("odtrack_spherical_v5_ep0003.pth.tar", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\checkpoints\v5\checkpoints\train\odtrack\finetune_spherical_v5\ODTrack_ep0003.pth.tar")),
    ("sutrack_lora_ep0005.pth", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\checkpoints\sutrack_lora\sutrack_lora_ep0005.pth")),
    ("sutrack_b224_ep0180.pth.tar", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\SUTRACK_b224_ep0180.pth.tar")),
    ("sutrack_t224_ep0180.pth.tar", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\SUTRACK_t224_ep0180.pth.tar")),
    ("sutrack_ep0300.pth.tar", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\SUTRACK_ep0300.pth.tar")),
    ("uetrack_base.tar", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\uetrack_base.tar")),
    ("lorat_base.bin", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\lorat_base.bin")),
    ("dinov2_vitb14_pretrain.pth", Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827\weights\dinov2_vitb14_pretrain.pth")),
)
MAX_ASSET_BYTES = 2_000_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=r"D:\instan\grt360_storage\manifests")
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for release_name, path in ASSETS:
        if not path.is_file():
            raise SystemExit(f"missing release asset: {path}")
        size = path.stat().st_size
        if size >= MAX_ASSET_BYTES:
            raise SystemExit(f"asset exceeds conservative Release limit: {path} ({size})")
        rows.append({"release_name": release_name, "local_path": str(path), "size": size,
                     "sha256": sha256(path), "status": "staged"})
    payload = {"release": "grt360-server-exit-20260827", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "upload_performed": False, "assets": rows}
    (output / "RELEASE_ASSETS_20260827.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "RELEASE_ASSETS_20260827.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["release_name", "local_path", "size", "sha256", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"release": payload["release"], "assets": len(rows), "upload_performed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
