#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge per-channel server-exit transfer manifests and verify closure."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def merge(root: Path, verify_local: bool = True) -> tuple[Path, Path, Path]:
    manifests = sorted(root.glob("transfer_manifest_*.json"))
    # Include the original default name if a previous run used it.
    default = root / "transfer_manifest.json"
    if default.is_file() and default not in manifests:
        manifests.append(default)
    records = {}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for record in payload.get("records", []):
            remote = str(record.get("remote", ""))
            if remote:
                records[remote] = dict(record)
    partials = [str(path) for path in root.rglob("*.partial") if path.is_file()]
    if partials:
        raise RuntimeError(f"archive is incomplete; partial files remain: {partials[:5]}")
    if not records:
        raise RuntimeError("no transfer records found")
    if verify_local:
        for remote, record in records.items():
            local = Path(record["local"])
            expected = int(record["size"])
            if not local.is_file() or local.stat().st_size != expected:
                raise RuntimeError(f"local artifact missing or size mismatch: {remote} -> {local}")
            if record.get("status") not in {"verified_existing", "transferred_verified"}:
                raise RuntimeError(f"artifact not verified: {remote} status={record.get('status')}")
    rows = sorted(records.values(), key=lambda record: str(record.get("remote", "")))
    manifest_path = root / "server_exit_manifest.json"
    manifest_path.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_root": str(root),
        "complete": True,
        "channels": [path.name for path in manifests],
        "records": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    sums_path = root / "SHA256SUMS.csv"
    with sums_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["remote", "local", "size", "sha256", "status"])
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    complete_path = root / "MIGRATION_COMPLETE.json"
    complete_path.write_text(json.dumps({
        "complete": True,
        "manifest": str(manifest_path),
        "sha256_csv": str(sums_path),
        "files": len(rows),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path, sums_path, complete_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args(argv)
    paths = merge(Path(args.root).resolve(), verify_local=not args.allow_unverified)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
