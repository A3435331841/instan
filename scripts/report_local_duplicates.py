#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report duplicate local files by SHA256 without deleting anything."""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path


CHUNK = 8 * 1024 * 1024


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(roots: list[Path]):
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name.endswith(".partial"):
                continue
            parts = {part.lower() for part in path.parts}
            if "failed_partials_20260827" in parts or ".tar_staging" in parts:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            yield path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=r"D:\instan\grt360_storage\manifests\DUPLICATES_BY_SHA256.csv")
    parser.add_argument("--root", action="append", dest="roots", default=None,
                        help="directory to scan; may be repeated")
    args = parser.parse_args(argv)
    roots = [Path(item).resolve() for item in args.roots] if args.roots else [
        Path(r"D:\instan\grt360_storage\experiments\server_exit_20260827"),
        Path(r"D:\instan\grt360_storage\experiments\legacy_artifacts"),
        Path(r"D:\instan\grt360_storage\datasets\360vot_legacy"),
        Path(r"D:\instan\grt360_deliverables"),
    ]
    by_size: dict[int, list[Path]] = defaultdict(list)
    enumerated = 0
    for path in iter_files(roots):
        by_size[path.stat().st_size].append(path)
        enumerated += 1
        if enumerated % 1000 == 0:
            print(f"enumerated={enumerated}", flush=True)
    groups: dict[tuple[int, str], list[str]] = defaultdict(list)
    hashed = 0
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        for path in paths:
            groups[(size, file_hash(path))].append(str(path))
            hashed += 1
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256", "size", "count", "path"])
        writer.writeheader()
        duplicate_groups = 0
        for (size, checksum), paths in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1])):
            if len(paths) < 2:
                continue
            duplicate_groups += 1
            for path in sorted(paths):
                writer.writerow({"sha256": checksum, "size": size, "count": len(paths), "path": path})
    print(f"files={enumerated} size_collision_candidates={hashed} duplicate_groups={duplicate_groups} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
