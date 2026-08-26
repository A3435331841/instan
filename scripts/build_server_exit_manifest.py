#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the final non-destructive server-exit manifest.

The SFTP manifests contain per-file hashes for checkpoint/weight/control
files.  Tar manifests contain the verified archive hash and the number of
files independently checked during extraction.  This combines both without
re-hashing tens of gigabytes a second time, and refuses to claim completion
when an expected transfer record or a ``.partial`` file is missing/unfinished.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    control = root / "server_control"
    sftp_files = [root / "sftp_p0_manifest.json", root / "sftp_p1_checkpoints_weights_manifest.json"]
    tar_files = [root / "tar_p2_manifest.json", root / "tar_sources_manifest.json",
                 root / "tar_control_manifest_retry.json"]
    records: list[dict] = []
    channels: list[str] = []
    for path in sftp_files:
        if not path.is_file():
            raise SystemExit(f"missing transfer manifest: {path}")
        channels.append(path.name)
        payload = read_json(path)
        for item in payload.get("records", []):
            status = item.get("status")
            if status not in {"verified_existing", "transferred_verified"}:
                raise SystemExit(f"unverified SFTP record: {item}")
            local = Path(item["local"])
            if not local.is_file() or local.stat().st_size != int(item["size"]):
                raise SystemExit(f"missing/size-mismatched SFTP artifact: {local}")
            records.append({"type": "file", **item})
    tar_records: list[dict] = []
    for path in tar_files:
        if not path.is_file():
            raise SystemExit(f"missing tar manifest: {path}")
        channels.append(path.name)
        payload = read_json(path)
        for item in payload.get("records", []):
            if item.get("status") != "verified":
                raise SystemExit(f"unverified tar record: {item}")
            archive = Path(item["archive"])
            if not archive.is_file() or archive.stat().st_size != int(item["archive_size"]):
                raise SystemExit(f"missing/size-mismatched archive: {archive}")
            tar_records.append({"type": "tree_archive", **item})
    # Abandoned transfers are deliberately retained under the quarantine
    # directory for forensic recovery.  They are not part of the deliverable
    # closure; every partial outside that directory must be resolved first.
    partials = [str(path) for path in root.rglob("*.partial")
                if path.is_file() and "failed_partials_20260827" not in path.parts]
    if partials:
        raise SystemExit(f"unfinished partial files remain: {partials[:5]}")
    audit = read_json(control / "server_exit_audit.json") if (control / "server_exit_audit.json").is_file() else {}
    rows = sorted(records, key=lambda item: str(item.get("remote", "")))
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "complete": True,
        "artifact_root": str(root),
        "channels": channels,
        "file_records": len(records),
        "tree_records": len(tar_records),
        "records": rows,
        "tree_archives": tar_records,
        "remote_audit": audit,
        "remote_excluded": audit.get("excluded_roots", []),
        "partials": [],
        "credentials_recorded": False,
    }
    manifest_path = root / "server_exit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    sums_path = root / "SHA256SUMS.csv"
    with sums_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["type", "remote", "local", "size", "sha256", "status",
                                                     "archive", "archive_size", "archive_sha256", "file_count"])
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
        for item in tar_records:
            writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
    complete = {
        "complete": True,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(manifest_path),
        "sha256_csv": str(sums_path),
        "sftp_files_verified": len(records),
        "trees_verified": len(tar_records),
        "remote_inventory_files": audit.get("inventory_files"),
        "remote_hashed_files": audit.get("hashed_files"),
        "excluded_roots": audit.get("excluded_roots", []),
        "no_remote_delete": True,
        "docker_push": False,
    }
    (root / "MIGRATION_COMPLETE.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(complete, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
