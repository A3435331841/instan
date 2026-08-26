#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive high-file-count server trees through one pipelined SFTP transfer.

The server's SFTP directory listing can stall on ``runs`` (about 9k files),
although ordinary ``sftp.get`` is reliable.  This helper makes a read-only
remote tar archive, downloads it atomically, records the archive SHA256, and
extracts it into the requested local directory.  The remote archive is left
in place intentionally: the migration is non-destructive and the server is
nearing expiry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import shlex
import stat
import tarfile
import time
from pathlib import Path

import paramiko


CHUNK = 8 * 1024 * 1024


def connect() -> paramiko.SSHClient:
    required = ("GRT_SSH_HOST", "GRT_SSH_PORT", "GRT_SSH_USER", "GRT_SSH_PASS")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit("missing SSH environment variables: " + ",".join(missing))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(os.environ["GRT_SSH_HOST"], port=int(os.environ["GRT_SSH_PORT"]),
                   username=os.environ["GRT_SSH_USER"], password=os.environ["GRT_SSH_PASS"],
                   timeout=30, banner_timeout=30, auth_timeout=30)
    return client


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def remote_hashes(client: paramiko.SSHClient, remote_root: str, top_level: bool = False) -> dict[str, str]:
    qroot = shlex.quote(remote_root)
    depth = "-maxdepth 1 " if top_level else ""
    command = f"find {qroot} {depth}-type f -print0 | xargs -0 -r sha256sum"
    _, stdout, stderr = client.exec_command(command, timeout=1800)
    text = stdout.read().decode("utf-8", "surrogateescape")
    error = stderr.read().decode("utf-8", "replace").strip()
    if error:
        raise RuntimeError(f"remote file hash index failed: {error}")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) >= 66 and len(line[:64]) == 64:
            result[line[66:]] = line[:64].lower()
    return result


def safe_extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r:") as stream:
        for member in stream.getmembers():
            name = member.name.replace("\\", "/")
            if name.startswith("/") or name == ".." or name.startswith("../") or "/../" in name:
                raise RuntimeError(f"unsafe tar member: {member.name}")
            target = (destination / name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"tar member escapes destination: {member.name}")
            stream.extract(member, destination)
            count += 1
    return count


def verify_tree(root: Path, remote_root: str, hashes: dict[str, str]) -> tuple[int, int]:
    checked = mismatches = 0
    prefix = remote_root.rstrip("/") + "/"
    for remote_file, expected in hashes.items():
        if not remote_file.startswith(prefix):
            continue
        relative = Path(*remote_file[len(prefix):].split("/"))
        local = root / relative
        checked += 1
        if not local.is_file() or local.stat().st_size == 0 and expected != digest(local):
            mismatches += 1
            continue
        if digest(local) != expected:
            mismatches += 1
    return checked, mismatches


def archive_one(client: paramiko.SSHClient, remote_root: str, local_root: Path,
                staging: Path, record: dict, top_level: bool = False) -> None:
    sftp = client.open_sftp()
    base = posixpath.basename(remote_root.rstrip("/")) or "root"
    remote_tar = f"/tmp/grt360_server_exit_{base}_20260827.tar"
    local_tar = staging / (base + ".tar")
    local_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    qtar, qroot = shlex.quote(remote_tar), shlex.quote(remote_root)
    if top_level:
        command = (f"find {qroot} -maxdepth 1 -type f -printf '%f\\0' | "
                   f"tar --null -cf {qtar} -C {qroot} -T -")
    else:
        command = f"tar -cf {qtar} -C {qroot} ."
    _, out, err = client.exec_command(command, timeout=1800)
    out.read()
    error = err.read().decode("utf-8", "replace").strip()
    if error:
        raise RuntimeError(f"remote tar failed for {remote_root}: {error}")
    remote_size = int(sftp.stat(remote_tar).st_size)
    _, out, err = client.exec_command(f"sha256sum -- {qtar}", timeout=1800)
    remote_digest = out.read().decode("utf-8", "replace").split()[0].lower()
    hash_error = err.read().decode("utf-8", "replace").strip()
    if hash_error or len(remote_digest) != 64:
        raise RuntimeError(f"remote tar hash failed for {remote_root}: {hash_error or 'empty'}")
    if not local_tar.is_file() or local_tar.stat().st_size != remote_size or digest(local_tar) != remote_digest:
        local_tar.with_name(local_tar.name + ".partial").unlink(missing_ok=True)
        sftp.get(remote_tar, str(local_tar.with_name(local_tar.name + ".partial")), prefetch=True)
        local_tar.with_name(local_tar.name + ".partial").replace(local_tar)
    if local_tar.stat().st_size != remote_size or digest(local_tar) != remote_digest:
        raise RuntimeError(f"archive SHA256 mismatch for {remote_root}")
    hashes = remote_hashes(client, remote_root, top_level=top_level)
    extracted = safe_extract(local_tar, local_root)
    checked = mismatches = 0
    for remote_file, expected in hashes.items():
        prefix = remote_root.rstrip("/") + "/"
        if not remote_file.startswith(prefix):
            continue
        relative = Path(*remote_file[len(prefix):].split("/"))
        local = local_root / relative
        checked += 1
        if not local.is_file() or digest(local) != expected:
            mismatches += 1
    if mismatches:
        raise RuntimeError(f"extracted tree hash mismatches for {remote_root}: {mismatches}/{checked}")
    record.update({"remote_root": remote_root, "local_root": str(local_root),
                   "remote_archive": remote_tar, "archive": str(local_tar),
                   "archive_size": remote_size, "archive_sha256": remote_digest,
                   "file_count": len(hashes), "extracted_members": extracted,
                   "status": "verified", "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    sftp.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827")
    parser.add_argument("--spec", action="append", required=True,
                        help="remote path=local relative path (repeatable)")
    parser.add_argument("--manifest-name", default="tar_archive_manifest.json")
    parser.add_argument("--top-level-control", action="store_true",
                        help="archive only regular files directly under /data")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    staging = root / ".tar_staging"
    records: list[dict] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    client = connect()
    try:
        for item in args.spec:
            if "=" not in item:
                raise SystemExit("--spec must be REMOTE=LOCAL")
            remote, local_rel = item.split("=", 1)
            print(f"[tar] {remote} -> {local_rel}", flush=True)
            record: dict = {"remote_root": remote, "local_root": str(root / local_rel)}
            archive_one(client, remote, root / local_rel, staging, record,
                        top_level=args.top_level_control and remote.rstrip("/") == "/data")
            records.append(record)
            (root / args.manifest_name).write_text(
                json.dumps({"started": started, "records": records}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            print(f"  verified {record['file_count']} files, {record['archive_size'] / 1e9:.2f} GB", flush=True)
    finally:
        client.close()
    (root / args.manifest_name).write_text(
        json.dumps({"started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "TAR_ARCHIVE_SHA256SUMS.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["remote_root", "local_root", "remote_archive", "archive",
                                                     "archive_size", "archive_sha256", "file_count",
                                                     "extracted_members", "status", "checked_at"])
        writer.writeheader()
        writer.writerows(records)
    print(f"manifest={root / args.manifest_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
