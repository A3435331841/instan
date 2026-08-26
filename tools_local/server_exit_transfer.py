#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resumable SSH/SFTP server-exit sync with per-file SHA256 verification.

Credentials are read only from GRT_SSH_HOST/GRT_SSH_PORT/GRT_SSH_USER/
GRT_SSH_PASS.  They are never written to manifests or printed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import socket
import time
from dataclasses import dataclass
from pathlib import Path

import paramiko


CHUNK = 1 * 1024 * 1024
REOPEN_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Spec:
    remote: str
    local: str
    kind: str


class RemoteSession:
    """Reconnectable SSH/SFTP session for hosts with long-read limits."""

    def __init__(self):
        self.client = None
        self.sftp = None
        self.reset()

    def reset(self):
        if self.sftp is not None:
            try:
                self.sftp.close()
            except OSError:
                pass
        if self.client is not None:
            self.client.close()
        self.client = connect()
        transport = self.client.get_transport()
        if transport is not None and transport.sock is not None:
            transport.sock.settimeout(60.0)
        self.sftp = self.client.open_sftp()

    def close(self):
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()


def default_specs(root: Path) -> list[Spec]:
    return [
        Spec("/data/sutrack_lora_training", "checkpoints/sutrack_lora", "checkpoint"),
        Spec("/data/training_spherical_v5_20260826", "checkpoints/v5", "checkpoint"),
        Spec("/data/training_headonly_v3_20260825", "checkpoints/v3", "checkpoint"),
        Spec("/data/training_headonly_v4_cont_20260825", "checkpoints/v4", "checkpoint"),
        Spec("/data/weights", "weights", "weight"),
        Spec("/data/runs", "runs", "result"),
        Spec("/data/pano360", "remote_workspace", "workspace"),
        Spec("/data/wheels", "environment/wheels", "environment"),
        Spec("/data/uetrack_src_20260825", "upstream_sources/uetrack", "source"),
        Spec("/data/sutrack_src_20260825", "upstream_sources/sutrack", "source"),
        Spec("/data/lorat_src_20260825", "upstream_sources/lorat", "source"),
    ]


def connect():
    required = ("GRT_SSH_HOST", "GRT_SSH_PORT", "GRT_SSH_USER", "GRT_SSH_PASS")
    missing = [key for key in required if not os.environ.get(key)]
    if missing:
        raise SystemExit("missing SSH environment variables: " + ",".join(missing))
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        os.environ["GRT_SSH_HOST"],
        port=int(os.environ["GRT_SSH_PORT"]),
        username=os.environ["GRT_SSH_USER"],
        password=os.environ["GRT_SSH_PASS"],
        timeout=25,
        banner_timeout=25,
        auth_timeout=25,
    )
    return client


def local_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_hash(client, path: str) -> str:
    command = "sha256sum -- " + "'" + path.replace("'", "'\\''") + "'"
    _, stdout, stderr = client.exec_command(command, timeout=300)
    error = stderr.read().decode("utf-8", "replace").strip()
    if error:
        raise RuntimeError(f"remote hash failed for {path}: {error}")
    value = stdout.read().decode("utf-8", "replace").split()
    if not value:
        raise RuntimeError(f"remote hash returned no value for {path}")
    return value[0]


def iter_files(sftp, root: str):
    """Yield (remote_file, size, mtime) recursively, without following links."""
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            attrs = sftp.lstat(current)
        except OSError as exc:
            raise RuntimeError(f"remote path unavailable: {current}: {exc}") from exc
        mode = attrs.st_mode
        if mode & 0o170000 == 0o100000:
            yield current, int(attrs.st_size), float(attrs.st_mtime)
            continue
        if mode & 0o170000 != 0o040000:
            continue
        for child in sftp.listdir_attr(current):
            path = posixpath.join(current, child.filename)
            child_mode = child.st_mode
            if child_mode & 0o170000 == 0o100000:
                yield path, int(child.st_size), float(child.st_mtime)
            elif child_mode & 0o170000 == 0o040000:
                stack.append(path)


def safe_relative(remote_root: str, remote_file: str) -> str:
    root = remote_root.rstrip("/")
    if remote_file == root:
        return Path(posixpath.basename(remote_file)).name
    prefix = root + "/"
    if not remote_file.startswith(prefix):
        raise RuntimeError(f"remote file escaped root: {remote_file}")
    return remote_file[len(prefix):]


def sync_file(session: RemoteSession, remote_file: str, local_file: Path,
              expected_size: int, record: dict):
    local_file.parent.mkdir(parents=True, exist_ok=True)
    record.update({"remote": remote_file, "local": str(local_file), "size": expected_size})
    remote_digest = remote_hash(session.client, remote_file)
    record["sha256"] = remote_digest
    record["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if local_file.is_file() and local_file.stat().st_size == expected_size:
        local_digest = local_hash(local_file)
        if local_digest == remote_digest:
            record["status"] = "verified_existing"
            return record
        conflict = local_file.with_name(local_file.name + ".conflict")
        conflict.unlink(missing_ok=True)
        local_file.replace(conflict)

    partial = local_file.with_name(local_file.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > expected_size:
        partial.unlink()
        offset = 0
    with partial.open("ab") as target:
        # Reopen the SFTP file periodically.  Some hosted SSH endpoints stop
        # servicing a long-lived read channel after a few hundred MiB even
        # though the connection itself remains alive; bounded sessions make
        # the operation restartable without losing the verified partial file.
        while offset < expected_size:
            block_end = min(expected_size, offset + REOPEN_BYTES)
            attempts = 0
            while offset < block_end:
                try:
                    with session.sftp.open(remote_file, "rb") as source:
                        source.settimeout(30.0)
                        source.seek(offset)
                        while offset < block_end:
                            data = source.read(min(CHUNK, block_end - offset))
                            if not data:
                                raise RuntimeError(
                                    f"remote file ended early: {remote_file} at {offset}/{expected_size}")
                            target.write(data)
                            offset += len(data)
                            target.flush()
                    attempts = 0
                except (OSError, EOFError, RuntimeError, socket.timeout):
                    attempts += 1
                    if attempts > 3:
                        raise
                    session.reset()
            # A fresh SSH connection prevents a long-lived SFTP channel from
            # becoming unresponsive after a few hundred MiB.
            if offset < expected_size:
                session.reset()
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"local partial size mismatch: {partial}")
    partial.replace(local_file)
    local_digest = local_hash(local_file)
    if local_digest != remote_digest:
        raise RuntimeError(f"SHA256 mismatch after transfer: {remote_file}")
    record["status"] = "transferred_verified"
    return record


def write_manifest(output: Path, records: list[dict], started: str):
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "transfer_manifest.json"
    json_path.write_text(json.dumps({
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.environ.get("GRT_SSH_HOST", ""),
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output / "SHA256SUMS.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["remote", "local", "size", "sha256", "status", "checked_at"])
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827")
    parser.add_argument("--only", action="append", default=[], help="remote prefix to sync; repeatable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    specs = default_specs(root)
    if args.only:
        specs = [spec for spec in specs if any(spec.remote.startswith(prefix) for prefix in args.only)]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records = []
    session = RemoteSession()
    try:
        for spec in specs:
            remote_files = list(iter_files(session.sftp, spec.remote))
            print(f"[{spec.kind}] {spec.remote}: {len(remote_files)} files")
            for remote_file, size, mtime in remote_files:
                relative = safe_relative(spec.remote, remote_file)
                local_file = root / spec.local / Path(*relative.split("/"))
                record = {"kind": spec.kind, "mtime": mtime}
                if args.dry_run:
                    record.update({"remote": remote_file, "local": str(local_file),
                                   "size": size, "status": "dry_run"})
                else:
                    sync_file(session, remote_file, local_file, size, record)
                records.append(record)
                print(f"  {record['status']}: {relative} ({size / 1e9:.2f} GB)" if size >= 1e9
                      else f"  {record['status']}: {relative} ({size} bytes)", flush=True)
    finally:
        session.close()
    json_path, csv_path = write_manifest(root, records, started)
    print(f"manifest={json_path}")
    print(f"sha256_csv={csv_path}")
    print(f"files={len(records)}")


if __name__ == "__main__":
    main()
