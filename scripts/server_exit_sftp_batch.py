#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-writer, resumable SFTP archive for the GRT-360 server exit.

The earlier exec/dd transport is useful on hosts that throttle SFTP, but some
hosted SSH endpoints expose a short-read channel.  This utility deliberately
uses one normal SFTP reader, resumes only from a verified byte boundary, and
renames a ``.partial`` file only after its size and SHA256 match the remote
file.  It never removes a remote file and never prints credentials.

Credentials are read from ``GRT_SSH_HOST``, ``GRT_SSH_PORT``,
``GRT_SSH_USER`` and ``GRT_SSH_PASS``.  The default profile archives only
reproducibility-critical directories; the large, locally reproducible
``traindata`` and ``finetune`` trees are intentionally excluded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import posixpath
import socket
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import paramiko


READ_BYTES = 8 * 1024 * 1024
REOPEN_BYTES = 512 * 1024 * 1024
HASH_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class RootSpec:
    remote: str
    local: str
    kind: str


DEFAULT_ROOTS = (
    RootSpec("/data/sutrack_lora_training", "checkpoints/sutrack_lora", "checkpoint"),
    RootSpec("/data/training_spherical_v5_20260826", "checkpoints/v5", "checkpoint"),
    RootSpec("/data/training_headonly_v3_20260825", "checkpoints/v3", "checkpoint"),
    RootSpec("/data/training_headonly_v4_cont_20260825", "checkpoints/v4", "checkpoint"),
    RootSpec("/data/weights", "weights", "weight"),
    RootSpec("/data/runs", "runs", "result"),
    RootSpec("/data/pano360", "remote_workspace", "workspace"),
    RootSpec("/data/wheels", "environment/wheels", "environment"),
    RootSpec("/data/uetrack_src_20260825", "upstream_sources/uetrack", "source"),
    RootSpec("/data/sutrack_src_20260825", "upstream_sources/sutrack", "source"),
    RootSpec("/data/lorat_src_20260825", "upstream_sources/lorat", "source"),
)


class Session:
    def __init__(self) -> None:
        self.client: paramiko.SSHClient | None = None
        self.sftp: paramiko.SFTPClient | None = None
        self.connect()

    def connect(self) -> None:
        self.close()
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
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        self.client = client
        self.sftp = client.open_sftp()

    def close(self) -> None:
        if self.sftp is not None:
            try:
                self.sftp.close()
            except OSError:
                pass
            self.sftp = None
        if self.client is not None:
            try:
                self.client.close()
            except OSError:
                pass
            self.client = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def require_sftp(self) -> paramiko.SFTPClient:
        if self.sftp is None:
            raise RuntimeError("SFTP session is not connected")
        return self.sftp

    def remote_hash(self, remote: str) -> str:
        if self.client is None:
            raise RuntimeError("SSH session is not connected")
        quoted = "'" + remote.replace("'", "'\\''") + "'"
        _, stdout, stderr = self.client.exec_command("sha256sum -- " + quoted, timeout=900)
        value = stdout.read().decode("utf-8", "replace").split()
        error = stderr.read().decode("utf-8", "replace").strip()
        if error or not value:
            raise RuntimeError(f"remote SHA256 failed for {remote}: {error or 'empty output'}")
        return value[0].lower()

    def remote_hashes(self, root: str, top_level: bool = False) -> dict[str, str]:
        """Compute a whole-tree SHA256 index in one remote shell call."""
        if self.client is None:
            raise RuntimeError("SSH session is not connected")
        quoted = "'" + root.replace("'", "'\\''") + "'"
        depth = "-maxdepth 1 " if top_level else ""
        command = f"find {quoted} {depth}-type f -print0 | xargs -0 -r sha256sum"
        _, stdout, stderr = self.client.exec_command(command, timeout=1800)
        raw = stdout.read().decode("utf-8", "surrogateescape")
        error = stderr.read().decode("utf-8", "replace").strip()
        if error:
            raise RuntimeError(f"remote SHA256 index failed for {root}: {error}")
        hashes: dict[str, str] = {}
        for line in raw.splitlines():
            if len(line) < 66:
                continue
            digest, name = line[:64].lower(), line[66:]
            if len(digest) == 64 and name:
                hashes[name] = digest
        return hashes

    def remote_files(self, root: str, top_level: bool = False) -> list[tuple[str, int, float]]:
        """Ask the remote shell for an index instead of recursively listing SFTP.

        The provider used for this machine occasionally stalls on a large
        SFTP ``OPENDIR`` response even though ordinary file reads work.  The
        remote ``find`` index is small, deterministic, and avoids that failure
        mode.  Names are emitted as a NUL-delimited stream so spaces are safe.
        """
        if self.client is None:
            raise RuntimeError("SSH session is not connected")
        quoted = "'" + root.replace("'", "'\\''") + "'"
        depth = "-maxdepth 1 " if top_level else ""
        command = (
            "find " + quoted + " " + depth + "-type f -printf '%p\\t%s\\t%T@\\0'"
        )
        _, stdout, stderr = self.client.exec_command(command, timeout=300)
        raw = stdout.read()
        error = stderr.read().decode("utf-8", "replace").strip()
        if error:
            raise RuntimeError(f"remote file index failed for {root}: {error}")
        records: list[tuple[str, int, float]] = []
        for item in raw.split(b"\0"):
            if not item:
                continue
            try:
                name, size, mtime = item.decode("utf-8", "surrogateescape").split("\t")
                records.append((name, int(size), float(mtime)))
            except (ValueError, UnicodeDecodeError) as exc:
                raise RuntimeError(f"malformed remote file index entry for {root}") from exc
        return records


def local_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_files(sftp: paramiko.SFTPClient, root: str) -> Iterable[tuple[str, int, float]]:
    """Yield regular files without following symlinks."""
    stack = [root.rstrip("/")]
    while stack:
        current = stack.pop()
        attrs = sftp.lstat(current)
        mode = attrs.st_mode
        if stat.S_ISREG(mode):
            yield current, int(attrs.st_size), float(attrs.st_mtime)
            continue
        if not stat.S_ISDIR(mode):
            continue
        for child in sftp.listdir_attr(current):
            child_path = posixpath.join(current, child.filename)
            if stat.S_ISREG(child.st_mode):
                yield child_path, int(child.st_size), float(child.st_mtime)
            elif stat.S_ISDIR(child.st_mode):
                stack.append(child_path)


def walk_top_level(sftp: paramiko.SFTPClient, root: str) -> Iterable[tuple[str, int, float]]:
    for child in sftp.listdir_attr(root.rstrip("/")):
        if stat.S_ISREG(child.st_mode):
            yield posixpath.join(root.rstrip("/"), child.filename), int(child.st_size), float(child.st_mtime)


def relative_to(remote_root: str, remote_file: str) -> str:
    root = remote_root.rstrip("/")
    prefix = root + "/"
    if not remote_file.startswith(prefix):
        raise RuntimeError(f"remote path escaped root: {remote_file}")
    return remote_file[len(prefix):]


def download_one(session: Session, remote: str, target: Path, size: int, remote_digest: str) -> str:
    """Download one file through Paramiko's pipelined SFTP getter.

    This endpoint has exhibited stalled ``SFTPFile.read`` calls, while
    Paramiko's built-in ``get`` (which enables read-ahead/prefetch) is stable.
    A failed attempt leaves the partial file in place for inspection; the next
    attempt restarts that individual file from byte zero rather than risking a
    hole or duplicate append.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == size:
        if local_hash(target) == remote_digest:
            return "verified_existing"
        conflict = target.with_name(target.name + ".conflict")
        if conflict.exists():
            conflict.unlink()
        target.replace(conflict)

    partial = target.with_name(target.name + ".partial")
    if partial.exists() and partial.stat().st_size > size:
        partial.unlink()
    attempts = 0
    while True:
        try:
            # SFTPClient.get uses pipelined reads.  It truncates the partial
            # destination at the start of each attempt, so a retry cannot
            # duplicate bytes from a previous failed attempt.
            session.require_sftp().get(remote, str(partial), prefetch=True)
            break
        except (OSError, EOFError, paramiko.SSHException, socket.timeout) as exc:
            attempts += 1
            if attempts > 5:
                raise RuntimeError(f"download failed after retries: {remote}: {exc}") from exc
            session.connect()
            time.sleep(min(5 * attempts, 20))
    if partial.stat().st_size != size:
        raise RuntimeError(f"partial size mismatch for {remote}: {partial.stat().st_size} != {size}")
    if local_hash(partial) != remote_digest:
        raise RuntimeError(f"SHA256 mismatch after transfer: {remote}")
    partial.replace(target)
    return "transferred_verified"

def write_manifest(root: Path, records: list[dict], started: str, name: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.environ.get("GRT_SSH_HOST", ""),
        "records": records,
    }
    (root / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["kind", "remote", "local", "size", "mtime", "sha256", "status", "checked_at"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827")
    parser.add_argument("--only", action="append", default=[], help="sync only this remote prefix; repeatable")
    parser.add_argument("--top-level", action="store_true", help="also archive regular files directly under /data")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-name", default="sftp_transfer_manifest.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    specs = [spec for spec in DEFAULT_ROOTS if not args.only or any(spec.remote.startswith(p) for p in args.only)]
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    records: list[dict] = []
    with Session() as session:
        for spec in specs:
            try:
                files = session.remote_files(spec.remote)
            except (OSError, paramiko.SSHException) as exc:
                print(f"skip unavailable root {spec.remote}: {exc}", flush=True)
                continue
            total = sum(size for _, size, _ in files)
            print(f"[{spec.kind}] {spec.remote}: {len(files)} files, {total / 1e9:.2f} GB", flush=True)
            hashes = {} if args.dry_run else session.remote_hashes(spec.remote)
            for remote, size, mtime in files:
                local = root / spec.local / Path(*relative_to(spec.remote, remote).split("/"))
                record = {"kind": spec.kind, "remote": remote, "local": str(local), "size": size, "mtime": mtime}
                if args.dry_run:
                    record.update({"sha256": "", "status": "dry_run", "checked_at": ""})
                else:
                    digest = hashes.get(remote) or session.remote_hash(remote)
                    status = download_one(session, remote, local, size, digest)
                    record.update({"sha256": digest, "status": status,
                                   "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                    print(f"  {status}: {relative_to(spec.remote, remote)} ({size / 1e9:.2f} GB)", flush=True)
                records.append(record)
                if not args.dry_run and len(records) % 20 == 0:
                    write_manifest(root, records, started, args.manifest_name)
        if args.top_level:
            files = session.remote_files("/data", top_level=True)
            print(f"[control] /data: {len(files)} files", flush=True)
            hashes = {} if args.dry_run else session.remote_hashes("/data", top_level=True)
            for remote, size, mtime in files:
                local = root / "server_control" / "top_level" / Path(posixpath.basename(remote))
                record = {"kind": "control", "remote": remote, "local": str(local), "size": size, "mtime": mtime}
                if args.dry_run:
                    record.update({"sha256": "", "status": "dry_run", "checked_at": ""})
                else:
                    digest = hashes.get(remote) or session.remote_hash(remote)
                    status = download_one(session, remote, local, size, digest)
                    record.update({"sha256": digest, "status": status,
                                   "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                    print(f"  {status}: {posixpath.basename(remote)} ({size / 1e6:.1f} MB)", flush=True)
                records.append(record)
                if not args.dry_run and len(records) % 20 == 0:
                    write_manifest(root, records, started, args.manifest_name)
    write_manifest(root, records, started, args.manifest_name)
    print(f"manifest={root / args.manifest_name}")
    print(f"files={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
