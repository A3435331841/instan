#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Capture a reproducible, credential-free server-exit audit bundle."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import time
from pathlib import Path

try:
    from scripts.server_exit_sftp_batch import Session
except ImportError:  # direct execution from the scripts directory
    from server_exit_sftp_batch import Session


ARCHIVED_ROOTS = (
    "/data/sutrack_lora_training",
    "/data/training_spherical_v5_20260826",
    "/data/training_headonly_v3_20260825",
    "/data/training_headonly_v4_cont_20260825",
    "/data/weights",
    "/data/runs",
    "/data/pano360",
    "/data/wheels",
    "/data/uetrack_src_20260825",
    "/data/sutrack_src_20260825",
    "/data/lorat_src_20260825",
)
EXCLUDED_ROOTS = (
    ("/data/traindata", "local official dataset already present; do not duplicate"),
    ("/data/finetune", "rebuildable derived set; keep manifest only"),
    ("/home/wuyou/grt_env", "recreate from freeze and wheels; do not copy venv"),
)


def run(session: Session, command: str, timeout: int = 900) -> str:
    if session.client is None:
        raise RuntimeError("SSH session is not connected")
    _, stdout, stderr = session.client.exec_command(command, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace").strip()
    if err:
        out += "\n[stderr]\n" + err + "\n"
    return out


def remote_index(session: Session, roots: list[str]) -> tuple[list[dict], dict[str, str]]:
    rows: list[dict] = []
    hashes: dict[str, str] = {}
    for root in roots:
        qroot = shlex.quote(root)
        command = f"find {qroot} -type f -printf '%p\\t%s\\t%T@\\n'"
        text = run(session, command, timeout=900)
        for line in text.splitlines():
            if "\t" not in line:
                continue
            path, size, mtime = line.rsplit("\t", 2)
            try:
                rows.append({"path": path, "size": int(size), "mtime": float(mtime), "root": root,
                             "scope": "archived"})
            except ValueError:
                continue
        hash_text = run(session, f"find {qroot} -type f -print0 | xargs -0 -r sha256sum", timeout=1800)
        for line in hash_text.split("\n"):
            if len(line) >= 66 and len(line[:64]) == 64:
                hashes[line[66:]] = line[:64].lower()
    return rows, hashes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\instan\grt360_storage\experiments\server_exit_20260827")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    control = root / "server_control"
    control.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with Session() as session:
        (control / "server_processes.txt").write_text(
            run(session, "date -Is; ps -eo pid,ppid,stat,etime,pcpu,pmem,args --sort=-pcpu | head -120"),
            encoding="utf-8")
        (control / "server_disk_usage.txt").write_text(
            run(session, "df -h / /data; echo '--- data top-level ---'; du -sh /data/* 2>/dev/null | sort -h"),
            encoding="utf-8")
        (control / "python-pip-freeze.txt").write_text(
            run(session, "python3 --version; /home/wuyou/grt_env/bin/python --version; "
                "/home/wuyou/grt_env/bin/python -m pip freeze"),
            encoding="utf-8")
        (control / "cuda-system-info.txt").write_text(
            run(session, "nvidia-smi; echo '--- query ---'; "
                "nvidia-smi --query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu "
                "--format=csv,noheader"), encoding="utf-8")
        rows, hashes = remote_index(session, list(ARCHIVED_ROOTS))
        # Include direct /data control files in the inventory/hash bundle.
        top_text = run(session, "find /data -maxdepth 1 -type f -printf '%p\\t%s\\t%T@\\n'")
        for line in top_text.splitlines():
            if "\t" not in line:
                continue
            path, size, mtime = line.rsplit("\t", 2)
            rows.append({"path": path, "size": int(size), "mtime": float(mtime),
                         "root": "/data", "scope": "control"})
        top_hash_text = run(session, "find /data -maxdepth 1 -type f -print0 | xargs -0 -r sha256sum")
        for line in top_hash_text.split("\n"):
            if len(line) >= 66 and len(line[:64]) == 64:
                hashes[line[66:]] = line[:64].lower()
        for path, reason in EXCLUDED_ROOTS:
            output = run(session, f"du -sb -- {shlex.quote(path)} 2>/dev/null || true")
            value = output.strip().split()[0] if output.strip() else ""
            rows.append({"path": path, "size": int(value) if value.isdigit() else "",
                         "mtime": "", "root": path, "scope": "excluded", "reason": reason})

    with (control / "remote_file_inventory.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["path", "size", "mtime", "root", "scope", "reason"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (control / "remote_SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in sorted(hashes):
            stream.write(f"{hashes[path]}  {path}\n")
    audit = {
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "archived_roots": list(ARCHIVED_ROOTS),
        "excluded_roots": [{"path": path, "reason": reason} for path, reason in EXCLUDED_ROOTS],
        "inventory_files": len(rows),
        "hashed_files": len(hashes),
        "credentials_recorded": False,
    }
    (control / "server_exit_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
