#!/usr/bin/env python3
"""Validate a source-only Git checkout or a local delivery package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY-----"),
    # Capture the right hand side so environment-variable indirections such
    # as password=os.environ["GRT_SSH_PASS"] are not mistaken for secrets.
    re.compile(r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*([^\s#]+)"),
)
TEXT_SUFFIXES = {".py", ".sh", ".ps1", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".csv", ".ini"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_text(root: Path) -> list[str]:
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 8 * 1024 * 1024:
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(_is_secret_match(pattern, text) for pattern in SECRET_PATTERNS):
            findings.append(str(path))
    return findings


def _is_secret_match(pattern: re.Pattern, text: str) -> bool:
    for match in pattern.finditer(text):
        if not match.groups():
            return True
        rhs = match.group(1).strip().lower()
        if rhs.startswith(("os.environ", "os.getenv", "getenv(", "get_global_constant", "$env:", "${")):
            continue
        if rhs in {"none", "null", "required", "key", "value"}:
            continue
        if re.fullmatch(r"[a-z_][a-z0-9_]*(?:\[[^]]+\])?", rhs):
            continue
        return True
    return False


def check_repo(repo: Path) -> dict:
    tracked = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True).splitlines()
    large = []
    for name in tracked:
        path = repo / name
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024:
            large.append({"path": name, "bytes": path.stat().st_size})
    secrets = []
    for name in tracked:
        path = repo / name
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path.stat().st_size <= 8 * 1024 * 1024:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(_is_secret_match(pattern, text) for pattern in SECRET_PATTERNS):
                secrets.append(str(path))
    docker_push = []
    for path in (repo / "docker").rglob("*") if (repo / "docker").is_dir() else []:
        if path.is_file() and path.suffix.lower() in {".dockerfile", ".sh", ".ps1"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"\bdocker\s+push\b", text, re.I):
                docker_push.append(str(path))
    return {"tracked_count": len(tracked), "tracked_over_50MiB": large,
            "secret_findings": secrets, "docker_push_in_build_files": docker_push}


def check_package(package: Path) -> dict:
    manifest_path = package / "asset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest.get("files", []):
        path = package / item["path"]
        if not path.is_file():
            failures.append({"path": item["path"], "error": "missing"})
            continue
        if path.stat().st_size != item["bytes"]:
            failures.append({"path": item["path"], "error": "size", "actual": path.stat().st_size})
        digest = sha256_file(path)
        if digest != item["sha256"]:
            failures.append({"path": item["path"], "error": "sha256", "actual": digest})
    secrets = scan_text(package)
    return {"package": str(package), "files": len(manifest.get("files", [])),
            "checksum_failures": failures, "secret_findings": secrets,
            "ok": not failures and not secrets}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--package", action="append", default=[])
    args = parser.parse_args(argv)
    repo = Path(args.repo or Path(__file__).resolve().parents[1]).resolve()
    result = {"repo": check_repo(repo), "packages": [check_package(Path(item).resolve()) for item in args.package]}
    result["ok"] = (not result["repo"]["tracked_over_50MiB"] and
                     not result["repo"]["secret_findings"] and
                     all(item["ok"] for item in result["packages"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
