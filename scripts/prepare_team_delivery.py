#!/usr/bin/env python3
"""Materialize non-destructive local GRT-360 hand-off packages.

The Git repository remains source-only.  This script creates four local
directory packages and hard-links large assets from the existing storage
archive whenever the source and destination are on the same volume.  A hard
link is intentionally used instead of a second 30--40 GB copy; the original
files are never removed or modified.  ``PACK_TO_TAR.ps1`` in each package can
make a portable archive later when a second physical copy is wanted.

The package manifests are self-contained: every payload file gets a SHA256,
while the manifest and checksum file themselves are excluded from their own
hash list to avoid a circular digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


VERSION = "20260829"
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA|PGP) PRIVATE KEY-----"),
    re.compile(r"(?i)(?:password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*([^\s#]+)"),
)
TEXT_SUFFIXES = {".py", ".sh", ".ps1", ".yaml", ".yml", ".json", ".toml", ".md", ".txt", ".csv", ".ini"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"asset {pattern!r} not found below {root}")
    files = [item for item in matches if item.is_file()]
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one {pattern!r} below {root}, found {len(files)}")
    return files[0]


def _extract_git_archive(repo: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="grt360-source-", suffix=".tar", delete=False) as temp:
        archive_path = Path(temp.name)
    try:
        with archive_path.open("wb") as handle:
            proc = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "HEAD"],
                                  stdout=handle, stderr=subprocess.PIPE, check=False, text=False)
        if proc.returncode != 0:
            raise RuntimeError(f"git archive failed: {proc.stderr.decode(errors='replace')}")
        with tarfile.open(archive_path, "r") as archive:
            root = target.resolve()
            for member in archive.getmembers():
                destination = (target / member.name).resolve()
                if os.path.commonpath((str(root), str(destination))) != str(root):
                    raise RuntimeError(f"unsafe git archive member: {member.name}")
                archive.extract(member, target)
    finally:
        archive_path.unlink(missing_ok=True)


def _copy_tree_links(source: Path, target: Path, ignored_dirs: set[str] | None = None) -> list[tuple[Path, Path, str]]:
    ignored_dirs = ignored_dirs or set()
    records: list[tuple[Path, Path, str]] = []
    if not source.is_dir():
        raise FileNotFoundError(source)
    for current, dirs, files in os.walk(source):
        dirs[:] = sorted(name for name in dirs if name not in ignored_dirs and name not in {".git", "__pycache__"})
        files = sorted(files)
        current_path = Path(current)
        relative = current_path.relative_to(source)
        for name in files:
            source_file = current_path / name
            target_file = target / relative / name
            method = _link_or_copy(source_file, target_file)
            records.append((source_file, target_file, method))
    return records


class PackageBuilder:
    def __init__(self, root: Path, repo: Path, package_name: str):
        self.root = root
        self.repo = repo
        self.name = package_name
        self.records: list[dict] = []
        root.mkdir(parents=True, exist_ok=True)

    def add_file(self, source: Path, relative: str | Path, category: str = "asset") -> Path:
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self.root / Path(relative)
        method = _link_or_copy(source, target)
        self.records.append({"source": str(source), "path": target.relative_to(self.root).as_posix(),
                             "bytes": source.stat().st_size, "category": category, "method": method})
        return target

    def add_tree(self, source: Path, relative: str | Path, category: str = "asset",
                 ignored_dirs: set[str] | None = None) -> None:
        target = self.root / Path(relative)
        for source_file, target_file, method in _copy_tree_links(source.resolve(), target, ignored_dirs):
            self.records.append({"source": str(source_file.resolve()),
                                 "path": target_file.relative_to(self.root).as_posix(),
                                 "bytes": source_file.stat().st_size, "category": category, "method": method})

    def add_source_snapshot(self) -> None:
        _extract_git_archive(self.repo, self.root / "src")

    def write_text(self, relative: str | Path, content: str) -> None:
        target = self.root / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")

    def _scan_secrets(self, payload: Iterable[Path]) -> None:
        for path in payload:
            if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 8 * 1024 * 1024:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for pattern in SECRET_PATTERNS:
                for match in pattern.finditer(text):
                    if not match.groups():
                        raise RuntimeError(f"possible secret in delivery payload: {path}")
                    rhs = match.group(1).strip().lower()
                    if rhs.startswith(("os.environ", "os.getenv", "getenv(", "get_global_constant", "$env:", "${")):
                        continue
                    if rhs in {"none", "null", "required", "key", "value"}:
                        continue
                    if re.fullmatch(r"[a-z_][a-z0-9_]*(?:\[[^]]+\])?", rhs):
                        continue
                    raise RuntimeError(f"possible secret in delivery payload: {path}")

    def finalize(self, readme: str) -> dict:
        self.write_text("README.md", readme)
        pack_script = r'''[CmdletBinding()]
param([string]$Output = "")
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path (Split-Path -Parent $PSScriptRoot) ("$((Split-Path -Leaf $PSScriptRoot)).tar") }
$parent = Split-Path -Parent $Output
New-Item -ItemType Directory -Force -Path $parent | Out-Null
tar -cf $Output -C $PSScriptRoot .
Write-Host "Wrote $Output"
'''
        self.write_text("PACK_TO_TAR.ps1", pack_script)
        excluded = {"asset_manifest.json", "SHA256SUMS"}
        payload = sorted(path for path in self.root.rglob("*")
                         if path.is_file() and path.name not in excluded)
        self._scan_secrets(payload)
        files = []
        for path in payload:
            files.append({"path": path.relative_to(self.root).as_posix(),
                          "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        manifest = {
            "schema": "grt360.delivery_package.v1", "package": self.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_commit": subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip(),
            "large_files_policy": "local-only; hardlinks may reference grt360_storage; never push to GitHub",
            "files": files, "asset_records": self.records,
        }
        self.write_text("asset_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        sums = "\n".join(f"{entry['sha256']}  {entry['path']}" for entry in files) + "\n"
        self.write_text("SHA256SUMS", sums)
        return {"package": self.name, "path": str(self.root), "file_count": len(files),
                "bytes": sum(item["bytes"] for item in files),
                "hardlinks": sum(item["method"] == "hardlink" for item in self.records),
                "copies": sum(item["method"] == "copy" for item in self.records),
                "manifest": str(self.root / "asset_manifest.json"),
                "sha256sums": str(self.root / "SHA256SUMS")}


def _readme(title: str, purpose: str, build: str, notes: str = "") -> str:
    return f"""# {title}

{purpose}

## Reproduce

1. Read `SHA256SUMS` and verify every payload file.
2. Keep the package directory intact; large files may be NTFS hardlinks to
   `D:\\instan\\grt360_storage` and are therefore not a second backup.
3. Follow `src/docs/REPRODUCE_V5.md` and `src/docs/BUILD_ARENA_CUDA128.md`.

## Build

{build}

The build scripts only run a local `docker build`; they never run `docker push`.
{notes}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=None)
    parser.add_argument("--storage", default=None)
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-history", action="store_true",
                        help="omit the full lineage directory (useful for a quick package smoke)")
    args = parser.parse_args(argv)

    repo = Path(args.repo or Path(__file__).resolve().parents[1]).resolve()
    root_parent = repo.parent
    storage = Path(args.storage or root_parent / "grt360_storage").resolve()
    scratch = Path(args.scratch or root_parent / "grt360_scratch").resolve()
    out = Path(args.out or root_parent / "grt360_deliverables" / f"team_v5_{VERSION}").resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty delivery root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    exit_root = storage / "experiments" / "server_exit_20260827"
    weights = exit_root / "weights"
    checkpoints = exit_root / "checkpoints"
    v5_candidates = sorted(checkpoints.rglob("finetune_spherical_v5/ODTrack_ep0006.pth.tar"))
    if len(v5_candidates) != 1:
        raise RuntimeError(f"expected exactly one v5 ep0006 checkpoint, found {len(v5_candidates)}")
    v5_checkpoint = v5_candidates[0]
    lora_checkpoint = _find_one(checkpoints, "sutrack_lora_ep0005.pth")
    b_weight = weights / "SUTRACK_b224_ep0180.pth.tar"
    t_weight = weights / "SUTRACK_t224_ep0180.pth.tar"
    base_od = repo / "artifacts" / "server_snapshot" / "weights" / "ODTrack_ep0300.pth.tar"
    if not base_od.is_file():
        raise FileNotFoundError(base_od)
    sutrack_src = exit_root / "upstream_sources" / "sutrack" / "SUTrack"
    if not (sutrack_src / "lib").is_dir():
        raise FileNotFoundError(sutrack_src / "lib")
    lorat_src = exit_root / "upstream_sources" / "lorat" / "lorat"
    uetrack_src = exit_root / "upstream_sources" / "uetrack" / "UETrack"
    if not lorat_src.is_dir():
        raise FileNotFoundError(lorat_src)
    if not uetrack_src.is_dir():
        raise FileNotFoundError(uetrack_src)
    onnx_root = scratch / "onnx"
    onnx_names = ["sutrack_b224_frame.onnx", "sutrack_b224_s224_t128.onnx",
                  "sutrack_t224_s224_t112.onnx", "odtrack_first.onnx", "odtrack_state.onnx",
                  "odtrack_v5_ep6_first.onnx", "odtrack_v5_ep6_state.onnx"]
    onnx_assets = {name: onnx_root / name for name in onnx_names}
    for asset in onnx_assets.values():
        if not asset.is_file():
            raise FileNotFoundError(asset)
    v5_artifacts = scratch / "geometry_recovery_v5_artifacts_20260829"
    for artifact in ("autonomous_run_manifest.json", "failure_matrix_130.csv", "full130_summary.json",
                     "latency_summary.json", "route_policy.json", "scenario_summary.csv", "valid35_summary.json",
                     "MIGRATION_COMPLETE.json", "RESTORE.md"):
        if not (v5_artifacts / artifact).is_file():
            raise FileNotFoundError(v5_artifacts / artifact)

    packages: list[dict] = []
    ort = PackageBuilder(out / "GRT360_FINAL_ORT_CUDA128", repo, "GRT360_FINAL_ORT_CUDA128")
    ort.add_source_snapshot()
    for name, source in onnx_assets.items():
        ort.add_file(source, Path("models") / name, "onnx_model")
    for artifact in ("full130_summary.json", "valid35_summary.json", "latency_summary.json", "route_policy.json"):
        ort.add_file(v5_artifacts / artifact, Path("results") / artifact, "key_result")
    packages.append(ort.finalize(_readme(
        "GRT-360 v5 final — ONNX Runtime CUDA 12.8",
        "Primary submission/runtime package.  It contains the exact exported graphs for the locked v5 causal geometry route; no raw dataset or checkpoint lineage is included.",
        "From this directory run `powershell -ExecutionPolicy Bypass -File src/scripts/build_image.ps1 -Backend ort -Context . -Tag grt360-v5-ort:cu128`.\n\nThen run `docker run --rm --gpus device=0 -v <dataset>:/mnt/dataset:ro -v <result>:/mnt/result grt360-v5-ort:cu128`.",
        "The authoritative full130 reference is AUC 0.7007805295, SR 0.8535501637, weighted end-to-end 36.2231 FPS; 5090 must be remeasured.")))

    torch = PackageBuilder(out / "GRT360_FINAL_TORCH_CUDA128", repo, "GRT360_FINAL_TORCH_CUDA128")
    torch.add_source_snapshot()
    torch.add_tree(sutrack_src, "sutrack_src", "upstream_source", {".git", "__pycache__"})
    torch.add_file(b_weight, Path("models") / b_weight.name, "pytorch_weight")
    torch.add_file(t_weight, Path("models") / t_weight.name, "pytorch_weight")
    torch.add_file(v5_artifacts / "full130_summary.json", Path("results") / "full130_summary.json", "key_result")
    packages.append(torch.finalize(_readme(
        "GRT-360 PyTorch CUDA reference — CUDA 12.8",
        "Backend comparison/fallback package.  It runs the upstream SUTrack B224 or T224 ERP three-tile path; it is deliberately documented as a reference and is not claimed to be bit-for-bit identical to the ORT v5 multi-expert route.",
        "Run `powershell -ExecutionPolicy Bypass -File src/scripts/build_image.ps1 -Backend torch -Context . -Tag grt360-sutrack-torch:cu128`.\n\nSelect the model with `GRT360_TORCH_PROFILE=b224_erp` or pass `--profile t224_erp` to the entrypoint.",
        "Use `scripts/benchmark_cuda_backends.py` from the source snapshot to compare this image and the ORT image on exactly the same dataset and GPU.")))

    train = PackageBuilder(out / "GRT360_CONTINUE_TRAINING", repo, "GRT360_CONTINUE_TRAINING")
    train.add_source_snapshot()
    train.add_tree(sutrack_src, "upstream/sutrack", "upstream_source", {".git", "__pycache__"})
    train.add_tree(lorat_src, "upstream/lorat", "upstream_source", {".git", "__pycache__"})
    train.add_tree(uetrack_src, "upstream/uetrack", "upstream_source", {".git", "__pycache__"})
    train.add_file(v5_checkpoint, Path("checkpoints") / "ODTrack_ep0006.pth.tar", "training_checkpoint")
    train.add_file(lora_checkpoint, Path("checkpoints") / lora_checkpoint.name, "training_checkpoint")
    for item in (b_weight, t_weight, base_od):
        train.add_file(item, Path("weights") / item.name, "baseline_weight")
    for artifact in ("autonomous_run_manifest.json", "failure_matrix_130.csv", "full130_summary.json",
                     "latency_summary.json", "route_policy.json", "scenario_summary.csv", "valid35_summary.json",
                     "RESTORE.md"):
        train.add_file(v5_artifacts / artifact, Path("results") / artifact, "key_result")
    train.add_file(exit_root / "server_control" / "remote_SHA256SUMS", "provenance/remote_SHA256SUMS", "provenance")
    train.add_file(exit_root / "server_control" / "server_exit_audit.json", "provenance/server_exit_audit.json", "provenance")
    packages.append(train.finalize(_readme(
        "GRT-360 continuation package",
        "This package is for teammates who want to continue training or develop a new expert.  It includes the source snapshot, upstream provenance, v5 ep6 training checkpoint, LoRA ep5, baseline weights, key result matrices and server audit records.",
        "Install the environment described by `src/docs/REPRODUCE_V5.md`; use `src/scripts/train_presence_calibrator.py`, the spherical training scripts and `src/configs/repro/v5_final.json` as the starting contract.  Keep valid35 locked and never route by sequence name or ground truth.",
        "The v5 ODTrack training file is larger than its inference state because it contains optimizer moments.  Use `src/scripts/extract_inference_weights.py` to create a separate net-only copy when deploying; the original remains untouched.")))

    if not args.skip_history:
        history = PackageBuilder(out / "GRT360_HISTORY_ARCHIVE", repo, "GRT360_HISTORY_ARCHIVE")
        history.add_source_snapshot()
        history.add_tree(exit_root / "checkpoints", "server_exit/checkpoints", "checkpoint_lineage", {"__pycache__"})
        history.add_tree(exit_root / "weights", "server_exit/weights", "weight_lineage", {"__pycache__"})
        history.add_tree(exit_root / "runs", "server_exit/runs", "run_lineage", {"__pycache__"})
        history.add_tree(scratch / "onnx", "exports/onnx", "onnx_export", {"__pycache__"})
        history.add_tree(scratch / "openvino", "exports/openvino", "openvino_export", {"__pycache__"})
        history.add_tree(exit_root / "remote_workspace", "server_exit/remote_workspace_source", "remote_source",
                         {".git", "__pycache__", "artifacts", "models", "weights", "data360", "runs", "offline", ".codex_tmp"})
        for name in ("remote_file_inventory.csv", "remote_SHA256SUMS", "server_exit_audit.json",
                     "server_disk_usage.txt", "cuda-system-info.txt", "python-pip-freeze.txt", "system_and_environment.txt"):
            history.add_file(exit_root / "server_control" / name, Path("server_exit/server_control") / name, "provenance")
        packages.append(history.finalize(_readme(
            "GRT-360 history archive",
            "Complete local lineage for checkpoints, exported graphs, remote runs and source/provenance needed to audit older experiments.  It intentionally omits raw train data (already present locally) and rebuildable caches.",
            "This is an archive, not a direct Docker build context.  Use one of the two final packages for image builds; use the source snapshot and manifests here to inspect history.",
            "Large directories are normally hardlinks to the existing server-exit archive.  Run `PACK_TO_TAR.ps1` only when a portable physical archive is required.")))

    index = {
        "schema": "grt360.local_packages_index.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": str(repo), "source_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "github_policy": "source/config/docs/manifests only; no large GitHub blobs and no competition docker push",
        "packages": packages,
    }
    (out / "LOCAL_PACKAGES_INDEX.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "MIGRATION_COMPLETE.json").write_text(json.dumps({
        "schema": "grt360.delivery_migration.v1", "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(), "source_commit": index["source_commit"],
        "large_files_uploaded_to_github": False, "competition_push_performed": False,
        "packages": [item["package"] for item in packages],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
