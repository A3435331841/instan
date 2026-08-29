#!/usr/bin/env python3
"""Benchmark the two delivered CUDA images on the same Arena dataset.

The script does not build or push an image.  It records the exact commands,
wall-clock elapsed time and image output under a new output directory so the
RTX 5090 comparison is auditable.  Accuracy is scored separately with the
official scorer because Arena datasets need not contain ground truth.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _run(command: list[str], log_path: Path, dry_run: bool) -> dict:
    started = time.perf_counter()
    if dry_run:
        return {"command": command, "returncode": None, "seconds": 0.0, "dry_run": True}
    with log_path.open("w", encoding="utf-8", newline="\n") as handle:
        proc = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return {"command": command, "returncode": proc.returncode,
            "seconds": time.perf_counter() - started, "log": str(log_path)}


def _docker_command(image: str, dataset: Path, result: Path, gpu: str,
                    entry_args: list[str]) -> list[str]:
    return ["docker", "run", "--rm", "--gpus", f"device={gpu}",
            "-v", f"{dataset}:/mnt/dataset:ro", "-v", f"{result}:/mnt/result",
            image, *entry_args]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ort-image", required=True)
    parser.add_argument("--torch-image", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seqs", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and shutil.which("docker") is None:
        raise RuntimeError("docker is not on PATH")
    dataset = Path(args.dataset).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty benchmark output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    shared: list[str] = []
    if args.seqs:
        shared += ["--seqs", args.seqs]
    if args.max_frames is not None:
        shared += ["--max-frames", str(args.max_frames)]
    jobs = {
        "ort_v5": (args.ort_image, ["--profile", "v5_final", *shared]),
        "torch_b224": (args.torch_image, ["--profile", "b224_erp", *shared]),
    }
    results: dict[str, object] = {
        "schema": "grt360.cuda_backend_benchmark.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "gpu": str(args.gpu), "dry_run": bool(args.dry_run), "jobs": {},
    }
    failed = False
    for name, (image, entry_args) in jobs.items():
        result_dir = out / name / "result"
        result_dir.mkdir(parents=True, exist_ok=True)
        command = _docker_command(image, dataset, result_dir, str(args.gpu), entry_args)
        record = _run(command, out / f"{name}.log", args.dry_run)
        record["result_dir"] = str(result_dir)
        results["jobs"][name] = record
        failed = failed or (record["returncode"] not in (0, None))
    (out / "benchmark.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
