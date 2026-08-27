#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically compare the exported B224 graph with the PyTorch graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval_official import build_sutrack_tracker  # noqa: E402
from scripts.export_sutrack_b224_onnx import FrameGraph  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--provider", choices=["cpu", "dml"], default="cpu")
    args = parser.parse_args(argv)
    namespace = argparse.Namespace(
        gpu="0", force_cpu=True, sutrack_workspace=args.workspace,
        sutrack_ckpt=args.checkpoint, sutrack_config="sutrack_b224",
        sutrack_amp=False, sutrack_lora_ckpt=None,
    )
    adapter = build_sutrack_tracker(namespace)
    network = adapter.tracker.network.eval().cpu()
    with torch.no_grad():
        text_src = network.forward_textencoder(torch.zeros((1, 77), dtype=torch.long)).cpu()
    graph = FrameGraph(network, text_src).eval()
    rng = np.random.default_rng(123)
    arrays = {
        "template0": rng.standard_normal((1, 6, 112, 112), dtype=np.float32),
        "template1": rng.standard_normal((1, 6, 112, 112), dtype=np.float32),
        "anno0": np.asarray([[0.25, 0.25, 0.50, 0.50]], dtype=np.float32),
        "anno1": np.asarray([[0.25, 0.25, 0.50, 0.50]], dtype=np.float32),
        "search": rng.standard_normal((1, 6, 224, 224), dtype=np.float32),
    }
    with torch.no_grad():
        ref = [value.detach().numpy() for value in graph(
            *(torch.from_numpy(arrays[name]) for name in
              ("template0", "template1", "anno0", "anno1", "search")))]
    import onnxruntime as ort
    providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if args.provider == "dml" else ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    if providers[0] not in available:
        raise SystemExit(f"requested provider {providers[0]} unavailable; available={available}")
    session = ort.InferenceSession(args.onnx, providers=providers)
    got = session.run(None, arrays)
    diffs = [float(np.max(np.abs(a - b))) for a, b in zip(ref, got)]
    rel = [float(np.mean(np.abs(a - b)) / max(1e-8, np.mean(np.abs(a)))) for a, b in zip(ref, got)]
    print({"providers": session.get_providers(), "max_abs": diffs, "relative_mean_abs": rel})
    if max(diffs) > 2e-3:
        raise SystemExit("ONNX numerical validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
