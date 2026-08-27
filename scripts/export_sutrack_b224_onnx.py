#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the SUTrack-B224 per-frame encoder/decoder to a static ONNX graph.

The tracker state (template bank, box crop and geometry) remains in Python;
the exported graph consumes two preprocessed template tensors, their normalized
annotations, and one preprocessed search tensor.  Text conditioning is fixed
to the same all-zero token used by the official no-language evaluation path,
so CLIP is not part of the exported graph.  This shape is suitable for a later
OpenVINO GPU/NPU compilation while retaining the exact B224 weights.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.eval_official import build_sutrack_tracker  # noqa: E402


class FrameGraph(nn.Module):
    def __init__(self, network: nn.Module, text_src: torch.Tensor):
        super().__init__()
        self.network = network
        self.register_buffer("text_src", text_src.detach().cpu())

    def forward(self, template0, template1, anno0, anno1, search):
        features = self.network.forward_encoder(
            [template0, template1], [search], [anno0, anno1], self.text_src, None)
        output = self.network.forward_decoder(features)
        return output["score_map"], output["size_map"], output["offset_map"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="sutrack_b224")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--search-size", type=int, default=224)
    parser.add_argument("--template-size", type=int, default=112)
    args = parser.parse_args(argv)

    # Reuse the production adapter's strict checkpoint loading and CPU clip
    # reconstruction, then export only the vision encoder/decoder path.
    namespace = argparse.Namespace(
        gpu="0", force_cpu=True,
        sutrack_workspace=args.workspace, sutrack_ckpt=args.checkpoint,
        sutrack_config=args.config, sutrack_amp=False, sutrack_lora_ckpt=None,
        sutrack_search_size=args.search_size if args.search_size != 224 else None,
        sutrack_template_size=args.template_size if args.template_size != 112 else None,
    )
    adapter = build_sutrack_tracker(namespace)
    network = adapter.tracker.network.eval().cpu()
    with torch.no_grad():
        tokens = torch.zeros((1, 77), dtype=torch.long)
        text_src = network.forward_textencoder(tokens).cpu()
    graph = FrameGraph(network, text_src).eval()
    template0 = torch.zeros((1, 6, args.template_size, args.template_size), dtype=torch.float32)
    template1 = torch.zeros((1, 6, args.template_size, args.template_size), dtype=torch.float32)
    anno0 = torch.tensor([[0.25, 0.25, 0.50, 0.50]], dtype=torch.float32)
    anno1 = anno0.clone()
    search = torch.zeros((1, 6, args.search_size, args.search_size), dtype=torch.float32)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] output={output} opset={args.opset}", flush=True)
    started = time.time()
    with torch.no_grad():
        torch.onnx.export(
            graph, (template0, template1, anno0, anno1, search), str(output),
            input_names=["template0", "template1", "anno0", "anno1", "search"],
            output_names=["score_map", "size_map", "offset_map"],
            opset_version=args.opset, do_constant_folding=True,
            dynamic_axes=None, export_params=True, training=torch.onnx.TrainingMode.EVAL,
        )
    print(f"[export] done seconds={time.time() - started:.1f} bytes={output.stat().st_size}", flush=True)
    try:
        import onnx
        model = onnx.load(str(output), load_external_data=True)
        onnx.checker.check_model(model)
        print(f"[onnx] checker=ok nodes={len(model.graph.node)}", flush=True)
    except ImportError:
        print("[onnx] checker=skipped (onnx package unavailable)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
