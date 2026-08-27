#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export an ODTrack CENTER graph with an explicit causal track-query state.

The upstream tracker stores ``track_query`` on the Python object.  This
wrapper makes it an ONNX input/output so an OpenVINO expert can preserve the
same one-frame state transition without embedding GT or sequence lookup.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


class ODFrameGraph(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.backbone = network.backbone
        self.box_head = network.box_head
        self.feat_len_s = int(network.feat_len_s)
        self.feat_sz_s = int(network.feat_sz_s)
        self.token_len = int(network.token_len)

    def forward(self, template0, template1, template2, search, track_query):
        templates = [template0, template1, template2]
        x, _aux = self.backbone(
            z=templates, x=search, ce_template_mask=None,
            ce_keep_rate=None, return_last_attn=False,
            track_query=track_query, token_len=self.token_len)
        feat_last = x[-1] if isinstance(x, (list, tuple)) else x
        enc_opt = feat_last[:, -self.feat_len_s:]
        new_query = x[:, :self.token_len]
        att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(0, 3, 2, 1).contiguous()
        opt_feat = opt.view(-1, opt.shape[2], self.feat_sz_s, self.feat_sz_s)
        score_map, bbox, size_map, offset_map = self.box_head(opt_feat, None)
        return score_map, bbox, size_map, offset_map, new_query


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="baseline")
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args(argv)
    root = Path(args.workspace).resolve()
    sys.path.insert(0, str(root))
    # ODTrack's construction path unconditionally moves modules to CUDA.  A
    # CPU export is sufficient because the serialized graph is device-neutral.
    torch.nn.Module.cuda = lambda self, *a, **k: self
    torch.Tensor.cuda = lambda self, *a, **k: self
    from scripts.odtrack_360vot import _patch_torch_six  # type: ignore
    _patch_torch_six()
    from lib.config.odtrack.config import cfg, update_config_from_file
    from lib.models.odtrack import build_odtrack

    update_config_from_file(root / "experiments" / "odtrack" / f"{args.config}.yaml")
    network = build_odtrack(cfg, training=False).cpu().eval()
    payload = torch.load(str(Path(args.checkpoint).resolve()), map_location="cpu", weights_only=False)
    network.load_state_dict(payload["net"], strict=True)
    graph = ODFrameGraph(network).eval()
    template_shape = (1, 3, int(cfg.TEST.TEMPLATE_SIZE), int(cfg.TEST.TEMPLATE_SIZE))
    search_shape = (1, 3, int(cfg.TEST.SEARCH_SIZE), int(cfg.TEST.SEARCH_SIZE))
    query_shape = (1, 1, int(cfg.MODEL.BACKBONE.HIDDEN_DIM)) if hasattr(cfg.MODEL.BACKBONE, "HIDDEN_DIM") else (1, 1, 768)
    tensors = tuple(torch.zeros(s, dtype=torch.float32) for s in (*([template_shape] * 3), search_shape, query_shape))
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        graph, tensors, str(output), opset_version=args.opset,
        input_names=["template0", "template1", "template2", "search", "track_query"],
        output_names=["score_map", "bbox", "size_map", "offset_map", "next_track_query"],
        dynamic_axes=None, do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL,
    )
    print(f"exported {output} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
