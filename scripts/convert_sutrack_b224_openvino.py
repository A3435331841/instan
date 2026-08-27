#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the exported B224 ONNX graph and probe Intel devices."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", required=True, help="OpenVINO IR .xml path")
    args = parser.parse_args(argv)
    import openvino as ov

    core = ov.Core()
    devices = list(core.available_devices)
    print(json.dumps({"available_devices": devices}, ensure_ascii=False), flush=True)
    started = time.time()
    model = ov.convert_model(args.onnx)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(model, str(output), compress_to_fp16=False)
    result = {"xml": str(output), "bin": str(output.with_suffix(".bin")),
              "conversion_seconds": time.time() - started, "devices": {}}
    for device in ("CPU", "GPU", "NPU"):
        if device not in devices:
            result["devices"][device] = {"available": False}
            continue
        try:
            compiled = core.compile_model(model, device)
            result["devices"][device] = {"available": True, "compiled": True,
                                          "inputs": [str(x.any_name) for x in compiled.inputs],
                                          "outputs": [str(x.any_name) for x in compiled.outputs]}
        except Exception as exc:  # noqa: BLE001 - report device-specific support
            result["devices"][device] = {"available": True, "compiled": False,
                                          "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
