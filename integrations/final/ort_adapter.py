"""Small ONNX Runtime adapter for the existing OpenVINO tracker kernels.

The B224 and ODTrack geometry trackers only require a compiled-model object
with named input/output ports and a callable inference method.  This adapter
provides that contract on CUDA or CPU without duplicating the tracker state
machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class OrtPort:
    """Minimal OpenVINO-port-compatible metadata."""

    any_name: str
    shape: tuple[int, ...]


def _shape(meta: Any) -> tuple[int, ...]:
    values: list[int] = []
    for value in meta.shape:
        if isinstance(value, int):
            values.append(value)
        elif value is None:
            values.append(-1)
        else:
            try:
                values.append(int(value))
            except (TypeError, ValueError):
                values.append(-1)
    return tuple(values)


class OrtCompiledModel:
    """Expose an ONNX Runtime session through the tracker graph contract."""

    def __init__(self, model_path: str | Path, device: str = "cuda",
                 device_id: int = 0, strict: bool = True):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised in image
            raise RuntimeError("onnxruntime is required for the ORT backend") from exc

        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        normalized = str(device).lower()
        available = set(ort.get_available_providers())
        if normalized == "cuda":
            requested: list[Any] = [("CUDAExecutionProvider", {"device_id": int(device_id)}),
                                    "CPUExecutionProvider"]
            if strict and "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider is unavailable; install onnxruntime-gpu "
                    "inside the CUDA image or use --force-cpu for smoke testing")
        elif normalized == "cpu":
            requested = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"unsupported ORT device: {device}")

        self.path = path
        self.session = ort.InferenceSession(str(path), providers=requested)
        self.providers = tuple(self.session.get_providers())
        self.inputs = [OrtPort(item.name, _shape(item))
                       for item in self.session.get_inputs()]
        self.outputs = [OrtPort(item.name, _shape(item))
                        for item in self.session.get_outputs()]
        self._output_names = [item.any_name for item in self.outputs]

    def __call__(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {str(name): np.asarray(value) for name, value in inputs.items()}
        values = self.session.run(self._output_names, feed)
        return dict(zip(self._output_names, values))


def required_model_paths(model_root: str | Path) -> dict[str, Path]:
    """Return the exact ONNX asset names required by the final ORT profile."""
    root = Path(model_root).resolve()
    return {
        "b": root / "sutrack_b224_frame.onnx",
        "b_high": root / "sutrack_b224_s224_t128.onnx",
        "t": root / "sutrack_t224_s224_t112.onnx",
        "od": root / "odtrack_state.onnx",
        "od_first": root / "odtrack_first.onnx",
        "od_v5": root / "odtrack_v5_ep6_state.onnx",
        "od_v5_first": root / "odtrack_v5_ep6_first.onnx",
    }
