#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only OpenVINO NPU capability probe for the local GRT-360 runtime.

The probe deliberately does not install drivers, change device settings, or
start a tracker.  It records enough information to distinguish an absent NPU,
an installed-but-not-exposed NPU runtime, and a graph that the NPU compiler
cannot accept.  A static IR may be compiled with ``--force-compile`` for a
real capability check; the result is written to a fresh JSON file.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path(r"D:\instan\grt360_scratch\openvino\sutrack_b224.xml")
DEFAULT_OUTPUT = Path(r"D:\instan\grt360_scratch\npu_probe_20260827.json")


def _run_powershell(script: str) -> tuple[str | None, str | None]:
    """Run a bounded, read-only Windows query and return stdout/stderr."""
    if platform.system() != "Windows":
        return None, "not_windows"
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return completed.stdout.strip() or None, completed.stderr.strip() or f"exit_{completed.returncode}"
    return completed.stdout.strip() or None, completed.stderr.strip() or None


def windows_os_info() -> dict[str, Any]:
    """Read the Windows build from the registry (ProductName may say Windows 10)."""
    if platform.system() != "Windows":
        return {"available": False, "reason": "not_windows"}
    script = r'''
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue
[pscustomobject]@{
  product_name = [string]$cv.ProductName
  display_version = [string]$cv.DisplayVersion
  current_build = [string]$cv.CurrentBuild
  ubr = [int]($cv.UBR)
} | ConvertTo-Json -Compress
'''
    stdout, error = _run_powershell(script)
    if stdout is None:
        return {"available": False, "reason": error or "query_failed"}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"available": False, "reason": "invalid_query_output", "raw": stdout[:500]}
    if not isinstance(parsed, dict):
        return {"available": False, "reason": "invalid_query_shape"}
    try:
        parsed["build_number"] = int(str(parsed.get("current_build", "0")))
    except ValueError:
        parsed["build_number"] = None
    parsed["windows_11_kernel"] = bool(parsed.get("build_number") and parsed["build_number"] >= 22000)
    parsed["available"] = True
    return parsed


def windows_npu_devices() -> dict[str, Any]:
    """Collect PnP status without exposing instance IDs or credentials."""
    if platform.system() != "Windows":
        return {"available": False, "reason": "not_windows", "devices": []}
    script = r'''
$rows = @(
  Get-PnpDevice -Class ComputeAccelerator -ErrorAction SilentlyContinue |
  ForEach-Object {
    $version = $null
    try {
      $p = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName DEVPKEY_Device_DriverVersion -ErrorAction Stop
      $version = [string]$p.Data
    } catch {}
    [pscustomobject]@{
      friendly_name = [string]$_.FriendlyName
      status = [string]$_.Status
      problem = [string]$_.Problem
      driver_version = $version
    }
  }
)
if ($rows.Count -eq 0) { '[]' } else { $rows | ConvertTo-Json -Compress }
'''
    stdout, error = _run_powershell(script)
    if stdout is None:
        return {"available": False, "reason": error or "query_failed", "devices": []}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"available": False, "reason": "invalid_query_output", "raw": stdout[:500], "devices": []}
    if isinstance(parsed, dict):
        parsed = [parsed]
    devices = parsed if isinstance(parsed, list) else []
    return {"available": bool(devices), "reason": "pnp_query", "devices": devices}


def plugin_inventory(ov_module: Any) -> list[str]:
    root = Path(ov_module.__file__).resolve().parent
    names: list[str] = []
    for pattern in ("*npu*.dll", "*NPU*.dll", "*npu*.so", "*NPU*.so"):
        for path in root.rglob(pattern):
            if path.is_file():
                names.append(str(path))
    return sorted(set(names))


def model_inventory(core: Any, model_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(model_path), "exists": model_path.is_file()}
    if not info["exists"]:
        return info
    try:
        model = core.read_model(model_path)
        inputs = []
        for port in model.inputs:
            shape = str(port.partial_shape)
            inputs.append({"name": port.get_any_name(), "shape": shape, "static": bool(port.partial_shape.is_static)})
        outputs = [{"name": port.get_any_name(), "shape": str(port.partial_shape)} for port in model.outputs]
        info.update({"inputs": inputs, "outputs": outputs, "all_inputs_static": all(row["static"] for row in inputs)})
    except Exception as exc:  # noqa: BLE001
        info["read_error"] = f"{type(exc).__name__}: {exc}"
    return info


def probe(model_path: Path, cache_dir: Path | None, force_compile: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": "grt360.npu_probe.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "model": str(model_path),
    }
    result["windows_os"] = windows_os_info()
    result["windows_pnp"] = windows_npu_devices()
    try:
        import openvino as ov
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "openvino_missing", "openvino_error": f"{type(exc).__name__}: {exc}"})
        return result

    result["openvino"] = ov.__version__
    result["plugin_files"] = plugin_inventory(ov)
    try:
        core = ov.Core()
        devices = list(core.available_devices)
        result["available_devices"] = devices
        result["device_properties"] = {}
        for name in devices:
            try:
                full_name = str(core.get_property(name, "FULL_DEVICE_NAME"))
            except Exception as exc:  # noqa: BLE001
                full_name = f"property_error:{type(exc).__name__}: {exc}"
            result["device_properties"][name] = {"full_name": full_name}
        npu_properties: dict[str, Any] = {}
        for key in (
            "SUPPORTED_PROPERTIES",
            "AVAILABLE_DEVICES",
            "NPU_DRIVER_VERSION",
            "NPU_COMPILER_VERSION",
            "NPU_PLATFORM",
            "NPU_COMPILER_TYPE",
            "FULL_DEVICE_NAME",
            "DEVICE_ID",
        ):
            try:
                value = core.get_property("NPU", key)
                # Keep the report JSON-friendly while preserving useful
                # plugin diagnostics such as an empty AVAILABLE_DEVICES list.
                npu_properties[key] = str(value) if key == "NPU_COMPILER_TYPE" else value
            except Exception as exc:  # noqa: BLE001
                npu_properties[key] = f"unavailable:{type(exc).__name__}: {exc}"
        result["npu_properties"] = npu_properties
        result["model_inventory"] = model_inventory(core, model_path)
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "openvino_probe_failed", "openvino_error": f"{type(exc).__name__}: {exc}"})
        return result

    npu_exposed = "NPU" in result.get("available_devices", [])
    pnp_devices = result.get("windows_pnp", {}).get("devices", [])
    hardware_present = bool(pnp_devices)
    windows11_kernel = bool(result.get("windows_os", {}).get("windows_11_kernel"))
    if not npu_exposed:
        if platform.system() == "Windows" and not windows11_kernel:
            result["status"] = "os_unsupported_npu_not_exposed"
        elif hardware_present:
            result["status"] = "hardware_present_driver_present_runtime_missing"
        else:
            result["status"] = "npu_not_exposed"
    else:
        result["status"] = "available"

    should_compile = bool(force_compile or npu_exposed)
    if should_compile and result.get("model_inventory", {}).get("exists"):
        if not result.get("model_inventory", {}).get("all_inputs_static", False):
            result["compile_status"] = "skipped_dynamic_input"
        else:
            try:
                model = core.read_model(model_path)
                config: dict[str, Any] = {}
                if cache_dir is not None:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    config["CACHE_DIR"] = str(cache_dir)
                started = time.perf_counter()
                core.compile_model(model, "NPU", config)
                result["compile_status"] = "success"
                result["compile_seconds"] = round(time.perf_counter() - started, 3)
                result["status"] = "available_and_compiled"
            except Exception as exc:  # noqa: BLE001
                result["compile_status"] = "failed"
                result["compile_error"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "npu_compile_failed" if npu_exposed else result["status"]
    else:
        result["compile_status"] = "not_attempted"
        if not result.get("model_inventory", {}).get("exists"):
            result["status"] = "model_missing"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Static OpenVINO IR XML to inspect/compile")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Fresh JSON report path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Optional OpenVINO NPU compile cache directory")
    parser.add_argument("--force-compile", action="store_true", help="Try NPU compilation even when NPU is not enumerated")
    args = parser.parse_args()
    report = probe(args.model, args.cache_dir, args.force_compile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
