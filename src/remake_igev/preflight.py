"""Read-only AutoDL checks; CUDA failure must never silently select CPU training."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import sys


def inspect(resource_root: Path, minimum_free_gib: float = 80) -> dict:
    report = {
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "resource_root": str(resource_root),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "ready_for_gpu_execution": False,
        "errors": [],
    }
    errors = report["errors"]
    try:
        root = resource_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("resource_root must be an existing directory")
        usage = shutil.disk_usage(root)
        report["storage"] = {
            "resolved_root": str(root),
            "device": root.stat().st_dev,
            "system_device": Path("/").stat().st_dev,
            "free_bytes": usage.free,
            "writable": os.access(root, os.W_OK),
        }
        if root.stat().st_dev == Path("/").stat().st_dev:
            errors.append("resource root resolves to the system filesystem")
        if not os.access(root, os.W_OK):
            errors.append("resource root is not writable")
        if usage.free < minimum_free_gib * 2**30:
            errors.append("insufficient free storage for the declared minimum")
    except (OSError, ValueError) as exc:
        errors.append(f"storage: {type(exc).__name__}: {exc}")

    devices = {}
    for name in ("nvidiactl", "nvidia-uvm"):
        try:
            fd = os.open(f"/dev/{name}", os.O_RDWR)
            os.close(fd)
            devices[name] = {"open_ok": True}
        except OSError as exc:
            devices[name] = {"open_ok": False, "errno": exc.errno, "error": str(exc)}
            errors.append(f"/dev/{name}: {exc.strerror}")
    report["devices"] = devices
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        code = driver.cuInit(0)
        report["cuInit"] = code
        if code:
            errors.append(f"CUDA driver initialization failed: {code}")
    except OSError as exc:
        errors.append(f"CUDA driver load: {exc}")

    try:
        import torch
        report["torch"] = torch.__version__
        report["torch_cuda_build"] = torch.version.cuda
        available = torch.cuda.is_available()
        report["cuda_available"] = available
        if not available:
            errors.append("torch.cuda.is_available() is false; CPU fallback forbidden")
        elif not errors:
            # Exercise allocation and autograd, not just NVML/device enumeration.
            value = torch.arange(1, 5, dtype=torch.float32, device="cuda", requires_grad=True)
            value.square().sum().backward()
            torch.cuda.synchronize()
            if not torch.equal(value.grad, 2 * value.detach()):
                errors.append("CUDA autograd smoke returned incorrect gradients")
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["capability"] = list(torch.cuda.get_device_capability(0))
            report["allocation_and_backward_passed"] = not errors
    except (ImportError, RuntimeError) as exc:
        errors.append(f"PyTorch: {type(exc).__name__}: {exc}")
    report["ready_for_gpu_execution"] = not errors
    # Dataset transfer and camera calibration are separate launch gates.
    report["dataset_and_geometry_verified"] = False
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--minimum-free-gib", type=float, default=80)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 0:
        parser.error("minimum-free-gib must be finite and non-negative")
    report = inspect(args.resource_root, args.minimum_free_gib)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready_for_gpu_execution"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
