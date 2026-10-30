"""Small, dependency-light helpers for reproducible GPU experiment records."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the GPU experiment dependencies: pip install -r requirements-gpu.txt") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Run this workflow in the configured GPU environment.")
    return torch


def hardware_record(torch):
    """Return hardware and software details needed to reproduce a GPU result."""
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_index": i,
        "gpu_name": props.name,
        "gpu_total_memory_gib": round(props.total_memory / 2**30, 2),
    }


def write_hardware(path, torch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(hardware_record(torch), indent=2) + "\n", encoding="utf-8")


def reset_peak(torch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak_memory_gib(torch):
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**30
