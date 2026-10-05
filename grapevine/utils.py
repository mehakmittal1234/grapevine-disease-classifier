"""Small shared helpers: seeding, device selection, JSON I/O and environment capture."""

from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA and MPS)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds the MPS and CUDA generators


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that derives each worker's seeds from the loader's seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(preferred: str | None = None) -> torch.device:
    """Return ``preferred`` if given, otherwise CUDA, then Apple-silicon MPS, then CPU."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Block until queued kernels finish, so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def project_relative(path: Path | str) -> str:
    """Path relative to the project folder, so reports carry no machine-specific prefixes."""
    from .config import PROJECT_ROOT

    return Path(os.path.relpath(Path(path).resolve(), PROJECT_ROOT)).as_posix()


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def environment_info() -> dict:
    import torchvision

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
