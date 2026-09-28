"""Seeding and config loading."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs; optionally force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_config(path: str | Path) -> dict:
    """Load a YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)
