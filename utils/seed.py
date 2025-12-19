"""Utility for deterministic seeding across common libraries."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, env_var: str = "PYTHONHASHSEED") -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: The seed value to apply across libraries.
        deterministic: When ``True``, request deterministic algorithms in PyTorch.
        env_var: Environment variable name for hashing seed (default ``PYTHONHASHSEED``).
    """

    os.environ[env_var] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["set_seed"]
