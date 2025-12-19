"""Device selection and tensor transfer helpers."""

from __future__ import annotations

from typing import Any

import torch


def choose_device(prefer_gpu: bool = True) -> torch.device:
    """Return an available ``torch.device``.

    Args:
        prefer_gpu: If ``True`` and CUDA is available, return the first GPU.

    Returns:
        A ``torch.device`` pointing to ``cuda`` when available (and preferred),
        otherwise ``cpu``.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors within a batch to the target device.

    Handles common nested structures such as ``dict``, ``list`` and ``tuple``.

    Args:
        batch: A tensor or nested structure containing tensors.
        device: The target device.

    Returns:
        The batch with all tensors moved to ``device``.
    """

    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        converted = [move_batch_to_device(item, device) for item in batch]
        return type(batch)(converted)
    return batch


__all__ = ["choose_device", "move_batch_to_device"]
