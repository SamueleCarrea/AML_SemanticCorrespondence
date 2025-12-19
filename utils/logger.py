"""Utility helpers for configuring consistent application logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str = "app",
    log_dir: str | Path = "outputs",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Create (or return) a configured logger.

    The logger writes to both stdout and a log file under ``outputs/`` by default.
    Handlers are only added once per logger name to avoid duplicate entries when
    the helper is called multiple times.

    Args:
        name: Name of the logger to create or retrieve.
        log_dir: Directory where log files should be stored. Created if missing.
        log_level: Logging level for the handlers (defaults to ``logging.INFO``).
        log_file: Optional explicit file name. Defaults to ``"{name}.log"``.

    Returns:
        A configured ``logging.Logger`` instance.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    file_name = log_file or f"{name}.log"

    file_handler = logging.FileHandler(log_dir_path / file_name)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


__all__ = ["get_logger"]
