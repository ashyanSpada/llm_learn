"""Utility helpers: device selection, reproducibility, logging."""

from __future__ import annotations

import logging
import random

import torch


def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Args:
        prefer_mps: When *True* (default), use MPS if CUDA is unavailable.
            Set to *False* to force CPU even on Apple Silicon.

    Returns:
        A :class:`torch.device` instance.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    try:
        import numpy as np  # optional dependency

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logger(name: str = "llm_learn", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Args:
        name: Logger name (shown in log lines).
        level: Logging level, e.g. ``logging.DEBUG``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
