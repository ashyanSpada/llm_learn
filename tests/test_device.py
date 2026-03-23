"""Tests for device selection logic in utils.py."""

from __future__ import annotations

from unittest.mock import patch

import torch

from llm_learn.utils import get_device, get_logger, set_seed

# ---------------------------------------------------------------------------
# get_device tests
# ---------------------------------------------------------------------------


def test_get_device_returns_torch_device() -> None:
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_cpu_when_nothing_available() -> None:
    """When neither CUDA nor MPS is available, get_device should return CPU."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device = get_device(prefer_mps=True)
    assert device.type == "cpu"


def test_get_device_mps_preferred_over_cpu() -> None:
    """When MPS is available and prefer_mps=True, get_device returns MPS."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device = get_device(prefer_mps=True)
    assert device.type == "mps"


def test_get_device_no_mps_when_prefer_mps_false() -> None:
    """When prefer_mps=False, MPS should not be selected even if available."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device = get_device(prefer_mps=False)
    assert device.type == "cpu"


def test_get_device_cuda_takes_priority_over_mps() -> None:
    """CUDA should be preferred over MPS even when both are 'available'."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device = get_device(prefer_mps=True)
    assert device.type == "cuda"


def test_get_device_cuda_when_available() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device = get_device()
    assert device.type == "cuda"


# ---------------------------------------------------------------------------
# set_seed tests
# ---------------------------------------------------------------------------


def test_set_seed_runs_without_error() -> None:
    set_seed(0)
    set_seed(42)
    set_seed(12345)


def test_set_seed_reproducibility() -> None:
    """Two runs with the same seed should produce the same random tensor."""
    set_seed(7)
    t1 = torch.rand(10)
    set_seed(7)
    t2 = torch.rand(10)
    assert torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# get_logger tests
# ---------------------------------------------------------------------------


def test_get_logger_returns_logger() -> None:
    import logging

    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name() -> None:
    logger = get_logger("my_module")
    assert logger.name == "my_module"


def test_get_logger_idempotent() -> None:
    """Calling get_logger twice with the same name should not add duplicate handlers."""
    logger1 = get_logger("dup_test")
    logger2 = get_logger("dup_test")
    assert logger1 is logger2
    assert len(logger1.handlers) == 1
