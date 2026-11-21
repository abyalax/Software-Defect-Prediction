"""Utility functions for the project."""

from __future__ import annotations


def get_device() -> str:
    """Detect the device being used (CUDA or CPU).

    Returns:
        Device string: 'cuda' if CUDA is available, otherwise 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
