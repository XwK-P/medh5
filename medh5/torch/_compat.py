"""Torch availability and array conversion, in one place."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

try:  # pragma: no cover - exercised by whichever environment runs the tests
    import torch

    AVAILABLE = True
except Exception:  # pragma: no cover - torch is an optional extra
    torch = None
    AVAILABLE = False


def require_torch() -> None:
    """Raise a message that says how to fix it, rather than an ImportError."""
    if not AVAILABLE:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyTorch is required for medh5.torch. Install it with: "
            "pip install 'medh5[torch]'"
        )


def to_tensor(array: npt.NDArray[Any]) -> Any:
    """One numpy array to one tensor, without a copy where numpy allows it."""
    require_torch()
    return torch.from_numpy(np.ascontiguousarray(array))


def dataset_base() -> type:
    """``torch.utils.data.Dataset`` when torch is installed, else ``object``.

    The datasets subclass it so ``isinstance`` checks and ``DataLoader``
    introspection behave, without making torch a hard dependency of
    ``import medh5``.
    """
    if not AVAILABLE:  # pragma: no cover - depends on the environment
        return object
    from torch.utils.data import Dataset

    base: type = Dataset
    return base


__all__ = ["AVAILABLE", "dataset_base", "require_torch", "to_tensor"]
