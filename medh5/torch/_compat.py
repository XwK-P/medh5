"""Torch availability and array conversion, in one place."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

_ERROR: Exception | None = None
"""Why ``import torch`` failed, when it did."""

try:  # pragma: no cover - exercised by whichever environment runs the tests
    import torch

    AVAILABLE = True
except Exception as error:  # pragma: no cover - torch is an optional extra
    torch = None
    AVAILABLE = False
    _ERROR = error


def require_torch() -> None:
    """Without PyTorch, raise an ``ImportError`` that says how to install it ---
    or, when it is installed and fails to import, why."""
    if AVAILABLE:
        return
    absent = isinstance(_ERROR, ModuleNotFoundError) and _ERROR.name == "torch"
    if _ERROR is None or absent:
        raise ImportError(
            "PyTorch is required for medh5.torch. Install it with: "
            "pip install 'medh5[torch]'"
        ) from _ERROR
    # A broken install (a CUDA library, an ABI or NumPy mismatch) was reported
    # as a missing one, and the cause was lost.
    raise ImportError(
        f"PyTorch is installed but failed to import "
        f"({type(_ERROR).__name__}: {_ERROR}); medh5.torch needs a working torch"
    ) from _ERROR


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


def sampler_base() -> type:
    """``torch.utils.data.Sampler`` when torch is installed, else ``object``."""
    if not AVAILABLE:  # pragma: no cover - depends on the environment
        return object
    from torch.utils.data import Sampler

    base: type = Sampler
    return base


__all__ = ["AVAILABLE", "dataset_base", "require_torch", "sampler_base", "to_tensor"]
