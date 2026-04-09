"""PyTorch Dataset wrapper for ``.medh5`` files.

Requires ``torch`` (install with ``pip install medh5[torch]``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from medh5.core import MEDH5File

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "PyTorch is required for MEDH5TorchDataset. "
            "Install it with: pip install medh5[torch]"
        )


class MEDH5TorchDataset:
    """A PyTorch-compatible map-style dataset over ``.medh5`` files.

    Each sample returns a dict with:

    - ``"images"``: ``dict[str, Tensor]`` keyed by modality name
    - ``"seg"``: ``dict[str, Tensor]`` (if present)
    - ``"bboxes"``: ``Tensor`` (if present)
    - ``"bbox_scores"``: ``Tensor`` (if present)
    - ``"bbox_labels"``: ``list[str]`` (if present)
    - ``"label"``: ``int | str | None``
    - ``"meta"``: :class:`~medh5.meta.SampleMeta`

    Parameters
    ----------
    paths : list of str or Path
        Paths to ``.medh5`` files (one per sample).
    transform : callable, optional
        Applied to the sample dict after loading.
    """

    def __init__(
        self,
        paths: list[str | Path],
        transform: Any = None,
    ) -> None:
        _require_torch()
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = MEDH5File.read(self.paths[idx])

        out: dict[str, Any] = {
            "images": {
                name: torch.from_numpy(np.ascontiguousarray(arr))
                for name, arr in sample.images.items()
            },
            "label": sample.meta.label,
            "meta": sample.meta,
        }

        if sample.seg is not None:
            out["seg"] = {
                name: torch.from_numpy(np.ascontiguousarray(arr))
                for name, arr in sample.seg.items()
            }

        if sample.bboxes is not None:
            out["bboxes"] = torch.from_numpy(np.ascontiguousarray(sample.bboxes))
        if sample.bbox_scores is not None:
            out["bbox_scores"] = torch.from_numpy(
                np.ascontiguousarray(sample.bbox_scores)
            )
        if sample.bbox_labels is not None:
            out["bbox_labels"] = sample.bbox_labels

        if self.transform is not None:
            out = self.transform(out)

        return out
