"""Batching items whose parts do not all stack (implementation plan §2.3).

A MEDH5 item is a nested dict of tensors, lists and metadata.  The default
``torch`` collate stacks anything it recognises and raises on anything it does
not, which for this shape means a per-object list of detections aborts the
batch.  This collate stacks what is stackable, keeps lists as lists, and
carries metadata through as a list of dicts --- so a detection head gets its
ragged targets and a segmentation head gets its stacked ones from the same
dataloader.

Shape mismatches are refused with the tensor's name and both shapes.  The
default message names neither, and "stack expects each tensor to be equal size"
in a training log is an hour of bisecting a dataset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from medh5.errors import MEDH5ValidationError
from medh5.torch._compat import require_torch


def collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``DataLoader(collate_fn=...)`` for the MEDH5 item shape."""
    require_torch()
    if not batch:
        raise MEDH5ValidationError("cannot collate an empty batch")
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = _collate_value(key, values)
    return out


def _collate_value(name: str, values: Sequence[Any]) -> Any:
    import torch

    first = values[0]
    if isinstance(first, torch.Tensor):
        return _stack(name, values)
    if isinstance(first, Mapping):
        return {
            key: _collate_value(f"{name}.{key}", [v[key] for v in values])
            for key in first
        }
    return list(values)


def _stack(name: str, values: Sequence[Any]) -> Any:
    import torch

    shapes = {tuple(v.shape) for v in values}
    if len(shapes) > 1:
        raise MEDH5ValidationError(
            f"cannot stack {name!r}: the batch mixes shapes {sorted(shapes)}. "
            "Sample fixed-size patches, or collate this key as a list."
        )
    return torch.stack(list(values))


def stack_images(batch: Sequence[Mapping[str, Any]], order: Sequence[str]) -> Any:
    """Channel-stack several images into one ``(B, C, *patch)`` tensor.

    Multi-modal models want one input tensor, not a dict.  The channel order is
    the caller's, never the file's, so channel *i* is the same modality across a
    cohort assembled from different sources.
    """
    require_torch()
    import torch

    planes = []
    for item in batch:
        images = item["images"]
        missing = [k for k in order if k not in images]
        if missing:
            raise MEDH5ValidationError(
                f"item is missing image(s) {missing}; present: {sorted(images)}"
            )
        planes.append(torch.stack([images[k] for k in order]))
    return torch.stack(planes)


__all__ = ["collate", "stack_images"]
