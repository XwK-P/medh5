"""Batching items whose parts do not all stack.

A MEDH5 item is a nested dict of tensors, lists and metadata.  The default
``torch`` collate stacks anything it recognises and raises on anything it does
not, which for this shape means a per-object list of detections aborts the
batch.  This collate stacks what is stackable, keeps lists as lists, and
carries everything else through key by key, as a list over the batch --- so a
detection head gets its ragged targets and a segmentation head gets its
stacked ones from the same dataloader.

Shape mismatches are refused with the tensor's name and both shapes.  The
default message names neither, and "stack expects each tensor to be equal size"
in a training log is an hour of bisecting a dataset.  So is a key some items
carry and others do not: the keys were taken from the first item, so a batch
led by an item without a label dropped every other item's, and one led by an
item with a label raised a bare ``KeyError``.  Such a key is refused by name.
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
    return {
        key: _collate_value(key, [item[key] for item in batch])
        for key in _shared_keys("the batch", batch)
    }


def _collate_value(name: str, values: Sequence[Any]) -> Any:
    import torch

    first = values[0]
    if isinstance(first, torch.Tensor):
        return _stack(name, values)
    if isinstance(first, Mapping):
        return {
            key: _collate_value(f"{name}.{key}", [v[key] for v in values])
            for key in _shared_keys(repr(name), values)
        }
    return list(values)


def _shared_keys(name: str, values: Sequence[Mapping[str, Any]]) -> list[str]:
    """The keys of *values*, which every one of them must carry.

    The items of a :class:`~medh5.torch.PairedPatchDataset` are keyed by their
    visits' timepoints, so pairs of different visits do not batch together.
    """
    keys = list(dict.fromkeys(key for value in values for key in value))
    for key in keys:
        lacking = [i for i, value in enumerate(values) if key not in value]
        if lacking:
            raise MEDH5ValidationError(
                f"cannot collate {name}: item(s) {lacking} of {len(values)} have "
                f"no {key!r}, which the others carry. A batch takes each key from "
                "every item --- batch items that carry the same keys (a pair's "
                "images are keyed by its visits), or collate them yourself"
            )
    return keys


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
