"""PyTorch Dataset wrappers for ``.medh5`` files.

Requires ``torch`` (install with ``pip install medh5[torch]``).

Two dataset classes are provided:

- :class:`MEDH5TorchDataset` — eager, full-volume read per ``__getitem__``.
  Best for small samples that comfortably fit in memory.
- :class:`MEDH5PatchDataset` — lazy, patch-based read using a
  :class:`~medh5.sampling.PatchSampler`. Best for large volumes where
  patch-based training is the norm.

Both share a per-process LRU file-handle cache (:class:`_HandleCache`)
so repeated ``__getitem__`` calls against the same file reuse one
``h5py.File`` instead of opening it from scratch every time. PyTorch
DataLoader workers each get their own copy of the cache (forked
process), so no cross-worker synchronization is needed.
"""

from __future__ import annotations

import atexit
import contextlib
from collections import OrderedDict
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


# ---------------------------------------------------------------------------
# Per-worker file handle cache
# ---------------------------------------------------------------------------


class _HandleCache:
    """LRU cache of open :class:`MEDH5File` handles.

    Sized by *maxsize*. Eviction closes the handle. Per-process state —
    PyTorch DataLoader workers each get their own copy because they are
    forked.
    """

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = int(maxsize)
        self._items: OrderedDict[str, MEDH5File] = OrderedDict()
        self.opens = 0  # exposed for tests / diagnostics

    def get(self, path: str | Path) -> MEDH5File:
        key = str(path)
        cached = self._items.get(key)
        if cached is not None:
            self._items.move_to_end(key)
            return cached
        handle = MEDH5File(key)
        self.opens += 1
        self._items[key] = handle
        if len(self._items) > self.maxsize:
            _, evicted = self._items.popitem(last=False)
            with contextlib.suppress(Exception):  # pragma: no cover
                evicted.close()
        return handle

    def close_all(self) -> None:
        while self._items:
            _, h = self._items.popitem(last=True)
            with contextlib.suppress(Exception):  # pragma: no cover
                h.close()


_HANDLE_CACHE = _HandleCache(maxsize=32)


@atexit.register
def _close_handle_cache() -> None:  # pragma: no cover - process exit hook
    _HANDLE_CACHE.close_all()


def _open_cached(path: str | Path) -> MEDH5File:
    """Module-level helper used by both dataset classes."""
    return _HANDLE_CACHE.get(path)


# ---------------------------------------------------------------------------
# Eager dataset
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Patch-based dataset
# ---------------------------------------------------------------------------


def _to_tensors(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert nested numpy arrays in *sample* to torch tensors."""
    out = dict(sample)
    if isinstance(out.get("images"), dict):
        out["images"] = {
            name: torch.from_numpy(np.ascontiguousarray(arr))
            for name, arr in out["images"].items()
        }
    if isinstance(out.get("seg"), dict):
        out["seg"] = {
            name: torch.from_numpy(np.ascontiguousarray(arr))
            for name, arr in out["seg"].items()
        }
    return out


class MEDH5PatchDataset:
    """Lazy patch-based PyTorch dataset.

    On every ``__getitem__`` a fresh patch is sampled from one of the
    underlying files via the supplied :class:`~medh5.sampling.PatchSampler`.
    Each file is opened once per worker (cached) so reads stay
    chunk-aligned and full volumes are never materialized.

    The virtual length is ``len(paths) * samples_per_volume`` so each
    epoch sees that many random patches.

    Parameters
    ----------
    paths : list of str or Path
        Source ``.medh5`` files.
    sampler : PatchSampler
        Patch sampling strategy (uniform / foreground / balanced).
    transform : callable, optional
        Pure-numpy transform applied to the sample dict before tensor
        conversion. See :mod:`medh5.transforms`.
    samples_per_volume : int
        Virtual oversampling factor (default 1).
    """

    def __init__(
        self,
        paths: list[str | Path],
        sampler: Any,
        *,
        transform: Any = None,
        samples_per_volume: int = 1,
    ) -> None:
        _require_torch()
        self.paths = [Path(p) for p in paths]
        self.sampler = sampler
        self.transform = transform
        self.samples_per_volume = int(samples_per_volume)

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_volume

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx % len(self.paths)]
        f = _open_cached(path)
        sample = self.sampler.sample(f)

        if self.transform is not None:
            sample = self.transform(sample)

        return _to_tensors(sample)
