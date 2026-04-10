"""Pure-numpy preprocessing transforms for sample dicts.

Each transform is callable as ``transform(sample) -> sample``, where
*sample* is the dict produced by :meth:`MEDH5PatchDataset.__getitem__`
or by a :class:`PatchSampler` (image keys live in ``sample["images"]``,
mask keys in ``sample["seg"]``).

Transforms are deliberately minimal — clip, normalize, z-score, random
flip, and a Compose helper. Anything more elaborate (rotation, elastic
deformation, …) is better served by MONAI / torchio in a downstream
pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def _walk_images(sample: dict[str, Any]) -> Iterable[tuple[str, np.ndarray]]:
    images = sample.get("images")
    if isinstance(images, dict):
        yield from images.items()


class Compose:
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: list[Any]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class Clip:
    """Clip image intensities to a range.

    Applies to all modalities by default; pass ``modalities`` to
    restrict.
    """

    def __init__(
        self,
        min: float | None = None,
        max: float | None = None,
        modalities: list[str] | None = None,
    ) -> None:
        self.min = min
        self.max = max
        self.modalities = set(modalities) if modalities else None

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for name, arr in _walk_images(sample):
            if self.modalities is not None and name not in self.modalities:
                continue
            sample["images"][name] = np.clip(arr, self.min, self.max)
        return sample


class Normalize:
    """Subtract a mean and divide by std per modality.

    ``mean`` and ``std`` may be a single float (applied to every
    modality) or a dict keyed by modality name. Modalities with no
    matching entry are left untouched.
    """

    def __init__(
        self,
        mean: float | dict[str, float],
        std: float | dict[str, float],
    ) -> None:
        self.mean = mean
        self.std = std

    def _resolve(self, table: float | dict[str, float], name: str) -> float | None:
        if isinstance(table, dict):
            return table.get(name)
        return float(table)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for name, arr in _walk_images(sample):
            m = self._resolve(self.mean, name)
            s = self._resolve(self.std, name)
            if m is None or s is None or s == 0:
                continue
            sample["images"][name] = (arr.astype(np.float32) - float(m)) / float(s)
        return sample


class ZScore:
    """Per-volume z-scoring (subtract mean, divide by std of *this* sample)."""

    def __init__(self, modalities: list[str] | None = None, eps: float = 1e-6) -> None:
        self.modalities = set(modalities) if modalities else None
        self.eps = eps

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for name, arr in _walk_images(sample):
            if self.modalities is not None and name not in self.modalities:
                continue
            f = arr.astype(np.float32)
            sample["images"][name] = (f - float(f.mean())) / (float(f.std()) + self.eps)
        return sample


class RandomFlip:
    """Randomly flip along one or more axes with probability *p* (per axis)."""

    def __init__(
        self,
        axes: tuple[int, ...] | list[int] = (0,),
        p: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.axes = tuple(axes)
        self.p = float(p)
        self._rng = np.random.default_rng(seed)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        flip_axes: list[int] = []
        for ax in self.axes:
            if self._rng.random() < self.p:
                flip_axes.append(ax)
        if not flip_axes:
            return sample

        for name, arr in _walk_images(sample):
            sample["images"][name] = np.flip(arr, axis=tuple(flip_axes)).copy()
        seg = sample.get("seg")
        if isinstance(seg, dict):
            for name, arr in seg.items():
                seg[name] = np.flip(arr, axis=tuple(flip_axes)).copy()
        return sample
