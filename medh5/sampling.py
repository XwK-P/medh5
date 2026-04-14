"""Patch samplers for lazy patch-based training over .medh5 files.

A :class:`PatchSampler` takes an open :class:`MEDH5File` and returns a
single patch as a sample dict (``{"images": {...}, "seg": {...}, ...}``).
All slicing goes through the lazy h5py datasets, so memory use is
bounded by the patch size — not the volume size.

Three strategies are supported:

- ``"uniform"``: random crop within the volume bounds.
- ``"foreground"``: bias the crop center toward voxels inside a named
  segmentation mask. Falls back to uniform when the mask is empty.
- ``"balanced"``: alternate between foreground and uniform with a
  configurable probability (``foreground_prob``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5ValidationError


def _coerce_patch_size(
    patch_size: int | tuple[int, ...] | list[int], ndim: int
) -> tuple[int, ...]:
    if isinstance(patch_size, int):
        return (patch_size,) * ndim
    p = tuple(int(v) for v in patch_size)
    if len(p) != ndim:
        raise MEDH5ValidationError(f"patch_size length {len(p)} != volume ndim {ndim}")
    return p


def _build_slices(starts: tuple[int, ...], patch: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(s, s + p) for s, p in zip(starts, patch, strict=True))


def _pad_to(arr: np.ndarray, target: tuple[int, ...], value: float) -> np.ndarray:
    """Pad *arr* to *target* shape with constant *value* (used for edge crops)."""
    if arr.shape == target:
        return arr
    pad_width = [(0, t - s) for s, t in zip(arr.shape, target, strict=True)]
    return np.pad(arr, pad_width, mode="constant", constant_values=value)


class PatchSampler:
    """Lazy patch sampler for .medh5 files.

    Parameters
    ----------
    patch_size : int or tuple
        Spatial extent of the patch. Scalars are broadcast to every
        spatial axis.
    strategy : {"uniform", "foreground", "balanced"}
        Sampling strategy. ``foreground`` and ``balanced`` require
        ``foreground_seg``.
    foreground_seg : str, optional
        Name of the segmentation mask to bias toward.
    foreground_prob : float
        For ``"balanced"``: probability that a patch is foreground-biased.
        Defaults to 0.5.
    pad_value : float
        Value used to pad patches that fall partially outside the
        volume bounds.
    include_bboxes : bool
        If True and the source file contains bounding boxes, return
        patch-local bboxes in ``sample["bboxes"]`` (and the matching
        scores/labels).  Boxes are translated to patch-local
        coordinates and filtered to those that intersect the patch;
        bounds are not clipped.
    seed : int, optional
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, ...] | list[int],
        *,
        strategy: str = "uniform",
        foreground_seg: str | None = None,
        foreground_prob: float = 0.5,
        pad_value: float = 0.0,
        include_bboxes: bool = False,
        seed: int | None = None,
    ) -> None:
        if strategy not in ("uniform", "foreground", "balanced"):
            raise MEDH5ValidationError(
                f"unknown strategy '{strategy}'. "
                "Choose from: uniform, foreground, balanced."
            )
        if strategy in ("foreground", "balanced") and foreground_seg is None:
            raise MEDH5ValidationError(
                f"strategy='{strategy}' requires foreground_seg=<mask name>"
            )
        self.patch_size = patch_size
        self.strategy = strategy
        self.foreground_seg = foreground_seg
        self.foreground_prob = float(foreground_prob)
        self.pad_value = float(pad_value)
        self.include_bboxes = bool(include_bboxes)
        self._rng = np.random.default_rng(seed)
        self._fg_indices_cache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Internal start-position picking
    # ------------------------------------------------------------------

    def _uniform_start(
        self, shape: tuple[int, ...], patch: tuple[int, ...]
    ) -> tuple[int, ...]:
        starts = []
        for s, p in zip(shape, patch, strict=True):
            high = max(s - p, 0) + 1
            starts.append(int(self._rng.integers(0, high)))
        return tuple(starts)

    def _foreground_start(
        self,
        f: MEDH5File,
        shape: tuple[int, ...],
        patch: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Pick a start such that some foreground voxel is inside the patch.

        Returns ``None`` if the mask is missing or has no foreground.
        """
        assert self.foreground_seg is not None
        seg_grp = f.seg
        if seg_grp is None or self.foreground_seg not in seg_grp:
            return None

        cache_key = f"{f._path}::{self.foreground_seg}"
        coords = self._fg_indices_cache.get(cache_key)
        if coords is None:
            mask = np.asarray(seg_grp[self.foreground_seg][...], dtype=bool)
            idx = np.argwhere(mask)
            if idx.size == 0:
                self._fg_indices_cache[cache_key] = idx
                return None
            self._fg_indices_cache[cache_key] = idx
            coords = idx

        if coords.size == 0:
            return None

        # Pick a random foreground voxel and center the patch on it.
        center = coords[self._rng.integers(0, coords.shape[0])]
        starts: list[int] = []
        for c, s, p in zip(center, shape, patch, strict=True):
            lo = int(c) - p // 2
            lo = max(0, min(lo, max(s - p, 0)))
            starts.append(lo)
        return tuple(starts)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, f: MEDH5File) -> dict[str, Any]:
        """Read one patch from an open :class:`MEDH5File`."""
        meta = f.meta
        img_grp = f.images
        first_name = sorted(img_grp.keys())[0]
        shape = tuple(img_grp[first_name].shape)
        patch = _coerce_patch_size(self.patch_size, len(shape))

        if self.strategy == "uniform":
            starts = self._uniform_start(shape, patch)
        elif self.strategy == "foreground":
            starts = self._foreground_start(f, shape, patch) or self._uniform_start(
                shape, patch
            )
        else:  # balanced
            if self._rng.random() < self.foreground_prob:
                starts = self._foreground_start(f, shape, patch) or self._uniform_start(
                    shape, patch
                )
            else:
                starts = self._uniform_start(shape, patch)

        slices = _build_slices(starts, patch)

        out: dict[str, Any] = {
            "images": {},
            "label": meta.label,
            "meta": meta,
            "patch_origin": starts,
        }
        for name in sorted(img_grp.keys()):
            arr = np.asarray(img_grp[name][slices])
            out["images"][name] = _pad_to(arr, patch, self.pad_value)

        if f.seg is not None:
            out["seg"] = {}
            for name in f.seg:
                arr = np.asarray(f.seg[name][slices])
                out["seg"][name] = _pad_to(arr, patch, 0)

        if self.include_bboxes:
            boxes, scores, labels = f.bbox_arrays()
            if boxes is not None:
                if boxes.shape[0] > 0:
                    shift = np.asarray(starts, dtype=boxes.dtype)
                    boxes_local = boxes - shift[None, :, None]
                    patch_arr = np.asarray(patch, dtype=boxes.dtype)
                    keep = np.all(
                        (boxes_local[..., 1] >= 0)
                        & (boxes_local[..., 0] < patch_arr[None, :]),
                        axis=1,
                    )
                else:
                    boxes_local = boxes
                    keep = np.zeros((0,), dtype=bool)
                out["bboxes"] = boxes_local[keep]
                if scores is not None:
                    out["bbox_scores"] = scores[keep]
                if labels is not None:
                    keep_list = keep.tolist()
                    out["bbox_labels"] = [
                        lbl for lbl, k in zip(labels, keep_list, strict=True) if k
                    ]

        return out
