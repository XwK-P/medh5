"""Chunk-size optimizer for HDF5 datasets.

Ported from mlarray's ``MLArray.comp_blosc2_params``, adapted for HDF5's
single-level chunking (no sub-chunk blocks).  The optimizer targets the
**L3 cache** because HDF5 decompresses whole chunks atomically.

The default cache constants come from an Intel Xeon Silver 4110:
  L1 = 32 KiB, L3 ≈ 1.375 MiB per core.
"""

from __future__ import annotations

import math
import os

import numpy as np

_CHUNK_OVERSHOOT_LIMIT = 1.5
"""Maximum allowed ratio of chunk size to patch size (per-axis average).

When growing chunks to fill the L3 cache budget, we stop expanding if
the mean ratio ``chunk / patch_size`` across spatial axes exceeds this
value.  Keeping chunks close to the patch size avoids reading excessive
unused data during patch-based training.
"""

_DEFAULT_L3_BYTES = 1_441_792


def _detect_l3_cache_bytes() -> int | None:
    """Try to detect L3 cache size from the OS.  Returns *None* on failure."""
    try:
        raw = os.popen("sysctl -n hw.l3cachesize 2>/dev/null").read().strip()
        if raw.isdigit():
            return int(raw)
    except OSError:
        pass
    try:
        with open("/sys/devices/system/cpu/cpu0/cache/index3/size") as fh:
            raw = fh.read().strip()
        if raw.endswith("K"):
            return int(raw[:-1]) * 1024
        if raw.endswith("M"):
            return int(raw[:-1]) * 1024 * 1024
        if raw.isdigit():
            return int(raw)
    except (FileNotFoundError, OSError, ValueError):
        pass
    return None


def optimize_chunks(
    image_shape: tuple[int, ...],
    patch_size: tuple[int, ...] | int,
    bytes_per_element: int = 4,
    spatial_axis_mask: list[bool] | None = None,
    l3_bytes: int | None = None,
    safety: float = 0.8,
) -> tuple[int, ...]:
    """Return an HDF5 chunk shape optimized for patch-based reads.

    Parameters
    ----------
    image_shape:
        Full shape of the array (including non-spatial axes).
    patch_size:
        Spatial patch size used during training.  An ``int`` is broadcast
        to all spatial axes.
    bytes_per_element:
        ``dtype.itemsize`` of the array (4 for float32).
    spatial_axis_mask:
        Boolean mask of length ``len(image_shape)`` indicating which axes
        are spatial.  When *None* every axis is treated as spatial.
    l3_bytes:
        L3 cache per core in bytes.  When *None* the optimizer tries to
        auto-detect the L3 cache from the OS and falls back to a
        conservative default (≈ 1.375 MiB).
    safety:
        Fraction of the cache to target (avoids filling it to the brim).
    """
    if l3_bytes is None:
        l3_bytes = _detect_l3_cache_bytes() or _DEFAULT_L3_BYTES

    image_shape = tuple(int(s) for s in image_shape)
    ndim = len(image_shape)

    if spatial_axis_mask is None:
        spatial_axis_mask = [True] * ndim
    if len(spatial_axis_mask) != ndim:
        raise ValueError("spatial_axis_mask length must match image_shape")

    spatial_ndim = sum(spatial_axis_mask)
    if spatial_ndim < 2 or spatial_ndim > 3:
        raise NotImplementedError(
            "Chunk optimization is implemented for 2-D and 3-D spatial data. "
            "Set chunk sizes manually for other dimensionalities."
        )
    if sum(not b for b in spatial_axis_mask) > 1:
        raise NotImplementedError(
            "At most one non-spatial axis is supported by the optimizer."
        )

    if isinstance(patch_size, (int, float)):
        patch_size = (int(patch_size),) * spatial_ndim
    patch_size = tuple(int(p) for p in patch_size)
    if len(patch_size) != spatial_ndim:
        raise ValueError(
            f"patch_size length ({len(patch_size)}) must match the number of "
            f"spatial axes ({spatial_ndim})"
        )

    def _move(lst: list[int], src: int, dst: int) -> list[int]:
        lst = list(lst)
        val = lst.pop(src)
        lst.insert(dst, val)
        return lst

    num_squeezes = 0
    img = list(image_shape)

    if ndim == 2:
        img = [1, 1] + img
        num_squeezes = 2
    elif ndim == 3:
        img = [1] + img
        num_squeezes = 1

    non_spatial_axis = None
    if spatial_axis_mask is not None:
        ns_mask = [not b for b in spatial_axis_mask]
        non_spatial_axis = next((i for i, v in enumerate(ns_mask) if v), None)
        if non_spatial_axis is not None:
            img = _move(img, non_spatial_axis + num_squeezes, 0)

    if len(img) != 4:
        raise RuntimeError("Internal error: image must be 4-D after expansion.")

    ps = list(patch_size)
    if len(ps) == 2:
        ps = [1] + ps
    ps_arr = np.array(ps)

    chunk = np.array((img[0], *[2 ** max(0, math.ceil(math.log2(p))) for p in ps_arr]))
    chunk = np.array([min(c, s) for c, s in zip(chunk, img, strict=True)])

    budget = l3_bytes * safety

    estimated = int(np.prod(chunk)) * bytes_per_element
    while estimated < budget:
        tail_match = all(chunk[2 + i] == img[2 + i] for i in range(len(ps_arr) - 1))
        if ps_arr[0] == 1 and tail_match:
            break
        if all(chunk[i] == img[i] for i in range(4)):
            break

        base = np.copy(chunk)
        axis_order = np.argsort(chunk[1:] / ps_arr)
        picked = None
        for ax in axis_order:
            if chunk[ax + 1] < img[ax + 1] and ps_arr[ax] != 1:
                picked = ax
                break
        if picked is None:
            break

        step = 2 ** max(0, math.ceil(math.log2(ps_arr[picked])))
        chunk[picked + 1] = min(chunk[picked + 1] + step, img[picked + 1])
        estimated = int(np.prod(chunk)) * bytes_per_element

        if np.mean(chunk[1:] / ps_arr) > _CHUNK_OVERSHOOT_LIMIT:
            chunk = base
            break

    chunk_list: list[int] = [
        min(int(c), int(s)) for c, s in zip(chunk, img, strict=True)
    ]

    if non_spatial_axis is not None:
        chunk_list = _move(chunk_list, 0, non_spatial_axis + num_squeezes)

    chunk_list = chunk_list[num_squeezes:]
    return tuple(int(v) for v in chunk_list)
