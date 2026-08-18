"""The index-world affine and the box/slice convention (spec §3.3, §8.1).

Two conventions are fixed here and nowhere else, because a format that leaves
either implicit produces silent half-voxel errors that survive every unit test
and show up as a systematic bias in a trained model:

**Index coordinates.**  Integer index *i* is the **centre** of voxel *i*.  A
volume of extent *n* therefore spans the closed region ``[-0.5, n - 0.5]``, and

.. code-block:: text

    x_world = origin + direction @ (spacing * i)

**Box corners.**  Boxes are measured at voxel **edges**, so a box maps onto a
numpy slice without rounding:

.. code-block:: text

    numpy slice a:b   <->   lo = a - 0.5,  hi = b - 0.5     (extent hi - lo = b - a)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError

ORTHONORMAL_TOL = 1e-4
"""Tolerance on ``direction.T @ direction == I`` (spec §3.2)."""


def build_affine(
    spacing: npt.ArrayLike,
    origin: npt.ArrayLike,
    direction: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Assemble the ``(S+1, S+1)`` index-to-world affine of spec §3.3."""
    sp = np.asarray(spacing, dtype=np.float64)
    org = np.asarray(origin, dtype=np.float64)
    dirn = np.asarray(direction, dtype=np.float64)
    s = sp.shape[0]
    if org.shape != (s,) or dirn.shape != (s, s):
        raise MEDH5ValidationError(
            f"spacing {sp.shape}, origin {org.shape} and direction {dirn.shape} "
            "must agree on the number of spatial axes",
            code="E109",
        )
    affine = np.eye(s + 1, dtype=np.float64)
    affine[:s, :s] = dirn @ np.diag(sp)
    affine[:s, s] = org
    return affine


def decompose_affine(
    affine: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split an affine back into ``(spacing, origin, direction)``.

    Exact inverse of :func:`build_affine` for any affine whose linear part has
    non-zero columns; the column norms are the spacings by construction.
    """
    mat = np.asarray(affine, dtype=np.float64)
    s = mat.shape[0] - 1
    linear = mat[:s, :s]
    spacing = np.linalg.norm(linear, axis=0)
    if np.any(spacing <= 0):
        raise MEDH5ValidationError("affine has a degenerate spatial axis", code="E104")
    direction = linear / spacing
    return spacing, mat[:s, s].copy(), direction


def is_orthonormal(direction: npt.ArrayLike, tol: float = ORTHONORMAL_TOL) -> bool:
    """Whether ``direction`` is orthonormal within *tol*."""
    dirn = np.asarray(direction, dtype=np.float64)
    if dirn.ndim != 2 or dirn.shape[0] != dirn.shape[1]:
        return False
    return bool(np.allclose(dirn.T @ dirn, np.eye(dirn.shape[0]), atol=tol, rtol=0.0))


def check_orthonormal(
    direction: npt.ArrayLike, tol: float = ORTHONORMAL_TOL, *, what: str = "direction"
) -> npt.NDArray[np.float64]:
    """Validate orthonormality, raising E102 when it fails."""
    dirn = np.asarray(direction, dtype=np.float64)
    if not is_orthonormal(dirn, tol):
        residual = float(np.max(np.abs(dirn.T @ dirn - np.eye(dirn.shape[0]))))
        raise MEDH5ValidationError(
            f"{what} is not orthonormal to {tol:g} (max residual {residual:.3g})",
            code="E102",
        )
    return dirn


def is_proper_rotation(matrix: npt.ArrayLike, tol: float = ORTHONORMAL_TOL) -> bool:
    """Whether *matrix* is orthonormal **and** has determinant +1."""
    mat = np.asarray(matrix, dtype=np.float64)
    if not is_orthonormal(mat, tol):
        return False
    return bool(abs(float(np.linalg.det(mat)) - 1.0) <= max(tol, 1e-6))


def index_to_world(
    affine: npt.ArrayLike, indices: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Map continuous index coordinates ``(..., S)`` to world coordinates."""
    mat = np.asarray(affine, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.float64)
    s = mat.shape[0] - 1
    flat = idx.reshape(-1, s)
    out = np.asarray(flat @ mat[:s, :s].T + mat[:s, s], dtype=np.float64)
    return out.reshape(idx.shape)


def world_to_index(
    affine: npt.ArrayLike, points: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Map world coordinates ``(..., S)`` back to continuous index coordinates."""
    mat = np.asarray(affine, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    s = mat.shape[0] - 1
    inv = np.linalg.inv(mat[:s, :s])
    flat = pts.reshape(-1, s)
    out = np.asarray((flat - mat[:s, s]) @ inv.T, dtype=np.float64)
    return out.reshape(pts.shape)


def box_to_slices(
    box: npt.ArrayLike, shape: Sequence[int] | None = None
) -> tuple[slice, ...]:
    """Convert one ``(S, 2)`` box in continuous index coordinates to numpy slices.

    ``start = round(lo + 0.5)``, ``stop = round(hi + 0.5)``.  When *shape* is
    given the result is clipped to the array, which is what a reader wants for a
    box that a resample pushed slightly out of bounds.
    """
    arr = np.asarray(box, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise MEDH5ValidationError(f"box must have shape (S, 2), got {arr.shape}")
    if np.any(arr[:, 0] > arr[:, 1]):
        raise MEDH5ValidationError("box has lo > hi on at least one axis", code="E406")
    start = np.rint(arr[:, 0] + 0.5).astype(np.int64)
    stop = np.rint(arr[:, 1] + 0.5).astype(np.int64)
    if shape is not None:
        extent = np.asarray(shape, dtype=np.int64)
        start = np.clip(start, 0, extent)
        stop = np.clip(stop, start, extent)
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop, strict=True))


def slices_to_box(slices: Sequence[slice]) -> npt.NDArray[np.float32]:
    """Convert numpy slices to one ``(S, 2)`` box in continuous index coordinates."""
    out = np.empty((len(slices), 2), dtype=np.float32)
    for axis, sl in enumerate(slices):
        if sl.start is None or sl.stop is None:
            raise MEDH5ValidationError("slices must have explicit start and stop")
        if sl.step not in (None, 1):
            raise MEDH5ValidationError("strided slices have no box representation")
        out[axis, 0] = sl.start - 0.5
        out[axis, 1] = sl.stop - 0.5
    return out


def box_corners(box: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """The ``(2**S, S)`` corners of an axis-aligned box, in odometer order.

    Converting a box to world coordinates is only meaningful through its
    corners: an axis-aligned box in index space is in general an oriented box
    in world space, and taking ``min``/``max`` of the transformed ``lo``/``hi``
    silently inflates it under an oblique ``direction``.
    """
    arr = np.asarray(box, dtype=np.float64)
    s = arr.shape[0]
    selectors = np.indices((2,) * s).reshape(s, -1).T
    corners: npt.NDArray[np.float64] = arr[np.arange(s)[None, :], selectors]
    return corners


def apply_affine_to_box(
    affine: npt.ArrayLike, box: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Axis-aligned bounds of a box after an affine, computed over its corners."""
    corners = index_to_world(affine, box_corners(box))
    return np.stack([corners.min(axis=0), corners.max(axis=0)], axis=1)


def voxel_volume(spacing: Sequence[float]) -> float:
    """Volume of one voxel in the grid's units^S."""
    return float(np.prod(np.asarray(spacing, dtype=np.float64)))


def affine_summary(affine: npt.ArrayLike) -> dict[str, Any]:
    """Human-facing summary used by ``medh5 info``."""
    spacing, origin, direction = decompose_affine(affine)
    return {
        "spacing": [float(v) for v in spacing],
        "origin": [float(v) for v in origin],
        "direction": [[float(v) for v in row] for row in direction],
        "determinant": float(np.linalg.det(direction)),
        "orthonormal": is_orthonormal(direction),
    }


__all__ = [
    "ORTHONORMAL_TOL",
    "affine_summary",
    "apply_affine_to_box",
    "box_corners",
    "box_to_slices",
    "build_affine",
    "check_orthonormal",
    "decompose_affine",
    "index_to_world",
    "is_orthonormal",
    "is_proper_rotation",
    "slices_to_box",
    "voxel_volume",
    "world_to_index",
]
