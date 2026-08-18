"""Evaluating transforms: interpolation, Jacobians and registration error (§10).

Interpolation is implemented here rather than delegated, because the two things
a reader needs from a displacement field --- its value between samples and the
determinant of its Jacobian --- must agree with each other, and a field whose
values come from one library and whose derivatives come from another produces a
folding fraction nobody can reproduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid

EXTRAPOLATIONS = ("zero", "nearest", "error")


def _check_extrapolation(extrapolation: str) -> None:
    if extrapolation not in EXTRAPOLATIONS:
        raise MEDH5ValidationError(f"unknown extrapolation {extrapolation!r}")


def _inside_field(
    field: npt.NDArray[Any], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    """Which *points* lie within the field's sampled domain, edges included."""
    spatial = np.asarray(field.shape[1:], dtype=np.int64)
    inside: npt.NDArray[np.bool_] = np.all(
        (points >= -0.5) & (points <= spatial - 0.5), axis=1
    )
    return inside


def _refuse_outside(inside: npt.NDArray[np.bool_]) -> None:
    """``extrapolation='error'`` is a refusal, not a quieter fill value."""
    if bool(inside.all()):
        return
    raise MEDH5ValidationError(
        f"{int((~inside).sum())} point(s) fall outside the field and "
        "extrapolation='error'"
    )


def linear_sample(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]:
    """Multilinear interpolation of ``(S, *spatial)`` data at continuous indices.

    *coords* is ``(N, S)`` in the field's continuous index coordinates.  Points
    outside the field follow *extrapolation*: ``zero`` (no displacement),
    ``nearest`` (clamp to the edge), or ``error``.
    """
    components = field.shape[0]
    spatial = np.asarray(field.shape[1:], dtype=np.int64)
    dim = spatial.size
    points = np.asarray(coords, dtype=np.float64).reshape(-1, dim)

    _check_extrapolation(extrapolation)
    inside = _inside_field(field, points)
    if extrapolation == "error":
        _refuse_outside(inside)

    clamped = np.clip(points, 0.0, spatial - 1.0)
    base = np.floor(clamped).astype(np.int64)
    base = np.minimum(base, np.maximum(spatial - 2, 0))
    frac = clamped - base

    out = np.zeros((points.shape[0], components), dtype=np.float64)
    for corner in range(1 << dim):
        offsets = np.array(
            [(corner >> axis) & 1 for axis in range(dim)], dtype=np.int64
        )
        weight = np.prod(np.where(offsets == 1, frac, 1.0 - frac), axis=1)
        index = np.minimum(base + offsets, spatial - 1)
        gathered = field[(slice(None), *index.T)].T
        out += weight[:, None] * gathered
    if extrapolation == "zero":
        out[~inside] = 0.0
    return out


def cubic_sample(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]:
    """Cubic interpolation, which needs SciPy.

    Rather than ship a second, subtly different cubic kernel, this defers to
    ``scipy.ndimage.map_coordinates``; a file declaring ``interpolation: cubic``
    that is read without SciPy fails loudly instead of quietly falling back to
    linear and shifting every warped voxel.
    """
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:  # pragma: no cover - exercised only without SciPy
        raise MEDH5ValidationError(
            "this transform declares interpolation='cubic', which needs SciPy; "
            "install it, or read the field with interpolation='linear' explicitly"
        ) from exc
    dim = field.ndim - 1
    points = np.asarray(coords, dtype=np.float64).reshape(-1, dim)
    _check_extrapolation(extrapolation)
    if extrapolation == "error":
        # SciPy has no raising mode, so the domain check happens here, against
        # the same bounds `linear_sample` uses.  Folding 'error' into SciPy's
        # constant-zero would answer an out-of-domain query with "no
        # displacement" --- precisely the silence the declared contract exists
        # to break, and only for cubic fields.
        _refuse_outside(_inside_field(field, points))
    mode = {"zero": "constant", "nearest": "nearest", "error": "constant"}[
        extrapolation
    ]
    return np.stack(
        [
            map_coordinates(field[component], points.T, order=3, mode=mode, cval=0.0)
            for component in range(field.shape[0])
        ],
        axis=1,
    )


def sample_field(
    field: npt.NDArray[Any],
    coords: npt.NDArray[np.float64],
    *,
    interpolation: str = "linear",
    extrapolation: str = "zero",
) -> npt.NDArray[np.float64]:
    if interpolation == "linear":
        return linear_sample(field, coords, extrapolation=extrapolation)
    if interpolation == "cubic":
        return cubic_sample(field, coords, extrapolation=extrapolation)
    raise MEDH5ValidationError(f"unknown interpolation {interpolation!r}")


def linear_part(grid: Grid) -> npt.NDArray[np.float64]:
    """``direction @ diag(spacing)`` --- index displacement to world displacement."""
    return np.asarray(grid.direction, dtype=np.float64) @ np.diag(
        np.asarray(grid.spacing, dtype=np.float64)
    )


def to_world_vectors(
    vectors: npt.NDArray[Any], grid: Grid, vector_space: str
) -> npt.NDArray[np.float64]:
    """Convert displacement components to world units."""
    values = np.asarray(vectors, dtype=np.float64)
    if vector_space == "world":
        return values
    if vector_space == "index":
        return values @ linear_part(grid).T
    raise MEDH5ValidationError(f"unknown vector_space {vector_space!r}")


def jacobian_determinant(
    field: npt.NDArray[Any],
    grid: Grid,
    *,
    vector_space: str = "world",
) -> npt.NDArray[np.float64]:
    """``det(I + du/dx)`` per voxel of the field grid.

    Values at or below zero mark **folding**: the transform is not a
    diffeomorphism there, and a warped label map will tear or overlap itself.
    Reporting the fraction is what makes a registration comparable between runs.
    """
    dim = field.shape[0]
    if dim != grid.n_spatial:
        raise MEDH5ValidationError(
            f"field has {dim} components for a {grid.n_spatial}-D grid", code="E503"
        )
    world = np.asarray(field, dtype=np.float64)
    linear = linear_part(grid)
    if vector_space == "index":
        world = np.tensordot(linear, world, axes=([1], [0]))
    elif vector_space != "world":
        raise MEDH5ValidationError(f"unknown vector_space {vector_space!r}")

    inverse_linear = np.linalg.inv(linear)
    # du/di, then du/dx = du/di @ di/dx
    gradients = np.empty((dim, dim, *world.shape[1:]), dtype=np.float64)
    for component in range(dim):
        for axis in range(dim):
            gradients[component, axis] = np.gradient(world[component], axis=axis)
    jac = np.einsum("ca...,ab->cb...", gradients, inverse_linear)
    jac = np.moveaxis(jac, (0, 1), (-2, -1)) + np.eye(dim)
    return np.asarray(np.linalg.det(jac), dtype=np.float64)


def folding_fraction(determinants: npt.NDArray[np.float64]) -> float:
    """Fraction of voxels where the transform folds (``det <= 0``)."""
    values = np.asarray(determinants)
    return float(np.count_nonzero(values <= 0.0) / values.size) if values.size else 0.0


def target_registration_error(
    transform: Any,
    fixed_points: npt.ArrayLike,
    moving_points: npt.ArrayLike,
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """TRE: ``‖T(p_i^F) − p_i^M‖``, the number landmark ground truth exists to give.

    Both point sets are world coordinates with matching row order (§10.6).
    """
    fixed = np.asarray(fixed_points, dtype=np.float64)
    moving = np.asarray(moving_points, dtype=np.float64)
    if fixed.shape != moving.shape:
        raise MEDH5ValidationError(
            f"landmark sets disagree: {fixed.shape} vs {moving.shape}; §10.6 requires "
            "equal N and matching row order"
        )
    warped = transform.transform_points(fixed)
    errors = np.linalg.norm(warped - moving, axis=1)
    w = (
        np.asarray(weights, dtype=np.float64)
        if weights is not None
        else np.ones_like(errors)
    )
    total = float(w.sum())
    return {
        "mean": float((errors * w).sum() / total) if total else 0.0,
        "median": float(np.median(errors)),
        "max": float(errors.max()) if errors.size else 0.0,
        "n": int(errors.size),
    }


__all__ = [
    "cubic_sample",
    "folding_fraction",
    "jacobian_determinant",
    "linear_part",
    "linear_sample",
    "sample_field",
    "target_registration_error",
    "to_world_vectors",
]
