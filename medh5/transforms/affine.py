"""``identity`` and ``affine`` transforms (spec §10.3)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.transforms.base import Transform

LAST_ROW_TOL = 1e-9


def encode_identity() -> AnnotationPayload:
    """The identity transform, which stores nothing but its endpoints."""
    return AnnotationPayload(kind="identity")


def encode_affine(matrix: npt.ArrayLike) -> AnnotationPayload:
    """Pack a homogeneous world-to-world affine (spec §10.3).

    Index-space affines are deliberately not storable: an affine that means
    something only in one grid's index space silently breaks when the image is
    resampled, and composing with the grids' affines (§3.3) recovers it whenever
    it is actually wanted.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"affine `matrix` must be square (S+1, S+1), got {array.shape}", code="E504"
        )
    dim = array.shape[0] - 1
    expected = np.zeros(dim + 1)
    expected[-1] = 1.0
    if not np.allclose(array[-1], expected, atol=LAST_ROW_TOL):
        raise MEDH5ValidationError(
            f"affine last row must be [0 … 0 1], got {array[-1].tolist()}", code="E504"
        )
    if abs(float(np.linalg.det(array[:dim, :dim]))) < LAST_ROW_TOL:
        raise MEDH5ValidationError(
            "affine linear part is singular, so the transform maps space onto a "
            "lower-dimensional set",
            code="E504",
        )
    return AnnotationPayload(kind="affine", datasets={"matrix": array})


class IdentityTransform(Transform):
    """Reader for ``kind = "identity"``: two frames declared to coincide."""

    __slots__ = ()

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(points, dtype=np.float64)

    @property
    def is_invertible(self) -> bool:
        return True


class AffineTransform(Transform):
    """Reader for ``kind = "affine"``."""

    __slots__ = ()

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        if "matrix" not in self.group:
            raise MEDH5ValidationError(
                f"transform {self.transform_id!r}: `affine` requires a `matrix` "
                f"dataset",
                code="E502",
            )
        return np.asarray(self.group["matrix"][...], dtype=np.float64)

    @property
    def n_spatial(self) -> int:
        return int(self.matrix.shape[0] - 1)

    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        matrix = self.matrix
        dim = matrix.shape[0] - 1
        values = np.asarray(points, dtype=np.float64)
        flat = values.reshape(-1, dim)
        out = flat @ matrix[:dim, :dim].T + matrix[:dim, dim]
        return np.asarray(out.reshape(values.shape), dtype=np.float64)

    @property
    def is_invertible(self) -> bool:
        if self.header.invertible is not None:
            return bool(self.header.invertible)
        dim = self.matrix.shape[0] - 1
        return bool(abs(float(np.linalg.det(self.matrix[:dim, :dim]))) > LAST_ROW_TOL)

    def inverse_matrix(self) -> npt.NDArray[np.float64]:
        """``T⁻¹`` as a matrix, computed rather than stored."""
        return np.asarray(np.linalg.inv(self.matrix), dtype=np.float64)

    def inverse_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        matrix = self.inverse_matrix()
        dim = matrix.shape[0] - 1
        values = np.asarray(points, dtype=np.float64)
        flat = values.reshape(-1, dim)
        out = flat @ matrix[:dim, :dim].T + matrix[:dim, dim]
        return np.asarray(out.reshape(values.shape), dtype=np.float64)

    @property
    def jacobian_determinant_value(self) -> float:
        """An affine's Jacobian determinant is constant, so it is one number."""
        dim = self.matrix.shape[0] - 1
        return float(np.linalg.det(self.matrix[:dim, :dim]))

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["jacobian_determinant"] = self.jacobian_determinant_value
        return out


__all__ = [
    "AffineTransform",
    "IdentityTransform",
    "encode_affine",
    "encode_identity",
]
