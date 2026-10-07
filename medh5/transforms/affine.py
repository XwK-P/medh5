"""``identity`` and ``affine`` transforms (spec §10.3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.transforms.base import Transform

LAST_ROW_TOL: float = _core.LAST_ROW_TOL

encode_identity = _core.encode_identity
encode_affine = _core.encode_affine


class IdentityTransform(Transform):
    """Two frames that coincide, said explicitly."""

    __slots__ = ()


class AffineTransform(Transform):
    """A homogeneous ``(S+1, S+1)`` matrix in world coordinates."""

    __slots__ = ()

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.matrix
        return found

    @property
    def n_spatial(self) -> int:
        return int(self._handle.n_spatial)

    def inverse_matrix(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_matrix()
        return found

    def inverse_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.inverse_points(points)
        return found

    @property
    def jacobian_determinant_value(self) -> float:
        return float(self._handle.jacobian_determinant_value)


__all__ = [
    "LAST_ROW_TOL",
    "AffineTransform",
    "IdentityTransform",
    "encode_affine",
    "encode_identity",
]
