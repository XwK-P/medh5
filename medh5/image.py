"""Images: dense arrays defined on exactly one grid (spec §4).

An :class:`Image` is a view of one stored image; every answer it gives ---
geometry, attributes, rescaling, region reads --- comes from the format
engine, which reads the file directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.geometry.grid import Grid
from medh5.geometry.multiscale import Pyramid

VALUE_TYPES: tuple[str, ...] = _core.VALUE_TYPES

SPEC_IMAGE_ATTRS: tuple[str, ...] = _core.SPEC_IMAGE_ATTRS


class Image:
    """Lazy access to one image and its geometry.

    Reads go through :meth:`read`, which reads exactly the region asked for in
    one call.  The two-step form ``dataset[k][roi]`` materialises the whole
    sub-array first and is far slower (spec §14.5); this class does not expose
    an interface that makes the slow form natural.
    """

    __slots__ = ("_handle", "image_id")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.image_id: str = handle.image_id

    # -- structure ---------------------------------------------------------

    @property
    def is_multiscale(self) -> bool:
        return bool(self._handle.is_multiscale)

    @property
    def levels(self) -> int:
        return int(self._handle.levels)

    @property
    def dataset(self) -> Any:
        """The stored dataset (level 0 of a pyramid)."""
        return self._handle.dataset

    def level(self, index: int) -> Image:
        """A view of one pyramid level, sharing this image's attributes."""
        return Image(self._handle.level(index))

    @property
    def pyramid(self) -> Pyramid | None:
        found: Pyramid | None = self._handle.pyramid
        return found

    # -- attributes --------------------------------------------------------

    @property
    def attrs(self) -> dict[str, Any]:
        """The stored attributes, as a ``dict``."""
        found: dict[str, Any] = self._handle.attrs
        return found

    @property
    def grid_id(self) -> str:
        return str(self._handle.grid_id)

    @property
    def grid(self) -> Grid:
        found: Grid = self._handle.grid
        return found

    @property
    def timepoint(self) -> str | None:
        found: str | None = self._handle.timepoint
        return found

    @property
    def modality(self) -> str:
        return str(self._handle.modality)

    @property
    def value_type(self) -> str:
        return str(self._handle.value_type)

    @property
    def value_units(self) -> str | None:
        found: str | None = self._handle.value_units
        return found

    @property
    def channel_names(self) -> tuple[str, ...] | None:
        found: tuple[str, ...] | None = self._handle.channel_names
        return found

    @property
    def rescale(self) -> tuple[float, float]:
        """``(slope, intercept)``; ``(1.0, 0.0)`` when the image declares none."""
        slope, intercept = self._handle.rescale
        return float(slope), float(intercept)

    @property
    def is_rescaled(self) -> bool:
        return bool(self._handle.is_rescaled)

    @property
    def window(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        found: tuple[tuple[float, ...], tuple[float, ...]] | None = self._handle.window
        return found

    @property
    def valid_mask(self) -> str | None:
        found: str | None = self._handle.valid_mask
        return found

    @property
    def prov(self) -> str | None:
        found: str | None = self._handle.prov
        return found

    @property
    def digest(self) -> str | None:
        found: str | None = self._handle.digest
        return found

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._handle.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        found: np.dtype[Any] = self._handle.dtype
        return found

    @property
    def nbytes(self) -> int:
        return int(self._handle.nbytes)

    @property
    def chunks(self) -> tuple[int, ...] | None:
        found: tuple[int, ...] | None = self._handle.chunks
        return found

    # -- data --------------------------------------------------------------

    def read(
        self,
        roi: Sequence[slice | int] | None = None,
        *,
        physical: bool = False,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray[Any]:
        """Read the image or a region of it.

        *roi* covers every axis, or only the spatial axes (leading channel or
        time axes are then read whole).  ``physical=True`` applies
        ``rescale_slope``/``rescale_intercept`` (into ``float32`` unless
        *dtype* says otherwise).
        """
        found: npt.NDArray[Any] = self._handle.read(roi, physical=physical, dtype=dtype)
        return found

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def __repr__(self) -> str:
        return str(self._handle.__repr__())


def check_value_type(value_type: str) -> str:
    """A known ``value_type``, or E203."""
    return str(_core.check_value_type(value_type))


def lossless_as_int16(array: npt.NDArray[Any]) -> bool:
    """Whether a float array would survive ``int16`` storage unchanged (W907).

    CT stored as ``float32`` is the most common avoidable waste in medical
    imaging datasets: HU are integers, and the measured cost is 3.0x on disk for
    zero information (spec §4.2).
    """
    return bool(_core.lossless_as_int16(np.asarray(array)))


def is_probability(array: npt.NDArray[Any]) -> bool:
    """Whether every value lies in ``[0, 1]`` (and there is at least one)."""
    return bool(_core.is_probability(np.asarray(array)))


__all__ = [
    "SPEC_IMAGE_ATTRS",
    "VALUE_TYPES",
    "Image",
    "check_value_type",
    "is_probability",
    "lossless_as_int16",
]
