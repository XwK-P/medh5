"""Images: dense arrays defined on exactly one grid (spec §4)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5._hdf5 import (
    as_float,
    as_float_tuple,
    as_int,
    as_str,
    as_str_tuple,
    require_attr,
)
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.geometry.multiscale import Pyramid

VALUE_TYPES = (
    "intensity",
    "quantitative",
    "rgb",
    "probability",
    "displacement",
    "mask",
)

SPEC_IMAGE_ATTRS = (
    "grid",
    "modality",
    "value_type",
    "channel_names",
    "rescale_slope",
    "rescale_intercept",
    "value_units",
    "window_center",
    "window_width",
    "valid_mask",
    "digest",
    "prov",
    "levels",
    "downsample_factors",
    "downsample_method",
    "grid_levels",
)


class Image:
    """Lazy access to one image and its geometry.

    Reads go through :meth:`read`, which slices in a **single** h5py call.  The
    two-step form ``dset[k][roi]`` materialises the whole sub-array first and is
    40x slower for a 64^3 ROI out of a 160^3 plane (spec §14.5); this class does
    not expose an interface that makes the slow form natural.
    """

    __slots__ = ("_grids", "_node", "image_id")

    def __init__(
        self, image_id: str, node: h5py.Dataset | h5py.Group, grids: Mapping[str, Grid]
    ) -> None:
        self.image_id = image_id
        self._node = node
        self._grids = grids

    # -- structure ---------------------------------------------------------

    @property
    def is_multiscale(self) -> bool:
        return isinstance(self._node, h5py.Group)

    @property
    def levels(self) -> int:
        if not self.is_multiscale:
            return 1
        return as_int(self._node.attrs.get("levels", len(self._node)))

    @property
    def dataset(self) -> h5py.Dataset:
        """The level-0 dataset."""
        if self.is_multiscale:
            return self._node["0"]
        return self._node

    def level(self, index: int) -> Image:
        """A view of one pyramid level, sharing this image's attributes."""
        if not self.is_multiscale:
            if index != 0:
                raise MEDH5ValidationError(
                    f"image {self.image_id!r} is single-scale; level {index} does not "
                    f"exist"
                )
            return self
        return _ImageLevel(self.image_id, self._node, self._grids, index)

    @property
    def pyramid(self) -> Pyramid | None:
        if not self.is_multiscale:
            return None
        attrs = self._node.attrs
        return Pyramid(
            levels=as_int(attrs["levels"]),
            downsample_factors=np.asarray(
                attrs["downsample_factors"], dtype=np.float64
            ),
            downsample_method=as_str(attrs["downsample_method"]),
            grid_levels=as_str_tuple(attrs["grid_levels"]),
        )

    # -- attributes --------------------------------------------------------

    @property
    def attrs(self) -> Any:
        return self._node.attrs

    @property
    def grid_id(self) -> str:
        return as_str(require_attr(self._node, "grid", code="E205"))

    @property
    def grid(self) -> Grid:
        try:
            return self._grids[self.grid_id]
        except KeyError:
            raise MEDH5ValidationError(
                f"image {self.image_id!r} names grid {self.grid_id!r}, which does not "
                f"exist",
                code="E101",
            ) from None

    @property
    def timepoint(self) -> str | None:
        """Inherited from the grid --- an image never declares its own (§3.7)."""
        return self.grid.timepoint

    @property
    def modality(self) -> str:
        return as_str(require_attr(self._node, "modality", code="E205"))

    @property
    def value_type(self) -> str:
        return as_str(require_attr(self._node, "value_type", code="E205"))

    @property
    def value_units(self) -> str | None:
        attrs = self._node.attrs
        return as_str(attrs["value_units"]) if "value_units" in attrs else None

    @property
    def channel_names(self) -> tuple[str, ...] | None:
        attrs = self._node.attrs
        return (
            as_str_tuple(attrs["channel_names"]) if "channel_names" in attrs else None
        )

    @property
    def rescale(self) -> tuple[float, float]:
        """``(slope, intercept)``, defaulting to the identity rescale."""
        attrs = self._node.attrs
        slope = as_float(attrs["rescale_slope"]) if "rescale_slope" in attrs else 1.0
        intercept = (
            as_float(attrs["rescale_intercept"])
            if "rescale_intercept" in attrs
            else 0.0
        )
        return slope, intercept

    @property
    def is_rescaled(self) -> bool:
        slope, intercept = self.rescale
        return slope != 1.0 or intercept != 0.0

    @property
    def window(self) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        attrs = self._node.attrs
        if "window_center" not in attrs or "window_width" not in attrs:
            return None
        return as_float_tuple(attrs["window_center"]), as_float_tuple(
            attrs["window_width"]
        )

    @property
    def valid_mask(self) -> str | None:
        attrs = self._node.attrs
        return as_str(attrs["valid_mask"]) if "valid_mask" in attrs else None

    @property
    def prov(self) -> str | None:
        attrs = self._node.attrs
        return as_str(attrs["prov"]) if "prov" in attrs else None

    @property
    def digest(self) -> str | None:
        attrs = self.dataset.attrs
        return as_str(attrs["digest"]) if "digest" in attrs else None

    # -- data --------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.dataset.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dataset.dtype)

    @property
    def nbytes(self) -> int:
        return int(self.dataset.nbytes)

    @property
    def chunks(self) -> tuple[int, ...] | None:
        chunks = self.dataset.chunks
        return tuple(int(c) for c in chunks) if chunks else None

    def _index(self, roi: Sequence[slice] | None) -> tuple[Any, ...]:
        grid = self.grid
        if roi is None:
            return (slice(None),) * grid.ndim
        roi_t = tuple(roi)
        if len(roi_t) == grid.ndim:
            return roi_t
        if len(roi_t) == grid.n_spatial:
            lead = grid.ndim - grid.n_spatial
            return (slice(None),) * lead + roi_t
        raise MEDH5ValidationError(
            f"roi has {len(roi_t)} axes; image {self.image_id!r} has {grid.ndim} "
            f"({grid.n_spatial} spatial)"
        )

    def read(
        self,
        roi: Sequence[slice] | None = None,
        *,
        physical: bool = False,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.NDArray[Any]:
        """Read the image, or a region of it.

        ``physical=True`` applies ``stored * slope + intercept``.  The rescale is
        never applied silently: a caller asking for raw numbers gets raw numbers,
        because a model trained on rescaled input and evaluated on raw input
        fails in a way no test catches.
        """
        data = np.asarray(self.dataset[self._index(roi)])
        if physical and self.is_rescaled:
            slope, intercept = self.rescale
            out_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
            scaled = data.astype(out_dtype) * slope + intercept
            return np.asarray(scaled, dtype=out_dtype)
        if dtype is not None:
            return data.astype(dtype)
        return data

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.image_id,
            "grid": self.grid_id,
            "timepoint": self.timepoint,
            "modality": self.modality,
            "value_type": self.value_type,
            "value_units": self.value_units,
            "shape": list(self.shape),
            "dtype": self.dtype.str,
            "chunks": list(self.chunks) if self.chunks else None,
            "levels": self.levels,
            "rescale": list(self.rescale),
            "nbytes": self.nbytes,
        }

    def __repr__(self) -> str:
        return (
            f"Image({self.image_id!r}, {self.modality}, shape={self.shape}, "
            f"dtype={self.dtype.str}, grid={self.grid_id!r})"
        )


class _ImageLevel(Image):
    """One level of a multiscale image; attributes come from the group."""

    __slots__ = ("_level",)

    def __init__(
        self,
        image_id: str,
        node: h5py.Group,
        grids: Mapping[str, Grid],
        level: int,
    ) -> None:
        super().__init__(image_id, node, grids)
        self._level = level

    @property
    def dataset(self) -> h5py.Dataset:
        return self._node[str(self._level)]

    @property
    def grid_id(self) -> str:
        grid_levels = as_str_tuple(self._node.attrs["grid_levels"])
        return grid_levels[self._level]

    def level(self, index: int) -> Image:
        return _ImageLevel(self.image_id, self._node, self._grids, index)

    def __repr__(self) -> str:
        return f"Image({self.image_id!r} level {self._level}, shape={self.shape})"


def check_value_type(value_type: str) -> str:
    if value_type not in VALUE_TYPES:
        raise MEDH5ValidationError(
            f"unknown value_type {value_type!r}; expected one of {list(VALUE_TYPES)}",
            code="E203",
        )
    return value_type


def lossless_as_int16(array: npt.NDArray[Any]) -> bool:
    """Whether a float array would survive ``int16`` storage unchanged (W907).

    CT stored as ``float32`` is the most common avoidable waste in medical
    imaging datasets: HU are integers, and the measured cost is 3.0x on disk for
    zero information (spec §4.2).
    """
    if array.dtype.kind != "f":
        return False
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return False
    if not np.array_equal(finite, np.rint(finite)):
        return False
    return bool(
        finite.min() >= np.iinfo(np.int16).min
        and finite.max() <= np.iinfo(np.int16).max
    )


def is_probability(array: npt.NDArray[Any]) -> bool:
    return bool(array.size and array.min() >= 0.0 and array.max() <= 1.0)


__all__ = [
    "SPEC_IMAGE_ATTRS",
    "VALUE_TYPES",
    "Image",
    "check_value_type",
    "is_probability",
    "lossless_as_int16",
]
