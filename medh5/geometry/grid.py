"""Grids: the named sampling lattices a sample's arrays live on (spec §3.1-§3.2).

A grid is an **empty HDF5 group carrying only attributes**.  That makes grids
essentially free, which is what lets the format say "two acquisitions with the
same lattice in different timepoints are two grids, not one shared grid" ---
keeping ``timepoint`` and ``frame_uid`` single-valued per grid and removing any
need for per-image geometry overrides.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5._hdf5 import (
    as_float_tuple,
    as_int_tuple,
    as_matrix,
    as_str,
    as_str_tuple,
    require_attr,
    set_attrs,
    validate_id,
)
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import (
    build_affine,
    check_orthonormal,
    index_to_world,
    world_to_index,
)

AXIS_KINDS = ("spatial", "channel", "time", "other")

KNOWN_COORD_SYSTEMS = ("LPS", "RAS", "RAI", "LAS", "custom")
KNOWN_UNITS = ("mm", "um", "m", "px")

MIN_SPATIAL, MAX_SPATIAL = 2, 3


@dataclass(frozen=True, eq=False)
class Grid:
    """One discrete sampling lattice with a complete index-world geometry."""

    grid_id: str
    shape: tuple[int, ...]
    axis_names: tuple[str, ...]
    axis_kinds: tuple[str, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: npt.NDArray[np.float64]
    coord_system: str = "LPS"
    units: str = "mm"
    timepoint: str | None = None
    frame_uid: str | None = None
    time_values: tuple[float, ...] | None = None
    time_units: str | None = None
    chunk_hint: tuple[int, ...] | None = None
    patch_hint: tuple[int, ...] | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "axis_names", tuple(str(a) for a in self.axis_names))
        object.__setattr__(self, "axis_kinds", tuple(str(a) for a in self.axis_kinds))
        object.__setattr__(self, "spacing", tuple(float(v) for v in self.spacing))
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        direction = np.ascontiguousarray(np.asarray(self.direction, dtype=np.float64))
        object.__setattr__(self, "direction", direction)
        direction.flags.writeable = False
        if self.time_values is not None:
            object.__setattr__(
                self, "time_values", tuple(float(v) for v in self.time_values)
            )
        if self.chunk_hint is not None:
            object.__setattr__(
                self, "chunk_hint", tuple(int(v) for v in self.chunk_hint)
            )
        if self.patch_hint is not None:
            object.__setattr__(
                self, "patch_hint", tuple(int(v) for v in self.patch_hint)
            )
        self.check()

    def check(self) -> None:
        """Validate every normative rule of spec §3.1-§3.2."""
        validate_id(self.grid_id, what="grid id")
        ndim = len(self.shape)
        if len(self.axis_names) != ndim or len(self.axis_kinds) != ndim:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: axis_names/axis_kinds must have {ndim} "
                f"entries",
                code="E109",
            )
        if any(s <= 0 for s in self.shape):
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: shape {self.shape} must be positive",
                code="E109",
            )
        unknown = set(self.axis_kinds) - set(AXIS_KINDS)
        if unknown:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: unknown axis kinds {sorted(unknown)}",
                code="E110",
            )
        n_spatial = self.axis_kinds.count("spatial")
        if not MIN_SPATIAL <= n_spatial <= MAX_SPATIAL:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: {n_spatial} spatial axes; "
                "the spec allows 2 or 3",
                code="E110",
            )
        if self.axis_kinds.count("time") > 1 or self.axis_kinds.count("channel") > 1:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: at most one time and one channel axis",
                code="E110",
            )
        spatial_idx = self.spatial_axes
        expected = tuple(range(ndim - n_spatial, ndim))
        if spatial_idx != expected:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: spatial axes {spatial_idx} must be contiguous "
                f"and trailing (expected {expected})",
                code="E103",
            )
        if len(self.spacing) != n_spatial or len(self.origin) != n_spatial:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: spacing/origin must have {n_spatial} entries",
                code="E109",
            )
        if any(v <= 0 for v in self.spacing):
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: spacing {self.spacing} must be > 0",
                code="E104",
            )
        if self.direction.shape != (n_spatial, n_spatial):
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: direction must be "
                f"{n_spatial}x{n_spatial}, got {self.direction.shape}",
                code="E109",
            )
        check_orthonormal(self.direction, what=f"grid {self.grid_id!r} direction")
        time_axis = self.time_axis
        if time_axis is not None and self.time_values is not None:
            extent = self.shape[time_axis]
            if len(self.time_values) != extent:
                raise MEDH5ValidationError(
                    f"grid {self.grid_id!r}: time_values has {len(self.time_values)} "
                    f"entries for a time axis of extent {extent}",
                    code="E109",
                )
        if self.patch_hint is not None and len(self.patch_hint) != n_spatial:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: patch_hint must have {n_spatial} entries",
                code="E109",
            )
        if self.chunk_hint is not None and len(self.chunk_hint) != ndim:
            raise MEDH5ValidationError(
                f"grid {self.grid_id!r}: chunk_hint must have {ndim} entries",
                code="E109",
            )

    # -- axis views --------------------------------------------------------

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spatial_axes(self) -> tuple[int, ...]:
        """Stored indices of the spatial axes, ascending."""
        return tuple(i for i, k in enumerate(self.axis_kinds) if k == "spatial")

    @property
    def n_spatial(self) -> int:
        return len(self.spatial_axes)

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.shape[i] for i in self.spatial_axes)

    @property
    def spatial_names(self) -> tuple[str, ...]:
        return tuple(self.axis_names[i] for i in self.spatial_axes)

    @property
    def channel_axis(self) -> int | None:
        return next((i for i, k in enumerate(self.axis_kinds) if k == "channel"), None)

    @property
    def time_axis(self) -> int | None:
        return next((i for i, k in enumerate(self.axis_kinds) if k == "time"), None)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.spatial_shape, dtype=np.int64))

    # -- geometry ----------------------------------------------------------

    @property
    def affine(self) -> npt.NDArray[np.float64]:
        """The ``(S+1, S+1)`` index-to-world affine (spec §3.3)."""
        return build_affine(self.spacing, self.origin, self.direction)

    def index_to_world(self, indices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return index_to_world(self.affine, indices)

    def world_to_index(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return world_to_index(self.affine, points)

    @property
    def extent(self) -> npt.NDArray[np.float64]:
        """Continuous index bounds ``(S, 2)``: every axis spans ``[-0.5, n - 0.5]``."""
        return np.stack(
            [
                np.full(self.n_spatial, -0.5),
                np.asarray(self.spatial_shape, dtype=np.float64) - 0.5,
            ],
            axis=1,
        )

    @property
    def physical_size(self) -> tuple[float, ...]:
        """Field of view per spatial axis, in :attr:`units`."""
        return tuple(
            float(n * s) for n, s in zip(self.spatial_shape, self.spacing, strict=True)
        )

    def comparable_with(self, other: Grid) -> bool:
        """Whether two grids are physically comparable without a transform (§3.3.4)."""
        return (
            self.frame_uid is not None
            and self.frame_uid == other.frame_uid
            and self.coord_system == other.coord_system
        )

    def is_congruent(self, other: Grid, tol: float = 1e-6) -> bool:
        """Whether two grids describe the same lattice, within *tol*."""
        return (
            self.shape == other.shape
            and self.axis_kinds == other.axis_kinds
            and np.allclose(self.spacing, other.spacing, atol=tol, rtol=0.0)
            and np.allclose(self.origin, other.origin, atol=tol, rtol=0.0)
            and np.allclose(self.direction, other.direction, atol=tol, rtol=0.0)
        )

    # -- identity ----------------------------------------------------------

    def _key(self) -> tuple[Any, ...]:
        return (
            self.grid_id,
            self.shape,
            self.axis_names,
            self.axis_kinds,
            self.spacing,
            self.origin,
            self.direction.tobytes(),
            self.coord_system,
            self.units,
            self.timepoint,
            self.frame_uid,
            self.time_values,
            self.time_units,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        tp = f", timepoint={self.timepoint!r}" if self.timepoint else ""
        return (
            f"Grid({self.grid_id!r}, shape={self.shape}, "
            f"spacing={tuple(round(v, 4) for v in self.spacing)}, "
            f"{self.coord_system}/{self.units}{tp})"
        )

    # -- serialization -----------------------------------------------------

    def attrs(self) -> dict[str, Any]:
        """The grid's HDF5 attribute dictionary (spec §3.2)."""
        out: dict[str, Any] = {
            "shape": np.asarray(self.shape, dtype=np.int64),
            "axis_names": list(self.axis_names),
            "axis_kinds": list(self.axis_kinds),
            "spacing": np.asarray(self.spacing, dtype=np.float64),
            "origin": np.asarray(self.origin, dtype=np.float64),
            "direction": np.asarray(self.direction, dtype=np.float64),
            "coord_system": self.coord_system,
            "units": self.units,
        }
        if self.timepoint is not None:
            out["timepoint"] = self.timepoint
        if self.frame_uid is not None:
            out["frame_uid"] = self.frame_uid
        if self.time_values is not None:
            out["time_values"] = np.asarray(self.time_values, dtype=np.float64)
        if self.time_units is not None:
            out["time_units"] = self.time_units
        if self.chunk_hint is not None:
            out["chunk_hint"] = np.asarray(self.chunk_hint, dtype=np.int64)
        if self.patch_hint is not None:
            out["patch_hint"] = np.asarray(self.patch_hint, dtype=np.int64)
        out.update(self.extra)
        return out

    def summary(self) -> dict[str, Any]:
        """JSON-safe description, for ``medh5 info --json``."""
        return {
            "id": self.grid_id,
            "shape": list(self.shape),
            "axis_names": list(self.axis_names),
            "axis_kinds": list(self.axis_kinds),
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "direction": [list(map(float, row)) for row in self.direction],
            "coord_system": self.coord_system,
            "units": self.units,
            "timepoint": self.timepoint,
            "frame_uid": self.frame_uid,
            "physical_size": list(self.physical_size),
        }


SPEC_GRID_ATTRS = (
    "shape",
    "axis_names",
    "axis_kinds",
    "spacing",
    "origin",
    "direction",
    "coord_system",
    "units",
    "timepoint",
    "frame_uid",
    "time_values",
    "time_units",
    "chunk_hint",
    "patch_hint",
)


def write_grid(grids_group: h5py.Group, grid: Grid) -> h5py.Group:
    """Create ``grids/<grid_id>`` as an empty group carrying the geometry."""
    group = grids_group.create_group(grid.grid_id)
    set_attrs(group, grid.attrs())
    return group


def read_grid(group: h5py.Group, grid_id: str | None = None) -> Grid:
    """Read one grid group back into a :class:`Grid`."""
    gid = grid_id if grid_id is not None else group.name.rsplit("/", 1)[-1]
    attrs = group.attrs
    extra = {key: attrs[key] for key in attrs if key not in SPEC_GRID_ATTRS}
    return Grid(
        grid_id=gid,
        shape=as_int_tuple(require_attr(group, "shape")),
        axis_names=as_str_tuple(require_attr(group, "axis_names")),
        axis_kinds=as_str_tuple(require_attr(group, "axis_kinds")),
        spacing=as_float_tuple(require_attr(group, "spacing")),
        origin=as_float_tuple(require_attr(group, "origin")),
        direction=as_matrix(require_attr(group, "direction")),
        coord_system=as_str(require_attr(group, "coord_system")),
        units=as_str(require_attr(group, "units")),
        timepoint=as_str(attrs["timepoint"]) if "timepoint" in attrs else None,
        frame_uid=as_str(attrs["frame_uid"]) if "frame_uid" in attrs else None,
        time_values=(
            as_float_tuple(attrs["time_values"]) if "time_values" in attrs else None
        ),
        time_units=as_str(attrs["time_units"]) if "time_units" in attrs else None,
        chunk_hint=as_int_tuple(attrs["chunk_hint"]) if "chunk_hint" in attrs else None,
        patch_hint=as_int_tuple(attrs["patch_hint"]) if "patch_hint" in attrs else None,
        extra=extra,
    )


def read_grids(root: h5py.Group) -> dict[str, Grid]:
    """Read every grid under ``<sample root>/grids``."""
    if "grids" not in root:
        raise MEDH5ValidationError("sample has no `grids` group", code="E008")
    grids_group = root["grids"]
    return {name: read_grid(grids_group[name], name) for name in sorted(grids_group)}


def iter_spatial_slices(
    grid: Grid, roi: Sequence[slice] | None
) -> Iterator[slice] | tuple[slice, ...]:
    """Expand a spatial ROI to a full-rank index tuple for *grid*."""
    if roi is None:
        return tuple(slice(None) for _ in range(grid.ndim))
    if len(roi) != grid.n_spatial:
        raise MEDH5ValidationError(
            f"roi has {len(roi)} axes; grid {grid.grid_id!r} has {grid.n_spatial} "
            f"spatial axes"
        )
    lead = grid.ndim - grid.n_spatial
    return (slice(None),) * lead + tuple(roi)


__all__ = [
    "AXIS_KINDS",
    "KNOWN_COORD_SYSTEMS",
    "KNOWN_UNITS",
    "SPEC_GRID_ATTRS",
    "Grid",
    "iter_spatial_slices",
    "read_grid",
    "read_grids",
    "write_grid",
]
