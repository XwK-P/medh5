"""Transforms map **points**, not images (spec §10).

One convention is fixed here and admits no attribute to switch it:

**A transform with ``from_frame = F`` and ``to_frame = M`` maps a point expressed
in F to the corresponding point in M: ``x_M = T(x_F)``.**

To warp a moving image defined in M onto a fixed grid in F --- the usual
operation --- evaluate T at each F-grid point and sample M at ``T(x)``.  That is
the ITK/SimpleITK ``TransformPoint`` convention, and the *inverse* of the
"forward warp" convention some optical-flow literature uses.  There is no flag to
select the other one: ambiguity here is the leading cause of silently mirrored
registration results, and a format that lets a file declare its own direction
just moves the ambiguity into the reader.

The format does not distinguish intra-timepoint (PET to CT) from inter-timepoint
(baseline to follow-up) registration --- both are a mapping between frames.  What
differs is necessity: grids in one session often share a frame and need no
transform at all, while grids in different timepoints never share one (§3.4), so
a longitudinal transform is always required to compare them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_bool, as_str, as_str_tuple, require_attr, validate_id
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.transforms.apply import EXTRAPOLATIONS

TRANSFORM_KINDS = ("identity", "affine", "displacement", "bspline", "composite")

VECTOR_SPACES = ("world", "index")
INTERPOLATIONS = ("linear", "cubic")

SPEC_TRANSFORM_ATTRS = (
    "kind",
    "from_frame",
    "to_frame",
    "from_grid",
    "to_grid",
    "units",
    "invertible",
    "inverse_id",
    "prov",
    "metrics",
    "digest",
    "field_grid",
    "vector_space",
    "interpolation",
    "extrapolation",
    "cp_grid",
    "order",
    "components",
)


@dataclass(slots=True)
class TransformHeader:
    """The attribute header every transform carries (spec §10.1)."""

    kind: str
    from_frame: str
    to_frame: str
    units: str = "mm"
    from_grid: str | None = None
    to_grid: str | None = None
    invertible: bool | None = None
    inverse_id: str | None = None
    prov: str | None = None
    metrics: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TRANSFORM_KINDS:
            raise MEDH5ValidationError(
                f"unknown transform kind {self.kind!r}; expected one of "
                f"{list(TRANSFORM_KINDS)}",
                code="E502",
            )
        if not self.from_frame or not self.to_frame:
            raise MEDH5ValidationError(
                "a transform requires both `from_frame` and `to_frame`", code="E502"
            )

    def attrs(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "units": self.units,
        }
        for key in ("from_grid", "to_grid", "inverse_id", "prov", "metrics"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.invertible is not None:
            out["invertible"] = bool(self.invertible)
        out.update(self.extra)
        return out

    @classmethod
    def read(cls, group: h5py.Group) -> TransformHeader:
        attrs = group.attrs
        return cls(
            kind=as_str(require_attr(group, "kind", code="E502")),
            from_frame=as_str(require_attr(group, "from_frame", code="E502")),
            to_frame=as_str(require_attr(group, "to_frame", code="E502")),
            units=as_str(attrs["units"]) if "units" in attrs else "mm",
            from_grid=as_str(attrs["from_grid"]) if "from_grid" in attrs else None,
            to_grid=as_str(attrs["to_grid"]) if "to_grid" in attrs else None,
            invertible=as_bool(attrs["invertible"]) if "invertible" in attrs else None,
            inverse_id=as_str(attrs["inverse_id"]) if "inverse_id" in attrs else None,
            prov=as_str(attrs["prov"]) if "prov" in attrs else None,
            metrics=as_str(attrs["metrics"]) if "metrics" in attrs else None,
        )


class Transform(ABC):
    """A mapping between two frames of reference."""

    __slots__ = ("_grids", "_siblings", "group", "header", "transform_id")

    def __init__(
        self,
        transform_id: str,
        group: h5py.Group,
        header: TransformHeader,
        grids: Mapping[str, Grid] | None = None,
        siblings: Mapping[str, h5py.Group] | None = None,
    ) -> None:
        self.transform_id = transform_id
        self.group = group
        self.header = header
        self._grids = grids or {}
        self._siblings = siblings or {}

    # -- header passthrough ------------------------------------------------

    @property
    def kind(self) -> str:
        return self.header.kind

    @property
    def from_frame(self) -> str:
        return self.header.from_frame

    @property
    def to_frame(self) -> str:
        return self.header.to_frame

    @property
    def units(self) -> str:
        return self.header.units

    @property
    def prov(self) -> str | None:
        return self.header.prov

    @property
    def metrics_key(self) -> str | None:
        return self.header.metrics

    @property
    def is_invertible(self) -> bool:
        """Whether an inverse is available --- declared, analytic, or neither."""
        if self.header.invertible is not None:
            return bool(self.header.invertible)
        return self.header.inverse_id is not None

    @property
    def timepoints(self) -> tuple[str, ...]:
        """The timepoints this transform relates, from the grids in its frames.

        A transform needs no ``timepoint`` attribute of its own: its endpoints
        are frames, and frames belong to grids, and grids declare the timepoint.
        """
        out: list[str] = []
        for frame in (self.from_frame, self.to_frame):
            for grid in self._grids.values():
                if grid.frame_uid == frame and grid.timepoint:
                    if grid.timepoint not in out:
                        out.append(grid.timepoint)
                    break
        return tuple(out)

    def grid_in(self, frame: str) -> Grid | None:
        """A representative grid living in *frame*, if the sample has one."""
        named = (
            self.header.from_grid if frame == self.from_frame else self.header.to_grid
        )
        if named is not None and named in self._grids:
            return self._grids[named]
        for grid in self._grids.values():
            if grid.frame_uid == frame:
                return grid
        return None

    # -- the contract ------------------------------------------------------

    @abstractmethod
    def transform_points(self, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map world points from ``from_frame`` to ``to_frame`` (``x_M = T(x_F)``)."""

    def inverse(self) -> Transform | None:
        """The stored inverse transform, when the file carries one."""
        target = self.header.inverse_id
        if target is None or target not in self._siblings:
            return None
        return open_transform(
            target, self._siblings[target], self._grids, self._siblings
        )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.transform_id,
            "kind": self.kind,
            "from_frame": self.from_frame,
            "to_frame": self.to_frame,
            "units": self.units,
            "timepoints": list(self.timepoints),
            "invertible": self.is_invertible,
            "inverse_id": self.header.inverse_id,
            "metrics": self.metrics_key,
            "prov": self.prov,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.transform_id!r}, "
            f"{self.from_frame!r} -> {self.to_frame!r})"
        )


def transform_readers() -> dict[str, Any]:
    """``kind`` -> reader class, assembled lazily to keep the imports acyclic."""
    from medh5.transforms.affine import AffineTransform, IdentityTransform
    from medh5.transforms.bspline import BSplineTransform
    from medh5.transforms.composite import CompositeTransform
    from medh5.transforms.displacement import DisplacementTransform

    return {
        "identity": IdentityTransform,
        "affine": AffineTransform,
        "displacement": DisplacementTransform,
        "bspline": BSplineTransform,
        "composite": CompositeTransform,
    }


def open_transform(
    transform_id: str,
    group: h5py.Group,
    grids: Mapping[str, Grid] | None = None,
    siblings: Mapping[str, h5py.Group] | None = None,
) -> Transform:
    """Open a transform group as the class matching its ``kind``."""
    header = TransformHeader.read(group)
    reader = transform_readers().get(header.kind)
    if reader is None:  # pragma: no cover - TransformHeader already rejects these
        raise MEDH5ValidationError(
            f"transform {transform_id!r}: unknown kind {header.kind!r}", code="E502"
        )
    opened: Transform = reader(transform_id, group, header, grids, siblings)
    return opened


def read_transforms(
    root: h5py.Group, grids: Mapping[str, Grid] | None = None
) -> dict[str, Transform]:
    """Every transform under ``<sample root>/transforms``."""
    node = root.get("transforms")
    if node is None:
        return {}
    siblings = {name: node[name] for name in sorted(node)}
    return {
        name: open_transform(name, group, grids, siblings)
        for name, group in siblings.items()
    }


def check_transform_id(transform_id: str) -> str:
    return validate_id(transform_id, what="transform id")


def frame_graph(transforms: Mapping[str, Transform]) -> dict[str, list[str]]:
    """Frame -> frames one hop away, as :func:`resolve_between` would walk them.

    A reverse edge is present only where the inverse can be *evaluated* ---
    the resolver's ``can_invert`` --- not merely declared.  This used to ask
    ``is_invertible``, so the graph a caller inspected and the paths the
    resolver would actually return disagreed for a displacement field written
    ``invertible=True`` with no stored inverse.
    """
    from medh5.transforms.resolve import InverseTransform

    graph: dict[str, list[str]] = {}
    for transform in transforms.values():
        graph.setdefault(transform.from_frame, []).append(transform.to_frame)
        graph.setdefault(transform.to_frame, [])
        if InverseTransform.can_invert(transform):
            graph[transform.to_frame].append(transform.from_frame)
    return graph


def component_ids(group: h5py.Group) -> tuple[str, ...]:
    """The ordered component ids of a composite transform."""
    if "components" not in group.attrs:
        return ()
    return as_str_tuple(group.attrs["components"])


__all__ = [
    "EXTRAPOLATIONS",
    "INTERPOLATIONS",
    "SPEC_TRANSFORM_ATTRS",
    "TRANSFORM_KINDS",
    "VECTOR_SPACES",
    "Transform",
    "TransformHeader",
    "check_transform_id",
    "component_ids",
    "frame_graph",
    "open_transform",
    "read_transforms",
    "transform_readers",
]
