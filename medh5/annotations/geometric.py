"""Geometric annotations: boxes, oriented boxes, keypoints, points, contours
and meshes (spec §8).

**Boxes sit at voxel edges, indices at voxel centres**: ``[a, b]`` is the
slice ``a+0.5 : b+0.5``.  The conversions here --- ``as_slices``,
``to_world``, ``to_index`` --- are the format engine's, and refuse a grid in
another frame of reference rather than guessing a transform.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import Annotation, Instance, _instance
from medh5.geometry import Grid

SPACES: tuple[str, ...] = _core.SPACES
CONTOUR_ROLES: tuple[str, ...] = _core.CONTOUR_ROLES
VISIBILITY: dict[int, str] = dict(_core.VISIBILITY)
ROTATION_TOL: float = _core.ROTATION_TOL

check_space = _core.check_space
check_slice_index = _core.check_slice_index
encode_boxes = _core.encode_boxes
encode_obb = _core.encode_obb
encode_keypoints = _core.encode_keypoints
encode_points = _core.encode_points
encode_contours = _core.encode_contours
encode_mesh = _core.encode_mesh


class GeometricAnnotation(Annotation):
    """Base for the §8 kinds: N objects whose coordinates live in one space."""

    __slots__ = ()

    @property
    def space(self) -> str:
        return str(self._handle.space)

    @property
    def frame_uid(self) -> str | None:
        found: str | None = self._handle.frame_uid
        return found

    @property
    def n_spatial(self) -> int:
        return int(self._handle.n_spatial)

    def to_world(
        self, coords: npt.ArrayLike, *, grid: Grid | str | None = None
    ) -> npt.NDArray[np.float64]:
        """Map coordinates from this annotation's ``space`` to world.

        Index coordinates count the voxels of the annotation's **own** grid;
        *grid* says whose world the caller wants, and is refused (E414) when
        its frame differs.
        """
        found: npt.NDArray[np.float64] = self._handle.to_world(coords, grid=grid)
        return found

    def to_index(
        self, coords: npt.ArrayLike, *, grid: Grid | str | None = None
    ) -> npt.NDArray[np.float64]:
        """Map coordinates from this annotation's ``space`` to *grid*'s
        continuous index (the annotation's own grid when none is named)."""
        found: npt.NDArray[np.float64] = self._handle.to_index(coords, grid=grid)
        return found

    def _read(
        self, name: str, dtype: npt.DTypeLike, required: bool = True
    ) -> npt.NDArray[Any] | None:
        if not required and not self._handle.has_dataset(name):
            return None
        return np.asarray(self._handle.dataset(name).read(), dtype=dtype)

    def _need(self, name: str, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        found = self._read(name, dtype)
        assert found is not None
        return found

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.object_class_ids
        return found

    @property
    def instance_ids(self) -> npt.NDArray[np.uint64] | None:
        found: npt.NDArray[np.uint64] | None = self._handle.instance_ids
        return found

    @property
    def scores(self) -> npt.NDArray[np.float32] | None:
        found: npt.NDArray[np.float32] | None = self._handle.scores
        return found

    @property
    def attributes(self) -> tuple[dict[str, Any], ...] | None:
        """Per-object free-form JSON, decoded."""
        found = self._handle.attributes
        return None if found is None else tuple(found)

    def __len__(self) -> int:
        return int(self._handle.n_items)


class BoxesAnnotation(GeometricAnnotation):
    """Axis-aligned boxes, ``(N, S, 2)`` ``[lo, hi]`` at voxel edges (§8.2)."""

    __slots__ = ()

    @property
    def ndim(self) -> int:
        return int(self._handle.box_ndim)

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        return self._need("boxes", np.float32)

    @property
    def slice_index(self) -> npt.NDArray[np.int32] | None:
        found: npt.NDArray[np.int32] | None = self._handle.slice_index
        return found

    def as_slices(self, grid: Grid | str | None = None) -> list[tuple[slice, ...]]:
        """Each box as voxel slices of *grid* (the annotation's own by default)."""
        return list(self._handle.as_slices(grid))

    def world_corners(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        """``(N, 2**S, S)`` world corners of every box."""
        found: npt.NDArray[np.float64] = self._handle.world_corners(grid)
        return found

    def as_world(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        """``(N, S, 2)`` world-axis-aligned bounds of every box."""
        found: npt.NDArray[np.float64] = self._handle.as_world(grid)
        return found

    def __iter__(self) -> Iterator[Instance]:
        for row in self._handle.instances():
            yield _instance(row)


class ObbAnnotation(GeometricAnnotation):
    """Oriented boxes: centre, full edge lengths, rotation (§8.3)."""

    __slots__ = ()

    @property
    def centers(self) -> npt.NDArray[np.float32]:
        return self._need("centers", np.float32)

    @property
    def sizes(self) -> npt.NDArray[np.float32]:
        return self._need("sizes", np.float32)

    @property
    def rotations(self) -> npt.NDArray[np.float32]:
        return self._need("rotations", np.float32)

    def corners(self) -> npt.NDArray[np.float64]:
        """``(N, 2**S, S)`` corners: ``center + R @ (size/2 * s)``."""
        found: npt.NDArray[np.float64] = self._handle.obb_corners()
        return found

    def as_aabb(self) -> npt.NDArray[np.float64]:
        """``(N, S, 2)`` axis-aligned bounds of every oriented box."""
        found: npt.NDArray[np.float64] = self._handle.obb_as_aabb()
        return found

    @property
    def volumes(self) -> npt.NDArray[np.float64]:
        found: npt.NDArray[np.float64] = self._handle.obb_volumes()
        return found


class KeypointsAnnotation(GeometricAnnotation):
    """``(N, K, S)`` keypoints with per-slot classes and visibility (§8.4)."""

    __slots__ = ()

    @property
    def points(self) -> npt.NDArray[np.float32]:
        return self._need("points", np.float32)

    @property
    def visibility(self) -> npt.NDArray[np.uint8]:
        found: npt.NDArray[np.uint8] = self._handle.visibility
        return found

    @property
    def keypoint_class_ids(self) -> npt.NDArray[np.uint16]:
        return self._need("keypoint_class_ids", np.uint16)

    @property
    def skeleton_id(self) -> str | None:
        found: str | None = self._handle.skeleton_id
        return found

    def skeleton(self) -> Any:
        """The label set's skeleton this annotation names, if any."""
        return self._handle.skeleton()

    def labelled(self) -> npt.NDArray[np.bool_]:
        found: npt.NDArray[np.bool_] = self._handle.labelled()
        return found


class PointsAnnotation(GeometricAnnotation):
    """A point set: landmarks, seeds, or half a correspondence (§8.5)."""

    __slots__ = ()

    @property
    def points(self) -> npt.NDArray[np.float32]:
        return self._need("points", np.float32)

    @property
    def names(self) -> tuple[str, ...] | None:
        found: tuple[str, ...] | None = self._handle.point_names
        return found

    @property
    def weights(self) -> npt.NDArray[np.float32] | None:
        return self._read("weights", np.float32, required=False)

    @property
    def correspondence(self) -> str | None:
        found: str | None = self._handle.correspondence
        return found

    def named(self) -> dict[str, npt.NDArray[np.float32]]:
        """``name -> point`` for a named point set (``{}`` when unnamed)."""
        return {
            name: np.asarray(point, dtype=np.float32)
            for name, point in self._handle.named_points().items()
        }

    def world_points(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        return self.to_world(self.points, grid=grid)


@dataclass(slots=True)
class Polygon:
    """One planar polygon handed to :func:`encode_contours`."""

    vertices: npt.NDArray[Any]
    class_id: int
    plane: tuple[int, int] = (-1, 0)
    """``(axis, index)`` of the plane it lies in; ``axis = -1`` for out-of-plane."""

    role: str = "outer"

    def __post_init__(self) -> None:
        _core.check_contour_role(self.role)


class ContoursAnnotation(GeometricAnnotation):
    """Planar polygons with an offset table (§8.6) --- RTSTRUCT-shaped."""

    __slots__ = ()

    @property
    def vertices(self) -> npt.NDArray[np.float32]:
        return self._need("vertices", np.float32)

    @property
    def offsets(self) -> npt.NDArray[np.int64]:
        return np.asarray(self._handle.contour_offsets, dtype=np.int64)

    @property
    def planes(self) -> npt.NDArray[np.int32]:
        found: npt.NDArray[np.int32] = self._handle.contour_planes
        return found

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(self._handle.contour_roles)

    def polygon(self, index: int) -> npt.NDArray[np.float32]:
        found: npt.NDArray[np.float32] = self._handle.polygon(int(index))
        return found

    def polygons(self) -> Iterator[Polygon]:
        for vertices, class_id, plane, role in self._handle.polygons():
            yield Polygon(
                vertices=vertices,
                class_id=int(class_id),
                plane=(int(plane[0]), int(plane[1])),
                role=role,
            )

    def by_plane(self) -> dict[tuple[int, int], list[int]]:
        found: dict[tuple[int, int], list[int]] = self._handle.by_plane()
        return found


class MeshAnnotation(GeometricAnnotation):
    """A triangle surface mesh, optionally several submeshes (§8.7)."""

    __slots__ = ()

    @property
    def vertices(self) -> npt.NDArray[np.float32]:
        return self._need("vertices", np.float32)

    @property
    def faces(self) -> npt.NDArray[np.int32]:
        return self._need("faces", np.int32)

    @property
    def normals(self) -> npt.NDArray[np.float32] | None:
        return self._read("normals", np.float32, required=False)

    @property
    def vertex_class_ids(self) -> npt.NDArray[np.uint16] | None:
        return self._read("vertex_class_ids", np.uint16, required=False)

    @property
    def n_submeshes(self) -> int:
        return int(self._handle.n_submeshes)

    def bounds(self) -> npt.NDArray[np.float64]:
        """``(S, 2)`` bounds of every vertex."""
        found: npt.NDArray[np.float64] = self._handle.mesh_bounds()
        return found


GEOMETRIC_READERS: dict[str, Any] = {
    "boxes": BoxesAnnotation,
    "obb": ObbAnnotation,
    "keypoints": KeypointsAnnotation,
    "points": PointsAnnotation,
    "contours": ContoursAnnotation,
    "mesh": MeshAnnotation,
}

__all__ = [
    "CONTOUR_ROLES",
    "GEOMETRIC_READERS",
    "ROTATION_TOL",
    "SPACES",
    "VISIBILITY",
    "BoxesAnnotation",
    "ContoursAnnotation",
    "GeometricAnnotation",
    "KeypointsAnnotation",
    "MeshAnnotation",
    "ObbAnnotation",
    "PointsAnnotation",
    "Polygon",
    "check_slice_index",
    "check_space",
    "encode_boxes",
    "encode_contours",
    "encode_keypoints",
    "encode_mesh",
    "encode_obb",
    "encode_points",
]
