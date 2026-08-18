"""Geometric annotations: boxes, oriented boxes, keypoints, points, contours,
meshes (spec §8).

Every coordinate in this module lives in a declared ``space`` --- continuous
index coordinates of a named grid, or physical coordinates in a named frame of
reference --- and readers convert through the affine rather than assuming a
convention (§8.1).  Two consequences are easy to get wrong and are handled here
once:

**Boxes are measured at voxel edges.**  ``numpy slice a:b`` is ``[a-0.5, b-0.5]``,
so a box converts to a slice without rounding.  Storing integer boxes, as 0.x
did, cannot represent a resampled or rotated box at all.

**An axis-aligned index box is not an axis-aligned world box.**  Under an oblique
``direction`` it is an oriented box, so :meth:`BoxesAnnotation.as_world` returns
the *enclosing* bounds and says so; :meth:`BoxesAnnotation.world_corners` is the
exact conversion.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_str, str_dtype
from medh5.annotations.base import Annotation, Instance
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import (
    box_corners,
    box_to_slices,
    is_proper_rotation,
)
from medh5.geometry.grid import Grid

SPACES = ("index", "world")

CONTOUR_ROLES = ("outer", "hole")

VISIBILITY = {0: "unlabelled", 1: "occluded", 2: "visible"}

ROTATION_TOL = 1e-4


def check_space(space: str) -> str:
    if space not in SPACES:
        raise MEDH5ValidationError(
            f"space {space!r} must be one of {list(SPACES)}", code="E412"
        )
    return space


# --------------------------------------------------------------------------
# Shared reader behaviour
# --------------------------------------------------------------------------


class GeometricAnnotation(Annotation):
    """Base for the §8 kinds: N objects whose coordinates live in one space."""

    __slots__ = ()

    # -- coordinate space --------------------------------------------------

    @property
    def space(self) -> str:
        return check_space(self.header.space or "index")

    @property
    def frame_uid(self) -> str | None:
        if self.header.frame_uid is not None:
            return self.header.frame_uid
        if self.header.grid is not None and self.header.grid in self._grids:
            return self._grids[self.header.grid].frame_uid
        return None

    @property
    def n_spatial(self) -> int:
        return self.grid.n_spatial

    def _resolve_grid(self, grid: Grid | str | None) -> Grid:
        if isinstance(grid, Grid):
            return grid
        if isinstance(grid, str):
            try:
                return self._grids[grid]
            except KeyError:
                raise MEDH5ValidationError(
                    f"annotation {self.ann_id!r}: grid {grid!r} does not exist",
                    code="E101",
                ) from None
        return self.grid

    def to_world(
        self, coords: npt.ArrayLike, *, grid: Grid | str | None = None
    ) -> npt.NDArray[np.float64]:
        """Map coordinates from this annotation's ``space`` to world."""
        values = np.asarray(coords, dtype=np.float64)
        if self.space == "world":
            return values
        return self._resolve_grid(grid).index_to_world(values)

    def to_index(
        self, coords: npt.ArrayLike, *, grid: Grid | str | None = None
    ) -> npt.NDArray[np.float64]:
        """Map coordinates from this annotation's ``space`` to continuous index."""
        values = np.asarray(coords, dtype=np.float64)
        if self.space == "index":
            return values
        target = self._resolve_grid(grid)
        if (
            target.frame_uid is not None
            and self.frame_uid is not None
            and target.frame_uid != self.frame_uid
        ):
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r} is in frame {self.frame_uid!r} and grid "
                f"{target.grid_id!r} is in {target.frame_uid!r}; a transform is "
                "required to relate them",
                code="E414",
            )
        return target.world_to_index(values)

    # -- per-object columns ------------------------------------------------

    def _dataset(self, name: str, required: bool = True) -> Any:
        if name in self.group:
            return self.group[name]
        if required:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: kind {self.kind!r} requires a "
                f"{name!r} dataset",
                code="E410",
            )
        return None

    def _optional(self, name: str, dtype: npt.DTypeLike) -> npt.NDArray[Any] | None:
        node = self._dataset(name, required=False)
        return None if node is None else np.asarray(node[...], dtype=dtype)

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._dataset("class_ids")[...], dtype=np.uint16)

    @property
    def instance_ids(self) -> npt.NDArray[np.uint64] | None:
        return self._optional("instance_ids", np.uint64)

    @property
    def scores(self) -> npt.NDArray[np.float32] | None:
        return self._optional("scores", np.float32)

    @property
    def attributes(self) -> tuple[dict[str, Any], ...] | None:
        """Per-object free-form JSON, decoded."""
        node = self._dataset("attributes", required=False)
        if node is None:
            return None
        return tuple(json.loads(as_str(v) or "{}") for v in node[...])

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.ann_id,
            "kind": self.kind,
            "task": self.task,
            "grid": self.grid_id,
            "space": self.space,
            "frame_uid": self.frame_uid,
            "timepoints": list(self.timepoints),
            "objects": len(self),
            "classes": len(self.class_ids),
            "annotated_classes": len(self.annotated_class_ids),
            "fully_covered": self.is_fully_covered,
            "quality": self.quality_key,
            "prov": self.prov,
        }

    def __len__(self) -> int:
        return int(self.object_class_ids.shape[0])


# --------------------------------------------------------------------------
# §8.2 boxes
# --------------------------------------------------------------------------


def _object_columns(
    n: int,
    class_ids: Sequence[int],
    instance_ids: Sequence[int] | None,
    scores: Sequence[float] | None,
    attributes: Sequence[Mapping[str, Any]] | None,
) -> dict[str, npt.NDArray[Any]]:
    if len(class_ids) != n:
        raise MEDH5ValidationError(
            f"class_ids has {len(class_ids)} entries for {n} objects", code="E405"
        )
    out: dict[str, npt.NDArray[Any]] = {
        "class_ids": np.asarray(class_ids, dtype=np.uint16)
    }
    if instance_ids is not None:
        out["instance_ids"] = np.asarray(instance_ids, dtype=np.uint32)
    if scores is not None:
        out["scores"] = np.asarray(scores, dtype=np.float32)
    if attributes is not None:
        out["attributes"] = np.array(
            [json.dumps(dict(a), sort_keys=True) for a in attributes],
            dtype=str_dtype(),
        )
    for name, column in out.items():
        if column.shape[0] != n:
            raise MEDH5ValidationError(
                f"{name} has {column.shape[0]} entries for {n} objects", code="E405"
            )
    return out


def encode_boxes(
    boxes: npt.ArrayLike,
    class_ids: Sequence[int],
    *,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    attributes: Sequence[Mapping[str, Any]] | None = None,
    slice_index: Sequence[int] | None = None,
) -> AnnotationPayload:
    """Pack axis-aligned boxes, ``(N, S, 2)`` in ``[lo, hi]`` form (spec §8.2)."""
    array = np.asarray(boxes, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 2:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"boxes must have shape (N, S, 2), got {array.shape}", code="E405"
        )
    if array.size and np.any(array[..., 0] > array[..., 1]):
        bad = int(np.sum(np.any(array[..., 0] > array[..., 1], axis=1)))
        raise MEDH5ValidationError(
            f"{bad} box(es) have lo > hi; boxes are stored [lo, hi] at voxel edges",
            code="E406",
        )
    datasets = {"boxes": array}
    datasets.update(
        _object_columns(array.shape[0], class_ids, instance_ids, scores, attributes)
    )
    if slice_index is not None:
        datasets["slice_index"] = np.asarray(slice_index, dtype=np.int32)
    return AnnotationPayload(
        kind="boxes",
        datasets=datasets,
        class_ids=tuple(sorted({int(c) for c in class_ids})),
    )


def _stacked(
    arrays: list[npt.NDArray[np.float64]], empty: tuple[int, ...]
) -> npt.NDArray[np.float64]:
    """``np.stack`` that survives an annotation holding no objects.

    An empty detection annotation is not a degenerate input to guard against ---
    it is the *verified negative* the coverage contract exists to record: no
    objects, and ``annotated_class_ids`` naming the classes that were searched
    for and not found (§6.4, §9).  ``np.stack([])`` raises, so the one case the
    model is designed to express was the one that crashed on read.
    """
    if not arrays:
        return np.empty(empty, dtype=np.float64)
    return np.stack(arrays)


class BoxesAnnotation(GeometricAnnotation):
    """Reader for ``kind = "boxes"``."""

    __slots__ = ()

    @property
    def ndim(self) -> int:
        """Spatial dimensionality, readable from the stored ``(N, S, 2)`` shape."""
        return int(self._dataset("boxes").shape[1])

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("boxes")[...], dtype=np.float32)

    @property
    def slice_index(self) -> npt.NDArray[np.int32] | None:
        return self._optional("slice_index", np.int32)

    def as_slices(self, grid: Grid | str | None = None) -> list[tuple[slice, ...]]:
        """Each box as a numpy index tuple, clipped to the grid."""
        target = self._resolve_grid(grid)
        if self.space == "index":
            boxes = self.boxes.astype(np.float64)
        else:
            boxes = self._boxes_in_index(target)
        return [
            box_to_slices(boxes[i], target.spatial_shape) for i in range(len(boxes))
        ]

    def _boxes_in_index(self, grid: Grid) -> npt.NDArray[np.float64]:
        return _stacked(
            [
                self._bounds(self.to_index(box_corners(b), grid=grid))
                for b in self.boxes
            ],
            (0, self.ndim, 2),
        )

    @staticmethod
    def _bounds(corners: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.stack([corners.min(axis=0), corners.max(axis=0)], axis=1)

    def world_corners(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        """Exact ``(N, 2**S, S)`` world corners of every box."""
        return _stacked(
            [self.to_world(box_corners(b), grid=grid) for b in self.boxes],
            (0, 2**self.ndim, self.ndim),
        )

    def as_world(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        """``(N, S, 2)`` **enclosing** world bounds.

        Under an oblique ``direction`` an axis-aligned index box is an oriented
        box in world space, so these bounds are larger than the box itself; use
        :meth:`world_corners` when that matters, or ``obb`` to store the box
        exactly.
        """
        if self.space == "world":
            return self.boxes.astype(np.float64)
        return _stacked(
            [self._bounds(c) for c in self.world_corners(grid)], (0, self.ndim, 2)
        )

    def __iter__(self) -> Iterator[Instance]:
        classes = self.object_class_ids
        ids = self.instance_ids
        scores = self.scores
        boxes = self.boxes
        for i in range(boxes.shape[0]):
            yield Instance(
                index=i,
                instance_id=int(ids[i]) if ids is not None else i,
                class_id=int(classes[i]),
                box=boxes[i],
                score=None if scores is None else float(scores[i]),
            )


# --------------------------------------------------------------------------
# §8.3 obb
# --------------------------------------------------------------------------


def encode_obb(
    centers: npt.ArrayLike,
    sizes: npt.ArrayLike,
    rotations: npt.ArrayLike,
    class_ids: Sequence[int],
    *,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    attributes: Sequence[Mapping[str, Any]] | None = None,
) -> AnnotationPayload:
    """Pack oriented boxes as centre, **full** edge lengths and a rotation matrix.

    Rotation *matrices*, not quaternions or Euler angles: they are
    dimension-generic, have no ordering convention to get wrong and no
    double-cover ambiguity, and S² floats per box is nothing beside image data.
    """
    c = np.asarray(centers, dtype=np.float32)
    s = np.asarray(sizes, dtype=np.float32)
    r = np.asarray(rotations, dtype=np.float32)
    n, dim = c.shape
    if s.shape != (n, dim) or r.shape != (n, dim, dim):
        raise MEDH5ValidationError(
            f"obb shapes disagree: centers {c.shape}, sizes {s.shape}, "
            f"rotations {r.shape}",
            code="E405",
        )
    if np.any(s < 0):
        raise MEDH5ValidationError("obb sizes must be non-negative", code="E406")
    for i in range(n):
        if not is_proper_rotation(r[i], ROTATION_TOL):
            raise MEDH5ValidationError(
                f"obb {i}: `rotations` must be orthonormal with det = +1 to "
                f"{ROTATION_TOL:g}",
                code="E407",
            )
    datasets = {"centers": c, "sizes": s, "rotations": r}
    datasets.update(_object_columns(n, class_ids, instance_ids, scores, attributes))
    return AnnotationPayload(
        kind="obb",
        datasets=datasets,
        class_ids=tuple(sorted({int(v) for v in class_ids})),
    )


class ObbAnnotation(GeometricAnnotation):
    """Reader for ``kind = "obb"``."""

    __slots__ = ()

    @property
    def centers(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("centers")[...], dtype=np.float32)

    @property
    def sizes(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("sizes")[...], dtype=np.float32)

    @property
    def rotations(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("rotations")[...], dtype=np.float32)

    def corners(self) -> npt.NDArray[np.float64]:
        """``(N, 2**S, S)`` corners: ``center + R @ (size/2 * s)`` for every sign."""
        centers = self.centers.astype(np.float64)
        sizes = self.sizes.astype(np.float64)
        rotations = self.rotations.astype(np.float64)
        dim = centers.shape[1]
        signs = np.indices((2,) * dim).reshape(dim, -1).T * 2.0 - 1.0
        out = np.empty((centers.shape[0], signs.shape[0], dim))
        for i in range(centers.shape[0]):
            offsets = (signs * (sizes[i] / 2.0)) @ rotations[i].T
            out[i] = centers[i] + offsets
        return out

    def as_aabb(self) -> npt.NDArray[np.float64]:
        """``(N, S, 2)`` axis-aligned bounds enclosing each oriented box."""
        corners = self.corners()
        return np.stack([corners.min(axis=1), corners.max(axis=1)], axis=2)

    @property
    def volumes(self) -> npt.NDArray[np.float64]:
        return np.asarray(
            np.prod(self.sizes.astype(np.float64), axis=1), dtype=np.float64
        )

    def __len__(self) -> int:
        return int(self.centers.shape[0])


# --------------------------------------------------------------------------
# §8.4 keypoints
# --------------------------------------------------------------------------


def encode_keypoints(
    points: npt.ArrayLike,
    keypoint_class_ids: Sequence[int],
    class_ids: Sequence[int],
    *,
    visibility: npt.ArrayLike | None = None,
    instance_ids: Sequence[int] | None = None,
    scores: Sequence[float] | None = None,
    skeleton: str | None = None,
) -> AnnotationPayload:
    """Pack ``(N, K, S)`` keypoints with per-slot classes and visibility."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 3:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"keypoints must have shape (N, K, S), got {array.shape}", code="E405"
        )
    n, k = array.shape[0], array.shape[1]
    if len(keypoint_class_ids) != k:
        raise MEDH5ValidationError(
            f"keypoint_class_ids has {len(keypoint_class_ids)} entries for "
            f"{k} keypoint slots",
            code="E405",
        )
    if visibility is None:
        vis = np.full((n, k), 2, dtype=np.uint8)
    else:
        vis = np.asarray(visibility, dtype=np.uint8)
        if vis.shape != (n, k):
            raise MEDH5ValidationError(
                f"visibility {vis.shape} must be (N, K) = ({n}, {k})", code="E405"
            )
        if np.any(vis > 2):  # noqa: PLR2004
            raise MEDH5ValidationError(
                "visibility values must be 0 (unlabelled), 1 (occluded) or 2 (visible)",
                code="E411",
            )
    datasets: dict[str, npt.NDArray[Any]] = {
        "points": array,
        "visibility": vis,
        "keypoint_class_ids": np.asarray(keypoint_class_ids, dtype=np.uint16),
    }
    datasets.update(_object_columns(n, class_ids, instance_ids, scores, None))
    return AnnotationPayload(
        kind="keypoints",
        datasets=datasets,
        attrs={"skeleton": skeleton} if skeleton else {},
        class_ids=tuple(sorted({int(c) for c in (*class_ids, *keypoint_class_ids)})),
    )


class KeypointsAnnotation(GeometricAnnotation):
    """Reader for ``kind = "keypoints"``."""

    __slots__ = ()

    @property
    def points(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("points")[...], dtype=np.float32)

    @property
    def visibility(self) -> npt.NDArray[np.uint8]:
        node = self._dataset("visibility", required=False)
        if node is None:
            return np.full(self.points.shape[:2], 2, dtype=np.uint8)
        return np.asarray(node[...], dtype=np.uint8)

    @property
    def keypoint_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._dataset("keypoint_class_ids")[...], dtype=np.uint16)

    @property
    def skeleton_id(self) -> str | None:
        value = self.group.attrs.get("skeleton")
        return as_str(value) if value is not None else None

    def skeleton(self) -> Any:
        """The skeleton this annotation names, resolved from the label set."""
        name = self.skeleton_id
        if name is None:
            return None
        if self._label_set is None:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r} names skeleton {name!r} but the sample "
                "has no label set to resolve it",
                code="E413",
            )
        return self._label_set.skeleton(name)

    def labelled(self) -> npt.NDArray[np.bool_]:
        """``(N, K)`` mask of keypoints that were actually annotated."""
        return self.visibility > 0

    def __len__(self) -> int:
        return int(self.points.shape[0])


# --------------------------------------------------------------------------
# §8.5 points
# --------------------------------------------------------------------------


def encode_points(
    points: npt.ArrayLike,
    *,
    class_ids: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    weights: Sequence[float] | None = None,
    correspondence: str | None = None,
) -> AnnotationPayload:
    """Pack a point set: landmarks, seeds, or one half of a correspondence pair."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"points must have shape (N, S), got {array.shape}", code="E405"
        )
    datasets: dict[str, npt.NDArray[Any]] = {"points": array}
    if class_ids is not None:
        datasets["class_ids"] = np.asarray(class_ids, dtype=np.uint16)
    if names is not None:
        if len(names) != array.shape[0]:
            raise MEDH5ValidationError(
                f"names has {len(names)} entries for {array.shape[0]} points",
                code="E405",
            )
        datasets["names"] = np.array(list(names), dtype=str_dtype())
    if weights is not None:
        datasets["weights"] = np.asarray(weights, dtype=np.float32)
    return AnnotationPayload(
        kind="points",
        datasets=datasets,
        attrs={"correspondence": correspondence} if correspondence else {},
        class_ids=tuple(sorted({int(c) for c in class_ids})) if class_ids else (),
    )


class PointsAnnotation(GeometricAnnotation):
    """Reader for ``kind = "points"``."""

    __slots__ = ()

    @property
    def points(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("points")[...], dtype=np.float32)

    @property
    def names(self) -> tuple[str, ...] | None:
        node = self._dataset("names", required=False)
        return None if node is None else tuple(as_str(v) for v in node[...])

    @property
    def weights(self) -> npt.NDArray[np.float32] | None:
        return self._optional("weights", np.float32)

    @property
    def correspondence(self) -> str | None:
        value = self.group.attrs.get("correspondence")
        return as_str(value) if value is not None else None

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        node = self._dataset("class_ids", required=False)
        if node is None:
            return np.zeros(self.points.shape[0], dtype=np.uint16)
        return np.asarray(node[...], dtype=np.uint16)

    def named(self) -> dict[str, npt.NDArray[np.float32]]:
        """``name -> point``, for landmark sets."""
        names = self.names
        if names is None:
            return {}
        return dict(zip(names, self.points, strict=True))

    def world_points(self, grid: Grid | str | None = None) -> npt.NDArray[np.float64]:
        return self.to_world(self.points, grid=grid)

    def __len__(self) -> int:
        return int(self.points.shape[0])


# --------------------------------------------------------------------------
# §8.6 contours
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Polygon:
    """One planar polygon handed to :func:`encode_contours`."""

    vertices: npt.NDArray[Any]
    class_id: int
    plane: tuple[int, int] = (-1, 0)
    """``(axis, index)`` of the plane it lies in; ``axis = -1`` for out-of-plane."""

    role: str = "outer"

    def __post_init__(self) -> None:
        if self.role not in CONTOUR_ROLES:
            raise MEDH5ValidationError(
                f"contour role {self.role!r} must be one of {list(CONTOUR_ROLES)}",
                code="E411",
            )


def encode_contours(polygons: Sequence[Polygon]) -> AnnotationPayload:
    """Concatenate planar polygons with an offset table (spec §8.6).

    Rasterising contours into a voxel annotation is an explicit,
    provenance-tracked activity --- never something a reader does implicitly,
    because the rasterisation rule (winding, hole handling, partial voxels) is a
    decision that belongs in the record.
    """
    if not polygons:
        raise MEDH5ValidationError("no polygons were supplied", code="E410")
    chunks = [np.asarray(p.vertices, dtype=np.float32) for p in polygons]
    dim = chunks[0].shape[1]
    for i, chunk in enumerate(chunks):
        if chunk.ndim != 2 or chunk.shape[1] != dim:  # noqa: PLR2004
            raise MEDH5ValidationError(
                f"polygon {i} has shape {chunk.shape}; expected (V, {dim})", code="E405"
            )
    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([c.shape[0] for c in chunks], dtype=np.int64)
    return AnnotationPayload(
        kind="contours",
        datasets={
            "vertices": np.concatenate(chunks),
            "contour_offsets": offsets,
            "contour_class_ids": np.asarray(
                [p.class_id for p in polygons], dtype=np.uint16
            ),
            "contour_plane": np.asarray([p.plane for p in polygons], dtype=np.int32),
            "contour_role": np.asarray(
                [CONTOUR_ROLES.index(p.role) for p in polygons], dtype=np.uint8
            ),
        },
        class_ids=tuple(sorted({int(p.class_id) for p in polygons})),
    )


class ContoursAnnotation(GeometricAnnotation):
    """Reader for ``kind = "contours"``."""

    __slots__ = ()

    @property
    def vertices(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("vertices")[...], dtype=np.float32)

    @property
    def offsets(self) -> npt.NDArray[np.int64]:
        return np.asarray(self._dataset("contour_offsets")[...], dtype=np.int64)

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._dataset("contour_class_ids")[...], dtype=np.uint16)

    @property
    def planes(self) -> npt.NDArray[np.int32]:
        node = self._dataset("contour_plane", required=False)
        if node is None:
            return np.full((len(self), 2), -1, dtype=np.int32)
        return np.asarray(node[...], dtype=np.int32)

    @property
    def roles(self) -> tuple[str, ...]:
        node = self._dataset("contour_role", required=False)
        if node is None:
            return ("outer",) * len(self)
        return tuple(CONTOUR_ROLES[int(v)] for v in node[...])

    def polygon(self, index: int) -> npt.NDArray[np.float32]:
        offsets = self.offsets
        return self.vertices[int(offsets[index]) : int(offsets[index + 1])]

    def polygons(self) -> Iterator[Polygon]:
        classes = self.object_class_ids
        planes = self.planes
        roles = self.roles
        for i in range(len(self)):
            yield Polygon(
                vertices=self.polygon(i),
                class_id=int(classes[i]),
                plane=(int(planes[i][0]), int(planes[i][1])),
                role=roles[i],
            )

    def by_plane(self) -> dict[tuple[int, int], list[int]]:
        """Plane -> polygon indices, which is how RTSTRUCT data is consumed."""
        out: dict[tuple[int, int], list[int]] = {}
        for i, plane in enumerate(self.planes):
            out.setdefault((int(plane[0]), int(plane[1])), []).append(i)
        return out

    def __len__(self) -> int:
        return int(self.offsets.shape[0] - 1)


# --------------------------------------------------------------------------
# §8.7 mesh
# --------------------------------------------------------------------------


def encode_mesh(
    vertices: npt.ArrayLike,
    faces: npt.ArrayLike,
    *,
    normals: npt.ArrayLike | None = None,
    vertex_class_ids: Sequence[int] | None = None,
    mesh_offsets: Sequence[int] | None = None,
    mesh_class_ids: Sequence[int] | None = None,
) -> AnnotationPayload:
    """Pack a triangle surface mesh (spec §8.7)."""
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"mesh vertices must have shape (V, 3), got {v.shape}", code="E405"
        )
    if f.ndim != 2 or f.shape[1] != 3:  # noqa: PLR2004
        raise MEDH5ValidationError(
            f"mesh faces must have shape (F, 3), got {f.shape}", code="E405"
        )
    if f.size and (f.min() < 0 or f.max() >= v.shape[0]):
        raise MEDH5ValidationError(
            f"mesh faces index outside the {v.shape[0]} vertices", code="E405"
        )
    datasets: dict[str, npt.NDArray[Any]] = {"vertices": v, "faces": f}
    if normals is not None:
        n = np.asarray(normals, dtype=np.float32)
        if n.shape != v.shape:
            raise MEDH5ValidationError(
                f"normals {n.shape} must match vertices {v.shape}", code="E405"
            )
        datasets["normals"] = n
    if vertex_class_ids is not None:
        datasets["vertex_class_ids"] = np.asarray(vertex_class_ids, dtype=np.uint16)
    if mesh_offsets is not None:
        datasets["mesh_offsets"] = np.asarray(mesh_offsets, dtype=np.int64)
    if mesh_class_ids is not None:
        datasets["mesh_class_ids"] = np.asarray(mesh_class_ids, dtype=np.uint16)
    declared = set()
    if vertex_class_ids is not None:
        declared |= {int(c) for c in vertex_class_ids}
    if mesh_class_ids is not None:
        declared |= {int(c) for c in mesh_class_ids}
    return AnnotationPayload(
        kind="mesh", datasets=datasets, class_ids=tuple(sorted(declared))
    )


class MeshAnnotation(GeometricAnnotation):
    """Reader for ``kind = "mesh"``.

    A mesh is a **surface**, not a cheaper voxel annotation: it does not satisfy
    the ``seg`` profile on its own, because you cannot ask it which class holds a
    voxel without rasterising, and rasterising is a decision.
    """

    __slots__ = ()

    @property
    def vertices(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("vertices")[...], dtype=np.float32)

    @property
    def faces(self) -> npt.NDArray[np.int32]:
        return np.asarray(self._dataset("faces")[...], dtype=np.int32)

    @property
    def normals(self) -> npt.NDArray[np.float32] | None:
        return self._optional("normals", np.float32)

    @property
    def vertex_class_ids(self) -> npt.NDArray[np.uint16] | None:
        return self._optional("vertex_class_ids", np.uint16)

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        node = self._dataset("mesh_class_ids", required=False)
        if node is not None:
            return np.asarray(node[...], dtype=np.uint16)
        return np.asarray(self.class_ids, dtype=np.uint16)

    @property
    def n_submeshes(self) -> int:
        node = self._dataset("mesh_offsets", required=False)
        return 1 if node is None else int(node.shape[0] - 1)

    def bounds(self) -> npt.NDArray[np.float64]:
        """``(3, 2)`` axis-aligned bounds of the surface."""
        v = self.vertices.astype(np.float64)
        return np.stack([v.min(axis=0), v.max(axis=0)], axis=1)

    def __len__(self) -> int:
        return int(self.faces.shape[0])


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
    "check_space",
    "encode_boxes",
    "encode_contours",
    "encode_keypoints",
    "encode_mesh",
    "encode_obb",
    "encode_points",
]
