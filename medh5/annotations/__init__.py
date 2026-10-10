"""Annotations: one coherent unit of ground truth per group (spec §6-§9).

:mod:`~medh5.annotations.base` is the header and the contract every kind
shares (§6); :mod:`~medh5.annotations.voxel` the voxel encodings (§7),
:mod:`~medh5.annotations.geometric` the geometric kinds (§8) and
:mod:`~medh5.annotations.classification` label assertions (§9).
"""

from __future__ import annotations

from typing import Any

from medh5.annotations.base import (
    ANNOTATION_KINDS,
    GEOMETRIC_KINDS,
    RESERVED_KINDS,
    TASKS,
    VOXEL_KINDS,
    Annotation,
    AnnotationHeader,
    AnnotationPayload,
    Instance,
    VoxelAnnotation,
)
from medh5.annotations.classification import (
    SCOPES,
    Assertion,
    ClassificationAnnotation,
    encode_classification,
)
from medh5.annotations.geometric import (
    GEOMETRIC_READERS,
    SPACES,
    BoxesAnnotation,
    ContoursAnnotation,
    GeometricAnnotation,
    KeypointsAnnotation,
    MeshAnnotation,
    ObbAnnotation,
    PointsAnnotation,
    Polygon,
    encode_boxes,
    encode_contours,
    encode_keypoints,
    encode_mesh,
    encode_obb,
    encode_points,
)
from medh5.annotations.voxel import READERS as VOXEL_READERS
from medh5.annotations.voxel import encode_voxels, select_encoding, transcode

READERS: dict[str, Any] = {
    **VOXEL_READERS,
    **GEOMETRIC_READERS,
    "classification": ClassificationAnnotation,
}
"""``kind`` -> reader class, for every kind."""


def open_annotation(handle: Any) -> Annotation:
    """The reader class for an annotation handle's ``kind``."""
    cls = READERS.get(handle.kind, Annotation)
    opened: Annotation = cls(handle)
    return opened


__all__ = [
    "ANNOTATION_KINDS",
    "GEOMETRIC_KINDS",
    "READERS",
    "RESERVED_KINDS",
    "SCOPES",
    "SPACES",
    "TASKS",
    "VOXEL_KINDS",
    "Annotation",
    "AnnotationHeader",
    "AnnotationPayload",
    "Assertion",
    "BoxesAnnotation",
    "ClassificationAnnotation",
    "ContoursAnnotation",
    "GeometricAnnotation",
    "Instance",
    "KeypointsAnnotation",
    "MeshAnnotation",
    "ObbAnnotation",
    "PointsAnnotation",
    "Polygon",
    "VoxelAnnotation",
    "encode_boxes",
    "encode_classification",
    "encode_contours",
    "encode_keypoints",
    "encode_mesh",
    "encode_obb",
    "encode_points",
    "encode_voxels",
    "open_annotation",
    "select_encoding",
    "transcode",
]
