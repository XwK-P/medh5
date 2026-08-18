"""Annotations: one coherent unit of ground truth per group (spec §6-§9)."""

from __future__ import annotations

from medh5.annotations.base import (
    ANNOTATION_KINDS,
    GEOMETRIC_KINDS,
    RESERVED_KINDS,
    TASKS,
    VOXEL_KINDS,
    Annotation,
    AnnotationHeader,
    VoxelAnnotation,
    open_annotation,
    readers,
)
from medh5.annotations.classification import (
    SCOPES,
    Assertion,
    ClassificationAnnotation,
    encode_classification,
)
from medh5.annotations.geometric import (
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
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel import (
    Instance,
    encode_voxels,
    select_encoding,
    transcode,
)

__all__ = [
    "ANNOTATION_KINDS",
    "GEOMETRIC_KINDS",
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
    "readers",
    "select_encoding",
    "transcode",
]
