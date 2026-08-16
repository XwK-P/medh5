"""Annotations: one coherent unit of ground truth per group (spec §6-§9)."""

from __future__ import annotations

from medh5.annotations.base import (
    ANNOTATION_KINDS,
    RESERVED_KINDS,
    TASKS,
    VOXEL_KINDS,
    Annotation,
    AnnotationHeader,
    VoxelAnnotation,
    open_annotation,
)
from medh5.annotations.voxel import (
    Instance,
    VoxelPayload,
    encode_voxels,
    select_encoding,
    transcode,
)

__all__ = [
    "ANNOTATION_KINDS",
    "RESERVED_KINDS",
    "TASKS",
    "VOXEL_KINDS",
    "Annotation",
    "AnnotationHeader",
    "Instance",
    "VoxelAnnotation",
    "VoxelPayload",
    "encode_voxels",
    "open_annotation",
    "select_encoding",
    "transcode",
]
