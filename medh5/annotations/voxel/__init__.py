"""The five voxel encodings, one uniform read contract (spec §7)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import Instance
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel.bitmask import BitmaskAnnotation, encode_bitmask
from medh5.annotations.voxel.instances import (
    InstanceInput,
    InstancesAnnotation,
    encode_instances,
    instances_from_masks,
)
from medh5.annotations.voxel.labelmap import LabelmapAnnotation, encode_labelmap
from medh5.annotations.voxel.layers import LayersAnnotation, encode_layers
from medh5.annotations.voxel.mask import MaskAnnotation, encode_mask
from medh5.annotations.voxel.payload import Masks, normalize_masks
from medh5.annotations.voxel.probmap import ProbmapAnnotation, encode_probmap
from medh5.annotations.voxel.select import (
    OverlapStats,
    analyse,
    cost_model,
    greedy_colour,
    select_encoding,
)
from medh5.annotations.voxel.transcode import (
    IN_BAND_IGNORE_KINDS,
    annotation_to_masks,
    check_roundtrip,
    encode_masks,
    masks_equal,
    payload_to_masks,
    transcode,
    transcode_payload,
)
from medh5.errors import MEDH5ValidationError

READERS: dict[str, Any] = {
    "labelmap": LabelmapAnnotation,
    "layers": LayersAnnotation,
    "bitmask": BitmaskAnnotation,
    "instances": InstancesAnnotation,
    "probmap": ProbmapAnnotation,
    "mask": MaskAnnotation,
}


def encode_voxels(
    masks: Mapping[int, npt.NDArray[np.bool_]],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    encoding: str = "auto",
    ignore: npt.NDArray[np.bool_] | None = None,
    **kwargs: Any,
) -> tuple[AnnotationPayload, OverlapStats]:
    """Encode class masks, choosing the encoding by measurement when asked to.

    Returns the payload **and** the statistics behind the choice, so a writer can
    report why it picked what it picked (spec §7.6).

    An ``ignore`` region rides in band only under ``labelmap`` and ``layers``.
    The other encodings express it as a separate ``mask`` annotation (§7.7),
    which one payload cannot hold, so the choice is refused rather than the
    region dropped: ``SampleWriter.add_segmentation`` writes the sibling mask
    for those, and this function --- which returns a payload and nothing
    else --- has nowhere to put it.
    """
    resolved, shape = normalize_masks(masks, spatial_shape)
    kind, stats = select_encoding(
        resolved,
        shape,
        prefer=None if encoding == "auto" else encoding,
        ignore=ignore is not None,
    )
    if ignore is not None:
        if kind not in IN_BAND_IGNORE_KINDS:
            raise MEDH5ValidationError(
                f"encode_voxels: {kind!r} cannot hold an ignore region in band; "
                "§7.7 puts it in a separate `mask` annotation, which a single "
                "payload cannot carry. Use SampleWriter.add_segmentation, which "
                "writes the sibling mask, or choose "
                f"{' or '.join(map(repr, IN_BAND_IGNORE_KINDS))}.",
                code="E404",
            )
        kwargs.setdefault("ignore", ignore)
    payload = encode_masks(resolved, kind, shape, **kwargs)
    return payload, stats


__all__ = [
    "READERS",
    "BitmaskAnnotation",
    "Instance",
    "InstanceInput",
    "InstancesAnnotation",
    "LabelmapAnnotation",
    "LayersAnnotation",
    "MaskAnnotation",
    "Masks",
    "OverlapStats",
    "ProbmapAnnotation",
    "AnnotationPayload",
    "IN_BAND_IGNORE_KINDS",
    "analyse",
    "annotation_to_masks",
    "check_roundtrip",
    "cost_model",
    "encode_bitmask",
    "encode_instances",
    "encode_labelmap",
    "encode_layers",
    "encode_mask",
    "encode_masks",
    "encode_probmap",
    "encode_voxels",
    "greedy_colour",
    "instances_from_masks",
    "masks_equal",
    "normalize_masks",
    "payload_to_masks",
    "select_encoding",
    "transcode",
    "transcode_payload",
]
