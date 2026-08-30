"""Lossless conversion between voxel encodings (spec §7.6).

Transcoding must preserve ``contains(class, voxel)`` for every class and every
voxel --- with the sole exception of ``probmap``, which is lossless only under a
declared threshold.  That property is what makes the encoding a storage decision
rather than a data-model decision: a file can be re-encoded for a different
access pattern without anyone re-deriving the ground truth.

Everything here works on :class:`AnnotationPayload` arrays as well as on open
annotations, so the round-trip matrix is testable without touching a file.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel.bitmask import BITS_PER_PLANE, encode_bitmask
from medh5.annotations.voxel.instances import InstanceInput, encode_instances
from medh5.annotations.voxel.labelmap import encode_labelmap
from medh5.annotations.voxel.layers import encode_layers
from medh5.annotations.voxel.mask import encode_mask
from medh5.annotations.voxel.probmap import DEFAULT_THRESHOLD, encode_probmap
from medh5.errors import MEDH5ValidationError
from medh5.geometry.affine import box_to_slices

TRANSCODABLE = ("labelmap", "layers", "bitmask", "instances", "probmap")


def payload_to_masks(
    payload: AnnotationPayload,
    *,
    spatial_shape: tuple[int, ...] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[int, npt.NDArray[np.bool_]]:
    """Decode any voxel payload to per-class boolean masks."""
    kind = payload.kind
    if kind == "labelmap":
        data = payload.data
        return {int(c): data == c for c in payload.class_ids}
    if kind == "layers":
        data = payload.data
        table = payload.datasets["layer_class_ids"]
        out: dict[int, npt.NDArray[np.bool_]] = {}
        for layer in range(table.shape[0]):
            for value in table[layer]:
                class_id = int(value)
                if class_id:
                    out[class_id] = data[layer] == class_id
        return dict(sorted(out.items()))
    if kind == "bitmask":
        data = payload.data
        ids = payload.datasets["bit_class_ids"]
        decoded: dict[int, npt.NDArray[np.bool_]] = {}
        for position, value in enumerate(ids):
            plane, bit = divmod(position, BITS_PER_PLANE)
            decoded[int(value)] = (
                data[plane] & (np.uint64(1) << np.uint64(bit))
            ) != np.uint64(0)
        return dict(sorted(decoded.items()))
    if kind == "probmap":
        data = payload.data
        return {
            int(c): np.asarray(data[i], dtype=np.float32) >= threshold
            for i, c in enumerate(payload.class_ids)
        }
    if kind == "mask":
        return {1: np.asarray(payload.data, dtype=bool)}
    if kind == "instances":
        if spatial_shape is None:
            raise MEDH5ValidationError(
                "decoding `instances` needs the grid's spatial shape", code="E405"
            )
        return _instances_to_masks(payload, spatial_shape)
    raise MEDH5ValidationError(f"cannot decode voxel kind {kind!r}", code="E401")


def _instances_to_masks(
    payload: AnnotationPayload, spatial_shape: tuple[int, ...]
) -> dict[int, npt.NDArray[np.bool_]]:
    boxes = payload.datasets["boxes"]
    classes = payload.datasets["class_ids"]
    # Seeded from `payload.class_ids`, not from the objects present.  A class
    # searched for and not found has no object and must still decode to an empty
    # mask: dropping it turns "verified absent" into "never looked for" (§11.3),
    # which is exactly what `encode_instances` keeps `class_ids` for.  It also
    # made `check_roundtrip` blind to the loss, since it decodes the original
    # through this same path before comparing.
    declared = tuple(payload.class_ids) or tuple(int(c) for c in np.unique(classes))
    out = {int(c): np.zeros(spatial_shape, dtype=bool) for c in declared}
    has_masks = "mask_data" in payload.datasets
    for index in range(boxes.shape[0]):
        slices = box_to_slices(boxes[index], spatial_shape)
        target = out[int(classes[index])]
        if has_masks:
            offsets = payload.datasets["mask_offsets"]
            shapes = payload.datasets["mask_shapes"]
            start, stop = int(offsets[index]), int(offsets[index + 1])
            shape = tuple(int(v) for v in shapes[index])
            n = int(np.prod(shape, dtype=np.int64))
            crop = (
                np.unpackbits(payload.datasets["mask_data"][start:stop])[:n]
                .astype(bool)
                .reshape(shape)
            )
            target[slices] |= crop
        else:
            target[slices] = True
    return dict(sorted(out.items()))


def annotation_to_masks(
    annotation: VoxelAnnotation, classes: Sequence[int | str] | None = None
) -> dict[int, npt.NDArray[np.bool_]]:
    """Decode an open annotation to per-class boolean masks."""
    ids = annotation.resolve_classes(classes)
    window = annotation._roi(None)  # noqa: SLF001 - same-package internal
    return {
        class_id: annotation._dense_class(class_id, window)  # noqa: SLF001
        for class_id in ids
    }


def encode_masks(
    masks: Mapping[int, npt.NDArray[np.bool_]],
    kind: str,
    spatial_shape: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> AnnotationPayload:
    """Encode per-class boolean masks into any voxel encoding."""
    if kind == "labelmap":
        return encode_labelmap(masks, spatial_shape, **kwargs)
    if kind == "layers":
        return encode_layers(masks, spatial_shape, **kwargs)
    if kind == "bitmask":
        return encode_bitmask(masks, spatial_shape)
    if kind == "probmap":
        return encode_probmap(
            {c: m.astype(np.float32) for c, m in masks.items()},
            spatial_shape,
            **kwargs,
        )
    if kind == "mask":
        planes = list(masks.values())
        if not planes:
            raise MEDH5ValidationError("no masks were supplied", code="E410")
        merged = planes[0].copy()
        for plane in planes[1:]:
            merged |= plane
        return encode_mask(merged)
    if kind == "instances":
        start_id = int(kwargs.pop("start_id", 1))
        objects = [
            InstanceInput(class_id=class_id, instance_id=start_id + i, mask=mask)
            for i, (class_id, mask) in enumerate(sorted(masks.items()))
            if mask.any()
        ]
        return encode_instances(
            objects, spatial_shape, class_ids=sorted(masks), **kwargs
        )
    raise MEDH5ValidationError(f"cannot encode voxel kind {kind!r}", code="E401")


def transcode_payload(
    payload: AnnotationPayload,
    to_kind: str,
    *,
    spatial_shape: tuple[int, ...] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    **kwargs: Any,
) -> AnnotationPayload:
    """Convert a payload to another encoding, preserving ``contains``."""
    if to_kind == payload.kind:
        return payload
    masks = payload_to_masks(payload, spatial_shape=spatial_shape, threshold=threshold)
    shape = spatial_shape or next(iter(masks.values())).shape
    return encode_masks(masks, to_kind, tuple(shape), **kwargs)


IN_BAND_IGNORE_KINDS = ("labelmap", "layers")
"""Encodings that can hold an ignore region in the data itself (spec §7.7)."""


def transcode(
    annotation: VoxelAnnotation, to_kind: str, **kwargs: Any
) -> AnnotationPayload:
    """Convert an open annotation to another encoding.

    Refuses rather than silently dropping what the target cannot express: an
    in-band ignore region, and object identity.  §7.6 calls transcoding
    lossless, so anything it cannot carry has to stop it.
    """
    if to_kind not in TRANSCODABLE and to_kind != "mask":
        raise MEDH5ValidationError(
            f"{to_kind!r} is not a voxel encoding; expected one of "
            f"{list(TRANSCODABLE)}",
            code="E401",
        )
    if to_kind == "instances" and annotation.kind != "instances":
        # A dense encoding records which voxels belong to a class, never which
        # object they belong to.  Going to `instances` from one merged every
        # object of a class into a single mask and minted a fresh
        # `instance_id` for it, so two lesions came back as one with an id
        # neither of them had -- in the field §7.4 makes the whole longitudinal
        # join.  `instances_from_masks` already refuses to split components for
        # the same reason; merging them is the same invention.
        raise MEDH5ValidationError(
            f"cannot transcode {annotation.kind!r} to 'instances': a dense "
            "encoding carries no object identity, so every object of a class "
            "would merge into one with a newly minted instance_id (spec §7.4). "
            "Re-derive the objects from the source that had them.",
            code="E404",
        )
    ignore = _ignore_region(annotation)
    if ignore is not None:
        if to_kind not in IN_BAND_IGNORE_KINDS:
            # §7.7: `bitmask` and `probmap` express ignore as a separate
            # `mask`-kind annotation, which this function cannot create -- it
            # returns one payload.  Dropping it turned "nobody examined these
            # voxels" into "verified absent for every annotated class", which
            # §7.7 names as the single most common cause of silently
            # mistrained segmentation models.
            raise MEDH5ValidationError(
                f"cannot transcode {annotation.kind!r} to {to_kind!r}: this "
                f"annotation carries an in-band ignore region, which "
                f"{to_kind!r} cannot hold (spec §7.7). Write the ignore region "
                "as a separate `mask` annotation and reference it with "
                "`ignore_mask=` first, or transcode to "
                f"{' or '.join(map(repr, IN_BAND_IGNORE_KINDS))} instead.",
                code="E404",
            )
        kwargs.setdefault("ignore", ignore)
    masks = annotation_to_masks(annotation)
    return encode_masks(masks, to_kind, annotation.spatial_shape, **kwargs)


def _ignore_region(annotation: VoxelAnnotation) -> npt.NDArray[np.bool_] | None:
    """The annotation's **in-band** ignore region, if it carries one (§7.7).

    Only the in-band case matters here.  An annotation whose header names a
    separate `mask` annotation via `ignore_mask` keeps that reference through
    the header, so nothing is lost; it is the voxels written as `ignore_id`
    inside `labelmap`/`layers` data that have nowhere to go in `bitmask` or
    `probmap`.
    """
    if not annotation._encodes_ignore():  # noqa: SLF001 - same-package internal
        return None
    reader = getattr(annotation, "ignore_mask", None)
    if reader is None:
        return None
    return np.asarray(reader(), dtype=bool)


def masks_equal(
    a: Mapping[int, npt.NDArray[np.bool_]], b: Mapping[int, npt.NDArray[np.bool_]]
) -> bool:
    """Whether two mask sets agree on every class --- the losslessness check."""
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def check_roundtrip(
    payload: AnnotationPayload,
    to_kind: str,
    *,
    spatial_shape: tuple[int, ...] | None = None,
) -> bool:
    """Whether ``A -> B -> A`` preserves every class mask."""
    shape = spatial_shape or (
        payload.data.shape[payload.stacked_axes :]
        if "data" in payload.datasets
        else None
    )
    original = payload_to_masks(payload, spatial_shape=shape)
    converted = transcode_payload(payload, to_kind, spatial_shape=shape)
    decoded = payload_to_masks(converted, spatial_shape=shape)
    return masks_equal(original, decoded)


__all__ = [
    "TRANSCODABLE",
    "annotation_to_masks",
    "check_roundtrip",
    "encode_masks",
    "masks_equal",
    "payload_to_masks",
    "transcode",
    "transcode_payload",
]
