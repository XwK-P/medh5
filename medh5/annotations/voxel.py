"""The voxel encodings, one uniform read contract (spec §7).

``labelmap`` (§7.1) holds mutually exclusive classes in one volume; ``layers``
(§7.2) packs overlapping classes into the fewest exclusive planes; ``bitmask``
(§7.3) gives every class a bit; ``instances`` (§7.4) stores objects with
identity, boxes and cropped masks; ``probmap`` (§7.5) holds soft labels; and
``mask`` (§7.7) is one boolean region with no classes --- a field of view or an
ignore region.  Every reader answers :class:`~medh5.annotations.VoxelAnnotation`'s
contract (``dense``, ``contains``, ``labelmap``, ...) whatever its encoding.

The encoders, the measurement that chooses an encoding (§7.6) and the
transcoder are the format engine's; the writer, the command line and this
module call the same functions.  A transcode refuses rather than dropping what
the target cannot express: an in-band ignore region, object identity, class
identity itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import (
    AnnotationPayload,
    Instance,
    VoxelAnnotation,
    _instance,
)

Masks = Mapping[int, npt.NDArray[np.bool_]]
"""class id -> boolean occupancy over the grid's spatial shape."""

SLAB_BYTES: int = _core.SLAB_BYTES
"""Read budget for a scan that only needs a yes/no or a count."""

BITS_PER_PLANE: int = _core.BITS_PER_PLANE
DEFAULT_THRESHOLD: float = _core.DEFAULT_THRESHOLD
STORAGE_DTYPES = (np.dtype(np.float16), np.dtype(np.float32))


def checked_class_id(class_id: Any) -> int:
    """A class id in the writable range ``1..65534``, or a coded refusal (E303)."""
    return int(_core.check_class_id(class_id))


def normalize_masks(
    masks: Masks, spatial_shape: tuple[int, ...] | None = None
) -> tuple[dict[int, npt.NDArray[np.bool_]], tuple[int, ...]]:
    """Coerce a mask mapping to ``bool`` arrays of one agreed shape."""
    resolved, shape = _core.normalize_masks(masks, spatial_shape)
    return dict(resolved), tuple(shape)


# -- the readers -------------------------------------------------------------------


class LabelmapAnnotation(VoxelAnnotation):
    """Mutually exclusive classes in one volume; ``65535`` marks ignore (§7.1)."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        """The in-band ignore region (the reserved id), ``False`` elsewhere."""
        found: npt.NDArray[np.bool_] = self._handle.ignore_mask(roi)
        return found


class LayersAnnotation(VoxelAnnotation):
    """Classes coloured onto layers so no two that overlap share one (§7.2)."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def layer_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.layer_class_ids
        return found

    @property
    def n_layers(self) -> int:
        return int(self._handle.n_layers)

    @property
    def layer_of(self) -> dict[int, int]:
        """class id -> the layer it is painted on."""
        found: dict[int, int] = self._handle.layer_of
        return found

    def layer_classes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(layer) for layer in self._handle.layer_classes())

    def read_layer(
        self, layer: int, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[Any]:
        found: npt.NDArray[Any] = self._handle.read_layer(int(layer), roi)
        return found

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        """The in-band ignore region (the reserved id on any layer)."""
        found: npt.NDArray[np.bool_] = self._handle.ignore_mask(roi)
        return found


class BitmaskAnnotation(VoxelAnnotation):
    """Classes as bits of ``uint64`` planes; position from ``bit_class_ids`` (§7.3)."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def bit_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.bit_class_ids
        return found

    @property
    def n_planes(self) -> int:
        return int(self._handle.n_planes)

    @property
    def position_of(self) -> dict[int, int]:
        """class id -> its bit position (plane * 64 + bit)."""
        found: dict[int, int] = self._handle.position_of
        return found

    def classes_at(self, voxel: Sequence[int]) -> tuple[int, ...]:
        """Every class present at one voxel."""
        return tuple(self._handle.classes_at([int(v) for v in voxel]))


class InstancesAnnotation(VoxelAnnotation):
    """Objects: each a class, a sample-scoped id, a box and maybe a mask (§7.4)."""

    __slots__ = ()

    @property
    def boxes(self) -> npt.NDArray[np.float32]:
        found: npt.NDArray[np.float32] = self._handle.boxes
        return found

    @property
    def object_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.object_class_ids
        return found

    @property
    def instance_ids(self) -> npt.NDArray[np.uint64]:
        found: npt.NDArray[np.uint64] = self._handle.instance_ids
        return found

    @property
    def scores(self) -> npt.NDArray[np.float32] | None:
        found: npt.NDArray[np.float32] | None = self._handle.scores
        return found

    @property
    def has_masks(self) -> bool:
        return bool(self._handle.has_masks)

    @property
    def n_objects(self) -> int:
        return int(self._handle.n_objects)

    def crop(self, index: int) -> npt.NDArray[np.bool_] | None:
        found: npt.NDArray[np.bool_] | None = self._handle.crop(int(index))
        return found

    def instance(self, instance_id: int) -> Instance:
        return _instance(self._handle.instance(int(instance_id)))

    def tracking(self) -> dict[int, int]:
        """``instance_id`` -> row index, the join key for §7.4 tracking."""
        found: dict[int, int] = self._handle.tracking()
        return found


class ProbmapAnnotation(VoxelAnnotation):
    """Soft labels; a voxel contains a class at or above ``threshold`` (§7.5)."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def normalized(self) -> bool:
        return bool(self._handle.normalized)

    @property
    def threshold(self) -> float:
        return float(self._handle.threshold)

    def probabilities(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.float32]:
        found: npt.NDArray[np.float32] = self._handle.probabilities(classes, roi)
        return found


class MaskAnnotation(VoxelAnnotation):
    """A region, not a class: what a field of view or an ignore mask is (§7.7)."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    def read(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        found: npt.NDArray[np.bool_] = self._handle.read_mask(roi)
        return found


READERS: dict[str, Any] = {
    "labelmap": LabelmapAnnotation,
    "layers": LayersAnnotation,
    "bitmask": BitmaskAnnotation,
    "instances": InstancesAnnotation,
    "probmap": ProbmapAnnotation,
    "mask": MaskAnnotation,
}
"""``kind`` -> reader class, for the voxel kinds."""

# -- the encoders --------------------------------------------------------------------

encode_labelmap = _core.encode_labelmap
encode_layers = _core.encode_layers
encode_bitmask = _core.encode_bitmask
encode_mask = _core.encode_mask
contains_at = _core.contains_at


@dataclass(slots=True)
class InstanceInput:
    """One object handed to :func:`encode_instances`."""

    class_id: int
    instance_id: int
    mask: npt.NDArray[np.bool_] | None = None
    box: npt.NDArray[np.float32] | None = None
    crop: npt.NDArray[np.bool_] | None = None
    score: float | None = None


def encode_instances(
    objects: Sequence[InstanceInput],
    spatial_shape: tuple[int, ...] | None = None,
    *,
    store_masks: bool = True,
    class_ids: Sequence[int] | None = None,
) -> Any:
    """Pack objects into boxes, ids and bit-packed crops (spec §7.4).

    ``class_ids`` declares the classes examined; with no objects it is what
    makes "examined, none found" expressible.
    """
    return _core.encode_instances(
        objects, spatial_shape, store_masks=store_masks, class_ids=class_ids
    )


def instances_from_masks(
    masks: Mapping[int, npt.NDArray[Any]], *, start_id: int = 1
) -> list[InstanceInput]:
    """One object per class mask, minting ids from *start_id*."""
    return [
        InstanceInput(class_id=class_id, instance_id=instance_id, mask=mask)
        for class_id, instance_id, mask in _core.instances_from_masks(
            masks, start_id=start_id
        )
    ]


def storage_dtype(
    planes: Any, threshold: float, requested: npt.DTypeLike = np.float16
) -> np.dtype[Any]:
    """The narrowest of ``float16``/``float32`` that keeps every voxel's
    containment decision (§7.5)."""
    found: np.dtype[Any] = _core.storage_dtype(planes, threshold, requested)
    return found


def encode_probmap(
    probabilities: Any,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    dtype: npt.DTypeLike = np.float16,
    normalized: bool = False,
    threshold: float | None = None,
) -> Any:
    """Pack per-class probability planes (spec §7.5)."""
    return _core.encode_probmap(
        probabilities,
        spatial_shape,
        dtype=dtype,
        normalized=normalized,
        threshold=threshold,
    )


# -- choosing an encoding by measurement (§7.6) --------------------------------------

LOCALIZED_BBOX_FRACTION: float = _core.LOCALIZED_BBOX_FRACTION
SPARSE_FILL: float = _core.SPARSE_FILL

OverlapStats = _core.OverlapStats
CostModel = _core.CostModel

analyse = _core.analyse
greedy_colour = _core.greedy_colour
layers_from_colouring = _core.layers_from_colouring
label_dtype_size = _core.label_dtype_size
cost_model = _core.cost_model
select_encoding = _core.select_encoding


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
    payload, stats = _core.encode_voxels(
        masks, spatial_shape, encoding=encoding, ignore=ignore, **kwargs
    )
    return payload, stats


# -- transcoding (§7.6) ----------------------------------------------------------------

TRANSCODABLE: tuple[str, ...] = _core.TRANSCODABLE
IN_BAND_IGNORE_KINDS: tuple[str, ...] = _core.IN_BAND_IGNORE_KINDS

payload_to_masks = _core.payload_to_masks
annotation_to_masks = _core.annotation_to_masks
encode_masks = _core.encode_masks
transcode_payload = _core.transcode_payload
transcode = _core.transcode
masks_equal = _core.masks_equal
check_roundtrip = _core.check_roundtrip

__all__ = [
    "BITS_PER_PLANE",
    "DEFAULT_THRESHOLD",
    "IN_BAND_IGNORE_KINDS",
    "LOCALIZED_BBOX_FRACTION",
    "READERS",
    "SLAB_BYTES",
    "SPARSE_FILL",
    "STORAGE_DTYPES",
    "TRANSCODABLE",
    "AnnotationPayload",
    "BitmaskAnnotation",
    "CostModel",
    "Instance",
    "InstanceInput",
    "InstancesAnnotation",
    "LabelmapAnnotation",
    "LayersAnnotation",
    "MaskAnnotation",
    "Masks",
    "OverlapStats",
    "ProbmapAnnotation",
    "analyse",
    "annotation_to_masks",
    "check_roundtrip",
    "checked_class_id",
    "contains_at",
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
    "label_dtype_size",
    "layers_from_colouring",
    "masks_equal",
    "normalize_masks",
    "payload_to_masks",
    "select_encoding",
    "storage_dtype",
    "transcode",
    "transcode_payload",
]
