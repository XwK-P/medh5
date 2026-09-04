"""``layers``: the default voxel encoding (spec §7.2).

*L* stacked labelmaps, mutually exclusive **within** a layer and freely
overlapping across layers.  The layer assignment comes from colouring the class
overlap graph (:mod:`medh5.annotations.voxel.select`), so *L* tracks the overlap
depth rather than the class count --- 5 volumes for 200 classes on the reference
phantom.

``data`` is chunked ``(1, *spatial_chunk)`` so one layer is readable without
decompressing the others, and reads index in a single call: ``data[(l, *roi)]``,
never ``data[l][roi]``, which materialises the whole layer first (spec §14.5).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.annotations.base import VoxelAnnotation
from medh5.annotations.payload import AnnotationPayload
from medh5.annotations.voxel.payload import Masks, contains_value, normalize_masks
from medh5.annotations.voxel.select import (
    label_dtype_size,
    layers_from_colouring,
)
from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import IGNORE_ID


def encode_layers(
    masks: Masks,
    spatial_shape: tuple[int, ...] | None = None,
    *,
    colouring: dict[int, int] | None = None,
    ignore: npt.NDArray[np.bool_] | None = None,
    ignore_id: int = IGNORE_ID,
) -> AnnotationPayload:
    """Colour the overlap graph and pack each colour into one labelmap."""
    from medh5.annotations.voxel.select import analyse

    resolved, shape = normalize_masks(masks, spatial_shape)
    class_ids = tuple(sorted(resolved))
    if colouring is None:
        colouring = analyse(resolved, shape).colouring
    else:
        missing = set(class_ids) - set(colouring)
        if missing:
            raise MEDH5ValidationError(
                f"colouring omits classes {sorted(missing)}", code="E404"
            )
    buckets = layers_from_colouring({c: colouring[c] for c in class_ids})
    n_layers = max(1, len(buckets))
    itemsize = label_dtype_size(class_ids, ignore=ignore is not None)
    dtype = np.uint8 if itemsize == 1 else np.uint16

    data = np.zeros((n_layers, *shape), dtype=dtype)
    for layer, bucket in enumerate(buckets):
        claimed = np.zeros(shape, dtype=bool)
        for class_id in bucket:
            mask = resolved[class_id]
            if bool(np.any(mask & claimed)):
                raise MEDH5ValidationError(
                    f"classes {bucket} were assigned to layer {layer} but overlap; "
                    "the colouring is not a valid colouring of the overlap graph",
                    code="E404",
                )
            data[layer][mask] = class_id
            claimed |= mask
        if ignore is not None:
            ignore_arr = np.asarray(ignore, dtype=bool)
            data[layer][ignore_arr & ~claimed] = dtype(ignore_id)

    width = max((len(b) for b in buckets), default=1)
    layer_class_ids = np.zeros((n_layers, max(1, width)), dtype=np.uint16)
    for layer, bucket in enumerate(buckets):
        layer_class_ids[layer, : len(bucket)] = np.asarray(bucket, dtype=np.uint16)

    return AnnotationPayload(
        kind="layers",
        datasets={"data": data, "layer_class_ids": layer_class_ids},
        attrs={},
        stacked_axes=1,
        class_ids=class_ids,
    )


class LayersAnnotation(VoxelAnnotation):
    """Reader for ``kind = "layers"``."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        try:
            return self.group["data"]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `layers` requires a `data` dataset",
                code="E410",
            ) from None

    @property
    def layer_class_ids(self) -> npt.NDArray[np.uint16]:
        try:
            return np.asarray(self.group["layer_class_ids"][...], dtype=np.uint16)
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `layers` requires `layer_class_ids`",
                code="E410",
            ) from None

    @property
    def n_layers(self) -> int:
        return int(self.data.shape[0])

    @property
    def layer_of(self) -> dict[int, int]:
        """class id -> layer index.  Every class appears in exactly one layer."""
        table = self.layer_class_ids
        out: dict[int, int] = {}
        for layer in range(table.shape[0]):
            for value in table[layer]:
                class_id = int(value)
                if class_id == 0:
                    continue
                if class_id in out:
                    raise MEDH5ValidationError(
                        f"annotation {self.ann_id!r}: class {class_id} appears in "
                        f"layers {out[class_id]} and {layer}",
                        code="E404",
                    )
                out[class_id] = layer
        return out

    def layer_classes(self) -> tuple[tuple[int, ...], ...]:
        table = self.layer_class_ids
        return tuple(
            tuple(int(v) for v in table[layer] if int(v) != 0)
            for layer in range(table.shape[0])
        )

    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        layer = self.layer_of.get(class_id)
        if layer is None:
            return np.zeros(self._roi_shape(roi), dtype=bool)
        block: npt.NDArray[np.bool_] = np.asarray(self.data[(layer, *roi)]) == class_id
        return block

    def dense(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """``(C, *roi)`` occupancy, reading each layer once rather than once
        per class.

        The generic implementation asks for one class at a time, which for a
        200-class annotation packed into four layers means 200 reads of four
        distinct planes --- 50× more decompression than the data requires.
        Grouping by layer first is the whole reason a 64³ multi-class label
        read costs milliseconds instead of the 117 ms the 0.x layout did.
        """
        ids = self.resolve_classes(classes)
        window = self._roi(roi)
        out = np.zeros((len(ids), *self._roi_shape(window)), dtype=bool)
        by_layer: dict[int, list[int]] = {}
        for position, class_id in enumerate(ids):
            layer = self.layer_of.get(class_id)
            if layer is not None:
                by_layer.setdefault(layer, []).append(position)
        for layer, positions in sorted(by_layer.items()):
            block = np.asarray(self.data[(layer, *window)])
            for position in positions:
                out[position] = block == ids[position]
        return out

    def read_layer(
        self, layer: int, roi: tuple[slice, ...] | None = None
    ) -> npt.NDArray[Any]:
        """One layer's labelmap, sliced in a single call (spec §14.5)."""
        window = self._roi(roi)
        return np.asarray(self.data[(layer, *window)])

    def _encodes_ignore(self) -> bool:
        """Whether any layer stores the ignore id, read in bounded slabs.

        ``has_ignore_region`` reads like a header lookup, and materialising
        every layer to answer it allocated ~1.3 GiB for a five-layer uint16
        annotation over a 512³ grid --- enough to end a per-sample loop in a
        dataloader or a CLI report on a property nobody expected to touch the
        data.  Layers chunk one per plane (§14.1), so a slab read costs a
        fraction of that, and the first hit ends the scan.
        """
        return contains_value(self.data, self.ignore_id)

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        """The in-band ignore region (§7.7), as ``labelmap`` also exposes it.

        A voxel is ignored where *any* layer marks it: the encoder writes the
        ignore id into every layer it is not already claimed in, and a caller
        asking "was this examined?" wants one answer for the voxel, not one per
        plane.

        This existed on ``labelmap`` and not here, so a ``layers`` annotation
        could report ``has_ignore_region`` while offering no way to read the
        region --- and callers written to `getattr(..., "ignore_mask", None)`
        silently concluded there was none.
        """
        window = self._roi(roi)
        block = np.asarray(self.data[(slice(None), *window)])
        out: npt.NDArray[np.bool_] = np.any(block == self.ignore_id, axis=0)
        return out

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        out["layers"] = self.n_layers
        return out


__all__ = ["LayersAnnotation", "encode_layers"]
