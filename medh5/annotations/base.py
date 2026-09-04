"""The common annotation model and the uniform read contract (spec §6, §7.6).

Every voxel encoding defines the same predicate::

    contains(annotation, class_id c, voxel v) -> bool

and a reader must expose it identically regardless of ``kind``.  That is the
load-bearing idea of the annotation model: the encoding is a **storage
decision**, chosen by measurement (§7.6) and changeable by transcoding, while
callers ask for classes and a region of interest and never for a layout.

The second load-bearing idea is the **coverage contract**.  ``class_ids`` is what
an annotation *can* express; ``annotated_class_ids`` is what the annotator
committed to finding.  The difference is the whole of partial labelling: a class
in the first set and not the second may be present in the patient and absent
from the file, so ``0`` at a voxel means "verified absent" only for classes in
``annotated_class_ids``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_int, as_str, as_str_tuple, require_attr
from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid
from medh5.labels.labelset import (
    BACKGROUND_ID,
    CLOSURES,
    IGNORE_ID,
    LabelClass,
    LabelSet,
)

VOXEL_KINDS = ("labelmap", "layers", "bitmask", "instances", "probmap", "mask")
GEOMETRIC_KINDS = ("boxes", "obb", "keypoints", "points", "contours", "mesh")
ANNOTATION_KINDS = (*VOXEL_KINDS, *GEOMETRIC_KINDS, "classification")


def instance_id_dtype(ids: Iterable[int]) -> Any:
    """``uint32`` unless an id needs the wider form (spec §7.4, §8.2).

    One helper for every kind that stores object identity.  ``instances`` had
    its own copy and the §8 encoders hard-cast to ``uint32``, so a box carrying
    id ``2**32 + 7`` was stored as ``7`` --- in the column that is the whole
    longitudinal join --- while the same id on an ``instances`` annotation
    survived.  The width follows the data, and it follows it everywhere.
    """
    return np.uint64 if any(int(i) > 0xFFFFFFFF for i in ids) else np.uint32


RESERVED_KINDS = ("rle",)
"""Names reserved by spec §16.  A 1.0 writer MUST NOT emit them."""

TASKS = ("segmentation", "detection", "classification", "registration", "other")

DEFAULT_TASK_FOR_KIND: dict[str, str] = {
    "labelmap": "segmentation",
    "layers": "segmentation",
    "bitmask": "segmentation",
    "probmap": "segmentation",
    "mask": "other",
    "instances": "segmentation",
    "contours": "segmentation",
    "mesh": "segmentation",
    "boxes": "detection",
    "obb": "detection",
    "keypoints": "detection",
    "points": "detection",
    "classification": "classification",
}

SPEC_ANNOTATION_ATTRS = (
    "kind",
    "task",
    "grid",
    "timepoints",
    "space",
    "frame_uid",
    "class_ids",
    "annotated_class_ids",
    "closure",
    "ignore_id",
    "ignore_mask",
    "prov",
    "quality",
    "derived_from",
    "digest",
    "scope",
    "scope_ids",
    "multilabel",
    "normalized",
    "skeleton",
)


@dataclass(slots=True)
class AnnotationHeader:
    """The fixed attribute header every annotation carries (spec §6.2)."""

    kind: str
    task: str
    grid: str | None = None
    timepoints: tuple[str, ...] | None = None
    space: str | None = None
    frame_uid: str | None = None
    class_ids: tuple[int, ...] = ()
    annotated_class_ids: tuple[int, ...] = ()
    closure: str = "explicit"
    ignore_id: int = IGNORE_ID
    ignore_mask: str | None = None
    prov: str | None = None
    quality: str | None = None
    derived_from: tuple[str, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind in RESERVED_KINDS:
            raise MEDH5ValidationError(
                f"annotation kind {self.kind!r} is reserved by spec §16 and "
                "must not be written by a 1.0 writer",
                code="E401",
            )
        if self.kind not in ANNOTATION_KINDS:
            raise MEDH5ValidationError(
                f"unknown annotation kind {self.kind!r}", code="E401"
            )
        if self.task not in TASKS:
            raise MEDH5ValidationError(
                f"unknown task {self.task!r}; expected one of {list(TASKS)}",
                code="E412",
            )
        if self.closure not in CLOSURES:
            raise MEDH5ValidationError(
                f"closure {self.closure!r} must be one of {list(CLOSURES)}", code="E412"
            )
        self.class_ids = tuple(int(c) for c in self.class_ids)
        self.annotated_class_ids = tuple(int(c) for c in self.annotated_class_ids)
        self.derived_from = tuple(self.derived_from)
        if self.timepoints is not None:
            self.timepoints = tuple(self.timepoints)
        missing = set(self.annotated_class_ids) - set(self.class_ids)
        if missing:
            raise MEDH5ValidationError(
                f"annotated_class_ids {sorted(missing)} are not in class_ids",
                code="E403",
            )
        reserved = {BACKGROUND_ID, IGNORE_ID} & set(self.class_ids)
        if reserved:
            raise MEDH5ValidationError(
                f"class_ids uses reserved id(s) {sorted(reserved)}", code="E303"
            )

    def attrs(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "task": self.task,
            "closure": self.closure,
        }
        if self.grid is not None:
            out["grid"] = self.grid
        if self.timepoints is not None:
            out["timepoints"] = list(self.timepoints)
        if self.space is not None:
            out["space"] = self.space
        if self.frame_uid is not None:
            out["frame_uid"] = self.frame_uid
        if self.kind != "mask":
            out["class_ids"] = np.asarray(self.class_ids, dtype=np.uint16)
        out["annotated_class_ids"] = np.asarray(
            self.annotated_class_ids, dtype=np.uint16
        )
        if self.ignore_id != IGNORE_ID:
            out["ignore_id"] = int(self.ignore_id)
        if self.ignore_mask is not None:
            out["ignore_mask"] = self.ignore_mask
        if self.prov is not None:
            out["prov"] = self.prov
        if self.quality is not None:
            out["quality"] = self.quality
        if self.derived_from:
            out["derived_from"] = list(self.derived_from)
        out.update(self.extra)
        return out

    @classmethod
    def read(cls, group: h5py.Group) -> AnnotationHeader:
        attrs = group.attrs
        kind = as_str(require_attr(group, "kind", code="E412"))
        return cls(
            kind=kind,
            task=as_str(attrs["task"])
            if "task" in attrs
            else DEFAULT_TASK_FOR_KIND[kind],
            grid=as_str(attrs["grid"]) if "grid" in attrs else None,
            timepoints=as_str_tuple(attrs["timepoints"])
            if "timepoints" in attrs
            else None,
            space=as_str(attrs["space"]) if "space" in attrs else None,
            frame_uid=as_str(attrs["frame_uid"]) if "frame_uid" in attrs else None,
            class_ids=tuple(int(c) for c in np.atleast_1d(attrs["class_ids"]))
            if "class_ids" in attrs
            else (),
            annotated_class_ids=tuple(
                int(c) for c in np.atleast_1d(attrs["annotated_class_ids"])
            )
            if "annotated_class_ids" in attrs
            else (),
            closure=as_str(attrs["closure"]) if "closure" in attrs else "explicit",
            ignore_id=as_int(attrs["ignore_id"]) if "ignore_id" in attrs else IGNORE_ID,
            ignore_mask=as_str(attrs["ignore_mask"])
            if "ignore_mask" in attrs
            else None,
            prov=as_str(attrs["prov"]) if "prov" in attrs else None,
            quality=as_str(attrs["quality"]) if "quality" in attrs else None,
            derived_from=as_str_tuple(attrs["derived_from"])
            if "derived_from" in attrs
            else (),
        )


@dataclass(frozen=True, slots=True)
class Instance:
    """One physical object: a box, a class, an id and optionally a cropped mask.

    ``instance_id`` is **sample-scoped**: the same lesion observed at several
    timepoints reuses its id, so lesion tracking is a join on this field rather
    than an additional structure.
    """

    index: int
    instance_id: int
    class_id: int
    box: npt.NDArray[np.float32]
    mask: npt.NDArray[np.bool_] | None = None
    score: float | None = None

    @property
    def slices(self) -> tuple[slice, ...]:
        from medh5.geometry.affine import box_to_slices

        return box_to_slices(self.box)

    @property
    def voxel_count(self) -> int:
        if self.mask is None:
            return int(np.prod([s.stop - s.start for s in self.slices], dtype=np.int64))
        return int(self.mask.sum())

    def __repr__(self) -> str:
        return (
            f"Instance(id={self.instance_id}, class={self.class_id}, "
            f"box={self.box.tolist()})"
        )


class Annotation(ABC):
    """Base class for every annotation kind."""

    __slots__ = ("_grids", "_label_set", "ann_id", "group", "header")

    def __init__(
        self,
        ann_id: str,
        group: h5py.Group,
        header: AnnotationHeader,
        grids: Mapping[str, Grid] | None = None,
        label_set: LabelSet | None = None,
    ) -> None:
        self.ann_id = ann_id
        self.group = group
        self.header = header
        self._grids = grids or {}
        self._label_set = label_set

    # -- header passthrough -----------------------------------------------

    @property
    def kind(self) -> str:
        return self.header.kind

    @property
    def task(self) -> str:
        return self.header.task

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self.header.class_ids

    @property
    def annotated_class_ids(self) -> tuple[int, ...]:
        return self.header.annotated_class_ids

    @property
    def closure(self) -> str:
        return self.header.closure

    @property
    def ignore_id(self) -> int:
        return self.header.ignore_id

    @property
    def prov(self) -> str | None:
        return self.header.prov

    @property
    def quality_key(self) -> str | None:
        return self.header.quality

    @property
    def label_set(self) -> LabelSet | None:
        return self._label_set

    @property
    def grid_id(self) -> str | None:
        return self.header.grid

    @property
    def grid(self) -> Grid:
        gid = self.header.grid
        if gid is None:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r} of kind {self.kind!r} has no grid"
            )
        try:
            return self._grids[gid]
        except KeyError:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r} names grid {gid!r}, which does not exist",
                code="E101",
            ) from None

    @property
    def timepoints(self) -> tuple[str, ...]:
        """Timepoints this annotation pertains to, inherited from the grid (§3.7).

        An annotation declares ``timepoints`` only when it spans them --- a
        response assessment, a change label.  Everything else inherits, so time
        is stated once, on the grid --- and a grid that names none in a
        single-timepoint sample belongs to that timepoint, which
        :attr:`Sample.grids` resolves before the grids reach here.
        """
        if self.header.timepoints is not None:
            return self.header.timepoints
        gid = self.header.grid
        if gid is not None and gid in self._grids:
            tp = self._grids[gid].timepoint
            return (tp,) if tp else ()
        return ()

    # -- coverage ----------------------------------------------------------

    def is_annotated(self, class_key: int | str) -> bool:
        """Whether the annotator committed to finding this class (§11.3).

        ``False`` means the class's absence carries **no information**: it may be
        present in the patient and simply not looked for.  Training code must
        consult this before treating ``0`` as a negative.
        """
        return self.resolve_class(class_key) in self.header.annotated_class_ids

    @property
    def is_fully_covered(self) -> bool:
        return set(self.header.annotated_class_ids) == set(self.header.class_ids)

    @property
    def has_ignore_region(self) -> bool:
        return self.header.ignore_mask is not None or self._encodes_ignore()

    def _encodes_ignore(self) -> bool:
        return False

    def resolve_class(self, class_key: int | str) -> int:
        """Resolve a class id or key to an id, using the label set when needed."""
        if isinstance(class_key, (int, np.integer)):
            return int(class_key)
        if self._label_set is None:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: cannot resolve class name "
                f"{class_key!r} without a label set"
            )
        return self._label_set[class_key].id

    def resolve_classes(self, keys: Sequence[int | str] | None) -> tuple[int, ...]:
        """Resolve requested class keys to ids, refusing the reserved ignore id.

        ``65535`` is not a class --- §5.2 says it **MUST NOT** appear in
        ``classes`` --- so nothing that returns per-class planes can answer for
        it.  What every encoding *could* do is return an all-zero plane, which is
        indistinguishable from a class examined and found absent.  Documentation
        shipped ``dense([65535])`` as the way to read the ignore region on the
        strength of that shape; under ``layers`` it silently yields nothing and
        the ignored voxels go into the loss.  Refusing here covers every encoding,
        because each one's ``dense`` resolves its classes through this --- and
        that includes ``mask``, which has no classes to resolve and so calls it
        for the check alone.  ``mask`` is the encoding that proves the placement:
        it overrides ``dense`` and used to drop the argument, so the guard reached
        five encodings of six while this paragraph claimed all six.

        :meth:`VoxelAnnotation.ignore_mask` reads the region where the encoding
        carries it in band; ``header.ignore_mask`` names the `mask` annotation
        holding it otherwise.
        """
        if keys is None:
            return self.header.class_ids
        ids = tuple(self.resolve_class(k) for k in keys)
        if IGNORE_ID in ids:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: {IGNORE_ID} is the reserved ignore "
                "id, not a class, so no plane can be returned for it; read the "
                "ignore region with `ignore_mask()` where the encoding carries it "
                "in band, or through the `mask` annotation named by "
                "`header.ignore_mask`",
                code="E404",
            )
        return ids

    @property
    def classes(self) -> tuple[LabelClass, ...]:
        """Resolved class entries; empty when the label set is unavailable."""
        if self._label_set is None:
            return ()
        return tuple(
            c for c in (self._label_set.get(i) for i in self.header.class_ids) if c
        )

    @property
    def annotated_classes(self) -> tuple[LabelClass, ...]:
        if self._label_set is None:
            return ()
        return tuple(
            c
            for c in (self._label_set.get(i) for i in self.header.annotated_class_ids)
            if c
        )

    def class_key(self, class_id: int) -> str:
        cls_ = self._label_set.get(class_id) if self._label_set else None
        return cls_.key if cls_ else str(class_id)

    # -- the contract ------------------------------------------------------

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """JSON-safe description for ``medh5 info``."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.ann_id!r}, kind={self.kind!r}, "
            f"{len(self.class_ids)} classes)"
        )


class VoxelAnnotation(Annotation):
    """An annotation defined on a grid's voxels (spec §7).

    Subclasses implement :meth:`_dense_class`; everything else --- ``contains``,
    multi-class ``dense``, ``labelmap`` flattening, voxel counts --- is derived
    once here so the five encodings cannot drift apart in behaviour.
    """

    __slots__ = ()

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return self.grid.spatial_shape

    def _roi(self, roi: Sequence[slice] | None) -> tuple[slice, ...]:
        shape = self.spatial_shape
        if roi is None:
            return tuple(slice(0, n) for n in shape)
        roi_t = tuple(roi)
        if len(roi_t) != len(shape):
            raise MEDH5ValidationError(
                f"roi has {len(roi_t)} axes; grid has {len(shape)} spatial axes"
            )
        return tuple(
            slice(
                0 if s.start is None else int(s.start),
                int(n) if s.stop is None else int(s.stop),
            )
            for s, n in zip(roi_t, shape, strict=True)
        )

    @staticmethod
    def _roi_shape(roi: Sequence[slice]) -> tuple[int, ...]:
        return tuple(max(0, s.stop - s.start) for s in roi)

    @abstractmethod
    def _dense_class(
        self, class_id: int, roi: tuple[slice, ...]
    ) -> npt.NDArray[np.bool_]:
        """Boolean occupancy of one class over *roi*."""

    def dense(
        self,
        classes: Sequence[int | str] | None = None,
        roi: Sequence[slice] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """``(C, *roi_shape)`` boolean occupancy, one plane per requested class.

        Asking for the reserved ignore id is refused rather than answered.  It is
        not a class --- §5.2 says it **MUST NOT** appear in ``classes`` --- so no
        encoding can return a plane for it, and every encoding could none the
        less return an all-zero one, which is indistinguishable from a class that
        was examined and found absent.  Documentation shipped `dense([65535])` as
        the way to read the ignore region on the strength of that shape: under
        `layers` it silently yields nothing, and the ignored voxels go into the
        loss.  :meth:`ignore_mask` and ``header.ignore_mask`` are the answer.
        """
        ids = self.resolve_classes(classes)
        window = self._roi(roi)
        out = np.zeros((len(ids), *self._roi_shape(window)), dtype=bool)
        for i, class_id in enumerate(ids):
            out[i] = self._dense_class(class_id, window)
        return out

    def contains(self, class_key: int | str, voxel: Sequence[int]) -> bool:
        """The uniform predicate of §7.6: is *class* present at *voxel*?"""
        class_id = self.resolve_class(class_key)
        window = tuple(slice(int(v), int(v) + 1) for v in voxel)
        return bool(self._dense_class(class_id, window).reshape(-1)[0])

    def labelmap(
        self,
        roi: Sequence[slice] | None = None,
        priority: Sequence[int | str] | None = None,
        *,
        dtype: npt.DTypeLike = np.uint16,
    ) -> npt.NDArray[Any]:
        """Flatten to one integer volume, breaking overlap ties **explicitly**.

        *priority* is ordered highest-precedence first; classes it omits are
        painted first, in ``class_ids`` order.  Flattening an overlapping
        annotation is lossy, and which class survives is the caller's decision,
        not the format's --- so when voxels are actually lost and the caller
        expressed no preference, this warns rather than picking silently.

        The warning is the whole point.  ``layers``, ``bitmask`` and ``probmap``
        exist because §7.0 classes overlap, and three callers inside this package
        (NIfTI export, the torch loader's ``labelmap`` format, and the MONAI
        bridge) flatten on the way out.  Each one was quietly deleting the
        overlap region --- a lesion inside a liver stopped being liver --- and
        nothing in the output said so.
        """
        window = self._roi(roi)
        out = np.zeros(self._roi_shape(window), dtype=dtype)
        ordered = list(self.class_ids)
        if priority is not None:
            ranked = list(self.resolve_classes(priority))
            rest = [c for c in ordered if c not in ranked]
            ordered = rest + list(reversed(ranked))
        overwritten = 0
        for class_id in ordered:
            mask = self._dense_class(class_id, window)
            if priority is None:
                overwritten += int(np.count_nonzero(mask & (out != 0)))
            out[mask] = class_id
        if overwritten:
            warnings.warn(
                f"labelmap() flattened {overwritten} overlapping voxel(s) in "
                f"{self.ann_id!r}: classes {list(ordered)} overlap and one integer "
                "volume cannot hold both, so later classes overwrote earlier "
                "ones. Pass priority=[...] to choose which class survives, or "
                "use dense()/contains() to keep the overlap.",
                stacklevel=2,
            )
        return out

    def voxel_counts(
        self, classes: Sequence[int | str] | None = None
    ) -> dict[int, int]:
        """Foreground voxel count per class, over the whole grid."""
        ids = self.resolve_classes(classes)
        window = self._roi(None)
        return {
            class_id: int(self._dense_class(class_id, window).sum()) for class_id in ids
        }

    def class_bboxes(
        self, classes: Sequence[int | str] | None = None
    ) -> dict[int, npt.NDArray[np.float32] | None]:
        """Tight ``(S, 2)`` index-space bounds per class; ``None`` when empty."""
        from medh5.geometry.affine import slices_to_box

        ids = self.resolve_classes(classes)
        window = self._roi(None)
        out: dict[int, npt.NDArray[np.float32] | None] = {}
        for class_id in ids:
            mask = self._dense_class(class_id, window)
            if not mask.any():
                out[class_id] = None
                continue
            slices = []
            for axis in range(mask.ndim):
                axes = tuple(i for i in range(mask.ndim) if i != axis)
                present = np.flatnonzero(mask.any(axis=axes))
                slices.append(slice(int(present[0]), int(present[-1]) + 1))
            out[class_id] = slices_to_box(slices)
        return out

    def instances(self) -> Iterator[Instance]:
        """Iterate objects, where the encoding carries object identity."""
        raise MEDH5ValidationError(
            f"annotation {self.ann_id!r} of kind {self.kind!r} does not carry "
            "instance identity, and it cannot be recovered from a dense "
            "encoding: transcoding to `instances` would merge every object of a "
            "class into one and mint an id that belongs to none of them (§7.4). "
            "Re-derive the objects from whatever source had them."
        )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.ann_id,
            "kind": self.kind,
            "task": self.task,
            "grid": self.grid_id,
            "timepoints": list(self.timepoints),
            "classes": len(self.class_ids),
            "annotated_classes": len(self.annotated_class_ids),
            "fully_covered": self.is_fully_covered,
            "closure": self.closure,
            "quality": self.quality_key,
            "prov": self.prov,
        }


def readers() -> dict[str, Any]:
    """``kind`` -> reader class, assembled lazily to keep the imports acyclic."""
    from medh5.annotations.classification import ClassificationAnnotation
    from medh5.annotations.geometric import GEOMETRIC_READERS
    from medh5.annotations.voxel import READERS as VOXEL_READERS

    return {
        **VOXEL_READERS,
        **GEOMETRIC_READERS,
        "classification": ClassificationAnnotation,
    }


def open_annotation(
    ann_id: str,
    group: h5py.Group,
    grids: Mapping[str, Grid] | None = None,
    label_set: LabelSet | None = None,
) -> Annotation:
    """Open an annotation group as the class matching its ``kind``."""
    header = AnnotationHeader.read(group)
    reader = readers().get(header.kind)
    if reader is None:
        raise MEDH5ValidationError(
            f"annotation {ann_id!r}: kind {header.kind!r} is not implemented yet",
            code="E401",
        )
    opened: Annotation = reader(ann_id, group, header, grids, label_set)
    return opened


__all__ = [
    "ANNOTATION_KINDS",
    "DEFAULT_TASK_FOR_KIND",
    "GEOMETRIC_KINDS",
    "RESERVED_KINDS",
    "SPEC_ANNOTATION_ATTRS",
    "TASKS",
    "VOXEL_KINDS",
    "Annotation",
    "AnnotationHeader",
    "Instance",
    "VoxelAnnotation",
    "instance_id_dtype",
    "open_annotation",
    "readers",
]
