"""Measuring agreement between two annotations (spec §11.2).

`quality.agreement` is the field that makes a second rater worth storing: two
annotations of the same structure are only useful if the disagreement between
them is quantified.  This module computes those numbers from the annotations
themselves, so an :class:`~medh5.curation.quality.Agreement` record in a file is
reproducible from the file rather than copied in from a spreadsheet nobody kept.

Three decisions are deliberate:

* **Only shared, examined classes are scored.**  A class one rater never looked
  at (§11.3) contributes no measurement --- averaging a Dice of 0 for it would
  report disagreement where there was no comparison.  Those classes come back
  under ``skipped`` instead of quietly dragging the mean down.
* **Empty-on-both is not a disagreement.**  Dice is undefined when both masks
  are empty; scoring it as 0 punishes raters for agreeing that a structure is
  absent, and scoring it as 1 inflates every partially-labelled cohort.  It is
  reported as ``None`` and excluded from the mean.
* **Instances match by id first.**  Where both annotations carry ``instance_id``
  (§7.4) the correspondence is already stated and IoU matching would only
  second-guess it.  Greedy IoU matching is the fallback for annotations that
  have no shared ids, and it says so in the result.

The first two hold for objects as for voxels.  An object whose class one side
never examined is left out, not counted as missed: identical work scored 0.667
while the class nobody asked the second rater to look for counted against them.
And when neither side found anything, object F1 is undefined rather than 0 ---
the value is ``None``, and :meth:`~InstanceAgreement.to_record` refuses to write
a number that was never measured.

A record is keyed the way §11.2 keys it: ``per_class`` by class **id**.  Keyed
by class name, as the reports print it, no record this module produced could be
stored in the file it described --- the schema admits only numeric keys.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5.curation.quality import Agreement
from medh5.errors import MEDH5ValidationError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.annotations.base import Annotation, VoxelAnnotation

DEFAULT_IOU = 0.5


def dice(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float | None:
    """Sørensen--Dice, or ``None`` when both masks are empty."""
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return None
    return 2.0 * float(np.count_nonzero(a & b)) / total


def iou(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float | None:
    """Intersection over union, or ``None`` when both masks are empty."""
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return None
    return float(np.count_nonzero(a & b)) / union


def box_iou(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """IoU of two ``(S, 2)`` boxes in the same space."""
    first = np.asarray(a, dtype=np.float64)
    second = np.asarray(b, dtype=np.float64)
    lo = np.maximum(first[:, 0], second[:, 0])
    hi = np.minimum(first[:, 1], second[:, 1])
    overlap = float(np.prod(np.clip(hi - lo, 0.0, None)))
    if overlap == 0.0:
        return 0.0
    volume_a = float(np.prod(first[:, 1] - first[:, 0]))
    volume_b = float(np.prod(second[:, 1] - second[:, 0]))
    union = volume_a + volume_b - overlap
    return overlap / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class VoxelAgreement:
    """Per-class agreement between two voxel annotations."""

    metric: str
    per_class: Mapping[str, float]
    skipped: tuple[str, ...] = ()
    """Classes not scored: absent from one side's coverage, or empty in both."""
    against: str | None = None
    class_ids: Mapping[str, int] = field(default_factory=dict)
    """``per_class`` key -> class id, which is what a record is keyed by."""

    @property
    def value(self) -> float | None:
        """Mean over the classes that were actually comparable.

        ``None`` when no class was: 0 would report total disagreement for a
        comparison that measured nothing.
        """
        values = list(self.per_class.values())
        return float(np.mean(values)) if values else None

    def to_record(self) -> Agreement:
        """The :class:`Agreement` a ``quality`` record stores (§11.2)."""
        value = _measured(self.value, self.skipped)
        return Agreement(
            metric=self.metric,
            value=value,
            against=self.against,
            per_class={
                str(self._class_id(key)): score for key, score in self.per_class.items()
            },
        )

    def _class_id(self, key: str) -> int:
        if key in self.class_ids:
            return int(self.class_ids[key])
        if key.isdigit():
            return int(key)
        raise MEDH5ValidationError(
            f"per-class score {key!r} names no class id; a `quality.agreement` "
            "record is keyed by class id (§11.2)"
        )

    def to_json(self) -> dict[str, Any]:
        """The report, which states an undefined comparison rather than refusing.

        Built directly, not from :meth:`to_record`: the record refuses a value
        nobody measured, and a report of that comparison is exactly what a
        reader needs to see.  ``per_class`` is keyed as the attribute is, by
        class key; the record keys it by id.
        """
        out: dict[str, Any] = {"metric": self.metric, "value": self.value}
        if self.against is not None:
            out["against"] = self.against
        out["per_class"] = dict(self.per_class)
        out["skipped"] = list(self.skipped)
        out["compared"] = len(self.per_class)
        return out


@dataclass(frozen=True, slots=True)
class InstanceAgreement:
    """Object-level agreement: what matched, what did not, and how."""

    matched: tuple[tuple[int, int, float], ...] = ()
    """``(index in a, index in b, IoU)`` for every matched pair."""
    only_in_a: tuple[int, ...] = ()
    only_in_b: tuple[int, ...] = ()
    matched_by: str = "instance_id"
    threshold: float = DEFAULT_IOU
    against: str | None = None
    class_mismatches: tuple[tuple[int, int, int], ...] = ()
    """``(instance_id, class in a, class in b)`` --- matched but classed apart."""
    skipped: tuple[str, ...] = ()
    """Classes whose objects were left out: not examined by both sides."""

    @property
    def value(self) -> float | None:
        """F1 over objects: the number a detection reviewer actually wants.

        ``None`` when neither side has an object to compare.  Two raters who
        both found nothing agree; scoring that 0 punished them for it, and the
        record it wrote said so in the file.
        """
        true_positives = len(self.matched)
        if true_positives + len(self.only_in_a) + len(self.only_in_b) == 0:
            return None
        if true_positives == 0:
            return 0.0
        precision = true_positives / (true_positives + len(self.only_in_b))
        recall = true_positives / (true_positives + len(self.only_in_a))
        return 2 * precision * recall / (precision + recall)

    @property
    def mean_iou(self) -> float | None:
        """Mean IoU of the matched pairs; ``None`` when nothing matched."""
        return float(np.mean([m[2] for m in self.matched])) if self.matched else None

    def to_record(self) -> Agreement:
        """The :class:`Agreement` a ``quality`` record stores (§11.2).

        No ``per_class``: that map is keyed by class id, and the mean IoU this
        used to put there under the key ``mean_iou`` made every record fail the
        schema.
        """
        return Agreement(
            metric="object_f1",
            value=_measured(self.value, self.skipped),
            against=self.against,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": "object_f1",
            "value": self.value,
            "mean_iou": self.mean_iou,
            "matched": [list(m) for m in self.matched],
            "only_in_a": list(self.only_in_a),
            "only_in_b": list(self.only_in_b),
            "matched_by": self.matched_by,
            "threshold": self.threshold,
            "class_mismatches": [list(m) for m in self.class_mismatches],
            "skipped": list(self.skipped),
            "against": self.against,
        }


def _measured(value: float | None, skipped: Sequence[str]) -> float:
    """A value worth recording, or a refusal that says why there is none."""
    if value is None:
        reason = f" (not scored: {', '.join(skipped)})" if skipped else ""
        raise MEDH5ValidationError(
            "nothing was comparable, so there is no agreement to record"
            f"{reason}; agreement on an empty comparison is undefined, and "
            "recording 0 would report a disagreement nobody measured"
        )
    return value


@dataclass(slots=True)
class _Pair:
    classes: list[int] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def compare_voxel(
    a: VoxelAnnotation,
    b: VoxelAnnotation,
    *,
    metric: str = "dice",
    classes: Sequence[int | str] | None = None,
) -> VoxelAgreement:
    """Per-class Dice or IoU between two voxel annotations on the same grid."""
    if metric not in ("dice", "iou"):
        raise MEDH5ValidationError(f"unknown agreement metric {metric!r}")
    if a.grid_id != b.grid_id:
        raise MEDH5ValidationError(
            f"annotations {a.ann_id!r} and {b.ann_id!r} are on different grids "
            f"({a.grid_id!r} vs {b.grid_id!r}); resample before comparing",
            code="E101",
        )
    pair = _classes_to_compare(a, b, classes)
    scorer = dice if metric == "dice" else iou
    per_class: dict[str, float] = {}
    ids: dict[str, int] = {}
    skipped = list(pair.skipped)
    for class_id in pair.classes:
        left = a.dense([class_id])[0]
        right = b.dense([class_id])[0]
        score = scorer(left, right)
        key = a.class_key(class_id)
        if score is None:
            skipped.append(f"{key} (empty in both)")
            continue
        per_class[key] = score
        ids[key] = int(class_id)
    return VoxelAgreement(
        metric=metric,
        per_class=per_class,
        skipped=tuple(skipped),
        against=f"annotations/{b.ann_id}",
        class_ids=ids,
    )


def _classes_to_compare(
    a: Annotation, b: Annotation, classes: Sequence[int | str] | None
) -> _Pair:
    """Classes both sides committed to finding, in a stable order (§11.3)."""
    out = _Pair()
    if classes is not None:
        wanted: Iterable[int] = [a.resolve_class(c) for c in classes]
    else:
        wanted = sorted(set(a.class_ids) | set(b.class_ids))
    for class_id in wanted:
        examined = a.is_annotated(class_id) and b.is_annotated(class_id)
        if not examined:
            out.skipped.append(f"{a.class_key(class_id)} (not examined by both)")
            continue
        out.classes.append(class_id)
    return out


OBJECT_KINDS = ("instances", "boxes")
"""Kinds whose objects carry an axis-aligned box that IoU can be taken over."""


def compare_instances(
    a: Annotation,
    b: Annotation,
    *,
    threshold: float = DEFAULT_IOU,
    classes: Sequence[int | str] | None = None,
) -> InstanceAgreement:
    """Object-level agreement between two instance-carrying annotations.

    **A miss counts only where the other side looked** (§11.3).  An object left
    unmatched is a disagreement if the other annotation examined its class, and
    no measurement at all otherwise --- so it is dropped from ``only_in_a`` or
    ``only_in_b`` and its class is listed under ``skipped``.  A matched pair
    always counts, class mismatch included: both sides found the object.

    The two must also share a coordinate system: the same grid for ``index``
    boxes, the same frame for ``world`` boxes.  Boxes on a 4 mm and a 1 mm grid
    compared in raw index units overlap where the anatomy does not.
    """
    for ann in (a, b):
        if ann.kind not in OBJECT_KINDS:
            raise MEDH5ValidationError(
                f"annotation {ann.ann_id!r} is {ann.kind!r}; object agreement needs "
                f"objects with boxes ({', '.join(OBJECT_KINDS)}). Compare voxel "
                "annotations with `compare_voxel`"
            )
    _check_same_space(a, b)
    pair = _classes_to_compare(a, b, classes)
    left = list(_objects(a))
    right = list(_objects(b))
    if classes is not None:
        asked = {a.resolve_class(c) for c in classes}
        left = [o for o in left if o.class_id in asked]
        right = [o for o in right if o.class_id in asked]
    ids_a = {o.instance_id for o in left}
    ids_b = {o.instance_id for o in right}
    shared = ids_a & ids_b
    if shared and _declares_ids(a) and _declares_ids(b):
        result = _match_by_id(left, right, shared, threshold, b.ann_id)
    else:
        result = _match_by_iou(left, right, threshold, b.ann_id)
    class_a = {o.index: o.class_id for o in left}
    class_b = {o.index: o.class_id for o in right}
    looked_a = set(a.annotated_class_ids)
    looked_b = set(b.annotated_class_ids)
    return replace(
        result,
        only_in_a=tuple(i for i in result.only_in_a if class_a[i] in looked_b),
        only_in_b=tuple(j for j in result.only_in_b if class_b[j] in looked_a),
        skipped=tuple(pair.skipped),
    )


def _space_of(ann: Annotation) -> str:
    return str(ann.header.space or "index")


def _frame_of(ann: Annotation) -> str | None:
    if ann.header.frame_uid is not None:
        return ann.header.frame_uid
    try:
        return ann.grid.frame_uid
    except MEDH5ValidationError:
        return None


def _check_same_space(a: Annotation, b: Annotation) -> None:
    """Refuse two coordinate systems that cannot be compared number for number."""
    space_a, space_b = _space_of(a), _space_of(b)
    if space_a != space_b:
        raise MEDH5ValidationError(
            f"annotations {a.ann_id!r} and {b.ann_id!r} store boxes in different "
            f"spaces ({space_a!r} vs {space_b!r}); convert one before comparing",
            code="E414",
        )
    if space_a == "world":
        # World boxes compare within one frame; one grid is one frame.  Two
        # annotations without a grid would pass a grid test as `None == None`.
        if a.grid_id is not None and a.grid_id == b.grid_id:
            return
        frame_a, frame_b = _frame_of(a), _frame_of(b)
        if frame_a is not None and frame_a == frame_b:
            return
        raise MEDH5ValidationError(
            f"annotations {a.ann_id!r} and {b.ann_id!r} are in frames "
            f"{frame_a!r} and {frame_b!r}; a transform is required to relate them",
            code="E414",
        )
    if a.grid_id != b.grid_id:
        raise MEDH5ValidationError(
            f"annotations {a.ann_id!r} and {b.ann_id!r} are on different grids "
            f"({a.grid_id!r} vs {b.grid_id!r}); their index coordinates count "
            "different voxels, so resample or convert before comparing",
            code="E101",
        )


def _declares_ids(ann: Annotation) -> bool:
    from medh5.curation.tracking import carries_instance_ids

    return carries_instance_ids(ann)


def _objects(ann: Annotation) -> Iterable[Any]:
    from medh5.curation.tracking import _objects as objects_of

    return list(objects_of(ann))


def _match_by_id(
    left: Sequence[Any],
    right: Sequence[Any],
    shared: set[int],
    threshold: float,
    against: str,
) -> InstanceAgreement:
    by_id_b = {o.instance_id: o for o in right}
    matched: list[tuple[int, int, float]] = []
    mismatches: list[tuple[int, int, int]] = []
    for obj in left:
        if obj.instance_id not in shared:
            continue
        other = by_id_b[obj.instance_id]
        matched.append((obj.index, other.index, box_iou(obj.box, other.box)))
        if obj.class_id != other.class_id:
            mismatches.append((obj.instance_id, obj.class_id, other.class_id))
    return InstanceAgreement(
        matched=tuple(matched),
        only_in_a=tuple(o.index for o in left if o.instance_id not in shared),
        only_in_b=tuple(o.index for o in right if o.instance_id not in shared),
        matched_by="instance_id",
        threshold=threshold,
        against=f"annotations/{against}",
        class_mismatches=tuple(mismatches),
    )


def _match_by_iou(
    left: Sequence[Any],
    right: Sequence[Any],
    threshold: float,
    against: str,
) -> InstanceAgreement:
    """Greedy highest-IoU-first matching, one object to at most one object."""
    candidates: list[tuple[float, int, int]] = []
    for i, obj in enumerate(left):
        for j, other in enumerate(right):
            if obj.class_id != other.class_id:
                continue
            overlap = box_iou(obj.box, other.box)
            if overlap >= threshold:
                candidates.append((overlap, i, j))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for overlap, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((left[i].index, right[j].index, overlap))
    return InstanceAgreement(
        matched=tuple(sorted(matched)),
        only_in_a=tuple(o.index for i, o in enumerate(left) if i not in used_a),
        only_in_b=tuple(o.index for j, o in enumerate(right) if j not in used_b),
        matched_by="iou",
        threshold=threshold,
        against=f"annotations/{against}",
    )


def compare(
    a: Annotation,
    b: Annotation,
    *,
    metric: str | None = None,
    threshold: float | None = None,
    classes: Sequence[int | str] | None = None,
) -> VoxelAgreement | InstanceAgreement:
    """Compare two annotations, choosing the comparison their kinds support.

    Two object-carrying annotations (``instances``, ``boxes``) are compared
    object by object, at an IoU *threshold*; two voxel annotations class by
    class, by *metric*.  An argument the chosen comparison cannot use is
    refused rather than ignored, and so is a pair with no common comparison.
    """
    from medh5.annotations.base import VoxelAnnotation as _Voxel

    if a.kind in OBJECT_KINDS and b.kind in OBJECT_KINDS:
        if metric is not None:
            raise MEDH5ValidationError(
                f"metric {metric!r} scores voxels; {a.ann_id!r} and {b.ann_id!r} "
                "are compared object by object, as F1 at an IoU threshold"
            )
        return compare_instances(
            a,
            b,
            threshold=DEFAULT_IOU if threshold is None else threshold,
            classes=classes,
        )
    if isinstance(a, _Voxel) and isinstance(b, _Voxel):
        if threshold is not None:
            raise MEDH5ValidationError(
                f"threshold matches objects; {a.ann_id!r} and {b.ann_id!r} are "
                "compared voxel by voxel"
            )
        return compare_voxel(a, b, metric=metric or "dice", classes=classes)
    raise MEDH5ValidationError(
        f"cannot compare {a.kind!r} with {b.kind!r}: transcode them to a common "
        "kind first"
    )


__all__ = [
    "DEFAULT_IOU",
    "OBJECT_KINDS",
    "InstanceAgreement",
    "VoxelAgreement",
    "box_iou",
    "compare",
    "compare_instances",
    "compare_voxel",
    "dice",
    "iou",
]
