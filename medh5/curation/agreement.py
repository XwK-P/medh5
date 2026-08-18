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
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
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

    @property
    def value(self) -> float:
        """Mean over the classes that were actually comparable."""
        values = list(self.per_class.values())
        return float(np.mean(values)) if values else 0.0

    def to_record(self) -> Agreement:
        """The :class:`Agreement` a ``quality`` record stores (§11.2)."""
        return Agreement(
            metric=self.metric,
            value=self.value,
            against=self.against,
            per_class=dict(self.per_class),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            **self.to_record().to_json(),
            "skipped": list(self.skipped),
            "compared": len(self.per_class),
        }


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

    @property
    def value(self) -> float:
        """F1 over objects: the number a detection reviewer actually wants."""
        true_positives = len(self.matched)
        if true_positives == 0:
            return 0.0
        precision = true_positives / (true_positives + len(self.only_in_b))
        recall = true_positives / (true_positives + len(self.only_in_a))
        return 2 * precision * recall / (precision + recall)

    @property
    def mean_iou(self) -> float:
        return float(np.mean([m[2] for m in self.matched])) if self.matched else 0.0

    def to_record(self) -> Agreement:
        return Agreement(
            metric="object_f1",
            value=self.value,
            against=self.against,
            per_class={"mean_iou": self.mean_iou},
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
            "against": self.against,
        }


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
    return VoxelAgreement(
        metric=metric,
        per_class=per_class,
        skipped=tuple(skipped),
        against=f"annotations/{b.ann_id}",
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


def compare_instances(
    a: Annotation,
    b: Annotation,
    *,
    threshold: float = DEFAULT_IOU,
) -> InstanceAgreement:
    """Object-level agreement between two instance-carrying annotations."""
    left = list(_objects(a))
    right = list(_objects(b))
    ids_a = {o.instance_id for o in left}
    ids_b = {o.instance_id for o in right}
    shared = ids_a & ids_b
    if shared and _declares_ids(a) and _declares_ids(b):
        return _match_by_id(left, right, shared, threshold, b.ann_id)
    return _match_by_iou(left, right, threshold, b.ann_id)


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
    a: Annotation, b: Annotation, **kwargs: Any
) -> VoxelAgreement | InstanceAgreement:
    """Compare two annotations, choosing the comparison their kinds support."""
    from medh5.annotations.base import VoxelAnnotation as _Voxel

    if a.kind == "instances" or b.kind == "instances" or not isinstance(a, _Voxel):
        return compare_instances(a, b, **kwargs)
    if not isinstance(b, _Voxel):
        raise MEDH5ValidationError(
            f"cannot compare {a.kind!r} with {b.kind!r}: transcode them to a "
            "common kind first"
        )
    return compare_voxel(a, b, **kwargs)


__all__ = [
    "DEFAULT_IOU",
    "InstanceAgreement",
    "VoxelAgreement",
    "box_iou",
    "compare",
    "compare_instances",
    "compare_voxel",
    "dice",
    "iou",
]
