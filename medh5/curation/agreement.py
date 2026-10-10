"""Measuring agreement between two annotations (spec §11.2).

`quality.agreement` is the field that makes a second rater worth storing: two
annotations of the same structure are only useful if the disagreement between
them is quantified.  This module computes those numbers from the annotations
themselves, so an :class:`~medh5.curation.Agreement` record in a file is
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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5._core import Agreement

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.annotations.base import Annotation, VoxelAnnotation

DEFAULT_IOU: float = _core.AGREEMENT_DEFAULT_IOU

OBJECT_KINDS: tuple[str, ...] = _core.AGREEMENT_OBJECT_KINDS
"""Kinds whose objects carry an axis-aligned box that IoU can be taken over."""


def dice(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float | None:
    """Sørensen--Dice, or ``None`` when both masks are empty."""
    result: float | None = _core.agreement_dice(a, b)
    return result


def iou(a: npt.NDArray[np.bool_], b: npt.NDArray[np.bool_]) -> float | None:
    """Intersection over union, or ``None`` when both masks are empty."""
    result: float | None = _core.agreement_iou(a, b)
    return result


def box_iou(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """IoU of two ``(S, 2)`` boxes in the same space."""
    result: float = _core.agreement_box_iou(a, b)
    return result


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
        result: float | None = _core.agreement_voxel_value(self)
        return result

    def to_record(self) -> Agreement:
        """The :class:`Agreement` a ``quality`` record stores (§11.2).

        Keyed by class id, as the schema keys ``per_class``; refused when
        nothing was comparable, because 0 would record a disagreement nobody
        measured.
        """
        return _core.agreement_voxel_record(self)

    def to_json(self) -> dict[str, Any]:
        """The report, which states an undefined comparison rather than refusing.

        Built directly, not from :meth:`to_record`: the record refuses a value
        nobody measured, and a report of that comparison is exactly what a
        reader needs to see.  ``per_class`` is keyed as the attribute is, by
        class key; the record keys it by id.
        """
        result: dict[str, Any] = _core.agreement_voxel_json(self)
        return result


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
        result: float | None = _core.agreement_instance_value(self)
        return result

    @property
    def mean_iou(self) -> float | None:
        """Mean IoU of the matched pairs; ``None`` when nothing matched."""
        result: float | None = _core.agreement_instance_mean_iou(self)
        return result

    def to_record(self) -> Agreement:
        """The :class:`Agreement` a ``quality`` record stores (§11.2).

        No ``per_class``: that map is keyed by class id, and the mean IoU this
        used to put there under the key ``mean_iou`` made every record fail the
        schema.
        """
        return _core.agreement_instance_record(self)

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.agreement_instance_json(self)
        return result


def compare_voxel(
    a: VoxelAnnotation,
    b: VoxelAnnotation,
    *,
    metric: str = "dice",
    classes: Sequence[int | str] | None = None,
) -> VoxelAgreement:
    """Per-class Dice or IoU between two voxel annotations on the same grid."""
    return VoxelAgreement(
        **_core.agreement_compare_voxel(a, b, metric=metric, classes=classes)
    )


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
    return InstanceAgreement(
        **_core.agreement_compare_instances(a, b, threshold=threshold, classes=classes)
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
    kind, fields = _core.agreement_compare(
        a, b, metric=metric, threshold=threshold, classes=classes
    )
    if kind == "voxel":
        return VoxelAgreement(**fields)
    return InstanceAgreement(**fields)


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
