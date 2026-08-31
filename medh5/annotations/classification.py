"""Classification annotations (spec §9).

Labels at sample, timepoint, grid, ROI, slice or instance scope, including the
**change** labels that make longitudinal work expressible: a response category is
an ordinary classification with ``scope = "sample"`` and an explicit
``timepoints`` list naming the visits compared, so ``["tp0","tp2"]`` and
``["tp1","tp2"]`` are distinct assessments rather than two rows that look alike.

Three semantics matter more than the layout:

* A class in ``annotated_class_ids`` with no entry is a **negative** --- looked
  for, not found.  A class outside it is **unknown**.  The difference is the
  whole of partial labelling.
* ``value = 0.0`` is an explicit negative assertion, which is not the same as
  absence: "0 of 4 raters" and "not assessed" must not collapse.
* Ordinal scales are labels, not numbers.  ``BI-RADS 4b`` is stored verbatim in
  ``scheme_values``; comparing them is a reader's problem, and the file does not
  pretend ``4b`` is a float.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5._hdf5 import as_bool, as_str, as_str_tuple, str_dtype
from medh5.annotations.base import Annotation
from medh5.annotations.payload import AnnotationPayload
from medh5.errors import MEDH5ValidationError

SCOPES = ("sample", "timepoint", "grid", "roi", "slice", "instance")


def check_scope(scope: str) -> str:
    if scope not in SCOPES:
        raise MEDH5ValidationError(
            f"classification scope {scope!r} must be one of {list(SCOPES)}", code="E412"
        )
    return scope


@dataclass(frozen=True, slots=True)
class Assertion:
    """One label assertion: a class, its value, and what it is about."""

    class_id: int
    value: float
    scope_id: int | None = None
    scheme: str | None = None
    scheme_value: str | None = None

    @property
    def is_positive(self) -> bool:
        return self.value > 0.0

    @property
    def is_negative(self) -> bool:
        return self.value == 0.0

    def __repr__(self) -> str:
        scheme = f", {self.scheme}={self.scheme_value!r}" if self.scheme else ""
        target = f" @{self.scope_id}" if self.scope_id is not None else ""
        return f"Assertion({self.class_id}={self.value:g}{target}{scheme})"


def encode_classification(
    labels: Mapping[int, float],
    *,
    scope: str = "sample",
    multilabel: bool = True,
    scope_ids: Sequence[int] | None = None,
    schemes: Sequence[str] | None = None,
    scheme_values: Sequence[str] | None = None,
) -> AnnotationPayload:
    """Pack label assertions (spec §9).

    ``multilabel = False`` means exactly one positive class per scope unit, and
    that is checked here rather than left to a validator: a single-label
    annotation carrying two positives is a curation error, and finding it at
    write time costs nothing.
    """
    check_scope(scope)
    class_ids = [int(c) for c in labels]
    values = [float(v) for v in labels.values()]
    if any(not 0.0 <= v <= 1.0 for v in values):
        raise MEDH5ValidationError(
            "classification values must lie in [0, 1]; 1.0 is a hard positive, "
            "0.0 an explicit negative",
            code="E404",
        )
    if not multilabel:
        by_unit: dict[Any, int] = {}
        units: list[Any] = (
            list(scope_ids) if scope_ids is not None else [None] * len(class_ids)
        )
        for unit, value in zip(units, values, strict=True):
            if value > 0.0:
                by_unit[unit] = by_unit.get(unit, 0) + 1
        crowded = sorted(str(k) for k, n in by_unit.items() if n > 1)
        if crowded:
            raise MEDH5ValidationError(
                f"multilabel=False allows one positive class per scope unit, but "
                f"unit(s) {crowded} carry several",
                code="E404",
            )
    datasets: dict[str, npt.NDArray[Any]] = {
        "class_ids": np.asarray(class_ids, dtype=np.uint16),
        "values": np.asarray(values, dtype=np.float32),
    }
    for name, column, dtype in (
        ("scope_ids", scope_ids, np.int64),
        ("schemes", schemes, None),
        ("scheme_values", scheme_values, None),
    ):
        if column is None:
            continue
        if len(column) != len(class_ids):
            raise MEDH5ValidationError(
                f"{name} has {len(column)} entries for {len(class_ids)} assertions",
                code="E405",
            )
        datasets[name] = (
            np.asarray(column, dtype=dtype)
            if dtype is not None
            else np.array([str(v) for v in column], dtype=str_dtype())
        )
    return AnnotationPayload(
        kind="classification",
        datasets=datasets,
        attrs={"scope": scope, "multilabel": bool(multilabel)},
        class_ids=tuple(sorted(set(class_ids))),
    )


class ClassificationAnnotation(Annotation):
    """Reader for ``kind = "classification"``."""

    __slots__ = ()

    def _dataset(self, name: str, required: bool = True) -> Any:
        if name in self.group:
            return self.group[name]
        if required:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `classification` requires a "
                f"{name!r} dataset",
                code="E410",
            )
        return None

    # -- header ------------------------------------------------------------

    @property
    def scope(self) -> str:
        value = self.group.attrs.get("scope")
        if value is None:
            raise MEDH5ValidationError(
                f"annotation {self.ann_id!r}: `classification` requires `scope`",
                code="E412",
            )
        return check_scope(as_str(value))

    @property
    def multilabel(self) -> bool:
        value = self.group.attrs.get("multilabel")
        return True if value is None else as_bool(value)

    # -- assertions --------------------------------------------------------

    @property
    def asserted_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._dataset("class_ids")[...], dtype=np.uint16)

    @property
    def values(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._dataset("values")[...], dtype=np.float32)

    @property
    def scope_ids(self) -> npt.NDArray[np.int64] | None:
        node = self._dataset("scope_ids", required=False)
        return None if node is None else np.asarray(node[...], dtype=np.int64)

    @property
    def schemes(self) -> tuple[str, ...] | None:
        node = self._dataset("schemes", required=False)
        return None if node is None else tuple(as_str(v) for v in node[...])

    @property
    def scheme_values(self) -> tuple[str, ...] | None:
        node = self._dataset("scheme_values", required=False)
        return None if node is None else tuple(as_str(v) for v in node[...])

    def assertions(self) -> Iterator[Assertion]:
        classes = self.asserted_class_ids
        values = self.values
        units = self.scope_ids
        schemes = self.schemes
        scheme_values = self.scheme_values
        for i in range(classes.shape[0]):
            yield Assertion(
                class_id=int(classes[i]),
                value=float(values[i]),
                scope_id=None if units is None else int(units[i]),
                scheme=None if schemes is None else schemes[i],
                scheme_value=None if scheme_values is None else scheme_values[i],
            )

    # -- the questions callers actually ask --------------------------------

    @property
    def labels(self) -> dict[str, float]:
        """``key -> value`` for every assertion, keyed by label-set key when known.

        One entry per class, so it is only well defined when each class is
        asserted once.  §9 makes the opposite case ordinary --- ``scope_ids`` is
        "per assertion", and ``scope = "timepoint"`` means one assertion per
        visit --- and this comprehension silently kept the last of them, while
        :meth:`value` returned the first.  The two then disagreed about the same
        file: ``state()`` answered "negative" for a class ``positives`` listed.
        Ask :meth:`assertions` or :meth:`by_scope_id` when a class is asserted
        more than once.
        """
        self._require_unambiguous("labels")
        return {
            self.class_key(int(c)): float(v)
            for c, v in zip(self.asserted_class_ids, self.values, strict=True)
        }

    def _duplicated_classes(self) -> dict[int, int]:
        """Class id -> how many assertions carry it, for the repeated ones."""
        counts: dict[int, int] = {}
        for class_id in self.asserted_class_ids:
            counts[int(class_id)] = counts.get(int(class_id), 0) + 1
        return {c: n for c, n in counts.items() if n > 1}

    def _require_unambiguous(self, what: str, target: int | None = None) -> None:
        repeated = self._duplicated_classes()
        if target is not None:
            repeated = {c: n for c, n in repeated.items() if c == target}
        if not repeated:
            return
        named = ", ".join(
            f"{self.class_key(c)!r} ({n} assertions)"
            for c, n in sorted(repeated.items())
        )
        raise MEDH5ValidationError(
            f"annotation {self.ann_id!r}: {what} collapses one value per class, "
            f"but {named} with scope {self.scope!r}. §9 makes that ordinary --- "
            "`scope_ids` is per assertion --- so there is no single answer to "
            "give. Pass scope_id=, or read `assertions()` / `by_scope_id()`.",
            code="E412",
        )

    @property
    def positives(self) -> tuple[str, ...]:
        return tuple(key for key, value in self.labels.items() if value > 0.0)

    def value(
        self, class_key: int | str, *, scope_id: int | None = None
    ) -> float | None:
        """The asserted value, or ``None`` when the class was not asserted.

        *scope_id* selects the assertion when a class is asserted for several
        scope units (§9).  Without it, a class carrying more than one assertion
        is an error rather than a silent first-match: the first row is not more
        authoritative than the second.
        """
        target = self.resolve_class(class_key)
        if scope_id is None:
            self._require_unambiguous("value()", target)
        scope_ids = self.scope_ids
        for i, (class_id, value) in enumerate(
            zip(self.asserted_class_ids, self.values, strict=True)
        ):
            if int(class_id) != target:
                continue
            if scope_id is not None and (
                scope_ids is None or int(scope_ids[i]) != scope_id
            ):
                continue
            return float(value)
        return None

    def state(self, class_key: int | str, *, scope_id: int | None = None) -> str:
        """``"positive"``, ``"negative"`` or ``"unknown"`` for one class (§9).

        ``unknown`` is the answer whenever the class is outside
        ``annotated_class_ids``: nobody looked, so its absence carries no
        information and training code must not treat it as a negative.
        """
        target = self.resolve_class(class_key)
        value = self.value(target, scope_id=scope_id)
        if value is not None:
            return "positive" if value > 0.0 else "negative"
        return "negative" if self.is_annotated(target) else "unknown"

    def scheme(self, name: str) -> str | None:
        """The ordinal value recorded under a named scheme, e.g. ``"BI-RADS"``."""
        schemes = self.schemes
        scheme_values = self.scheme_values
        if schemes is None or scheme_values is None:
            return None
        for scheme, value in zip(schemes, scheme_values, strict=True):
            if scheme == name:
                return value
        return None

    def by_scope_id(self) -> dict[int | None, tuple[Assertion, ...]]:
        """Assertions grouped by scope unit --- per-slice, per-visit, per-lesion."""
        out: dict[int | None, list[Assertion]] = {}
        for assertion in self.assertions():
            out.setdefault(assertion.scope_id, []).append(assertion)
        return {k: tuple(v) for k, v in out.items()}

    @property
    def is_change_label(self) -> bool:
        """Whether this asserts something *about a set of timepoints* (§9)."""
        return self.scope == "sample" and len(self.header.timepoints or ()) > 1

    @property
    def compared_timepoints(self) -> tuple[str, ...]:
        return (
            as_str_tuple(self.group.attrs["timepoints"])
            if ("timepoints" in self.group.attrs)
            else ()
        )

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.ann_id,
            "kind": self.kind,
            "task": self.task,
            "grid": self.grid_id,
            "scope": self.scope,
            "multilabel": self.multilabel,
            "timepoints": list(self.timepoints),
            "change_label": self.is_change_label,
            "labels": self.labels,
            "classes": len(self.class_ids),
            "annotated_classes": len(self.annotated_class_ids),
            "fully_covered": self.is_fully_covered,
            "quality": self.quality_key,
            "prov": self.prov,
        }

    def __len__(self) -> int:
        return int(self.asserted_class_ids.shape[0])


__all__ = [
    "SCOPES",
    "Assertion",
    "ClassificationAnnotation",
    "check_scope",
    "encode_classification",
]
