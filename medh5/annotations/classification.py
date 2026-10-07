"""Classification annotations: label assertions about a scope unit (spec §9).

A **change label** is a classification with ``scope = "sample"`` and explicit
``timepoints`` naming the visits compared --- the format adds no ``change``
kind.  A class that was looked for and not found is a negative (value 0.0);
a class nobody asserted is not.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import Annotation

SCOPES: tuple[str, ...] = _core.SCOPES

check_scope = _core.check_scope
assertion_rows = _core.assertion_rows
encode_classification = _core.encode_classification

Labels = Mapping[Any, float] | Sequence[Sequence[Any]]
"""What a classification is built from: ``class -> value``, or assertion rows.

A row is ``(class, value)``, ``(class, value, scope_id)`` or
``(class, value, scope_id, scheme, scheme_value)``.
"""


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


def _assertion(row: tuple[Any, ...]) -> Assertion:
    class_id, value, scope_id, scheme, scheme_value = row
    return Assertion(
        class_id=int(class_id),
        value=float(value),
        scope_id=scope_id,
        scheme=scheme,
        scheme_value=scheme_value,
    )


class ClassificationAnnotation(Annotation):
    """Reader for ``kind = "classification"``."""

    __slots__ = ()

    @property
    def scope(self) -> str:
        return str(self._handle.scope)

    @property
    def multilabel(self) -> bool:
        return bool(self._handle.multilabel)

    @property
    def asserted_class_ids(self) -> npt.NDArray[np.uint16]:
        return np.asarray(self._handle.asserted_class_ids, dtype=np.uint16)

    @property
    def values(self) -> npt.NDArray[np.float32]:
        return np.asarray(self._handle.assertion_values, dtype=np.float32)

    @property
    def scope_ids(self) -> npt.NDArray[np.int64] | None:
        found = self._handle.scope_ids
        return None if found is None else np.asarray(found, dtype=np.int64)

    @property
    def schemes(self) -> tuple[str, ...] | None:
        found: tuple[str, ...] | None = self._handle.schemes
        return found

    @property
    def scheme_values(self) -> tuple[str, ...] | None:
        found: tuple[str, ...] | None = self._handle.scheme_values
        return found

    def assertions(self) -> Iterator[Assertion]:
        for row in self._handle.assertions():
            yield _assertion(row)

    @property
    def labels(self) -> dict[str, float]:
        """``key -> value`` for every assertion, keyed by label-set key when
        known.  Refused (E412) when a class is asserted more than once: ask
        :meth:`assertions` or :meth:`by_scope_id` then."""
        found: dict[str, float] = self._handle.labels
        return found

    @property
    def positives(self) -> tuple[str, ...]:
        return tuple(self._handle.positives)

    def value(
        self, class_key: int | str, *, scope_id: int | None = None
    ) -> float | None:
        """The asserted value, or ``None`` when the class was not asserted."""
        found: float | None = self._handle.value(class_key, scope_id=scope_id)
        return found

    def state(self, class_key: int | str, *, scope_id: int | None = None) -> str:
        """``"positive"``, ``"negative"`` or ``"unknown"`` for one class (§9).

        ``unknown`` is the answer whenever the class is outside
        ``annotated_class_ids``: nobody looked, so its absence carries no
        information and training code must not treat it as a negative.
        """
        return str(self._handle.state(class_key, scope_id=scope_id))

    def scheme(self, name: str) -> str | None:
        """The ordinal value recorded under a named scheme, e.g. ``"BI-RADS"``."""
        found: str | None = self._handle.scheme(name)
        return found

    def by_scope_id(self) -> dict[int | None, tuple[Assertion, ...]]:
        return {
            unit: tuple(_assertion(row) for row in rows)
            for unit, rows in self._handle.by_scope_id().items()
        }

    @property
    def is_change_label(self) -> bool:
        return bool(self._handle.is_change_label)

    @property
    def compared_timepoints(self) -> tuple[str, ...]:
        return tuple(self._handle.compared_timepoints)

    def __len__(self) -> int:
        return int(self._handle.n_items)


__all__ = [
    "SCOPES",
    "Assertion",
    "ClassificationAnnotation",
    "Labels",
    "assertion_rows",
    "check_scope",
    "encode_classification",
]
