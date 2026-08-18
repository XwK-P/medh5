"""Timepoints: the sample's observation occasions (spec §3.7).

A sample is one subject at one **or more** timepoints.  The declaration lives in
``/meta -> timepoints``; every grid names one, and images, annotations and
transforms inherit theirs rather than repeating it.  Binding time to the *grid*
is what keeps the rule single-valued: a grid belongs to exactly one acquisition
occasion, whereas an image or an annotation might plausibly be argued either
way.

``days_from_baseline`` rather than ``date`` is what models should consume: the
interval survives de-identification date shifting, and it is the clinically
load-bearing quantity.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5._hdf5 import ID_PATTERN
from medh5.errors import MEDH5ValidationError

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True, slots=True)
class Timepoint:
    """One observation occasion --- in DICOM terms, usually one study."""

    id: str
    index: int
    label: str | None = None
    date: str | None = None
    days_from_baseline: float | None = None
    study_uid: str | None = None
    series_uids: Mapping[str, str] = field(default_factory=dict)
    subject_age_years: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not ID_PATTERN.match(self.id):
            raise MEDH5ValidationError(
                f"timepoint id {self.id!r} must match [A-Za-z0-9_.-]{{1,128}}",
                code="E003",
            )
        if self.index < 0:
            raise MEDH5ValidationError(
                f"timepoint {self.id!r}: index must be >= 0", code="E108"
            )
        if self.date is not None and not DATE_PATTERN.match(self.date):
            raise MEDH5ValidationError(
                f"timepoint {self.id!r}: date {self.date!r} is not ISO 8601",
                code="E604",
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "index": self.index}
        for key in (
            "label",
            "date",
            "days_from_baseline",
            "study_uid",
            "subject_age_years",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.series_uids:
            out["series_uids"] = dict(self.series_uids)
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Timepoint:
        known = {
            "id",
            "index",
            "label",
            "date",
            "days_from_baseline",
            "study_uid",
            "series_uids",
            "subject_age_years",
        }
        return cls(
            id=str(doc["id"]),
            index=int(doc["index"]),
            label=doc.get("label"),
            date=doc.get("date"),
            days_from_baseline=doc.get("days_from_baseline"),
            study_uid=doc.get("study_uid"),
            series_uids=dict(doc.get("series_uids") or {}),
            subject_age_years=doc.get("subject_age_years"),
            extra={k: v for k, v in doc.items() if k not in known},
        )


class Timeline(Sequence[Timepoint]):
    """The sample's timepoints, in acquisition order.

    Indexable by position (``tl[0]``) or by id (``tl["tp1"]``), because both are
    natural: position is what an ordered loop wants, id is what a grid names.
    """

    __slots__ = ("_by_id", "_points")

    def __init__(self, timepoints: Sequence[Timepoint]) -> None:
        self._points = tuple(sorted(timepoints, key=lambda t: t.index))
        self._by_id = {t.id: t for t in self._points}
        self.check()

    def check(self) -> None:
        """Validate spec §3.7 rules 1 and 2 (E108)."""
        if not self._points:
            raise MEDH5ValidationError(
                "a sample must declare at least one timepoint", code="E108"
            )
        if len(self._by_id) != len(self._points):
            raise MEDH5ValidationError("duplicate timepoint id", code="E108")
        indices = [t.index for t in self._points]
        if indices != list(range(len(indices))):
            raise MEDH5ValidationError(
                f"timepoint indices {indices} must be dense and start at 0", code="E108"
            )
        days = [t.days_from_baseline for t in self._points]
        known = [(i, d) for i, d in enumerate(days) if d is not None]
        for (i0, d0), (i1, d1) in zip(known, known[1:], strict=False):
            if d1 < d0:
                raise MEDH5ValidationError(
                    f"days_from_baseline decreases between index {i0} and {i1} "
                    f"({d0} -> {d1}); `index` must be increasing with time",
                    code="E108",
                )

    # -- Sequence protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self) -> Iterator[Timepoint]:
        return iter(self._points)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self._by_id[key]
            except KeyError:
                raise KeyError(
                    f"undeclared timepoint {key!r}; declared: {list(self._by_id)}"
                ) from None
        return self._points[key]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._by_id
        return item in self._points

    def __repr__(self) -> str:
        return f"Timeline({[t.id for t in self._points]!r})"

    # -- convenience -------------------------------------------------------

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(t.id for t in self._points)

    @property
    def baseline(self) -> Timepoint:
        return self._points[0]

    @property
    def is_longitudinal(self) -> bool:
        return len(self._points) > 1

    def interval_days(self, a: str, b: str) -> float | None:
        """Days between two timepoints, or ``None`` when either lacks an interval."""
        da, db = self[a].days_from_baseline, self[b].days_from_baseline
        if da is None or db is None:
            return None
        return float(db - da)

    def require(self, timepoint_id: str, *, where: str = "") -> Timepoint:
        """Resolve an id, raising E409/E107 style errors with a useful message."""
        try:
            found: Timepoint = self[timepoint_id]
        except KeyError:
            prefix = f"{where}: " if where else ""
            raise MEDH5ValidationError(
                f"{prefix}timepoint {timepoint_id!r} is not declared "
                f"(declared: {list(self.ids)})",
                code="E409",
            ) from None
        return found

    def to_json(self) -> list[dict[str, Any]]:
        return [t.to_json() for t in self._points]

    @classmethod
    def from_json(cls, docs: Sequence[Mapping[str, Any]]) -> Timeline:
        return cls([Timepoint.from_json(d) for d in docs])

    @classmethod
    def single(cls, timepoint_id: str = "tp0", **kwargs: Any) -> Timeline:
        """The one-timepoint timeline a cross-sectional sample declares."""
        return cls([Timepoint(id=timepoint_id, index=0, **kwargs)])


__all__ = ["Timeline", "Timepoint"]
