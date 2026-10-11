"""The ``clinical`` profile (format 1.1): clinical context beside the images.

A 1.1 sample may carry the subject's source documents, asynchronous
observations and interventions, typed links between them and the imaging
objects, and explicit *information-availability* times, all under
``clinical/`` (``docs/spec/medh5-1.1.md`` §3--§8).  Nothing about an image,
grid, annotation or transform changes: the profile is additive, and a clinical
event needs no imaging timepoint.

Records are frozen dataclasses mirroring the logical-record JSON
(``medh5-clinical-1.schema.json``).  Times are signed microseconds on the
subject clock, and every time is a pair of **inclusive bounds**: an exact
instant is ``(t, t)``, a date known to the day spans the day, and an unknown
time is ``None`` --- never a guess.

.. code-block:: python

    from medh5.clinical import HOUR, Clock, Event, Link

    with medh5.create("case.medh5", sample_id="case", subject_id="P-01") as w:
        ...  # grids and images, as for any sample
        w.set_clock(Clock.relative("subject-clock", "baseline CT acquisition"))
        w.add_event(Event("ct0", "ct0", "imaging", "point", "final",
                          effective_start_us=0, available_us=HOUR, timepoint_id="tp0"))
        w.add_link(Link.between(("event", "ct0"), "describes", ("image", "CT")))

    with medh5.open("case.medh5") as s:
        s.clinical.select(24 * HOUR).event_ids     # what was known a day in

Selection is the engine's (``medh5.clinical.select``): the same rules decide
what a model may see at a cutoff from Rust, Python and the command line.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any

from medh5 import _core

Bounds = tuple[int, int]
"""Inclusive ``(lo, hi)`` bounds on one instant, in clock microseconds."""

MICROSECOND = 1
SECOND = 1_000_000
MINUTE = 60 * SECOND
HOUR = 60 * MINUTE
DAY = 24 * HOUR

PROFILE: str = _core.CLINICAL_PROFILE
SCHEMA: str = _core.CLINICAL_SCHEMA
EVENT_KINDS: tuple[str, ...] = _core.EVENT_KINDS
TEMPORAL_TYPES: tuple[str, ...] = _core.TEMPORAL_TYPES
STATUSES: tuple[str, ...] = _core.STATUSES
COMPARATORS: tuple[str, ...] = _core.COMPARATORS
ENDPOINT_TYPES: tuple[str, ...] = _core.ENDPOINT_TYPES
RELATIONS: tuple[str, ...] = _core.RELATIONS
CLOCK_REFERENCES: tuple[str, ...] = _core.CLOCK_REFERENCES
LESION_VALUES: tuple[str, ...] = _core.LESION_VALUES
SELECTION_POLICIES: tuple[str, ...] = _core.SELECTION_POLICIES
ASSESSMENT_SYSTEM = "org.medh5.assessment"
LESION_PRESENCE = "lesion_presence"


def hours(value: float) -> int:
    """``value`` hours, in clock microseconds."""
    return int(round(value * HOUR))


def days(value: float) -> int:
    """``value`` days, in clock microseconds."""
    return int(round(value * DAY))


def _bounds(value: Any) -> Bounds | None:
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    lo, hi = value
    return (int(lo), int(hi))


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def schema_text() -> str:
    """The clinical descriptor and logical-record JSON Schema."""
    return str(_core.clinical_schema_text())


@dataclass(frozen=True, slots=True)
class Clock:
    """The subject clock every clinical time is measured on (1.1 §3)."""

    id: str
    reference: str = "relative"
    origin_description: str | None = None
    unit: str = "us"

    @classmethod
    def relative(cls, clock_id: str, origin_description: str) -> Clock:
        """Signed microseconds from a documented, subject-specific origin."""
        return cls(clock_id, "relative", origin_description)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "unit": self.unit,
            "reference": self.reference,
        }
        if self.origin_description is not None:
            out["origin_description"] = self.origin_description
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Clock:
        return cls(
            str(doc["id"]),
            str(doc["reference"]),
            doc.get("origin_description"),
            str(doc.get("unit", "us")),
        )


@dataclass(frozen=True, slots=True)
class Event:
    """One immutable version of information about the subject (1.1 §5).

    ``available_us`` is when this whole version became available in the source
    workflow; ``None`` when unknown, and then strict selection never uses it.
    A revision is a new event with the same ``record_id``, linked by
    ``supersedes``.
    """

    event_id: str
    record_id: str
    kind: str
    temporal_type: str
    status: str
    effective_start_us: Bounds | None = None
    effective_end_us: Bounds | None = None
    available_us: Bounds | None = None
    timepoint_id: str | None = None
    encounter_id: str | None = None
    code_system: str | None = None
    code: str | None = None
    code_version: str | None = None
    value_num: float | None = None
    value_comparator: str | None = None
    unit: str | None = None
    value_text: str | None = None
    missing_reason: str | None = None
    prov: str | None = None

    def __post_init__(self) -> None:
        for name in ("effective_start_us", "effective_end_us", "available_us"):
            object.__setattr__(self, name, _bounds(getattr(self, name)))

    def to_json(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Event:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in known})

    @property
    def is_lesion_assessment(self) -> bool:
        return self.code_system == ASSESSMENT_SYSTEM and self.code == LESION_PRESENCE


@dataclass(frozen=True, slots=True)
class Document:
    """Canonical, de-identified source text (1.1 §6)."""

    document_id: str
    text: str
    media_type: str = "text/plain"
    language: str | None = None
    source_type: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Document:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in known})


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """A document's metadata; its text is read when asked for."""

    document_id: str
    media_type: str
    language: str | None
    source_type: str | None
    n_bytes: int


@dataclass(frozen=True, slots=True)
class Link:
    """A typed relationship between sample-relative objects (1.1 §7)."""

    source_type: str
    source_id: str
    relation: str
    target_type: str
    target_id: str
    source_span: tuple[int, int] | None = None
    target_annotation_id: str | None = None
    asserted_by_event_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_span is not None:
            object.__setattr__(
                self, "source_span", tuple(int(v) for v in self.source_span)
            )

    @classmethod
    def between(
        cls,
        source: tuple[str, str],
        relation: str,
        target: tuple[str, str],
        *,
        asserted_by: str | None = None,
        span: tuple[int, int] | None = None,
        target_annotation_id: str | None = None,
    ) -> Link:
        """``Link.between(("event", "ct0"), "describes", ("image", "CT"))``."""
        return cls(
            source[0],
            source[1],
            relation,
            target[0],
            str(target[1]),
            span,
            target_annotation_id,
            asserted_by,
        )

    def to_json(self) -> dict[str, Any]:
        return {f.name: _jsonable(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Link:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in doc.items() if k in known})


@dataclass(frozen=True, slots=True)
class ClinicalRecords:
    """A logical-record bundle: what ``augment`` adds and ``records()`` reads."""

    clock: Clock
    events: tuple[Event, ...] = ()
    documents: tuple[Document, ...] = ()
    links: tuple[Link, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "clinical": {"schema": SCHEMA, "clock": self.clock.to_json()},
            "events": [e.to_json() for e in self.events],
            "documents": [d.to_json() for d in self.documents],
            "links": [link.to_json() for link in self.links],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ClinicalRecords:
        checked = _core.clinical_records(dict(doc))
        return cls(
            Clock.from_json(checked["clinical"]["clock"]),
            tuple(Event.from_json(e) for e in checked.get("events", [])),
            tuple(Document.from_json(d) for d in checked.get("documents", [])),
            tuple(Link.from_json(link) for link in checked.get("links", [])),
        )


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """How inputs are chosen at a cutoff (task-and-cache contract §3.4).

    ``selection="strict_prospective"`` is the guarantee: only versions known
    to be available at the cutoff, and a row whose later revision has unknown
    or straddling availability is *uncertifiable*.  ``latest_provable`` is its
    one named alternative, and never claims the source's newest version.
    """

    selection: str = "strict_prospective"
    order_by: str = "effective"
    context_us: int | None = None
    context_boundary: str = "closed"
    uncertainty: str = "contained"
    kinds: tuple[str, ...] | None = None
    plans: bool = False
    static: bool = True
    max_events: int | None = None
    keep: str = "latest"
    ties: str = "keep_group"

    def to_json(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.kinds is not None:
            out["kinds"] = list(self.kinds)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SelectionPolicy:
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in doc.items() if k in known}
        if values.get("kinds") is not None:
            values["kinds"] = tuple(values["kinds"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class SelectedEvent:
    """One event version a selection admits, in input order."""

    event_id: str
    record_id: str
    kind: str
    order_us: Bounds | None
    tie_group: int
    plan: bool


@dataclass(frozen=True, slots=True)
class Selection:
    """What a cutoff admits (task-and-cache contract §4)."""

    cutoff_us: int
    policy: str
    status: str
    events: tuple[SelectedEvent, ...]
    links: tuple[int, ...]
    payloads: frozenset[tuple[int, str, str]]
    uncertain_records: tuple[str, ...]
    excluded: dict[str, int] = field(default_factory=dict)

    @property
    def certified(self) -> bool:
        """Whether the strict guarantee holds."""
        return self.status == "certified"

    @property
    def event_ids(self) -> list[str]:
        return [e.event_id for e in self.events]

    def admits(self, kind: str, object_id: str, fragment: int = 0) -> bool:
        """Whether ``(kind, id)`` in ``fragment`` may be read as input."""
        return (fragment, kind, str(object_id)) in self.payloads

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Selection:
        return cls(
            int(doc["cutoff_us"]),
            str(doc["policy"]),
            str(doc["status"]),
            tuple(
                SelectedEvent(
                    e["event_id"],
                    e["record_id"],
                    e["kind"],
                    _bounds(e.get("order_us")),
                    int(e["tie_group"]),
                    bool(e["plan"]),
                )
                for e in doc["events"]
            ),
            tuple(int(i) for i in doc["links"]),
            frozenset((int(f), str(k), str(i)) for f, k, i in doc["payloads"]),
            tuple(doc["uncertain_records"]),
            {str(k): int(v) for k, v in doc.get("excluded", {}).items()},
        )


def _policy(policy: SelectionPolicy | Mapping[str, Any] | str | None) -> Any:
    if isinstance(policy, SelectionPolicy):
        return policy.to_json()
    if isinstance(policy, Mapping):
        return dict(policy)
    return policy


def select(
    events: Iterable[Event | Mapping[str, Any]],
    links: Iterable[Link | Mapping[str, Any]],
    cutoff_us: int,
    policy: SelectionPolicy | Mapping[str, Any] | str | None = None,
) -> Selection:
    """Select at ``cutoff_us`` from records not read from a file."""
    as_json = [e.to_json() if hasattr(e, "to_json") else dict(e) for e in events]
    link_json = [
        link.to_json() if hasattr(link, "to_json") else dict(link) for link in links
    ]
    return Selection.from_json(
        _core.clinical_select(as_json, link_json, int(cutoff_us), _policy(policy))
    )


class Clinical:
    """One sample's clinical profile, read: ``Sample.clinical``.

    Events and links are read when the view is made --- selection needs all of
    them --- and document text one document at a time, when asked for.
    """

    __slots__ = ("_by_id", "_documents", "_events", "_handle", "_links")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._events: tuple[Event, ...] | None = None
        self._links: tuple[Link, ...] | None = None
        self._documents: tuple[DocumentInfo, ...] | None = None
        self._by_id: dict[str, Event] | None = None

    @property
    def descriptor(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.descriptor()
        return found

    @property
    def clock(self) -> Clock:
        return Clock.from_json(self.descriptor["clock"])

    @property
    def projection(self) -> bool:
        """Read from a higher minor version, as what this engine knows."""
        return bool(self._handle.projection)

    @property
    def events(self) -> tuple[Event, ...]:
        if self._events is None:
            self._events = tuple(Event.from_json(e) for e in self._handle.events())
        return self._events

    def event(self, event_id: str) -> Event:
        if self._by_id is None:
            self._by_id = {e.event_id: e for e in self.events}
        try:
            return self._by_id[event_id]
        except KeyError:
            raise KeyError(f"no event {event_id!r}") from None

    @property
    def links(self) -> tuple[Link, ...]:
        if self._links is None:
            self._links = tuple(Link.from_json(link) for link in self._handle.links())
        return self._links

    @property
    def documents(self) -> tuple[DocumentInfo, ...]:
        if self._documents is None:
            self._documents = tuple(DocumentInfo(**d) for d in self._handle.documents())
        return self._documents

    def text(self, document_id: str) -> str:
        """One document's text, read from the file now."""
        return str(self._handle.text(document_id))

    def document(self, document_id: str) -> Document:
        return Document.from_json(self._handle.document(document_id))

    def records(self) -> ClinicalRecords:
        """Everything, as a logical-record bundle (every text read)."""
        doc = self._handle.records()
        return ClinicalRecords(
            Clock.from_json(doc["clinical"]["clock"]),
            tuple(Event.from_json(e) for e in doc["events"]),
            tuple(Document.from_json(d) for d in doc["documents"]),
            tuple(Link.from_json(link) for link in doc["links"]),
        )

    def select(
        self,
        cutoff_us: int,
        policy: SelectionPolicy | Mapping[str, Any] | str | None = None,
    ) -> Selection:
        """What strict (or the named) selection admits at ``cutoff_us``."""
        return Selection.from_json(self._handle.select(int(cutoff_us), _policy(policy)))

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def __repr__(self) -> str:
        return (
            f"Clinical(clock={self.clock.id!r}, {len(self.events)} events, "
            f"{len(self.documents)} documents)"
        )


def augment(
    path: str | os.PathLike[str],
    records: ClinicalRecords | Mapping[str, Any],
    *,
    out: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Add clinical records to a sample, in place or into ``out`` (1.1 §10).

    Images, grids, annotations and transforms are copied as stored, so their
    digests do not change; the sample becomes 1.1 with a new ``content_id``.
    The report lists what the records leave unknown --- availability, precision
    --- which stays unknown.  A ``clinical`` group that is not the profile's is
    refused, never reinterpreted.
    """
    doc = records.to_json() if isinstance(records, ClinicalRecords) else dict(records)
    found: dict[str, Any] = _core.clinical_augment(
        os.fspath(path), doc, None if out is None else os.fspath(out)
    )
    return found


def strip(path: str | os.PathLike[str], out: str | os.PathLike[str]) -> dict[str, Any]:
    """Write the imaging projection of ``path`` to ``out``: a different sample,
    and a reported loss."""
    found: dict[str, Any] = _core.clinical_strip(os.fspath(path), os.fspath(out))
    return found


def imaging_events_from_timepoints(
    path: str | os.PathLike[str],
) -> tuple[list[Event], list[Link], list[str]]:
    """``imaging`` events from ``days_from_baseline``, known to the day.

    Their availability is unknown and stays so; pair them with
    :func:`baseline_day_clock`.  The notes say what was assumed and skipped.
    """
    events, links, notes = _core.imaging_events_from_timepoints(os.fspath(path))
    return (
        [Event.from_json(e) for e in events],
        [Link.from_json(link) for link in links],
        [str(n) for n in notes],
    )


def baseline_day_clock(clock_id: str) -> Clock:
    """The clock :func:`imaging_events_from_timepoints` measures on."""
    return Clock.from_json(_core.baseline_day_clock(clock_id))


def records_from(
    clock: Clock,
    events: Sequence[Event] = (),
    documents: Sequence[Document] = (),
    links: Sequence[Link] = (),
) -> ClinicalRecords:
    """A bundle from lists."""
    return ClinicalRecords(clock, tuple(events), tuple(documents), tuple(links))


__all__ = [
    "ASSESSMENT_SYSTEM",
    "CLOCK_REFERENCES",
    "COMPARATORS",
    "DAY",
    "ENDPOINT_TYPES",
    "EVENT_KINDS",
    "HOUR",
    "LESION_PRESENCE",
    "LESION_VALUES",
    "MICROSECOND",
    "MINUTE",
    "PROFILE",
    "RELATIONS",
    "SCHEMA",
    "SECOND",
    "SELECTION_POLICIES",
    "STATUSES",
    "TEMPORAL_TYPES",
    "Bounds",
    "Clinical",
    "ClinicalRecords",
    "Clock",
    "Document",
    "DocumentInfo",
    "Event",
    "Link",
    "SelectedEvent",
    "Selection",
    "SelectionPolicy",
    "augment",
    "baseline_day_clock",
    "days",
    "hours",
    "imaging_events_from_timepoints",
    "records_from",
    "schema_text",
    "select",
    "strip",
]
