"""Task manifests (``medh5.task/1``): which rows, at which cutoffs, see what.

A ``.medh5`` file is the patient source.  A *task* is a separate, versioned
JSON document (``docs/spec/task-cache-1.md``) naming

- the **sources** of each subject --- files or collection members, each pinned
  to the ``content_id`` it was built from (a URI says where bytes are; the pin
  says which bytes count);
- the **partition** of each subject, assigned before any row exists, so no
  window puts one patient on both sides of a split;
- the **rows**: a subject at a cutoff;
- the **selection policy** deciding what a row may read (strict prospective by
  default), the **modality slots** a row's images fill, and the **target**.

:func:`preflight` opens every source once, checks every pin, reconciles the
subject's fragments, and says per row whether it is *eligible*,
*uncertifiable*, *excluded* or in *error* --- and why --- before a voxel or a
report is read.  The engine does all of it; this module is the Python face.

.. code-block:: python

    from medh5.task import Slot, SourceRef, Target, TaskManifest

    task = TaskManifest.new("progression", "1", identity_namespace="site-a",
                            slots=[Slot("ct", "CT", required=True, patch=(32, 32, 16))],
                            split=("fold-0", ["train", "val"]))
    task.add_subject("P-01", [SourceRef.pin("p01.medh5")], partition="train")
    task.add_row("P-01@d30", "P-01", cutoff_us=30 * DAY)
    report = task.preflight()
    report.counts            # {"eligible": 1}
"""

from __future__ import annotations

import copy
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from medh5 import _core
from medh5.clinical import Event, Selection, SelectionPolicy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.sample import Sample

SCHEMA: str = _core.TASK_SCHEMA
"""The manifest's own version."""

CODES: dict[str, str] = dict(_core.COMPANION_CODES)
"""The companion contract's finding codes (T1xx manifest, T2xx identity and
splits, T3xx sources, T4xx caches) and what each means."""

ROI = ("center", "eligible_instances")
CENSORING = ("censor", "exclude")
STATUSES = ("eligible", "uncertifiable", "excluded", "error")
LABELS = ("positive", "negative", "censored", "prevalent", "none")

PathLike = str | os.PathLike[str]


def schema_text() -> str:
    """The ``medh5.task/1`` JSON Schema."""
    return str(_core.task_schema_text())


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong with a task, its sources, or a cache."""

    code: str
    location: str
    message: str

    @property
    def summary(self) -> str:
        return CODES.get(self.code, "")

    def __str__(self) -> str:
        return f"{self.code} {self.location}: {self.message}"

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Finding:
        return cls(str(doc["code"]), str(doc["location"]), str(doc["message"]))


def _findings(found: Iterable[Mapping[str, Any]]) -> tuple[Finding, ...]:
    return tuple(Finding.from_json(f) for f in found)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One sample --- a file, or a member of a ``.medh5c`` --- at a pinned version.

    ``uri`` is a path (absolute, or relative to the manifest's directory;
    ``file://`` is accepted).  ``content_id`` is the pin: the reference counts
    only while the sample's content address is this one.
    """

    uri: str
    content_id: str
    sample_key: str | None = None
    source_id: str = ""
    local_subject_id: str | None = None

    @classmethod
    def pin(
        cls,
        path: PathLike,
        *,
        sample_key: str | None = None,
        source_id: str | None = None,
        uri: str | None = None,
    ) -> SourceRef:
        """A reference pinned to what the sample at ``path`` is now.

        ``uri`` is what the manifest records (default: ``path`` as given) ---
        pass a relative one to keep a manifest movable with its data.
        """
        doc = _core.source_pin(os.fspath(path), sample_key, source_id, uri)
        found = cls.from_json(doc)
        if not found.source_id:
            key = f":{sample_key}" if sample_key else ""
            default = re.sub(r"[^A-Za-z0-9_.@:-]", "_", f"{Path(found.uri).stem}{key}")
            found = cls(
                found.uri,
                found.content_id,
                found.sample_key,
                default[:128],
                found.local_subject_id,
            )
        return found

    @property
    def locator(self) -> str:
        """``uri`` or ``uri::sample_key``."""
        return self.uri if self.sample_key is None else f"{self.uri}::{self.sample_key}"

    def resolve(self, base: PathLike | None = None) -> Path:
        """The file the URI names, relative to ``base`` when it is relative."""
        text = self.uri.removeprefix("file://")
        path = Path(text)
        if base is not None and not path.is_absolute():
            return Path(base) / path
        return path

    def open(self, base: PathLike | None = None) -> Sample:
        """Open the sample (the pin is not checked; see :meth:`check`)."""
        from medh5.collection import open_any
        from medh5.errors import MEDH5ValidationError
        from medh5.sample import Sample

        found = open_any(self.resolve(base), key=self.sample_key)
        if not isinstance(found, Sample):
            found.close()
            raise MEDH5ValidationError(
                f"{self.uri!r} is a collection; a source names one of its members "
                "with `sample_key`"
            )
        return found

    def check(
        self, base: PathLike | None = None, *, deep: bool = False
    ) -> list[Finding]:
        """Whether the sample is still the pinned version: its stored
        ``content_id`` is the pin, the root recomputes to it, and the clinical
        datasets' bytes match their digests (every dataset's with ``deep``)."""
        found = _core.source_check(
            self.to_json(), None if base is None else os.fspath(base), deep
        )
        return list(_findings(found))

    def to_json(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "uri": self.uri,
            "sample_key": self.sample_key,
            "content_id": self.content_id,
            "local_subject_id": self.local_subject_id,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SourceRef:
        return cls(
            str(doc["uri"]),
            str(doc["content_id"]),
            doc.get("sample_key"),
            str(doc.get("source_id") or ""),
            doc.get("local_subject_id"),
        )


@dataclass(frozen=True, slots=True)
class Slot:
    """A stable modality slot, filled per row by the newest *eligible* image of
    its modality --- never by a filename (contract §3.5).

    ``patch`` is the window read around the slot's centre, in voxels of the
    image's own grid (``None`` reads the whole volume); ``roi`` centres it on
    the grid (``center``) or on the first eligible instance
    (``eligible_instances``).  ``classes`` are read as labels from eligible
    voxel annotations on the same grid.
    """

    name: str
    modality: str
    required: bool = False
    patch: tuple[int, ...] | None = None
    roi: str = "center"
    classes: tuple[int, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "modality": self.modality,
            "required": self.required,
            "patch": None if self.patch is None else list(self.patch),
            "roi": self.roi,
            "classes": list(self.classes),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Slot:
        patch = doc.get("patch")
        return cls(
            str(doc["name"]),
            str(doc["modality"]),
            bool(doc.get("required", False)),
            None if patch is None else tuple(int(v) for v in patch),
            str(doc.get("roi", "center")),
            tuple(int(c) for c in doc.get("classes", ())),
        )


@dataclass(frozen=True, slots=True)
class Target:
    """What a row is labelled with (contract §5): an event concept whose final
    ``value_text`` falls in ``positive`` or ``negative`` inside
    ``(cutoff, cutoff + horizon_us]``, read from the full history.

    A negative needs an observation at least ``min_follow_up_us`` after the
    cutoff (default: the horizon).  A row it cannot label is censored ---
    kept with its target unobserved --- or, with ``censoring="exclude"``,
    excluded; a row whose outcome had already occurred is excluded as
    prevalent unless ``exclude_prevalent=False``.
    """

    id: str
    version: str
    code_system: str
    code: str
    positive: tuple[str, ...]
    negative: tuple[str, ...] = ()
    horizon_us: int = 0
    min_follow_up_us: int | None = None
    censoring: str = "censor"
    exclude_prevalent: bool = True
    kind: str | None = None

    def to_json(self) -> dict[str, Any]:
        event: dict[str, Any] = {"code_system": self.code_system, "code": self.code}
        if self.kind is not None:
            event["kind"] = self.kind
        out: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "event": event,
            "positive": list(self.positive),
            "negative": list(self.negative),
            "horizon_us": int(self.horizon_us),
            "censoring": self.censoring,
            "exclude_prevalent": self.exclude_prevalent,
        }
        if self.min_follow_up_us is not None:
            out["min_follow_up_us"] = int(self.min_follow_up_us)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Target:
        event = doc["event"]
        return cls(
            str(doc["id"]),
            str(doc["version"]),
            str(event["code_system"]),
            str(event["code"]),
            tuple(doc.get("positive", ())),
            tuple(doc.get("negative", ())),
            int(doc.get("horizon_us", 0)),
            doc.get("min_follow_up_us"),
            str(doc.get("censoring", "censor")),
            bool(doc.get("exclude_prevalent", True)),
            event.get("kind"),
        )


@dataclass(frozen=True, slots=True)
class Reconciled:
    """An event version several fragments hold, and the digest they share."""

    event_id: str
    digest: str
    sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Subject:
    """One subject: its fragments, its clock and its partition."""

    subject_id: str
    sources: tuple[SourceRef, ...]
    partition: str | None = None
    clock_id: str | None = None
    reconciled: tuple[Reconciled, ...] = ()

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Subject:
        return cls(
            str(doc["subject_id"]),
            tuple(SourceRef.from_json(s) for s in doc.get("sources", ())),
            doc.get("partition"),
            doc.get("clock_id"),
            tuple(
                Reconciled(str(r["event_id"]), str(r["digest"]), tuple(r["sources"]))
                for r in doc.get("reconciled", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class Row:
    """One example: a subject at a cutoff."""

    row_id: str
    subject_id: str
    cutoff_us: int


def _policy_json(policy: SelectionPolicy | Mapping[str, Any] | str | None) -> Any:
    if policy is None:
        return None
    if isinstance(policy, SelectionPolicy):
        return policy.to_json()
    if isinstance(policy, str):
        return {"selection": policy}
    return dict(policy)


class TaskManifest:
    """A ``medh5.task/1`` manifest: build it, check it, save it, preflight it.

    The document is held as JSON and normalised by the engine, which fills
    every default --- so two spellings of one task share a
    :attr:`task_fingerprint`.  ``base`` resolves relative source URIs: the
    manifest's directory when it was loaded from a file.
    """

    __slots__ = ("_doc", "_normal", "base")

    def __init__(self, doc: Mapping[str, Any], *, base: PathLike | None = None) -> None:
        self._doc: dict[str, Any] = copy.deepcopy(dict(doc))
        self._normal: dict[str, Any] | None = None
        self.base: Path | None = None if base is None else Path(base)
        self._normalised()

    # -- construction ------------------------------------------------------

    @classmethod
    def new(
        cls,
        task_id: str,
        version: str,
        *,
        identity_namespace: str,
        policy: SelectionPolicy | Mapping[str, Any] | str | None = None,
        slots: Sequence[Slot] = (),
        target: Target | None = None,
        split: tuple[str, Sequence[str]] | None = None,
        description: str | None = None,
        base: PathLike | None = None,
    ) -> TaskManifest:
        """An empty manifest.  ``split`` is ``(set_id, partitions)``, the
        training partition first: learned preprocessing is fitted on it."""
        task: dict[str, Any] = {"id": task_id, "version": str(version)}
        if description is not None:
            task["description"] = description
        doc: dict[str, Any] = {
            "schema": SCHEMA,
            "task": task,
            "identity_namespace": identity_namespace,
            "slots": [s.to_json() for s in slots],
            "target": None if target is None else target.to_json(),
            "subjects": [],
            "rows": [],
        }
        found = _policy_json(policy)
        if found is not None:
            doc["policy"] = found
        if split is not None:
            doc["split"] = {"set_id": split[0], "partitions": list(split[1])}
        return cls(doc, base=base)

    @classmethod
    def load(cls, path: PathLike) -> TaskManifest:
        """Read a manifest; relative source URIs resolve against its directory."""
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            from medh5.errors import MEDH5ValidationError

            raise MEDH5ValidationError(f"{path} is not JSON: {exc}", "T101") from None
        return cls(doc, base=path.parent)

    def save(self, path: PathLike) -> Path:
        """Write the manifest, pretty-printed, with its fingerprint."""
        path = Path(path)
        doc = self.to_json()
        doc["fingerprint"] = self.manifest_fingerprint
        path.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path

    def _normalised(self) -> dict[str, Any]:
        if self._normal is None:
            self._normal = _core.task_normalize(self._doc)
        return self._normal

    def _changed(self) -> None:
        self._normal = None
        self._doc.pop("fingerprint", None)

    def add_subject(
        self,
        subject_id: str,
        sources: Sequence[SourceRef],
        *,
        partition: str | None = None,
        clock_id: str | None = None,
    ) -> TaskManifest:
        """Declare a subject, its sources and its partition."""
        entry: dict[str, Any] = {
            "subject_id": subject_id,
            "sources": [s.to_json() for s in sources],
        }
        if partition is not None:
            entry["partition"] = partition
        if clock_id is not None:
            entry["clock_id"] = clock_id
        self._doc.setdefault("subjects", []).append(entry)
        self._changed()
        return self

    def add_row(self, row_id: str, subject_id: str, cutoff_us: int) -> TaskManifest:
        """Add one example: ``subject_id`` at ``cutoff_us`` on its clock."""
        self._doc.setdefault("rows", []).append(
            {"row_id": row_id, "subject_id": subject_id, "cutoff_us": int(cutoff_us)}
        )
        self._changed()
        return self

    def reconcile(self, base: PathLike | None = None) -> TaskManifest:
        """Record, per subject, the event versions several fragments hold and
        the digest they agree on --- what preflight checks duplicates against."""
        found = _core.task_reconcile(self._doc, self._base(base))
        self._doc["subjects"] = found["subjects"]
        self._changed()
        return self

    # -- reading -----------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        """The normalised manifest (every default filled), as a ``dict``."""
        return copy.deepcopy(self._normalised())

    @property
    def task_id(self) -> str:
        return str(self._normalised()["task"]["id"])

    @property
    def task_version(self) -> str:
        return str(self._normalised()["task"]["version"])

    @property
    def identity_namespace(self) -> str:
        return str(self._normalised()["identity_namespace"])

    @property
    def policy(self) -> SelectionPolicy:
        return SelectionPolicy.from_json(self._normalised()["policy"])

    @property
    def slots(self) -> tuple[Slot, ...]:
        return tuple(Slot.from_json(s) for s in self._normalised()["slots"])

    @property
    def target(self) -> Target | None:
        found = self._normalised().get("target")
        return None if found is None else Target.from_json(found)

    @property
    def split(self) -> tuple[str, tuple[str, ...]] | None:
        found = self._normalised().get("split")
        if found is None:
            return None
        return str(found["set_id"]), tuple(found["partitions"])

    @property
    def training_partition(self) -> str | None:
        """The split's first partition: what learned preprocessing fits on."""
        split = self.split
        return None if split is None or not split[1] else split[1][0]

    @property
    def subjects(self) -> tuple[Subject, ...]:
        return tuple(Subject.from_json(s) for s in self._normalised()["subjects"])

    @property
    def rows(self) -> tuple[Row, ...]:
        return tuple(
            Row(str(r["row_id"]), str(r["subject_id"]), int(r["cutoff_us"]))
            for r in self._normalised()["rows"]
        )

    def partition_of(self, subject_id: str) -> str | None:
        for s in self._normalised()["subjects"]:
            if s["subject_id"] == subject_id:
                found: str | None = s.get("partition")
                return found
        return None

    # -- identity ----------------------------------------------------------

    @property
    def task_fingerprint(self) -> str:
        """Identifies the task: the digest of its canonical definition."""
        return str(_core.task_fingerprints(self._doc)["task"])

    @property
    def manifest_fingerprint(self) -> str:
        """Identifies the whole manifest, rows and pins included."""
        return str(_core.task_fingerprints(self._doc)["manifest"])

    @property
    def declared_fingerprint(self) -> str | None:
        found: str | None = self._normalised().get("fingerprint")
        return found

    def row_fingerprint(self, row_id: str) -> str:
        """Identifies one example: task, subject, pinned versions, cutoff."""
        return str(_core.task_row_fingerprint(self._doc, row_id))

    def subjects_digest(self, partition: str) -> str:
        """Who is in a partition --- what learned preprocessing records."""
        return str(_core.task_subjects_digest(self._doc, partition))

    # -- checking ----------------------------------------------------------

    def validate(self) -> list[Finding]:
        """Everything wrong with the manifest that opening no file can find."""
        return list(_findings(_core.task_validate(self._doc)))

    def _base(self, base: PathLike | None) -> str | None:
        found = base if base is not None else self.base
        return None if found is None else os.fspath(found)

    def preflight(
        self, base: PathLike | None = None, *, deep: bool = False
    ) -> Preflight:
        """Open and check every source, and build every row's view."""
        return Preflight.from_json(
            _core.task_preflight(self._doc, self._base(base), deep)
        )

    def __getstate__(self) -> dict[str, Any]:
        return {
            "doc": self._doc,
            "base": None if self.base is None else os.fspath(self.base),
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._doc = state["doc"]
        self._normal = None
        self.base = None if state["base"] is None else Path(state["base"])

    def __repr__(self) -> str:
        n = self._normalised()
        return (
            f"TaskManifest({self.task_id!r} v{self.task_version}, "
            f"{len(n['subjects'])} subjects, {len(n['rows'])} rows)"
        )


# --------------------------------------------------------------------------
# Preflight: what a task admits, per row
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SlotFill:
    """How one slot was filled for one row --- or that it was not."""

    slot: str
    available: bool
    fragment: int | None
    image_id: str | None
    grid_id: str | None
    event_id: str | None
    center: tuple[int, ...] | None
    roi: str
    annotations: tuple[str, ...]
    """Eligible voxel annotations on the image's grid: admissible as inputs."""
    label_annotations: tuple[str, ...] = ()
    """Every voxel annotation on the grid: supervision, which --- like the
    target --- may come from after the cutoff and never enters an input."""

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SlotFill:
        center = doc.get("center")
        return cls(
            str(doc["slot"]),
            bool(doc["available"]),
            doc.get("fragment"),
            doc.get("image_id"),
            doc.get("grid_id"),
            doc.get("event_id"),
            None if center is None else tuple(int(v) for v in center),
            str(doc["roi"]),
            tuple(doc.get("annotations", ())),
            tuple(doc.get("label_annotations", ())),
        )


@dataclass(frozen=True, slots=True)
class TargetLabel:
    """A row's target: ``positive`` (1.0), ``negative`` (0.0), or unobserved
    (``censored``, ``prevalent``, ``none``) with the reason."""

    status: str
    value: float | None
    event_id: str | None
    reason: str | None

    @property
    def observed(self) -> bool:
        return self.value is not None

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> TargetLabel:
        value = doc.get("value")
        return cls(
            str(doc["status"]),
            None if value is None else float(value),
            doc.get("event_id"),
            doc.get("reason"),
        )


@dataclass(frozen=True, slots=True)
class RowView:
    """Everything a task admits for one row, and nothing else."""

    row_id: str
    subject_id: str
    partition: str | None
    cutoff_us: int
    fingerprint: str
    status: str
    reasons: tuple[str, ...]
    sources: tuple[SourceRef, ...]
    selection: Selection | None
    events: tuple[Event, ...]
    """The selected event versions, in input order."""
    event_fragments: tuple[int, ...]
    """For each selected version, the source (index into ``sources``) it was
    read from."""
    slots: dict[str, SlotFill]
    target: TargetLabel

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> RowView:
        selection = doc.get("selection")
        return cls(
            str(doc["row_id"]),
            str(doc["subject_id"]),
            doc.get("partition"),
            int(doc["cutoff_us"]),
            str(doc["fingerprint"]),
            str(doc["status"]),
            tuple(doc.get("reasons", ())),
            tuple(SourceRef.from_json(s) for s in doc.get("sources", ())),
            None if selection is None else Selection.from_json(selection),
            tuple(Event.from_json(e) for e in doc.get("events", ())),
            tuple(int(f) for f in doc.get("event_fragments", ())),
            {s["slot"]: SlotFill.from_json(s) for s in doc.get("slots", ())},
            TargetLabel.from_json(doc["target"]),
        )


@dataclass(frozen=True, slots=True)
class Preflight:
    """A task's preflight: manifest and source findings, and every row's view.

    Any finding makes the task unfit to train on (:attr:`ok` is ``False``);
    rows whose sources are wrong are in ``error``, with the reason.
    """

    task_fingerprint: str
    manifest_fingerprint: str
    findings: tuple[Finding, ...]
    rows: tuple[RowView, ...]

    @property
    def ok(self) -> bool:
        return not self.findings

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.rows:
            out[r.status] = out.get(r.status, 0) + 1
        return dict(sorted(out.items()))

    def row(self, row_id: str) -> RowView:
        for r in self.rows:
            if r.row_id == row_id:
                return r
        raise KeyError(f"no row {row_id!r}")

    def eligible(self, partition: str | None = None) -> tuple[RowView, ...]:
        """The eligible rows, of one partition when given."""
        return tuple(
            r
            for r in self.rows
            if r.eligible and (partition is None or r.partition == partition)
        )

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Preflight:
        return cls(
            str(doc["task_fingerprint"]),
            str(doc["manifest_fingerprint"]),
            _findings(doc.get("findings", ())),
            tuple(RowView.from_json(r) for r in doc.get("rows", ())),
        )


def preflight(
    task: TaskManifest | PathLike,
    base: PathLike | None = None,
    *,
    deep: bool = False,
) -> Preflight:
    """Preflight a task (a manifest, or the path of one)."""
    manifest = task if isinstance(task, TaskManifest) else TaskManifest.load(task)
    return manifest.preflight(base, deep=deep)


__all__ = [
    "CENSORING",
    "CODES",
    "LABELS",
    "ROI",
    "SCHEMA",
    "STATUSES",
    "Finding",
    "Preflight",
    "Reconciled",
    "Row",
    "RowView",
    "SlotFill",
    "Slot",
    "SourceRef",
    "Subject",
    "Target",
    "TargetLabel",
    "TaskManifest",
    "preflight",
    "schema_text",
]
