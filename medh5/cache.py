"""Feature caches (``medh5.cache/1``): derived features, pinned to their sources.

A cache is one HDF5 file beside the data (``docs/spec/task-cache-1.md``
§7--§8): a canonical-JSON **dependency manifest** --- the encoder and its
immutable revision, the preprocessing, the output dtype and shape, and for
learned preprocessing the training split it was fitted on --- and one array
per **entry**, each naming every source version it read and carrying its own
checksum.

Two failures are told apart, because they need different fixes:

- **stale** (T403): a source no longer has the content an entry pins --- the
  cache is right about a sample that changed; rebuild the entry;
- **corrupt** (T401, T402): the cache's own bytes no longer match their
  checksums --- the cache is wrong about itself; rebuild the cache.

A cache is derived.  It never redefines, overrides or invalidates a source;
validating one never touches the samples except to read them.

.. code-block:: python

    from medh5.cache import CacheWriter, HashingTextEncoder, validate_cache

    enc = HashingTextEncoder(dim=64)
    with CacheWriter("notes.medh5cache", level="event", **enc.header()) as w:
        w.add_event(source, "report.v1", enc.encode(text), document_id="rep1")
    validate_cache("notes.medh5cache").ok
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.task import Finding, SourceRef

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.task import RowView, TaskManifest

SCHEMA: str = _core.CACHE_SCHEMA
SUFFIX = ".medh5cache"
LEVELS = ("event", "patient")

PathLike = str | os.PathLike[str]


def schema_text() -> str:
    """The ``medh5.cache/1`` manifest JSON Schema."""
    return str(_core.cache_schema_text())


def event_entry_id(content_id: str, event_id: str) -> str:
    """The entry id of an event-level feature of ``(content_id, event_id)``."""
    return str(_core.cache_event_entry_id(content_id, event_id))


def fitted_on(task: TaskManifest, partition: str | None = None) -> dict[str, Any]:
    """The ``fitted_on`` record for learned preprocessing over ``partition``
    (default: the task's training partition --- the split's first)."""
    chosen = partition if partition is not None else task.training_partition
    if chosen is None:
        from medh5.errors import MEDH5ValidationError

        raise MEDH5ValidationError(
            "learned preprocessing is fitted on a partition, "
            "and the task declares no split",
            "T405",
        )
    found: dict[str, Any] = _core.cache_fitted_on(task.to_json(), chosen)
    return found


def fitted_on_mismatches(task: TaskManifest, fitted: Mapping[str, Any]) -> list[str]:
    """Why preprocessing recorded as *fitted* was not fitted on *task*'s
    training partition (T405): another task, split (``set_id``), membership or
    partition.  Empty when it was --- the one comparison caches and
    vocabularies are held to."""
    return list(_core.cache_fitted_on_mismatches(task.to_json(), dict(fitted)))


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cached feature and everything it depends on."""

    entry_id: str
    sources: tuple[SourceRef, ...]
    digest: str
    event_id: str | None = None
    document_id: str | None = None
    row_id: str | None = None
    cutoff_us: int | None = None
    event_versions: tuple[str, ...] | None = None
    row_fingerprint: str | None = None

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> CacheEntry:
        versions = doc.get("event_versions")
        return cls(
            str(doc["entry_id"]),
            tuple(
                SourceRef(str(s["uri"]), str(s["content_id"]), s.get("sample_key"))
                for s in doc["sources"]
            ),
            str(doc["digest"]),
            doc.get("event_id"),
            doc.get("document_id"),
            doc.get("row_id"),
            doc.get("cutoff_us"),
            None if versions is None else tuple(versions),
            doc.get("row_fingerprint"),
        )


class CacheWriter:
    """Write a cache; it appears at ``path`` only on :meth:`commit`.

    As a context manager it commits when the block succeeds and leaves nothing
    behind when it raises.  Every entry must match the declared output dtype
    and shape (T402), name the source versions it read (T403), and --- at the
    patient level --- pin its row (by id and fingerprint), cutoff and selected
    versions (T404); an event-level entry names no row.
    """

    __slots__ = ("_handle", "digest", "path")

    def __init__(
        self,
        path: PathLike,
        *,
        level: str,
        encoder: Mapping[str, Any],
        output: Mapping[str, Any],
        preprocessing: Mapping[str, Any] | None = None,
        task: TaskManifest | None = None,
        selection: str | None = None,
        fitted_on: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        header: dict[str, Any] = {
            "level": level,
            "encoder": dict(encoder),
            "output": dict(output),
            "preprocessing": dict(preprocessing or {}),
            "task_fingerprint": None if task is None else task.task_fingerprint,
            "selection": selection
            if selection is not None or task is None
            else task.policy.selection,
            "fitted_on": None if fitted_on is None else dict(fitted_on),
        }
        self._handle: Any = _core.cache_create(os.fspath(self.path), header)
        self.digest: str | None = None

    def add(
        self,
        entry_id: str,
        values: npt.ArrayLike,
        *,
        sources: Sequence[SourceRef],
        event_id: str | None = None,
        document_id: str | None = None,
        row_id: str | None = None,
        cutoff_us: int | None = None,
        event_versions: Sequence[str] | None = None,
        row_fingerprint: str | None = None,
    ) -> CacheEntry:
        """Add one feature."""
        entry = {
            "entry_id": entry_id,
            "sources": [s.to_json() for s in sources],
            "event_id": event_id,
            "document_id": document_id,
            "row_id": row_id,
            "cutoff_us": cutoff_us,
            "event_versions": None if event_versions is None else list(event_versions),
            "row_fingerprint": row_fingerprint,
        }
        return CacheEntry.from_json(self._handle.add(entry, np.asarray(values)))

    def add_row(
        self,
        row: RowView,
        values: npt.ArrayLike,
        *,
        entry_id: str | None = None,
        sources: Sequence[SourceRef] | None = None,
    ) -> CacheEntry:
        """Add the feature of one preflight row: its id, fingerprint, cutoff
        and admitted versions pinned from the row itself (§7.2).  *sources*
        are the source versions the feature read --- default, every one the
        row's subject pins."""
        return self.add(
            entry_id if entry_id is not None else row.row_id,
            values,
            sources=list(row.sources) if sources is None else sources,
            row_id=row.row_id,
            cutoff_us=row.cutoff_us,
            event_versions=[e.event_id for e in row.events],
            row_fingerprint=row.fingerprint,
        )

    def add_event(
        self,
        source: SourceRef,
        event_id: str,
        values: npt.ArrayLike,
        *,
        document_id: str | None = None,
    ) -> CacheEntry:
        """Add the feature of one event version (and its document) of ``source``."""
        return self.add(
            event_entry_id(source.content_id, event_id),
            values,
            sources=[source],
            event_id=event_id,
            document_id=document_id,
        )

    def commit(self) -> str:
        """Write the manifest and its checksum, and move the file into place."""
        self.digest = str(self._handle.commit())
        return self.digest

    def abort(self) -> None:
        """Discard the half-written cache."""
        self._handle.abort()

    def __enter__(self) -> CacheWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()


class FeatureCache:
    """An open cache.  Opening checks the manifest's checksum and schema;
    reading an entry checks the entry's."""

    __slots__ = ("_entries", "_handle")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self._entries: tuple[CacheEntry, ...] | None = None

    @classmethod
    def open(cls, path: PathLike) -> FeatureCache:
        return cls(_core.cache_open(os.fspath(path)))

    def __enter__(self) -> FeatureCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the file.  Safe to call twice."""
        self._handle.close()

    def abandon(self) -> None:
        """Drop the handle without closing it --- in a forked child, whose
        parent owns the descriptor (§14.4)."""
        self._handle.abandon()

    @property
    def path(self) -> str:
        return str(self._handle.path)

    @property
    def manifest_digest(self) -> str:
        return str(self._handle.manifest_digest)

    @property
    def header(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.header()
        return found

    @property
    def level(self) -> str:
        return str(self.header["level"])

    @property
    def entries(self) -> tuple[CacheEntry, ...]:
        if self._entries is None:
            self._entries = tuple(
                CacheEntry.from_json(e) for e in self._handle.entries()
            )
        return self._entries

    def get(self, entry_id: str) -> npt.NDArray[Any]:
        """An entry's payload, its checksum verified (T402 when it fails)."""
        found: npt.NDArray[Any] = self._handle.get(entry_id)
        return found

    def event_feature(self, content_id: str, event_id: str) -> npt.NDArray[Any] | None:
        """The feature of one event version of a pinned source, or ``None``."""
        entry = self._handle.event_entry(content_id, event_id)
        return None if entry is None else self.get(str(entry["entry_id"]))

    def row_feature(self, row_id: str) -> npt.NDArray[Any] | None:
        """The patient-level feature of a row, or ``None``."""
        entry = self._handle.row_entry(row_id)
        return None if entry is None else self.get(str(entry["entry_id"]))

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"FeatureCache({self.path!r}, {self.level}, {len(self)} entries)"


def open_cache(path: PathLike) -> FeatureCache:
    """Open a cache file, read-only."""
    return FeatureCache.open(path)


@dataclass(frozen=True, slots=True)
class CacheReport:
    """What validating a cache found."""

    path: str
    entries: int
    findings: tuple[Finding, ...]
    manifest_digest: str | None = None
    """The manifest checksum of the cache validated (``None`` when it could
    not be opened): what a reader that opens the file again checks it reads
    (§8)."""
    level: str | None = None
    """The level of the cache validated."""

    @property
    def ok(self) -> bool:
        return not self.findings

    def _with(self, codes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({f.location for f in self.findings if f.code in codes}))

    @property
    def stale(self) -> tuple[str, ...]:
        """Entries whose sources changed or vanished (T403)."""
        return self._with(("T403",))

    @property
    def corrupt(self) -> tuple[str, ...]:
        """Entries (or ``"manifest"``) whose own bytes are wrong (T401, T402)."""
        return self._with(("T401", "T402"))

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> CacheReport:
        return cls(
            str(doc["path"]),
            int(doc["entries"]),
            tuple(Finding.from_json(f) for f in doc["findings"]),
            doc.get("manifest_digest"),
            doc.get("level"),
        )


def validate_cache(
    path: PathLike,
    *,
    base: PathLike | None = None,
    task: TaskManifest | None = None,
    task_base: PathLike | None = None,
    check_rows: bool = True,
) -> CacheReport:
    """Validate a cache: its checksums, every source pin, and --- given a task
    --- that it was built for this task, fitted on this task's training
    partition, and (``check_rows``) that each row's entry was built at its
    cutoff, from versions it admits, of its own subject's sources.

    ``base`` resolves the entries' relative source URIs (default: the cache's
    directory); ``task_base`` the task's (default: the manifest's).  The row
    checks run the task's preflight; an event-level cache has no row entries,
    so ``check_rows=False`` spares it that.
    """
    if task is not None and task_base is None:
        task_base = task.base
    found = _core.cache_validate(
        os.fspath(path),
        None if base is None else os.fspath(base),
        None if task is None else task.to_json(),
        None if task_base is None else os.fspath(task_base),
        check_rows,
    )
    return CacheReport.from_json(found)


# --------------------------------------------------------------------------
# A deterministic text encoder, for fixtures and examples
# --------------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


class HashingTextEncoder:
    """Signed feature hashing of lower-cased words into ``dim`` floats.

    Not a clinical language model: a dependency-free, deterministic encoder
    whose output is a pure function of the text and ``dim``, so a cache built
    with it is reproducible on any machine --- what tests and the runnable
    examples need.  Swap in a real encoder by giving the cache its name and
    an immutable revision; nothing else changes.
    """

    NAME = "org.medh5.hashing-text"
    REVISION = "1"

    __slots__ = ("dim",)

    def __init__(self, dim: int = 64) -> None:
        if int(dim) < 1:
            raise ValueError("an encoder needs at least one output dimension")
        self.dim = int(dim)

    def encode(self, text: str) -> npt.NDArray[np.float32]:
        out = np.zeros(self.dim, dtype=np.float64)
        for word in _WORD.findall(text.lower()):
            h = hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(h[:4], "little") % self.dim
            out[index] += 1.0 if h[4] & 1 else -1.0
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out /= norm
        return out.astype(np.float32)

    def header(self) -> dict[str, Any]:
        """``encoder``, ``preprocessing`` and ``output`` for :class:`CacheWriter`."""
        return {
            "encoder": {
                "name": self.NAME,
                "revision": self.REVISION,
                "tokenizer": {"name": "unicode-words-lowercase", "revision": "1"},
            },
            "preprocessing": {"dim": self.dim, "normalise": "l2", "hash": "blake2b-64"},
            "output": {"dtype": "float32", "shape": [self.dim], "pooling": "sum"},
        }


def build_document_cache(
    task: TaskManifest,
    path: PathLike,
    encoder: HashingTextEncoder | Any,
    *,
    base: PathLike | None = None,
) -> CacheReport:
    """An event-level cache of every document event of the task's sources.

    One entry per ``(source, document event)``, encoding the document the
    event describes.  Event-level features are cutoff-free: one event version
    is immutable, so its feature is reused by every row --- and each row's
    selection decides which of them it may read.  Returns the new cache's
    validation report.
    """
    root = base if base is not None else task.base
    with CacheWriter(path, level="event", **encoder.header()) as writer:
        for subject in task.subjects:
            for source in subject.sources:
                with source.open(root) as sample:
                    clinical = sample.clinical
                    if clinical is None:
                        continue
                    kinds = {e.event_id: e.kind for e in clinical.events}
                    for link in clinical.links:
                        if (
                            link.relation == "describes"
                            and link.source_type == "event"
                            and link.target_type == "document"
                            and kinds.get(link.source_id) == "document"
                        ):
                            text = clinical.text(link.target_id)
                            writer.add_event(
                                source,
                                link.source_id,
                                encoder.encode(text),
                                document_id=link.target_id,
                            )
    return validate_cache(path, base=root)


__all__ = [
    "LEVELS",
    "SCHEMA",
    "SUFFIX",
    "CacheEntry",
    "CacheReport",
    "CacheWriter",
    "FeatureCache",
    "HashingTextEncoder",
    "build_document_cache",
    "event_entry_id",
    "fitted_on",
    "fitted_on_mismatches",
    "open_cache",
    "schema_text",
    "validate_cache",
]
