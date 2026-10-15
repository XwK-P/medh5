"""Cohort manifests: what a metadata-only scan of a directory tree knows.

Opening ten thousand samples to answer "how many have a liver mask" is a
minute of I/O for a question the ``/meta`` document can answer in a
millisecond.  A manifest is that answer, cached: one metadata-only pass writes
a JSON file, and splitting, stratification, filtering and cohort checks all run
against it without touching a single voxel.

The manifest is also the **authority for splits** (spec §12.3).  A sample
carries a ``SplitClaim``, but a claim is a claim; the manifest's ``sha256`` is
what lets a reader notice that a file's claim predates the current split
instead of quietly training on a stale partition.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias

from medh5 import _core, _deprecation
from medh5._core import __format_version__, __version__

SUFFIXES: tuple[str, ...] = _core.MANIFEST_SUFFIXES

SCHEMA: str = _core.MANIFEST_SCHEMA
"""What every manifest :meth:`Manifest.save` writes names under ``schema``."""

PathStyle: TypeAlias = Literal["keep", "relative", "absolute"]
"""How :meth:`Manifest.save` writes paths."""


@dataclass(slots=True)
class Entry:
    """One sample's metadata, as far as a cohort needs it."""

    path: str
    sample_id: str
    subject_id: str
    group_id: str
    content_id: str | None = None
    profiles: tuple[str, ...] = ()
    sex: str | None = None
    laterality: str | None = None
    bodypart: str | None = None
    dataset_id: str | None = None
    site_id: str | None = None
    scanner_id: str | None = None
    acquisition_protocol: str | None = None
    timepoints: tuple[str, ...] = ()
    days_from_baseline: tuple[float | None, ...] = ()
    images: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    class_ids: tuple[int, ...] = ()
    annotated_class_ids: tuple[int, ...] = ()
    class_keys: dict[int, str] = field(default_factory=dict)
    """The label set's key for each class id an annotation names --- what the
    id *means*, which ``dataset check`` compares between label sets (C102).
    Empty where the sample declares no label set, or one by reference."""
    label_set_id: str | None = None
    label_set_version: str | None = None
    label_set_digest: str | None = None
    quality: dict[str, str] = field(default_factory=dict)
    splits: tuple[dict[str, Any], ...] = ()
    deidentified: bool = False
    key: str | None = None
    """The sample's key inside a collection, or ``None`` for a plain file."""
    size: int = 0
    mtime: float = 0.0

    @property
    def is_longitudinal(self) -> bool:
        return len(self.timepoints) > 1

    def has_class(self, class_id: int) -> bool:
        """Whether any annotation *names* the class --- see :meth:`examined`."""
        return class_id in self.class_ids

    def examined(self, class_id: int) -> bool:
        """Whether the class was looked for (§11.3), found or not.

        The distinction matters for training: a sample that was never examined
        for a class is not a negative example of it.
        """
        return class_id in self.annotated_class_ids

    def _fields(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_entry_json(self._fields())
        return found

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Entry:
        """An entry from its JSON, read as the engine reads a manifest's."""
        return cls._from_fields(_core.dataset_entry_json(doc))

    @classmethod
    def _from_fields(cls, doc: Mapping[str, Any]) -> Entry:
        """An entry from the JSON the engine writes, which is already checked."""
        return cls(
            path=doc["path"],
            sample_id=doc["sample_id"],
            subject_id=doc["subject_id"],
            group_id=doc["group_id"],
            content_id=doc.get("content_id"),
            profiles=tuple(doc["profiles"]),
            sex=doc.get("sex"),
            laterality=doc.get("laterality"),
            bodypart=doc.get("bodypart"),
            dataset_id=doc.get("dataset_id"),
            site_id=doc.get("site_id"),
            scanner_id=doc.get("scanner_id"),
            acquisition_protocol=doc.get("acquisition_protocol"),
            timepoints=tuple(doc["timepoints"]),
            days_from_baseline=tuple(doc["days_from_baseline"]),
            images=tuple(doc["images"]),
            modalities=tuple(doc["modalities"]),
            annotations=dict(doc["annotations"]),
            class_ids=tuple(doc["class_ids"]),
            annotated_class_ids=tuple(doc["annotated_class_ids"]),
            class_keys={int(k): v for k, v in doc["class_keys"].items()},
            label_set_id=doc.get("label_set_id"),
            label_set_version=doc.get("label_set_version"),
            label_set_digest=doc.get("label_set_digest"),
            quality=dict(doc["quality"]),
            splits=tuple(doc["splits"]),
            deidentified=doc["deidentified"],
            key=doc.get("key"),
            size=doc["size"],
            mtime=doc["mtime"],
        )

    def field(self, dotted: str) -> Any:
        """Read a field by the dotted name the CLI accepts.

        ``cohort.site_id``, ``identity.sex`` and bare ``site_id`` all reach the
        same value, because a curator writing ``--stratify-by`` should not have
        to remember which document section a field came from.  A name that is
        not a field is refused.
        """
        _core.dataset_entry_field(self._fields(), dotted)
        return getattr(self, dotted.split(".")[-1])


GROUPABLE: tuple[str, ...] = _core.GROUPABLE
"""Fields worth grouping or stratifying on --- all single-valued and scannable."""


@dataclass(slots=True)
class Manifest:
    """A cohort as metadata, with the digest that makes a split checkable."""

    entries: list[Entry] = field(default_factory=list)
    root: str | None = None
    generator: str = f"medh5 {__version__}"
    format: str = __format_version__

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Entry]:
        return iter(self.entries)

    def __getitem__(self, index: int) -> Entry:
        return self.entries[index]

    def by_path(self, path: str | os.PathLike[str]) -> Entry | None:
        text = os.fspath(path)
        for entry in self.entries:
            if entry.path == text:
                return entry
        return None

    def filter(self, predicate: Callable[[Entry], bool]) -> Manifest:
        """A manifest over the entries that satisfy *predicate*.

        The digest changes with the contents, so a split made from a filtered
        manifest cannot be mistaken for one made from the whole cohort.
        """
        return Manifest(
            entries=[e for e in self.entries if predicate(e)],
            root=self.root,
            generator=self.generator,
            format=self.format,
        )

    def groups(self, by: str = "group_id") -> dict[str, list[Entry]]:
        """Entries by the string value of a field, in first-seen order."""
        out: dict[str, list[Entry]] = {}
        keys = _core.dataset_group_keys([e._fields() for e in self.entries], by)
        for key, entry in zip(keys, self.entries, strict=True):
            out.setdefault(key, []).append(entry)
        return out

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({e.subject_id for e in self.entries}))

    def _doc(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "generator": self.generator,
            "root": self.root,
            "entries": [e._fields() for e in self.entries],
        }

    def to_json(self) -> dict[str, Any]:
        found: dict[str, Any] = _core.dataset_manifest_json(self._doc())
        return found

    def sha256(self) -> str:
        """Digest of the cohort's *membership*: which samples, grouped how.

        Only ``sample_id``, ``subject_id`` and ``group_id`` are hashed --- not
        paths, sizes, mtimes, the generating version, and deliberately **not**
        ``content_id``.

        Membership and grouping by ``group_id`` are what a default split is
        computed from, so this is what a split claim should be checkable
        against.  A split grouped or stratified on another field also depends
        on that field's values, which this does not cover: re-curating them
        changes the split and not the digest, so re-split after changing a
        field you split on.  Including content would make the digest unusable
        for claims: writing a claim into a file changes the file's content, so
        every claim would be stale the instant it was written.  Content drift
        is a different question, and ``dataset check`` answers it separately
        (C401, and ``--deep``).
        """
        return str(_core.dataset_manifest_sha256(self._doc()))

    def save(self, path: str | os.PathLike[str], *, paths: PathStyle = "keep") -> Path:
        """Write the manifest (``medh5.dataset/1``) to *path*, atomically.

        In memory a relative path is relative to the working directory; in
        the file it is relative to the manifest's directory when it lies
        beneath it, and absolute otherwise, so the manifest opens from any
        directory and moves with its cohort.  ``paths="keep"`` (the default)
        leaves a path given absolute absolute, ``"relative"`` writes it
        relative too where it lies beneath, and ``"absolute"`` writes every
        path absolute.  The digest covers no path, so none of this changes it.
        """
        return Path(
            _core.dataset_manifest_save(self._doc(), os.fspath(path), paths=paths)
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        base: str | os.PathLike[str] | None = None,
    ) -> Manifest:
        """Read a manifest written by :meth:`save` or ``medh5 dataset index``.

        Each relative path is read against the manifest's directory and comes
        back relative to the working directory.  A manifest written before 2.1
        (no ``schema``) is read against the working directory, as it was
        written, with a ``FutureWarning`` where 3.0 would read it otherwise;
        *base* names the directory its relative paths are relative to.
        """
        return _load(path, base, stacklevel=4)

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Manifest:
        """A manifest from its JSON, with its paths as the document holds them.

        An unknown ``schema`` is refused.
        """
        _core.dataset_manifest_legacy(doc)
        return cls(
            entries=[Entry.from_json(e) for e in doc.get("entries", ())],
            root=doc.get("root"),
            generator=str(doc.get("generator", "")),
            format=str(doc.get("format", __format_version__)),
        )

    def stale(self) -> tuple[str, ...]:
        """Paths whose file no longer matches what the manifest recorded.

        Size and mtime are the cheap check.  They are not proof of equality ---
        ``content_id`` is, and ``medh5 dataset check --deep`` uses it --- but
        they catch the overwhelming majority of "somebody re-ran the converter"
        without opening anything.
        """
        return tuple(_core.dataset_manifest_stale(self._doc()))


def find(
    root: str | os.PathLike[str], *, suffixes: Sequence[str] = SUFFIXES
) -> list[Path]:
    """Every sample and collection under *root*, in a stable order."""
    return [
        Path(p) for p in _core.dataset_find(os.fspath(root), suffixes=list(suffixes))
    ]


def scan(
    root: str | os.PathLike[str],
    *,
    suffixes: Sequence[str] = SUFFIXES,
    on_error: str = "warn",
) -> tuple[Manifest, tuple[str, ...]]:
    """Build a manifest from a directory tree, and report what would not open.

    Returns the manifest and the paths that failed: files that would not open,
    and directories that could not be listed, whose samples the manifest
    lacks.  A cohort scan that dies on one broken file has told you nothing
    about the other 9 999; ``on_error="raise"`` re-raises the first failure
    instead (the default is ``"warn"``, which also issues a ``UserWarning``
    naming how many paths failed --- the manifest lacks their samples --- and
    any other value is refused).
    """
    doc, failures = _core.dataset_scan(
        os.fspath(root), suffixes=list(suffixes), on_error=on_error
    )
    if failures:
        more = f" (and {len(failures) - 1} more)" if len(failures) > 1 else ""
        warnings.warn(
            f"{len(failures)} path(s) under {os.fspath(root)!r} could not be "
            f"scanned, so the manifest lacks their samples: {failures[0]}{more}",
            UserWarning,
            stacklevel=2,
        )
    return Manifest.from_json(doc), tuple(failures)


def entries_for(path: str | os.PathLike[str]) -> list[Entry]:
    """The manifest entries in one file: one per sample, so a collection fans
    out into one entry per member (each with its ``key``)."""
    return [
        Entry._from_fields(doc) for doc in _core.dataset_entries_for(os.fspath(path))
    ]


def load(
    path: str | os.PathLike[str], *, base: str | os.PathLike[str] | None = None
) -> Manifest:
    """Read a manifest written by ``Manifest.save`` or ``medh5 dataset index``;
    see :meth:`Manifest.load`."""
    return _load(path, base, stacklevel=4)


def _load(
    path: str | os.PathLike[str],
    base: str | os.PathLike[str] | None,
    *,
    stacklevel: int,
) -> Manifest:
    doc, differs = _core.dataset_manifest_load(
        os.fspath(path), base=None if base is None else os.fspath(base)
    )
    if differs:
        _deprecation.warn(
            f"medh5 3.0 reads the paths of {os.fspath(path)!r} against its own "
            "directory: pass base=… or run `medh5 dataset upgrade`",
            category=FutureWarning,
            stacklevel=stacklevel,
        )
    return Manifest.from_json(doc)


def counts(entries: Iterable[Entry], by: str) -> dict[str, int]:
    """How many entries carry each value of *by* --- the stratification tally."""
    found: dict[str, int] = _core.dataset_counts([e._fields() for e in entries], by)
    return found


__all__ = [
    "GROUPABLE",
    "SCHEMA",
    "SUFFIXES",
    "PathStyle",
    "Entry",
    "Manifest",
    "counts",
    "entries_for",
    "find",
    "load",
    "scan",
]
