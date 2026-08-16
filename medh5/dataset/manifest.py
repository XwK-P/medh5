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

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from medh5.__about__ import __format_version__, __version__
from medh5.errors import MEDH5Error, MEDH5ValidationError
from medh5.labels.labelset import canonical_json

SUFFIXES = (".medh5", ".medh5c")


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
    days_from_baseline: tuple[int | None, ...] = ()
    images: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    annotations: dict[str, dict[str, Any]] = field(default_factory=dict)
    class_ids: tuple[int, ...] = ()
    annotated_class_ids: tuple[int, ...] = ()
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

    def to_json(self) -> dict[str, Any]:
        out = {
            "path": self.path,
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
            "group_id": self.group_id,
            "content_id": self.content_id,
            "profiles": list(self.profiles),
            "timepoints": list(self.timepoints),
            "days_from_baseline": list(self.days_from_baseline),
            "images": list(self.images),
            "modalities": list(self.modalities),
            "annotations": self.annotations,
            "class_ids": list(self.class_ids),
            "annotated_class_ids": list(self.annotated_class_ids),
            "quality": self.quality,
            "splits": [dict(s) for s in self.splits],
            "deidentified": self.deidentified,
            "size": self.size,
            "mtime": self.mtime,
        }
        for name in (
            "sex",
            "laterality",
            "bodypart",
            "dataset_id",
            "site_id",
            "scanner_id",
            "acquisition_protocol",
            "label_set_id",
            "label_set_version",
            "label_set_digest",
            "key",
        ):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Entry:
        return cls(
            path=str(doc["path"]),
            sample_id=str(doc["sample_id"]),
            subject_id=str(doc["subject_id"]),
            group_id=str(doc["group_id"]),
            content_id=doc.get("content_id"),
            profiles=tuple(doc.get("profiles", ())),
            sex=doc.get("sex"),
            laterality=doc.get("laterality"),
            bodypart=doc.get("bodypart"),
            dataset_id=doc.get("dataset_id"),
            site_id=doc.get("site_id"),
            scanner_id=doc.get("scanner_id"),
            acquisition_protocol=doc.get("acquisition_protocol"),
            timepoints=tuple(doc.get("timepoints", ())),
            days_from_baseline=tuple(doc.get("days_from_baseline", ())),
            images=tuple(doc.get("images", ())),
            modalities=tuple(doc.get("modalities", ())),
            annotations=dict(doc.get("annotations", {})),
            class_ids=tuple(int(v) for v in doc.get("class_ids", ())),
            annotated_class_ids=tuple(
                int(v) for v in doc.get("annotated_class_ids", ())
            ),
            label_set_id=doc.get("label_set_id"),
            label_set_version=doc.get("label_set_version"),
            label_set_digest=doc.get("label_set_digest"),
            quality=dict(doc.get("quality", {})),
            splits=tuple(dict(s) for s in doc.get("splits", ())),
            deidentified=bool(doc.get("deidentified", False)),
            key=doc.get("key"),
            size=int(doc.get("size", 0)),
            mtime=float(doc.get("mtime", 0.0)),
        )

    def field(self, dotted: str) -> Any:
        """Read a field by the dotted name the CLI accepts.

        ``cohort.site_id``, ``identity.sex`` and bare ``site_id`` all reach the
        same value, because a curator writing ``--stratify-by`` should not have
        to remember which document section a field came from.
        """
        name = dotted.split(".")[-1]
        if not hasattr(self, name):
            raise MEDH5ValidationError(
                f"{dotted!r} is not a manifest field; try one of "
                f"{', '.join(sorted(GROUPABLE))}"
            )
        return getattr(self, name)


GROUPABLE = (
    "subject_id",
    "group_id",
    "site_id",
    "scanner_id",
    "dataset_id",
    "acquisition_protocol",
    "sex",
    "laterality",
    "bodypart",
    "label_set_id",
)
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
        out: dict[str, list[Entry]] = {}
        for entry in self.entries:
            out.setdefault(str(entry.field(by)), []).append(entry)
        return out

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted({e.subject_id for e in self.entries}))

    def to_json(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "generator": self.generator,
            "root": self.root,
            "sha256": self.sha256(),
            "samples": len(self.entries),
            "entries": [e.to_json() for e in self.entries],
        }

    def sha256(self) -> str:
        """Digest of the cohort's *membership*: which samples, grouped how.

        Only ``sample_id``, ``subject_id`` and ``group_id`` are hashed --- not
        paths, sizes, mtimes, the generating version, and deliberately **not**
        ``content_id``.

        Membership and grouping are exactly what a split is computed from, so
        this is what a split claim should be checkable against.  Including
        content would make the digest unusable for that: writing a claim into a
        file changes the file's content, so every claim would be stale the
        instant it was written.  Content drift is a different question, and
        ``dataset check`` answers it separately (C401, and ``--deep``).
        """
        import hashlib

        payload = [
            {
                "sample_id": e.sample_id,
                "subject_id": e.subject_id,
                "group_id": e.group_id,
            }
            for e in sorted(self.entries, key=lambda e: (e.sample_id, e.path))
        ]
        return hashlib.sha256(canonical_json({"entries": payload})).hexdigest()

    def save(self, path: str | os.PathLike[str]) -> Path:
        target = Path(os.fspath(path))
        target.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Manifest:
        doc = json.loads(Path(os.fspath(path)).read_text())
        return cls.from_json(doc)

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Manifest:
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
        out: list[str] = []
        for entry in self.entries:
            path = Path(entry.path)
            if not path.exists():
                out.append(entry.path)
                continue
            stat = path.stat()
            if stat.st_size != entry.size or abs(stat.st_mtime - entry.mtime) > 1e-6:
                out.append(entry.path)
        return tuple(out)


def find(
    root: str | os.PathLike[str], *, suffixes: Sequence[str] = SUFFIXES
) -> list[Path]:
    """Every sample and collection under *root*, in a stable order."""
    base = Path(os.fspath(root))
    if base.is_file():
        return [base]
    found = [p for p in base.rglob("*") if p.is_file() and p.suffix in set(suffixes)]
    return sorted(found)


def scan(
    root: str | os.PathLike[str],
    *,
    suffixes: Sequence[str] = SUFFIXES,
    on_error: str = "warn",
) -> tuple[Manifest, tuple[str, ...]]:
    """Build a manifest from a directory tree, and report what would not open.

    Returns the manifest and the paths that failed.  A cohort scan that dies on
    one broken file has told you nothing about the other 9 999.
    """
    manifest = Manifest(root=os.fspath(root))
    failures: list[str] = []
    for path in find(root, suffixes=suffixes):
        try:
            manifest.entries.extend(entries_for(path))
        except (MEDH5Error, OSError) as exc:
            if on_error == "raise":
                raise
            failures.append(f"{path}: {exc}")
    return manifest, tuple(failures)


def entries_for(path: str | os.PathLike[str]) -> list[Entry]:
    """The manifest entries in one file: one per sample, so a collection fans out.

    A cohort directory holds whatever the curator put there, so the kind is read
    from the file rather than assumed from its extension.
    """
    from medh5.collection import Collection, open_any

    target = Path(os.fspath(path))
    stat = target.stat()
    with open_any(target) as opened:
        if isinstance(opened, Collection):
            return [
                _entry(sample, str(target), stat, key=key)
                for key, sample in opened.items()
            ]
        return [_entry(opened, str(target), stat)]


def _entry(
    sample: Any, path: str, stat: os.stat_result, *, key: str | None = None
) -> Entry:
    document = sample.document
    identity = document.identity
    cohort = document.cohort
    label_set = document.label_set
    annotations: dict[str, dict[str, Any]] = {}
    classes: set[int] = set()
    annotated: set[int] = set()
    for name, annotation in sample.annotations.items():
        annotations[name] = {
            "kind": annotation.kind,
            "task": annotation.task,
            "classes": [int(c) for c in annotation.class_ids],
            "annotated_classes": [int(c) for c in annotation.annotated_class_ids],
            "timepoints": list(annotation.timepoints),
        }
        classes.update(int(c) for c in annotation.class_ids)
        annotated.update(int(c) for c in annotation.annotated_class_ids)
    return Entry(
        path=path,
        sample_id=identity.sample_id,
        subject_id=identity.subject_id,
        group_id=document.group_id,
        content_id=sample.content_id,
        profiles=tuple(sorted(sample.profiles)),
        sex=identity.sex,
        laterality=identity.laterality,
        bodypart=identity.bodypart,
        dataset_id=cohort.dataset_id,
        site_id=cohort.site_id,
        scanner_id=cohort.scanner_id,
        acquisition_protocol=cohort.acquisition_protocol,
        timepoints=tuple(t.id for t in document.timepoints),
        days_from_baseline=tuple(t.days_from_baseline for t in document.timepoints),
        images=tuple(sample.images),
        modalities=tuple(
            sorted({i.modality for i in sample.images.values() if i.modality})
        ),
        annotations=annotations,
        class_ids=tuple(sorted(classes)),
        annotated_class_ids=tuple(sorted(annotated)),
        label_set_id=label_set.id if label_set else None,
        label_set_version=label_set.version if label_set else None,
        label_set_digest=label_set.digest() if label_set else None,
        quality={k: v.status for k, v in document.quality.items()},
        splits=tuple(s.to_json() for s in document.splits),
        deidentified=document.deidentification is not None,
        key=key,
        size=stat.st_size,
        mtime=stat.st_mtime,
    )


def load(path: str | os.PathLike[str]) -> Manifest:
    return Manifest.load(path)


def counts(entries: Iterable[Entry], by: str) -> dict[str, int]:
    """How many entries carry each value of *by* --- the stratification tally."""
    out: dict[str, int] = {}
    for entry in entries:
        out[str(entry.field(by))] = out.get(str(entry.field(by)), 0) + 1
    return dict(sorted(out.items()))


__all__ = [
    "GROUPABLE",
    "SUFFIXES",
    "Entry",
    "Manifest",
    "counts",
    "entries_for",
    "find",
    "load",
    "scan",
]
