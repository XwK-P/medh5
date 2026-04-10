"""Dataset index: scan directories of .medh5 files and cache lightweight metadata.

A :class:`Dataset` is a list-like collection of :class:`DatasetRecord`
entries. Each record holds a path plus the metadata fields from
:func:`MEDH5File.read_meta` — no array data is touched. Manifests can be
persisted as JSON and reloaded; staleness is detected via per-file
``mtime`` and ``size``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5Error
from medh5.meta import SCHEMA_VERSION

_MANIFEST_VERSION = 1


@dataclass
class DatasetRecord:
    """One entry in a :class:`Dataset` manifest.

    Holds the file path plus a flat snapshot of metadata fields needed
    for filtering and splitting. Array data is never loaded.
    """

    path: str
    label: int | str | None = None
    label_name: str | None = None
    image_names: list[str] = field(default_factory=list)
    shape: list[int] | None = None
    spacing: list[float] | None = None
    coord_system: str | None = None
    patch_size: list[int] | None = None
    has_seg: bool = False
    seg_names: list[str] | None = None
    has_bbox: bool = False
    review_status: str = "pending"
    extra: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION
    file_size: int = 0
    mtime: float = 0.0

    @classmethod
    def from_path(cls, path: str | Path) -> DatasetRecord:
        """Build a record by reading metadata-only from a ``.medh5`` file."""
        p = Path(path)
        meta = MEDH5File.read_meta(p)
        stat = p.stat()
        review_status = "pending"
        if meta.extra is not None:
            review = meta.extra.get("review")
            if isinstance(review, dict):
                raw_status = review.get("status")
                if isinstance(raw_status, str) and raw_status:
                    review_status = raw_status
        return cls(
            path=str(p),
            label=meta.label,
            label_name=meta.label_name,
            image_names=list(meta.image_names) if meta.image_names else [],
            shape=list(meta.shape) if meta.shape else None,
            spacing=(
                list(meta.spatial.spacing) if meta.spatial.spacing is not None else None
            ),
            coord_system=meta.spatial.coord_system,
            patch_size=list(meta.patch_size) if meta.patch_size else None,
            has_seg=meta.has_seg,
            seg_names=list(meta.seg_names) if meta.seg_names else None,
            has_bbox=meta.has_bbox,
            review_status=review_status,
            extra=dict(meta.extra) if meta.extra else None,
            schema_version=meta.schema_version,
            file_size=stat.st_size,
            mtime=stat.st_mtime,
        )

    def is_stale(self) -> bool:
        """Return True if the on-disk file no longer matches the cached stat."""
        p = Path(self.path)
        if not p.exists():
            return True
        stat = p.stat()
        return stat.st_size != self.file_size or stat.st_mtime != self.mtime


class Dataset:
    """List-like collection of :class:`DatasetRecord` entries.

    Build from a directory with :meth:`from_directory`, filter with
    :meth:`filter`, persist with :meth:`save` / :meth:`load`. Indexing
    and ``len()`` work as expected; iteration yields records.
    """

    def __init__(self, records: list[DatasetRecord]) -> None:
        self.records: list[DatasetRecord] = list(records)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        recursive: bool = True,
        skip_invalid: bool = False,
    ) -> Dataset:
        """Scan *root* for ``.medh5`` files and build a Dataset.

        Parameters
        ----------
        root : str or Path
            Directory to scan.
        recursive : bool
            Recurse into subdirectories (default ``True``).
        skip_invalid : bool
            If True, silently skip files that fail to read; otherwise
            re-raise the underlying :class:`MEDH5Error`.
        """
        root_p = Path(root)
        pattern = "**/*.medh5" if recursive else "*.medh5"
        records: list[DatasetRecord] = []
        for path in sorted(root_p.glob(pattern)):
            try:
                records.append(DatasetRecord.from_path(path))
            except MEDH5Error:
                if not skip_invalid:
                    raise
        return cls(records)

    @classmethod
    def from_paths(cls, paths: list[str | Path]) -> Dataset:
        """Build from an explicit list of file paths."""
        return cls([DatasetRecord.from_path(p) for p in paths])

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DatasetRecord:
        return self.records[idx]

    def __iter__(self) -> Iterator[DatasetRecord]:
        return iter(self.records)

    # ------------------------------------------------------------------
    # Filtering / projection
    # ------------------------------------------------------------------

    def filter(self, predicate: Callable[[DatasetRecord], bool]) -> Dataset:
        """Return a new Dataset containing only records matching *predicate*."""
        return Dataset([r for r in self.records if predicate(r)])

    @property
    def paths(self) -> list[str]:
        """Return the file paths in current order."""
        return [r.path for r in self.records]

    def stale(self) -> list[DatasetRecord]:
        """Return records whose underlying files have changed since indexing."""
        return [r for r in self.records if r.is_stale()]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the manifest as JSON."""
        payload = {
            "manifest_version": _MANIFEST_VERSION,
            "medh5_schema_version": SCHEMA_VERSION,
            "records": [asdict(r) for r in self.records],
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Load a manifest written by :meth:`save`."""
        payload = json.loads(Path(path).read_text())
        records = [DatasetRecord(**rec) for rec in payload.get("records", [])]
        return cls(records)
