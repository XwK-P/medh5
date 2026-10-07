"""Collections --- many sample roots in one file (spec §2.2).

One sample per file is the primary mode, and nothing here changes that.  A
collection exists for the one case where the primary mode fails operationally:
100 000 chest radiographs at 300 KiB each spend more time in ``open`` than in
``read``, and a directory of them is hostile to every filesystem and object
store.  Packing them into shards amortises the per-file cost.

The containment is strict:

.. code-block:: text

    /                       medh5_kind = "collection", medh5_version
    └── samples/
        ├── <sample_key>/   structurally identical to a standalone sample root
        └── <sample_key>/

Because a sample root inside a collection is *exactly* a sample root, extraction
is a pure copy --- :func:`unpack` moves the raw chunks, so the extracted file's
datasets are byte-identical to the packed ones and its ``content_id`` is
unchanged.  That is the property that makes packing safe to do late and safe to
undo: a shard is a container for samples, never a different encoding of them.

Splits stay per-sample for the same reason.  A shard is an I/O decision; a
partition is a cohort decision (§12.3), and the two must not be forced to agree.
Whole-subject splitting therefore still works on a packed dataset --- membership
is read from each sample root, not from the shard it happens to live in.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from medh5 import _core
from medh5.sample import Sample

SUFFIX: str = _core.COLLECTION_SUFFIX
"""Conventional extension for a collection file (spec §2.1)."""

SAMPLES_GROUP: str = _core.SAMPLES_GROUP
"""Where sample roots live inside a collection (spec §2.2)."""


def is_collection(root: Any) -> bool:
    """Whether a root declares itself a collection.

    *root* is a ``Group`` (``Sample.root``), an open sample or collection, or a
    path.
    """
    return bool(_core.is_collection(root))


class Collection(Mapping[str, Sample]):
    """A read-only mapping of ``sample_key -> Sample`` over one shard.

    Members are samples in the shard's own file; reading one is reading the
    shard.  A member stays readable until the last reference to it goes, even
    after :meth:`close`.
    """

    __slots__ = ("_handle", "_samples", "path")

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.path: str | None = handle.path
        self._samples: dict[str, Sample] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Collection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the shard.  Safe to call twice."""
        self._handle.close()

    def __repr__(self) -> str:
        return str(self._handle.repr())

    # -- mapping -----------------------------------------------------------

    @property
    def root(self) -> Any:
        """The shard's root group."""
        return self._handle.root

    @property
    def group(self) -> Any:
        """The ``samples`` group."""
        return self._handle.group

    def __iter__(self) -> Iterator[str]:
        return iter(self._handle.keys())

    def __len__(self) -> int:
        return len(self._handle)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and bool(self._handle.contains(key))

    def __getitem__(self, key: str) -> Sample:
        if key not in self._samples:
            self._samples[key] = Sample(self._handle.get(key))
        return self._samples[key]

    # -- header ------------------------------------------------------------

    @property
    def version(self) -> str:
        return str(self._handle.version)

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found

    def subject_ids(self) -> dict[str, str]:
        """``sample_key -> subject_id`` --- what a split has to group by (§12.2)."""
        found: dict[str, str] = self._handle.subject_ids()
        return found


def open_collection(path: str | os.PathLike[str]) -> Collection:
    """Open a ``.medh5c`` shard, read-only.

    There is no mode: :class:`Collection` has no mutating method, and neither
    has :class:`~medh5.sample.Sample`.  ``pack``, ``unpack`` and ``amend`` on an
    extracted member are the write paths.
    """
    return Collection(_core.open_collection(os.fspath(path)))


def default_key(path: str | os.PathLike[str]) -> str:
    """The sample key a file gets when none is supplied: its stem."""
    return str(_core.default_key(os.fspath(path)))


def pack(
    sources: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    keys: Sequence[str] | None = None,
) -> Path:
    """Copy sample files into one collection shard (spec §2.2).

    Chunks move as raw bytes, so ``unpack(pack(x)) == x`` down to the
    compressed bytes and every ``content_id`` is unchanged.  *keys* name the
    members (default: each file's stem); they must be unique and valid sample
    keys.
    """
    return Path(
        _core.pack(
            [os.fspath(s) for s in sources],
            os.fspath(out),
            keys=None if keys is None else list(keys),
        )
    )


def unpack(
    path: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    keys: Sequence[str] | None = None,
    suffix: str = ".medh5",
) -> list[Path]:
    """Extract sample roots back into standalone files (spec §2.2).

    Member names come from the file and each becomes a file name, so every one
    is validated as a sample key first.
    """
    return [
        Path(p)
        for p in _core.unpack(
            os.fspath(path),
            os.fspath(outdir),
            keys=None if keys is None else list(keys),
            suffix=suffix,
        )
    ]


def extract(
    path: str | os.PathLike[str], key: str, out: str | os.PathLike[str]
) -> Path:
    """Extract one member of a collection into a standalone sample file."""
    return Path(_core.extract(os.fspath(path), key, os.fspath(out)))


def open_any(
    path: str | os.PathLike[str], *, key: str | None = None
) -> Sample | Collection:
    """Open a file whatever its kind, resolving *key* inside a collection.

    A collection without *key* comes back as a :class:`Collection`; with it,
    as the member :class:`~medh5.sample.Sample`.
    """
    found = _core.open_any(os.fspath(path), key=key)
    if isinstance(found, _core.CollectionHandle):
        return Collection(found)
    return Sample(found)


__all__ = [
    "SAMPLES_GROUP",
    "SUFFIX",
    "Collection",
    "default_key",
    "extract",
    "is_collection",
    "open_any",
    "open_collection",
    "pack",
    "unpack",
]
