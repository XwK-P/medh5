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

import h5py

from medh5._hdf5 import (
    as_str,
    atomic_h5,
    copy_object,
    open_h5,
    validate_sample_key,
)
from medh5.errors import MEDH5FileError, MEDH5ValidationError
from medh5.sample import FORMAT_VERSION, Sample

SUFFIX = ".medh5c"
"""Conventional extension for a collection file (spec §2.1)."""

SAMPLES_GROUP = "samples"
"""Where sample roots live inside a collection (spec §2.2)."""


def is_collection(root: h5py.Group) -> bool:
    """Whether an open root declares itself a collection."""
    return as_str(root.attrs.get("medh5_kind", "sample")) == "collection"


class Collection(Mapping[str, Sample]):
    """A read-only mapping of ``sample_key -> Sample`` over one shard.

    Each value is an ordinary :class:`~medh5.sample.Sample`; every reader,
    validator and loader that works on a file works unchanged on a member,
    because the member *is* a sample root.
    """

    __slots__ = ("_handle", "_owns_handle", "_samples", "path", "root")

    def __init__(
        self,
        root: h5py.Group,
        *,
        handle: h5py.File | None = None,
        owns_handle: bool = False,
        path: str | None = None,
    ) -> None:
        self.root = root
        self.path = path
        self._handle = handle
        self._owns_handle = owns_handle
        if SAMPLES_GROUP not in root:
            raise MEDH5ValidationError(
                f"collection {path or '<memory>'} has no {SAMPLES_GROUP!r} group",
                code="E008",
            )
        self._samples: dict[str, Sample] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Collection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._samples.clear()
        if self._owns_handle and self._handle is not None:
            self._handle.close()
            self._handle = None

    def __repr__(self) -> str:
        return f"Collection({self.path!r}, {len(self)} samples)"

    # -- mapping -----------------------------------------------------------

    @property
    def group(self) -> h5py.Group:
        node: h5py.Group = self.root[SAMPLES_GROUP]
        return node

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self.group))

    def __len__(self) -> int:
        return len(self.group)

    def __getitem__(self, key: str) -> Sample:
        if key not in self._samples:
            group = self.group
            if key not in group:
                raise KeyError(
                    f"collection has no sample {key!r}; known keys: {sorted(group)}"
                )
            self._samples[key] = Sample(
                group[key], handle=self._handle, path=f"{self.path}::{key}"
            )
        return self._samples[key]

    # -- header ------------------------------------------------------------

    @property
    def version(self) -> str:
        return as_str(self.root.attrs["medh5_version"])

    @property
    def kind(self) -> str:
        return as_str(self.root.attrs.get("medh5_kind", "sample"))

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "kind": self.kind,
            "samples": [
                {
                    "key": key,
                    "sample_id": sample.identity.sample_id,
                    "subject_id": sample.identity.subject_id,
                    "profiles": sorted(sample.profiles),
                    "content_id": sample.content_id,
                    "timepoints": list(sample.timepoints.ids),
                    "images": sorted(sample.images),
                    "annotations": sorted(sample.annotations),
                }
                for key, sample in self.items()
            ],
        }

    def subject_ids(self) -> dict[str, str]:
        """``sample_key -> subject_id`` --- what a split has to group by (§12.2)."""
        return {key: sample.identity.subject_id for key, sample in self.items()}


def open_collection(path: str | os.PathLike[str], mode: str = "r") -> Collection:
    """Open a ``.medh5c`` shard."""
    handle = open_h5(path, mode)
    try:
        if not is_collection(handle):
            raise MEDH5ValidationError(
                f"{os.fspath(path)!r} declares medh5_kind="
                f"{as_str(handle.attrs.get('medh5_kind', 'sample'))!r}, not "
                "'collection'; open it with `medh5.open`",
                code="E006",
            )
        return Collection(handle, handle=handle, owns_handle=True, path=os.fspath(path))
    except BaseException:
        handle.close()
        raise


# --------------------------------------------------------------------------
# pack / unpack
# --------------------------------------------------------------------------


def default_key(path: str | os.PathLike[str]) -> str:
    """The sample key a file gets when none is supplied: its stem."""
    return validate_sample_key(Path(os.fspath(path)).name.split(".")[0])


def pack(
    sources: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    keys: Sequence[str] | None = None,
) -> Path:
    """Copy sample files into one collection shard (spec §2.2).

    Sample roots are copied object-by-object with HDF5's own copy, so chunks
    move as raw bytes: nothing is decompressed, re-chunked or re-encoded, and
    every ``content_id`` in the shard still addresses the same content it did in
    the standalone file.  Packing is therefore a container operation, and
    ``unpack(pack(x)) == x`` down to the compressed bytes.
    """
    paths = [Path(os.fspath(p)) for p in sources]
    if not paths:
        raise MEDH5ValidationError("pack needs at least one sample file", code="E008")
    if keys is not None and len(keys) != len(paths):
        raise MEDH5ValidationError(
            f"{len(keys)} keys for {len(paths)} sources", code="E003"
        )
    chosen = [
        validate_sample_key(str(keys[i])) if keys is not None else default_key(path)
        for i, path in enumerate(paths)
    ]
    duplicates = sorted({k for k in chosen if chosen.count(k) > 1})
    if duplicates:
        raise MEDH5ValidationError(
            f"sample key(s) {duplicates} are not unique in the collection; "
            "pass explicit --key values",
            code="E003",
        )
    target = Path(os.fspath(out))
    with atomic_h5(target) as handle:
        handle.attrs["medh5_version"] = FORMAT_VERSION
        handle.attrs["medh5_kind"] = "collection"
        group = handle.create_group(SAMPLES_GROUP)
        for key, source in zip(chosen, paths, strict=True):
            with open_h5(source, "r") as src:
                if is_collection(src):
                    raise MEDH5ValidationError(
                        f"{source} is already a collection; pack takes sample files",
                        code="E006",
                    )
                _copy_root(src, group.create_group(key))
    return target


def unpack(
    path: str | os.PathLike[str],
    outdir: str | os.PathLike[str],
    *,
    keys: Sequence[str] | None = None,
    suffix: str = ".medh5",
) -> list[Path]:
    """Extract sample roots back into standalone files (spec §2.2)."""
    directory = Path(os.fspath(outdir))
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with open_collection(path) as collection:
        wanted = (
            list(collection) if keys is None else [validate_sample_key(k) for k in keys]
        )
        missing = [k for k in wanted if k not in collection]
        if missing:
            raise MEDH5ValidationError(
                f"collection has no sample(s) {missing}; known keys: "
                f"{sorted(collection)}",
                code="E003",
            )
        group = collection.group
        for key in wanted:
            destination = directory / f"{key}{suffix}"
            _write_sample_root(group[key], destination)
            written.append(destination)
    return written


def _write_sample_root(src: h5py.Group, destination: Path) -> None:
    """Write one collection member out as a standalone sample file."""
    with atomic_h5(destination) as handle:
        _copy_root(src, handle)
        handle.attrs["medh5_kind"] = "sample"
        if "medh5_version" not in handle.attrs:
            handle.attrs["medh5_version"] = FORMAT_VERSION


def _copy_root(src: h5py.Group, dst: h5py.Group) -> None:
    """Copy every child and attribute of one sample root into another."""
    for name in src:
        copy_object(src, name, dst)
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def extract(
    path: str | os.PathLike[str], key: str, out: str | os.PathLike[str]
) -> Path:
    """Extract one member of a collection into a standalone sample file.

    Writes *out* and nothing else.  This used to go through ``unpack``, which
    names its output after the member key, and then move the result into place
    --- so extracting ``case1`` to ``renamed.medh5`` destroyed any unrelated
    ``case1.medh5`` already sitting in the same directory, silently and
    unrecoverably, on the way past.
    """
    target = Path(os.fspath(out))
    target.parent.mkdir(parents=True, exist_ok=True)
    wanted = validate_sample_key(key)
    with open_collection(path) as collection:
        if wanted not in collection:
            raise MEDH5ValidationError(
                f"collection has no sample {wanted!r}; known keys: "
                f"{sorted(collection)}",
                code="E003",
            )
        _write_sample_root(collection.group[wanted], target)
    return target


def open_any(
    path: str | os.PathLike[str], *, key: str | None = None
) -> Sample | Collection:
    """Open a file whatever its kind, resolving *key* inside a collection.

    A caller that already knows which it has should use ``medh5.open`` or
    :func:`open_collection`; this exists for tools that take a path from a user.
    """
    handle = open_h5(path, "r")
    try:
        if is_collection(handle):
            collection = Collection(
                handle, handle=handle, owns_handle=True, path=os.fspath(path)
            )
            if key is None:
                return collection
            member = collection[key]
            # The caller closes what it is given, so the member --- not the
            # collection it came from --- has to own the file handle.
            return Sample(
                member.root,
                handle=handle,
                owns_handle=True,
                path=f"{os.fspath(path)}::{key}",
            )
        if key is not None:
            raise MEDH5FileError(
                f"{os.fspath(path)!r} is a single sample; `key` applies to collections"
            )
        return Sample(handle, handle=handle, owns_handle=True, path=os.fspath(path))
    except BaseException:
        handle.close()
        raise


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
