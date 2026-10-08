"""The storage layer: chunking, codecs, sampling indices, recompression (spec §14).

**Chunks** (§14.1).  HDF5 decompresses a whole chunk to serve any element of it,
so the chunk is the real unit of I/O.  Sizing it to the L3 cache keeps a patch
read inside cache after decompression; sizing it near the training patch keeps
read amplification low.  The engine resolves the two by starting at the patch,
growing toward the cache budget, and stopping before the chunk is much larger
than the patch; stacked encodings chunk per plane so one layer reads without
the others.

**Codec profiles** (§14.2).  Four named profiles pair a codec for image data
with one for label and field data: ``training`` (lz4:1), ``balanced`` (zstd:3,
the default), ``archive`` (zstd:9) and ``portable`` (gzip:4, readable by any
HDF5).  The codec a dataset was written with is discoverable from the file
itself (:func:`describe_filters`); nothing records the profile name, because a
profile is a writer convenience and a file may mix codecs.  Blosc2 is compiled
into the engine, so writing and reading every profile needs no plugin package.

**Sampling indices** (§14.3) are derived, rebuildable acceleration data.  An
entry holds per-class voxel counts, boxes, a uniform subsample of foreground
coordinates and an occupancy map, so a patch sampler draws a foreground-centred
patch without scanning the annotation.  Each entry records the
``source_digest`` of the annotation it was built from; a stale entry is
ignored, never trusted.

**Recompression** is copy-on-write like every other mutation (§14.4), and
content-preserving by construction: digests cover decompressed content, so it
changes every stored byte and no digest.  The output is read back and verified.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5._optional import require

# -- chunking (§14.1) -----------------------------------------------------------

DEFAULT_L3_BYTES: int = _core.DEFAULT_L3_BYTES
MIN_CHUNK_BYTES: float = _core.MIN_CHUNK_BYTES
MAX_CHUNK_BYTES: float = _core.MAX_CHUNK_BYTES
CACHE_SAFETY: float = _core.CACHE_SAFETY
OVERSHOOT_LIMIT: float = _core.OVERSHOOT_LIMIT
DEFAULT_PATCH: int = _core.DEFAULT_PATCH

detect_l3_bytes = _core.detect_l3_bytes
spatial_chunk_for = _core.spatial_chunk_for
optimize_chunks = _core.optimize_chunks
grid_chunks = _core.grid_chunks
fit_chunks = _core.fit_chunks
field_chunks = _core.field_chunks
chunk_report = _core.chunk_report

# -- codec profiles (§14.2) ------------------------------------------------------

Role = Literal["image", "label", "aux"]

COMPRESS_MIN_BYTES: int = _core.COMPRESS_MIN_BYTES
"""Datasets smaller than this are stored contiguous and unfiltered."""

BULK_MIN_BYTES: int = _core.BULK_MIN_BYTES
"""Datasets at least this large are bulk: W902 wants them compressed."""

BLOSC2_FILTER_ID: int = _core.BLOSC2_FILTER_ID
BLOSC_FILTER_ID: int = _core.BLOSC_FILTER_ID
BUILTIN_FILTER_IDS: frozenset[int] = frozenset(_core.BUILTIN_FILTER_IDS)
DEFAULT_PROFILE: str = _core.DEFAULT_PROFILE


@dataclass(frozen=True, slots=True)
class Codec:
    """One codec setting."""

    name: str
    blosc2: tuple[str, int, str] | None = None
    gzip_level: int | None = None
    shuffle: bool = True

    def kwargs(self) -> dict[str, Any]:
        """``h5py.Group.create_dataset`` keywords for this codec.

        For writing a dataset with ``h5py`` directly in the profile's codec;
        medh5 itself does not need them.  A Blosc2 codec takes its filter
        options from ``hdf5plugin``, which is then required.
        """
        if self.blosc2 is not None:
            cname, clevel, shuffle_mode = self.blosc2
            hdf5plugin = require(
                "hdf5plugin", extra="h5py", purpose="Blosc2 keywords for h5py"
            )
            mode = {
                "shuffle": hdf5plugin.Blosc2.SHUFFLE,
                "bitshuffle": hdf5plugin.Blosc2.BITSHUFFLE,
                "none": hdf5plugin.Blosc2.NOFILTER,
            }[shuffle_mode]
            return dict(hdf5plugin.Blosc2(cname=cname, clevel=clevel, filters=mode))
        if self.gzip_level is not None:
            return {
                "compression": "gzip",
                "compression_opts": self.gzip_level,
                "shuffle": self.shuffle,
            }
        return {}  # pragma: no cover - no uncompressed profile is defined


@dataclass(frozen=True, slots=True)
class CodecProfile:
    """A named pairing of codecs for image data and for label/field data."""

    name: str
    image: Codec
    label: Codec
    description: str

    def codec(self, role: Role) -> Codec:
        return self.image if role == "image" else self.label


def _codec(doc: dict[str, Any]) -> Codec:
    blosc2 = doc.get("blosc2")
    return Codec(
        name=doc["name"],
        blosc2=None
        if blosc2 is None
        else (str(blosc2[0]), int(blosc2[1]), str(blosc2[2])),
        gzip_level=doc.get("gzip_level"),
        shuffle=bool(doc.get("shuffle", True)),
    )


PROFILES: dict[str, CodecProfile] = {
    p["name"]: CodecProfile(
        name=p["name"],
        image=_codec(p["image"]),
        label=_codec(p["label"]),
        description=p["description"],
    )
    for p in _core.codec_profiles()
}


def resolve_profile(profile: str | CodecProfile | None) -> CodecProfile:
    """A profile by name (``None`` is ``balanced``), or the profile given."""
    if isinstance(profile, CodecProfile):
        return profile
    return PROFILES[_core.resolve_profile_name(profile)]


def dataset_kwargs(
    shape: tuple[int, ...],
    dtype: npt.DTypeLike,
    *,
    profile: str | CodecProfile | None = None,
    role: Role = "image",
    chunks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """The layout the writer gives one dataset: ``{}`` when it is stored
    contiguous (small or empty), else ``{"chunks": ..., "codec": ...}``."""
    name = resolve_profile(profile).name
    found: dict[str, Any] = _core.dataset_layout(
        [int(n) for n in shape],
        np.dtype(dtype).itemsize,
        profile=name,
        role=role,
        chunks=None if chunks is None else [int(c) for c in chunks],
    )
    return found


def describe_filters(dataset: Any) -> str:
    """A dataset's actual filter pipeline, e.g. ``blosc2:zstd:3+shuffle``.

    *dataset* is a stored dataset as this package hands them out
    (``Image.dataset``, the return of ``SampleWriter.add_image``).
    """
    return str(_core.describe_filters(dataset))


def is_bulk(dataset: Any) -> bool:
    """Whether a dataset is large enough for the W902 warning."""
    return bool(_core.is_bulk(dataset))


def profile_family(path: str | os.PathLike[str]) -> str:
    """``portable`` when every dataset of the file needs only HDF5's own
    filters, else ``balanced`` --- what an amend of it defaults to."""
    return str(_core.profile_family(os.fspath(path)))


# -- sampling indices (§14.3) ------------------------------------------------------

DEFAULT_MAX_COORDS: int = _core.DEFAULT_MAX_COORDS
DEFAULT_OCCUPANCY_FACTOR: int = _core.DEFAULT_OCCUPANCY_FACTOR

SamplingIndex = _core.SamplingIndex
"""One stored index entry: ``voxel_counts``, ``bbox()``, ``coords()``,
``sample_foreground(class_id, n, rng)``, ``class_weights(mode)``."""

IndexPayload = _core.IndexPayload
build_index = _core.build_index
occupancy = _core.occupancy
read_indices = _core.read_indices
"""Every stored index entry under a sample root (``Sample.root``), by
annotation id --- stale ones included; ``Sample.fresh_indices`` says which are
current."""

# -- recompression (§14.2, §14.4) ---------------------------------------------------


@dataclass(slots=True)
class RecompressResult:
    """What one file's re-encoding did."""

    path: str
    profile: str
    datasets: int = 0
    bytes_before: int = 0
    """Whole-file size before and after; per-dataset codecs are in `changed`."""
    bytes_after: int = 0
    content_id: str | None = None
    content_id_preserved: bool = True
    verified: bool = True
    """Whether the *output* verifies: every object digest, and the root."""
    mismatched: list[str] = field(default_factory=list)
    unattested: list[str] = field(default_factory=list)
    """Undigested datasets inside objects a declared ``content_id`` covers."""
    changed: list[tuple[str, str, str]] = field(default_factory=list)
    """``(path, codec before, codec after)`` for each dataset re-encoded."""

    @classmethod
    def from_json(cls, doc: dict[str, Any]) -> RecompressResult:
        return cls(
            path=doc["path"],
            profile=doc["profile"],
            datasets=int(doc.get("datasets", 0)),
            bytes_before=int(doc.get("bytes_before", 0)),
            bytes_after=int(doc.get("bytes_after", 0)),
            content_id=doc.get("content_id"),
            content_id_preserved=bool(doc.get("content_id_preserved", True)),
            verified=bool(doc.get("verified", True)),
            mismatched=list(doc.get("mismatched") or ()),
            unattested=list(doc.get("unattested") or ()),
            changed=[(str(a), str(b), str(c)) for a, b, c in doc.get("changed") or ()],
        )

    @property
    def ratio(self) -> float:
        return self.bytes_after / self.bytes_before if self.bytes_before else 1.0

    @property
    def ok(self) -> bool:
        return self.verified and self.content_id_preserved

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "profile": self.profile,
            "datasets": self.datasets,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "ratio": self.ratio,
            "content_id": self.content_id,
            "content_id_preserved": self.content_id_preserved,
            "verified": self.verified,
            "mismatched": list(self.mismatched),
            "unattested": list(self.unattested),
            "ok": self.ok,
            "changed": [list(c) for c in self.changed],
        }

    def __str__(self) -> str:
        return (
            f"{self.path}: {self.profile}, {self.datasets} datasets, "
            f"{self.bytes_before} -> {self.bytes_after} bytes "
            f"({self.ratio:.2f}×)"
        )


def recompress(
    path: str | os.PathLike[str],
    profile: str,
    *,
    out: str | os.PathLike[str] | None = None,
    rechunk: bool = False,
) -> RecompressResult:
    """Rewrite *path* (or write *out*) with every bulk dataset re-encoded
    under *profile*; ``rechunk`` also re-derives chunk shapes from the grids."""
    return RecompressResult.from_json(
        _core.recompress(
            os.fspath(path),
            profile,
            out=None if out is None else os.fspath(out),
            rechunk=rechunk,
        )
    )


def recompress_paths(
    paths: Sequence[str | os.PathLike[str]], profile: str, *, rechunk: bool = False
) -> list[RecompressResult]:
    """Recompress many files, one result each."""
    return [
        RecompressResult.from_json(doc)
        for doc in _core.recompress_paths(
            [os.fspath(p) for p in paths], profile, rechunk=rechunk
        )
    ]


__all__ = [
    "BLOSC2_FILTER_ID",
    "BLOSC_FILTER_ID",
    "BUILTIN_FILTER_IDS",
    "BULK_MIN_BYTES",
    "CACHE_SAFETY",
    "COMPRESS_MIN_BYTES",
    "DEFAULT_L3_BYTES",
    "DEFAULT_MAX_COORDS",
    "DEFAULT_OCCUPANCY_FACTOR",
    "DEFAULT_PATCH",
    "DEFAULT_PROFILE",
    "MAX_CHUNK_BYTES",
    "MIN_CHUNK_BYTES",
    "OVERSHOOT_LIMIT",
    "PROFILES",
    "Codec",
    "CodecProfile",
    "IndexPayload",
    "RecompressResult",
    "Role",
    "SamplingIndex",
    "build_index",
    "chunk_report",
    "dataset_kwargs",
    "describe_filters",
    "detect_l3_bytes",
    "field_chunks",
    "fit_chunks",
    "grid_chunks",
    "is_bulk",
    "occupancy",
    "optimize_chunks",
    "profile_family",
    "read_indices",
    "recompress",
    "recompress_paths",
    "resolve_profile",
    "spatial_chunk_for",
]
