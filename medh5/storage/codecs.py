"""Codec profiles (spec §14.2).

A file's datasets need not share a codec.  Four named profiles cover the real
trade space, measured in ``docs/design/benchmarks/bench_io.py``:

===========  =====================  =========================================
Profile      Codec                  Intended use
===========  =====================  =========================================
``training`` Blosc2 lz4 L1          hot dataloader path; ~80x faster to write
``balanced`` Blosc2 zstd L3         general use (default)
``archive``  Blosc2 zstd L9         cold storage and distribution
``portable`` gzip L4                readers without ``hdf5plugin``
===========  =====================  =========================================

``portable`` exists because Blosc2 requires the ``hdf5plugin`` package on the
*reader*; a ``portable`` file opens in stock h5py, MATLAB, R and ``h5dump``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from medh5.errors import MEDH5ValidationError

Role = Literal["image", "label", "aux"]

COMPRESS_MIN_BYTES = 64 * 1024
"""Below this raw size a dataset is stored contiguous and uncompressed.

Chunking a 14-byte ``class_ids`` array costs a chunk index and a filter
pipeline to save nothing.  The validator's W902 threshold is deliberately
higher (:data:`BULK_MIN_BYTES`) so this policy never trips its own warning.
"""

BULK_MIN_BYTES = 1024 * 1024
"""At or above this raw size a dataset is 'bulk' and W902 applies."""


@dataclass(frozen=True, slots=True)
class Codec:
    """One codec setting, expressed as ``h5py.create_dataset`` keyword arguments."""

    name: str
    blosc2: tuple[str, int, str] | None = None
    gzip_level: int | None = None
    shuffle: bool = True

    def kwargs(self) -> dict[str, Any]:
        if self.blosc2 is not None:
            cname, clevel, shuffle_mode = self.blosc2
            import hdf5plugin

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


PROFILES: dict[str, CodecProfile] = {
    "training": CodecProfile(
        name="training",
        image=Codec("blosc2:lz4:1:shuffle", blosc2=("lz4", 1, "shuffle")),
        label=Codec("blosc2:lz4:1:shuffle", blosc2=("lz4", 1, "shuffle")),
        description="fastest decompression; hot dataloader path",
    ),
    "balanced": CodecProfile(
        name="balanced",
        image=Codec("blosc2:zstd:3:shuffle", blosc2=("zstd", 3, "shuffle")),
        label=Codec("blosc2:zstd:3:bitshuffle", blosc2=("zstd", 3, "bitshuffle")),
        description="general use; the default",
    ),
    "archive": CodecProfile(
        name="archive",
        image=Codec("blosc2:zstd:9:bitshuffle", blosc2=("zstd", 9, "bitshuffle")),
        label=Codec("blosc2:zstd:9:bitshuffle", blosc2=("zstd", 9, "bitshuffle")),
        description="smallest on disk; cold storage and distribution",
    ),
    "portable": CodecProfile(
        name="portable",
        image=Codec("gzip:4", gzip_level=4),
        label=Codec("gzip:4", gzip_level=4),
        description="readable without hdf5plugin",
    ),
}

DEFAULT_PROFILE = "balanced"


def resolve_profile(profile: str | CodecProfile | None) -> CodecProfile:
    """Resolve a profile name, or pass a profile through.  Defaults to ``balanced``."""
    if profile is None:
        return PROFILES[DEFAULT_PROFILE]
    if isinstance(profile, CodecProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise MEDH5ValidationError(
            f"unknown codec profile {profile!r}; expected one of {known}"
        ) from None


def dataset_kwargs(
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    *,
    profile: str | CodecProfile | None = None,
    role: Role = "image",
    chunks: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build ``create_dataset`` keyword arguments for one dataset.

    Datasets below :data:`COMPRESS_MIN_BYTES` are stored contiguous: chunking and
    a filter pipeline cost more than they save at that size, and a contiguous
    read is one seek.
    """
    nbytes = int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize) if shape else 0
    if nbytes < COMPRESS_MIN_BYTES or 0 in shape:
        return {}
    resolved = resolve_profile(profile)
    kwargs: dict[str, Any] = dict(resolved.codec(role).kwargs())
    kwargs["chunks"] = tuple(chunks) if chunks is not None else True
    return kwargs


BLOSC2_FILTER_ID = 32026
BLOSC_FILTER_ID = 32001

_BLOSC_CNAMES = {0: "blosclz", 1: "lz4", 2: "lz4hc", 3: "snappy", 4: "zlib", 5: "zstd"}
_BLOSC_SHUFFLE = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}


def describe_filters(dset: Any) -> str:
    """Describe a dataset's actual HDF5 filter pipeline, for ``medh5 info``.

    The codec used is discoverable from the file itself (spec §14.2); nothing
    records the profile name, because a profile is a writer convenience and a
    file may mix codecs.  Blosc2 parameters are decoded from the filter's client
    data, since h5py reports a registered third-party filter only as "unknown".
    """
    if dset.chunks is None:
        return "contiguous"
    parts: list[str] = []
    plist = dset.id.get_create_plist()
    for i in range(plist.get_nfilters()):
        filter_id, _, values, _ = plist.get_filter(i)
        if filter_id == BLOSC2_FILTER_ID:
            parts.append(_describe_blosc(values))
        elif filter_id == BLOSC_FILTER_ID:
            parts.append("blosc:" + _describe_blosc(values).partition(":")[2])
        elif dset.compression and dset.compression != "unknown":
            opts = dset.compression_opts
            parts.append(
                f"{dset.compression}" + (f":{opts}" if opts is not None else "")
            )
    if dset.shuffle:
        parts.append("shuffle")
    return "+".join(parts) if parts else "chunked"


def _describe_blosc(values: Any) -> str:
    """Decode ``(_, _, _, _, clevel, shuffle, cname)`` from the Blosc client data."""
    if len(values) < 7:  # noqa: PLR2004 - not a Blosc parameter block
        return "blosc2"
    clevel, shuffle, cname = int(values[4]), int(values[5]), int(values[6])
    return (
        f"blosc2:{_BLOSC_CNAMES.get(cname, cname)}:{clevel}"
        f"+{_BLOSC_SHUFFLE.get(shuffle, shuffle)}"
    )


def is_bulk(dset: Any) -> bool:
    """Whether a dataset is large enough for the W902 chunking/compression warning."""
    return int(dset.nbytes) >= BULK_MIN_BYTES


__all__ = [
    "BLOSC2_FILTER_ID",
    "BLOSC_FILTER_ID",
    "BULK_MIN_BYTES",
    "COMPRESS_MIN_BYTES",
    "DEFAULT_PROFILE",
    "PROFILES",
    "Codec",
    "CodecProfile",
    "Role",
    "dataset_kwargs",
    "describe_filters",
    "is_bulk",
    "resolve_profile",
]
