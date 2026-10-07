"""Codec profiles: how each dataset is compressed (spec §14.2).

Four named profiles pair a codec for image data with one for label and field
data.  The codec a dataset was written with is discoverable from the file
itself (:func:`describe_filters`); nothing records the profile name, because a
profile is a writer convenience and a file may mix codecs.

The profiles and the layout rules are the format engine's: Blosc2 is compiled
into it, so writing and reading every profile needs no plugin package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5._optional import require

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


__all__ = [
    "BLOSC2_FILTER_ID",
    "BLOSC_FILTER_ID",
    "BUILTIN_FILTER_IDS",
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
    "profile_family",
    "resolve_profile",
]
