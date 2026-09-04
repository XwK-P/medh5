"""Re-encoding a file's bulk data under a different codec profile (§14.2).

The operation exists because the right codec depends on where a file is in its
life, not on what it contains: `archive` for the copy that sits in cold storage
for five years, `training` for the copy a dataloader hammers, `portable` for
the copy that goes to a collaborator whose reader has no `hdf5plugin`.

It is safe to do at any point because **digests are computed over decompressed
content** (§13.1).  Recompression therefore changes every stored byte and no
digest: `content_id` before and after are identical, and a downstream cache
keyed on it correctly treats the two files as the same data.  A format that
hashed the compressed bytes would make this operation a data migration.

Chunking is preserved by default.  A codec change alters compression ratio, not
access pattern, and re-chunking would quietly undo the L3-aware sizing (§14.1)
that makes patch reads cheap.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from medh5._hdf5 import atomic_rewrite, open_h5
from medh5.errors import MEDH5ValidationError
from medh5.integrity.verify import verify_root
from medh5.storage.codecs import PROFILES, Role, dataset_kwargs, describe_filters


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
    """Whether the *output* verifies: every object digest, and the root.

    The re-encoded file is read back and checked against the digests it
    carries.  Without it, ``content_id_preserved`` compared the attribute this
    function had just copied with itself --- true by construction, including on
    a file whose bytes were corrupted before it ran, which then reported
    "content_id yes", exited 0, and failed ``verify()``.
    """
    mismatched: list[str] = field(default_factory=list)
    changed: list[tuple[str, str, str]] = field(default_factory=list)
    """``(path, codec before, codec after)`` for each dataset re-encoded."""

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
    """Rewrite *path* with every bulk dataset re-encoded under *profile*.

    Copy-on-write like every other mutation (§14.4): the new file is built
    beside the old one and atomically replaces it, so an interrupted run leaves
    the original intact.
    """
    if profile not in PROFILES:
        raise MEDH5ValidationError(
            f"unknown codec profile {profile!r}; expected one of {sorted(PROFILES)}"
        )
    source_path = Path(os.fspath(path))
    target = Path(os.fspath(out)) if out is not None else source_path
    result = RecompressResult(path=str(target), profile=profile)
    # File size, not the sum of `get_storage_size()`: a dataset in a file that
    # is still open reports whatever has been flushed, which is not its size.
    result.bytes_before = source_path.stat().st_size
    # `atomic_rewrite`, not `open_h5` + `atomic_h5`: nesting those exits
    # right-to-left, so the replace ran while the source was still open, which
    # Windows refuses when the target is the source.
    with atomic_rewrite(source_path, target) as (src, dst):
        before = src.attrs.get("content_id")
        # Root attributes are copied by `_copy_group`, which every group goes
        # through; copying them here as well was a second, identical pass.
        _copy_group(src, dst, profile, rechunk, result)
    result.bytes_after = target.stat().st_size
    # Verify the *output*, against the digests it carries.  §13.1 says digests
    # cover decompressed content, so a correct re-encode changes every stored
    # byte and no digest --- which is a claim worth checking rather than
    # asserting, and checking it is what makes the preserved `content_id`
    # evidence about the data instead of about the copy.
    with open_h5(target, "r") as check:
        after = check.attrs.get("content_id")
        from medh5.sample import attr_name_map_of

        verification = verify_root(check, attr_name_map_of(check))
    result.content_id = None if after is None else str(_text(after))
    result.mismatched = [*verification.mismatched, *verification.malformed]
    result.verified = verification.ok
    result.content_id_preserved = (
        _text(before) == _text(after) and verification.content_id_ok is not False
    )
    return result


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def _copy_group(
    src: h5py.Group,
    dst: h5py.Group,
    profile: str,
    rechunk: bool,
    result: RecompressResult,
) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for name, node in src.items():
        if isinstance(node, h5py.Group):
            _copy_group(node, dst.create_group(name), profile, rechunk, result)
            continue
        _copy_dataset(node, dst, name, profile, rechunk, result)


def _copy_dataset(
    node: h5py.Dataset,
    dst: h5py.Group,
    name: str,
    profile: str,
    rechunk: bool,
    result: RecompressResult,
) -> None:
    kwargs = (
        {}
        if node.dtype == object
        else dataset_kwargs(
            node.shape,
            node.dtype,
            profile=profile,
            role=_role(node),
            chunks=None if rechunk else node.chunks,
        )
    )
    if not kwargs:
        # Below the writer's compression threshold, or a variable-length string:
        # exactly the datasets a writer would leave contiguous, so copying them
        # through keeps the result identical to a fresh write (§14.2), and keeps
        # `/meta` byte-for-byte (§2.4).
        node.parent.copy(node.name.rsplit("/", 1)[-1], dst, name=name)
        return
    before = describe_filters(node)
    out = dst.create_dataset(name, shape=node.shape, dtype=node.dtype, **kwargs)
    for window in _slabs(node):
        out[window] = node[window]
    for key, value in node.attrs.items():
        out.attrs[key] = value
    after = describe_filters(out)
    result.datasets += 1
    if before != after:
        result.changed.append((node.name, before, after))


def _role(node: h5py.Dataset) -> Role:
    """Which half of a codec profile applies: image data or labels/fields."""
    path = node.name or ""
    if "/annotations/" in path or "/transforms/" in path or "/index/" in path:
        return "label"
    return "image"


def _slabs(node: h5py.Dataset, budget: int = 64 << 20) -> Iterator[tuple[slice, ...]]:
    """Windows covering the dataset, each under *budget* bytes.

    Copying through slabs rather than `dataset[...]` keeps peak memory bounded:
    recompressing a 40 GiB whole-body CT must not need 40 GiB of RAM.
    """
    if node.size == 0:
        yield tuple(slice(0, s) for s in node.shape)
        return
    itemsize = node.dtype.itemsize
    trailing = int(np.prod(node.shape[1:], dtype=np.int64)) if node.ndim > 1 else 1
    per_plane = max(1, trailing * itemsize)
    step = max(1, min(node.shape[0], budget // per_plane))
    for start in range(0, node.shape[0], step):
        stop = min(start + step, node.shape[0])
        yield (slice(start, stop), *(slice(0, s) for s in node.shape[1:]))


def recompress_paths(
    paths: Sequence[str | os.PathLike[str]],
    profile: str,
    *,
    rechunk: bool = False,
) -> list[RecompressResult]:
    return [recompress(p, profile, rechunk=rechunk) for p in paths]


__all__ = ["RecompressResult", "recompress", "recompress_paths"]
