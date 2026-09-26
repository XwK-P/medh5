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
that makes patch reads cheap.  ``rechunk=True`` re-derives chunks **the way the
writer does** --- from each grid's ``patch_hint``, ``(1, *spatial)`` for stacked
encodings --- rather than handing the choice to h5py, whose guess spanned the
stacked axis of a ``layers`` dataset and drew W902 from this package's own
validator.
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
from medh5.geometry.grid import Grid, read_grids
from medh5.integrity.verify import verify_root
from medh5.storage.chunking import field_chunks, fit_chunks, grid_chunks
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
    unattested: list[str] = field(default_factory=list)
    """Undigested datasets inside objects a declared ``content_id`` covers.

    Carried through from the source as they were, and the reason ``verified``
    is false when nothing mismatched: the output is not attested either.
    """
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
    from medh5.sample import require_major

    with atomic_rewrite(source_path, target) as (src, dst):
        # One gate for every rewrite (§16): a file from an unknown major is
        # refused here as `open` and `amend` refuse it, not re-encoded under
        # rules it may not follow.
        require_major(src, source_path)
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

        checked = [
            (prefix, verify_root(root, attr_name_map_of(root)))
            for prefix, root in _sample_roots(check)
        ]
    result.content_id = None if after is None else str(_text(after))
    result.mismatched = [
        f"{prefix}{name}"
        for prefix, verification in checked
        for name in (*verification.mismatched, *verification.malformed)
    ]
    result.unattested = [
        f"{prefix}{name}"
        for prefix, verification in checked
        for name in verification.unattested
    ]
    result.verified = all(verification.ok for _, verification in checked)
    result.content_id_preserved = _text(before) == _text(after) and all(
        verification.content_id_ok is not False for _, verification in checked
    )
    return result


def _sample_roots(root: h5py.Group) -> list[tuple[str, h5py.Group]]:
    """``(path prefix, sample root)`` for a sample, or for each collection member.

    A collection's root holds no ``/meta`` and no ``content_id`` of its own ---
    each member carries its own (§2.2) --- so its output is verified member by
    member.  Verifying the root as a sample raised a KeyError after the file had
    already been replaced.
    """
    from medh5.collection import SAMPLES_GROUP, is_collection

    if not is_collection(root):
        return [("", root)]
    members = root[SAMPLES_GROUP]
    return [(f"{SAMPLES_GROUP}/{key}/", members[key]) for key in sorted(members)]


def _text(value: Any) -> str | None:
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


class _Layout:
    """One sample root's grids, and the chunks the writer derives from them."""

    __slots__ = ("_grids", "_root", "name")

    def __init__(self, root: h5py.Group) -> None:
        self._root = root
        self.name = str(root.name).rstrip("/") + "/"
        self._grids: dict[str, Grid] | None = None

    def grid(self, grid_id: Any) -> Grid | None:
        if self._grids is None:
            try:
                self._grids = read_grids(self._root)
            except Exception:  # a malformed grid: fall back to h5py's choice
                self._grids = {}
        return self._grids.get(str(_text(grid_id))) if grid_id is not None else None

    def chunks_for(self, node: h5py.Dataset) -> tuple[int, ...] | None:
        """What the writer would chunk *node* as; ``None`` leaves it to h5py."""
        parts = str(node.name)[len(self.name) :].split("/")
        itemsize = int(node.dtype.itemsize)
        section = parts[0]
        if section == "images" and len(parts) == 2:
            grid = self.grid(node.attrs.get("grid"))
            return (
                None
                if grid is None
                else fit_chunks(grid_chunks(grid, itemsize), node.shape)
            )
        if section == "images" and len(parts) == 3 and parts[2].isdigit():
            levels = node.parent.attrs.get("grid_levels")
            level = int(parts[2])
            if levels is None or level >= len(levels):
                return None
            grid = self.grid(levels[level])
            return (
                None
                if grid is None
                else fit_chunks(grid_chunks(grid, itemsize), node.shape)
            )
        if section == "annotations" and len(parts) == 3 and parts[2] == "data":
            grid = self.grid(node.parent.attrs.get("grid"))
            if grid is None or node.ndim < grid.n_spatial:
                return None
            return fit_chunks(
                grid_chunks(grid, itemsize, leading=node.ndim - grid.n_spatial),
                node.shape,
            )
        if section == "transforms" and len(parts) == 3 and parts[2] == "field":
            grid = self.grid(node.parent.attrs.get("field_grid"))
            return None if grid is None else field_chunks(grid, node.shape, itemsize)
        if section == "index" and parts[-1] == "occupancy" and node.ndim > 1:
            return (1, *(int(n) for n in node.shape[1:]))
        return None


def _copy_group(
    src: h5py.Group,
    dst: h5py.Group,
    profile: str,
    rechunk: bool,
    result: RecompressResult,
    layout: _Layout | None = None,
) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    if rechunk and isinstance(src.get("grids"), h5py.Group):
        layout = _Layout(src)
    for name, node in src.items():
        if isinstance(node, h5py.Group):
            _copy_group(node, dst.create_group(name), profile, rechunk, result, layout)
            continue
        _copy_dataset(node, dst, name, profile, rechunk, result, layout)


def _copy_dataset(
    node: h5py.Dataset,
    dst: h5py.Group,
    name: str,
    profile: str,
    rechunk: bool,
    result: RecompressResult,
    layout: _Layout | None = None,
) -> None:
    if rechunk:
        chunks = layout.chunks_for(node) if layout is not None else None
    else:
        chunks = node.chunks
    kwargs = (
        {}
        if node.dtype == object
        else dataset_kwargs(
            node.shape,
            node.dtype,
            profile=profile,
            role=_role(node),
            chunks=chunks,
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
