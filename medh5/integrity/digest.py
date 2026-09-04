"""Per-object digests and the Merkle ``content_id`` (spec §13).

A digest covers one object's **decompressed** content, so recompressing a file
does not invalidate it.  The root ``content_id`` is a Merkle root over the sorted
per-object digests, which buys four things a single whole-file hash cannot:

* **incremental** --- adding one annotation rehashes one object plus the root;
* **partial verification** --- a reader that touched ``images/CT`` verifies only that;
* **content addressing** --- ``content_id`` is a cache key and a dedup key;
* **locality** --- a mismatch names the object that changed.

Digest input for a dataset (§13.1)::

    H( object_path || 0x00 || dtype_str || 0x00 || shape_csv
       || 0x00 || C-order little-endian bytes )

``dtype_str`` is the NumPy dtype string with byte order normalised to
little-endian, so a file written on a big-endian machine digests identically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import h5py
import numpy as np

from medh5._hdf5 import as_str
from medh5.errors import MEDH5ValidationError

DEFAULT_ALGO = "sha256"
DIGEST_ALGOS = ("sha256", "sha512", "blake2b")

STREAM_BYTES = 32 * 1024 * 1024
"""Slab size for streaming a large dataset through the hash."""


def _hasher(algo: str) -> Any:
    if algo not in DIGEST_ALGOS:
        raise MEDH5ValidationError(
            f"unsupported digest algorithm {algo!r}; expected one of "
            f"{list(DIGEST_ALGOS)}",
            code="E703",
        )
    return hashlib.new(algo)


def parse_digest(value: str) -> tuple[str, str]:
    """Split ``"<algo>:<hex>"``, raising E703 when malformed."""
    text = as_str(value)
    algo, _, hexdigest = text.partition(":")
    if not hexdigest or algo not in DIGEST_ALGOS:
        raise MEDH5ValidationError(f"malformed digest {text!r}", code="E703")
    try:
        int(hexdigest, 16)
    except ValueError:
        raise MEDH5ValidationError(f"malformed digest {text!r}", code="E703") from None
    return algo, hexdigest


def digest_bytes(payload: bytes, algo: str = DEFAULT_ALGO) -> str:
    """``"<algo>:<hex>"`` over a byte string."""
    hasher = _hasher(algo)
    hasher.update(payload)
    return f"{algo}:{hasher.hexdigest()}"


def _le_dtype_str(dtype: np.dtype[Any]) -> str:
    """The dtype string with byte order normalised to little-endian."""
    return dtype.newbyteorder("<").str if dtype.byteorder == ">" else dtype.str


def array_digest(
    path: str, array: np.ndarray[Any, np.dtype[Any]], algo: str = DEFAULT_ALGO
) -> str:
    """Digest of an in-memory array under the §13.1 canonical byte stream."""
    arr = np.ascontiguousarray(array)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    hasher = _hasher(algo)
    _feed_header(hasher, path, _le_dtype_str(arr.dtype), arr.shape)
    hasher.update(arr.tobytes())
    return f"{algo}:{hasher.hexdigest()}"


def _feed_header(hasher: Any, path: str, dtype_str: str, shape: Sequence[int]) -> None:
    hasher.update(path.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(dtype_str.encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(",".join(str(int(s)) for s in shape).encode("ascii"))
    hasher.update(b"\x00")


def _is_vlen_str(dset: h5py.Dataset) -> bool:
    return h5py.check_string_dtype(dset.dtype) is not None


def dataset_digest(
    dset: h5py.Dataset, path: str | None = None, algo: str = DEFAULT_ALGO
) -> str:
    """Digest an HDF5 dataset, streaming so a 4 GiB volume never lands in RAM.

    Variable-length string datasets hash their UTF-8 payloads separated by
    ``0x00``, because a vlen dataset has no meaningful raw byte stream.
    """
    obj_path = path if path is not None else _relative_name(dset)
    hasher = _hasher(algo)
    if _is_vlen_str(dset):
        _feed_header(hasher, obj_path, "|O", dset.shape)
        values = dset[()]
        items = [values] if dset.shape == () else np.asarray(values).ravel().tolist()
        for item in items:
            payload = item if isinstance(item, bytes) else str(item).encode("utf-8")
            hasher.update(payload)
            hasher.update(b"\x00")
        return f"{algo}:{hasher.hexdigest()}"

    dtype = dset.dtype
    _feed_header(hasher, obj_path, _le_dtype_str(dtype), dset.shape)
    if dset.size == 0:
        return f"{algo}:{hasher.hexdigest()}"
    if dset.ndim == 0:
        arr = np.ascontiguousarray(dset[()])
        hasher.update(_to_le(arr).tobytes())
        return f"{algo}:{hasher.hexdigest()}"

    row_bytes = max(1, int(np.prod(dset.shape[1:], dtype=np.int64)) * dtype.itemsize)
    step = max(1, STREAM_BYTES // row_bytes)
    for start in range(0, dset.shape[0], step):
        block = np.ascontiguousarray(dset[start : start + step])
        hasher.update(_to_le(block).tobytes())
    return f"{algo}:{hasher.hexdigest()}"


def _to_le(arr: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
    if arr.dtype.byteorder == ">":
        return np.ascontiguousarray(arr.astype(arr.dtype.newbyteorder("<")))
    return arr


def _relative_name(obj: h5py.HLObject) -> str:
    return str(obj.name).lstrip("/")


def _jsonable(value: Any) -> Any:
    """Convert an HDF5 attribute value to a JSON-serialisable form."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "OSU":
            return [_jsonable(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_attrs(obj: h5py.HLObject, names: Iterable[str]) -> str:
    """Canonical JSON over an object's spec-defined attributes (§13.2).

    Sorted keys, arrays as nested lists, floats in shortest round-trip form ---
    which is what ``json.dumps`` produces for Python floats.  Attributes not in
    *names* are excluded, so a third-party extension attribute cannot change a
    file's ``content_id``.
    """
    wanted = sorted(set(names) & set(obj.attrs.keys()))
    doc = {name: _jsonable(obj.attrs[name]) for name in wanted}
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def attrs_digest(
    obj: h5py.HLObject, names: Iterable[str], algo: str = DEFAULT_ALGO
) -> str:
    hasher = _hasher(algo)
    hasher.update(canonical_attrs(obj, names).encode("utf-8"))
    return f"{algo}:{hasher.hexdigest()}"


def stamp_digests(
    root: h5py.Group,
    *,
    algo: str = DEFAULT_ALGO,
    skip: Sequence[str] = ("index",),
    only_missing: bool = False,
) -> dict[str, str]:
    """Write a ``digest`` attribute on every dataset under *root*.

    ``index/`` is skipped by default: it is derived, regenerable and carries
    ``source_digest`` instead (§13.3).

    ``only_missing`` digests just the datasets that carry none, and returns the
    stored value for the rest.  That is what an amend wants: copy-on-write
    brings every untouched dataset across with its digest intact, so re-reading
    them all to recompute what they already say is the whole cost of the flag
    --- and the caller still gets a complete map, which is what ``content_id``
    is built from.
    """
    digests: dict[str, str] = {}
    blocked = tuple(skip)

    def visit(name: str, obj: h5py.HLObject) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if name == "meta" or name.split("/", 1)[0] in blocked:
            return
        stored = obj.attrs.get("digest")
        if only_missing and stored is not None:
            digests[name] = as_str(stored)
            return
        value = dataset_digest(obj, name, algo)
        obj.attrs["digest"] = np.array(value, dtype=h5py.string_dtype())
        digests[name] = value

    root.visititems(visit)
    return digests


def collect_digests(
    root: h5py.Group, skip: Sequence[str] = ("index",)
) -> dict[str, str]:
    """Read the ``digest`` attribute of every dataset that carries one."""
    out: dict[str, str] = {}
    blocked = tuple(skip)

    def visit(name: str, obj: h5py.HLObject) -> None:
        if (
            isinstance(obj, h5py.Dataset)
            and "digest" in obj.attrs
            and name.split("/", 1)[0] not in blocked
        ):
            out[name] = as_str(obj.attrs["digest"])

    root.visititems(visit)
    return out


def relative_path(node: h5py.HLObject, root: h5py.Group | None = None) -> str:
    """An object's path relative to a sample root.

    Digests hash the object path, so it must be *sample-root relative*: a sample
    extracted from a collection has to digest identically to the same sample
    standalone (spec §2.2).
    """
    base = "/" if root is None else str(root.name).rstrip("/") + "/"
    name = str(node.name)
    return name[len(base) :] if name.startswith(base) else name.lstrip("/")


def group_digest(
    group: h5py.Group, algo: str = DEFAULT_ALGO, *, root: h5py.Group | None = None
) -> str:
    """A digest over every dataset in a group --- an annotation's identity.

    Spec §13.3 requires an index entry to carry the ``digest`` of the annotation
    it derives from, but an annotation is a *group* of datasets and only datasets
    carry digests.  This defines the missing quantity: a mini-Merkle over the
    group's own datasets, sorted by name, computed the same way in every
    implementation.

    Datasets that have not been stamped yet are digested on the fly, using the
    same sample-root-relative path :func:`stamp_digests` would use --- otherwise
    an index built before the commit would look stale immediately after it.
    """
    lines: list[str] = []
    for name in sorted(group):
        node = group[name]
        if not isinstance(node, h5py.Dataset):
            continue
        stored = node.attrs.get("digest")
        value = (
            as_str(stored)
            if stored is not None
            else dataset_digest(node, relative_path(node, root), algo)
        )
        lines.append(f"{name}\t{value}\n")
    hasher = _hasher(algo)
    hasher.update("".join(lines).encode("utf-8"))
    return f"{algo}:{hasher.hexdigest()}"


def compute_content_id(
    root: h5py.Group,
    attr_names: Mapping[str, Sequence[str]],
    *,
    algo: str = DEFAULT_ALGO,
    digests: Mapping[str, str] | None = None,
) -> str:
    """The Merkle root over dataset digests, ``meta`` and canonical attributes.

    *attr_names* maps a sample-root-relative object path to the spec-defined
    attribute names of that object.  Paths are relative to the **sample root**,
    so a sample extracted from a collection keeps the same ``content_id``.

    An *algo* outside §2.1's vocabulary is E703 here, before any byte is
    hashed: a file declaring ``blake3`` used to reach ``hashlib.new`` and die
    with a ``ValueError`` no caller could catch by the documented type.
    """
    _hasher(algo)
    dataset_lines = sorted(
        f"{path}\t{value}\n"
        for path, value in (
            digests if digests is not None else collect_digests(root)
        ).items()
    )
    raw_meta = root["meta"][()]
    meta_text = (
        raw_meta.decode("utf-8") if isinstance(raw_meta, bytes) else str(raw_meta)
    )
    meta_hasher = _hasher(algo)
    meta_hasher.update(meta_text.encode("utf-8"))
    meta_hash = meta_hasher.hexdigest()
    attr_lines = sorted(
        f"@{path}\t{attrs_digest(root[path] if path else root, names, algo)}\n"
        for path, names in attr_names.items()
        if (path == "" or path in root)
    )
    payload = "".join(dataset_lines) + f"meta\t{meta_hash}\n" + "".join(attr_lines)
    root_hasher = _hasher(algo)
    root_hasher.update(payload.encode("utf-8"))
    return f"{algo}:{root_hasher.hexdigest()}"


__all__ = [
    "DEFAULT_ALGO",
    "DIGEST_ALGOS",
    "STREAM_BYTES",
    "array_digest",
    "attrs_digest",
    "canonical_attrs",
    "collect_digests",
    "compute_content_id",
    "dataset_digest",
    "digest_bytes",
    "group_digest",
    "parse_digest",
    "relative_path",
    "stamp_digests",
]
