"""Digests and ``content_id`` (spec §13.1--§13.2).

A dataset digest covers the object path, dtype, shape and the **decompressed**
little-endian bytes, so recompression changes every stored byte and no
digest.  ``content_id`` is a Merkle root over the *stored* digests, ``meta``
and canonical attributes: an edited dataset breaks its object digest and
leaves the root matching --- verify per object, never only the root.

The functions taking stored objects accept the ``Dataset`` and ``Group``
views this package hands out (``Sample.root``, ``Image.dataset``).
"""

from __future__ import annotations

from medh5 import _core

DEFAULT_ALGO: str = _core.DEFAULT_ALGO
DIGEST_ALGOS: tuple[str, ...] = _core.DIGEST_ALGOS
STREAM_BYTES: int = _core.STREAM_BYTES
"""Datasets are hashed in slabs of at most this many bytes."""

parse_digest = _core.parse_digest
digest_bytes = _core.digest_bytes
array_digest = _core.array_digest
dataset_digest = _core.dataset_digest
canonical_attrs = _core.canonical_attrs
attrs_digest = _core.attrs_digest
relative_path = _core.relative_path
group_digest = _core.group_digest
compute_content_id = _core.compute_content_id
collect_digests = _core.collect_digests

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
]
