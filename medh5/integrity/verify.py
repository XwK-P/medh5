"""Verification of digests, ``content_id`` and derived-index currency (spec §13)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import h5py

from medh5._hdf5 import as_str
from medh5.integrity.digest import (
    DEFAULT_ALGO,
    collect_digests,
    compute_content_id,
    dataset_digest,
    group_digest,
    parse_digest,
)


@dataclass(slots=True)
class VerifyResult:
    """Outcome of a verification pass."""

    checked: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    undigested: tuple[str, ...] = ()
    malformed: tuple[str, ...] = ()
    content_id_declared: str | None = None
    content_id_computed: str | None = None
    stale_index: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def content_id_ok(self) -> bool | None:
        """Whether the root ``content_id`` matches, or ``None`` when unjudged.

        ``None`` covers two cases that must not read as failure: a file that
        declares no ``content_id`` (it is a SHOULD), and a *partial* pass, which
        deliberately did not recompute the Merkle root.
        """
        if self.content_id_declared is None or self.content_id_computed is None:
            return None
        return self.content_id_declared == self.content_id_computed

    @property
    def ok(self) -> bool:
        return (
            not self.mismatched
            and not self.malformed
            and self.content_id_ok is not False
        )

    def summary(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": len(self.checked),
            "mismatched": list(self.mismatched),
            "undigested": list(self.undigested),
            "malformed": list(self.malformed),
            "content_id_ok": self.content_id_ok,
            "stale_index": list(self.stale_index),
        }


def verify_object(root: h5py.Group, path: str) -> bool:
    """Recompute one dataset's digest and compare it with the stored value."""
    dset = root[path]
    stored = dset.attrs.get("digest")
    if stored is None:
        return False
    algo, _ = parse_digest(as_str(stored))
    return dataset_digest(dset, path, algo) == as_str(stored)


def stale_index_entries(root: h5py.Group) -> tuple[str, ...]:
    """Index entries whose ``source_digest`` no longer matches their source (§13.3).

    A stale entry is **not** a file error: readers must ignore it and rebuild.
    Returning the list lets a validator raise W905 and ``medh5 fix`` act on it.
    """
    if "index" not in root:
        return ()
    stale: list[str] = []
    index = root["index"]
    for name in sorted(index):
        entry = index[name]
        declared = entry.attrs.get("source_digest")
        if declared is None:
            stale.append(name)
            continue
        source_path = f"annotations/{name}"
        if source_path not in root:
            stale.append(name)
            continue
        source = root[source_path]
        current = (
            group_digest(source, root=root)
            if isinstance(source, h5py.Group)
            else as_str(source.attrs.get("digest", ""))
        )
        if current != as_str(declared):
            stale.append(name)
    return tuple(stale)


def verify_root(
    root: h5py.Group,
    attr_names: Mapping[str, Sequence[str]] | None = None,
    *,
    partial: Sequence[str] | None = None,
    check_content_id: bool = True,
) -> VerifyResult:
    """Verify a sample root.

    *partial* restricts the pass to named objects --- the point of per-object
    digests is that a reader which touched one image can verify one image.
    """
    stored = collect_digests(root)
    targets = list(partial) if partial is not None else sorted(stored)
    checked: list[str] = []
    mismatched: list[str] = []
    malformed: list[str] = []
    for path in targets:
        value = stored.get(path)
        if value is None:
            malformed.append(path)
            continue
        try:
            algo, _ = parse_digest(value)
        except Exception:  # noqa: BLE001 - reported, not raised
            malformed.append(path)
            continue
        checked.append(path)
        if dataset_digest(root[path], path, algo) != value:
            mismatched.append(path)

    undigested: list[str] = []

    def visit(name: str, obj: h5py.HLObject) -> None:
        if (
            isinstance(obj, h5py.Dataset)
            and name != "meta"
            and not name.startswith("index/")
            and "digest" not in obj.attrs
        ):
            undigested.append(name)

    root.visititems(visit)

    declared = root.attrs.get("content_id")
    declared_str = as_str(declared) if declared is not None else None
    computed: str | None = None
    if check_content_id and partial is None and attr_names is not None:
        algo = (
            as_str(root.attrs["digest_algo"])
            if "digest_algo" in root.attrs
            else DEFAULT_ALGO
        )
        computed = compute_content_id(root, attr_names, algo=algo, digests=stored)

    return VerifyResult(
        checked=tuple(checked),
        mismatched=tuple(mismatched),
        undigested=tuple(sorted(undigested)),
        malformed=tuple(malformed),
        content_id_declared=declared_str,
        content_id_computed=computed,
        stale_index=stale_index_entries(root),
    )


__all__ = ["VerifyResult", "stale_index_entries", "verify_object", "verify_root"]
