"""Re-encoding a file under another codec profile (spec §14.2).

Copy-on-write like every other mutation (§14.4), and content-preserving by
construction: digests cover decompressed content, so recompression changes
every stored byte and no digest.  The output is read back and verified.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core


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


__all__ = ["RecompressResult", "recompress", "recompress_paths"]
