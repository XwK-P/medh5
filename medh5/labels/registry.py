"""Bundled vocabularies and the vocabulary registry (spec §5.1).

Three vocabularies ship with the package, chosen because they cover the shapes a
label set can take rather than because they are exhaustive: one class
(``binary-foreground``), a small hierarchy with overlap-capable sub-regions
(``brats-subregions``), and a flat multi-organ set (``amos22-organs``).

**No ontology codes are bundled.**  A wrong SNOMED-CT or FMA binding is a silent
data-integrity defect that propagates into every file written with the
vocabulary, and it is not detectable by any validator.  Bindings are the
curator's to add --- :class:`~medh5.labels.labelset.OntologyCode` exists for
exactly that --- and the validator's W912 says so when they are missing.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from medh5.errors import MEDH5ValidationError
from medh5.labels.labelset import LabelClass, LabelSet, Relation, Skeleton

VOCAB_DIR = Path(__file__).parent / "vocabularies"

_EXTRA: dict[str, LabelSet] = {}


def _bundled() -> dict[str, Path]:
    if not VOCAB_DIR.is_dir():  # pragma: no cover - only if package data is missing
        return {}
    return {p.stem: p for p in sorted(VOCAB_DIR.glob("*.json"))}


def available() -> tuple[str, ...]:
    """Every vocabulary name :func:`load` accepts, bundled and registered."""
    return tuple(sorted({*_bundled(), *_EXTRA}))


def _labelset_from_doc(doc: Mapping[str, Any]) -> LabelSet:
    return LabelSet(
        id=str(doc["id"]),
        version=str(doc.get("version", "1.0.0")),
        classes=[LabelClass.from_json(c) for c in doc.get("classes") or ()],
        relations=[Relation.from_json(r) for r in doc.get("relations") or ()],
        skeletons=[Skeleton.from_json(s) for s in doc.get("skeletons") or ()],
    )


def load(name: str) -> LabelSet:
    """Load a bundled or registered vocabulary by name."""
    if name in _EXTRA:
        return _EXTRA[name]
    path = _bundled().get(name)
    if path is None:
        raise MEDH5ValidationError(
            f"unknown vocabulary {name!r}; available: {list(available())}", code="E305"
        )
    return _labelset_from_doc(json.loads(path.read_text(encoding="utf-8")))


def load_file(path: str | Path) -> LabelSet:
    """Load a vocabulary from a JSON file on disk."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return _labelset_from_doc(doc)


def register(name: str, label_set: LabelSet) -> LabelSet:
    """Make a vocabulary loadable by name for the rest of the process."""
    _EXTRA[name] = label_set
    return label_set


def unregister(name: str) -> None:
    """Drop a registered vocabulary; bundled ones cannot be removed."""
    _EXTRA.pop(name, None)


def describe() -> dict[str, dict[str, Any]]:
    """Name -> ``{version, classes, digest}`` for ``medh5 labels registry list``."""
    out: dict[str, dict[str, Any]] = {}
    for name in available():
        ls = load(name)
        out[name] = {
            "id": ls.id,
            "version": ls.version,
            "classes": len(ls),
            "sha256": ls.digest(),
        }
    return out


__all__ = [
    "VOCAB_DIR",
    "available",
    "describe",
    "load",
    "load_file",
    "register",
    "unregister",
]
