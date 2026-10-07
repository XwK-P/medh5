"""Bundled vocabularies and the vocabulary registry (spec §5.1).

Three vocabularies ship with the engine, chosen because they cover the shapes a
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

import os
from typing import Any

from medh5 import _core
from medh5.labels.labelset import LabelSet

_REGISTERED: dict[str, LabelSet] = {}
"""The objects callers registered, so ``load`` hands back the same one."""


def available() -> tuple[str, ...]:
    """Every vocabulary name :func:`load` accepts, bundled and registered."""
    return tuple(_core.registry_available())


def load(name: str) -> LabelSet:
    """A bundled or registered vocabulary by name; E305 for an unknown one."""
    registered = _REGISTERED.get(name)
    if registered is not None:
        return registered
    return _core.registry_load(name)


def load_file(path: str | os.PathLike[str]) -> LabelSet:
    """A vocabulary from a JSON file on disk."""
    return _core.registry_load_file(path)


def register(name: str, label_set: LabelSet) -> LabelSet:
    """Make a vocabulary loadable by name for the rest of the process."""
    _core.registry_register(name, label_set)
    _REGISTERED[name] = label_set
    return label_set


def unregister(name: str) -> None:
    """Drop a registered vocabulary; bundled ones cannot be removed."""
    _REGISTERED.pop(name, None)
    _core.registry_unregister(name)


def describe() -> dict[str, dict[str, Any]]:
    """Name -> ``{id, version, classes, sha256}``."""
    return dict(_core.registry_describe())


__all__ = ["available", "describe", "load", "load_file", "register", "unregister"]
