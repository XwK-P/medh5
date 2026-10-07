"""The sample document: everything in ``/meta`` (spec §2.4).

The 1.0 rule for where a fact lives is exact, and this module is one half of it:

* **arrays and per-object facts live in HDF5**, as attributes on the object they
  describe;
* **documents live in** ``/meta``.

Nothing is mirrored.  0.x split metadata between typed attributes and a JSON
``extra`` blob and duplicated some values in both; mirrors drift, and a reader
then has to decide which copy to believe.

The document model, its serialisation and its JSON Schema check are the format
engine's; the schema is embedded in it (``crates/medh5/data/``), so checking
E005 needs no optional dependency.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from medh5 import _core

META_DATASET: str = _core.META_DATASET

SampleDocument = _core.SampleDocument
"""Typed access to ``/meta``: identity, timepoints, cohort, label set,
provenance, quality, splits, acquisition, de-identification and extensions."""

new_document = _core.new_document

read_document = _core.read_document
"""The sample document under a sample root (``sample.root``), parsed."""

read_document_text = _core.read_document_text
"""The raw ``/meta`` text under a sample root."""

SCHEMA_NAME = "medh5-sample-1.0.schema.json"
"""The schema's file name, as published beside the specification."""

SCHEMA_PATH = Path(__file__).parent / "schemas" / SCHEMA_NAME
"""The schema as a file, for tools that want one: the same bytes the engine
embeds (a test compares the copies)."""


def schema_text() -> str:
    """The sample-document JSON Schema (draft 2020-12), as published."""
    return str(_core.schema_text())


@lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    """The sample-document JSON Schema, parsed."""
    found: dict[str, Any] = json.loads(schema_text())
    return found


def schema_available() -> bool:
    """Whether E005 can be checked.  Always ``True``: the engine embeds the
    schema and its validator, so no optional dependency decides this."""
    return True


def validate_against_schema(doc: Mapping[str, Any]) -> list[str]:
    """Validate a document, returning human-readable messages (empty when valid).

    Messages read ``<path or <root>>: <message>``, sorted by path.
    """
    return list(_core.validate_against_schema(dict(doc)))


__all__ = [
    "META_DATASET",
    "SCHEMA_NAME",
    "SCHEMA_PATH",
    "SampleDocument",
    "new_document",
    "read_document",
    "read_document_text",
    "schema",
    "schema_available",
    "schema_text",
    "validate_against_schema",
]
