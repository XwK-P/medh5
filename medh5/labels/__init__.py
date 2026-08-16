"""The label space: controlled vocabularies of classes (spec §5)."""

from __future__ import annotations

from medh5.labels.labelset import (
    BACKGROUND_ID,
    CLOSURES,
    IGNORE_ID,
    MAX_CLASS_ID,
    LabelClass,
    LabelSet,
    OntologyCode,
    Relation,
    Skeleton,
    canonical_json,
)
from medh5.labels.registry import available, load, register

__all__ = [
    "BACKGROUND_ID",
    "CLOSURES",
    "IGNORE_ID",
    "MAX_CLASS_ID",
    "LabelClass",
    "LabelSet",
    "OntologyCode",
    "Relation",
    "Skeleton",
    "available",
    "canonical_json",
    "load",
    "register",
]
