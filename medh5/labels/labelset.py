"""Label sets: id -> meaning (spec §5).

Annotations reference classes by ``uint16`` id and never by name, so a label set
is the only thing standing between an integer and a diagnosis.  Two properties
matter more than the data model:

* The hierarchy is a **DAG, not a tree**.  ``left_kidney`` is a ``kidney`` and is
  part of the urinary system; forcing that into a tree loses one of the two.
* ``closure`` is declared per annotation, never inferred.  A reader that helpfully
  adds ``liver`` because ``liver_segment_iv`` is present has invented ground
  truth, so the spec forbids it unless ``closure = "implicit"`` says otherwise.

The classes are the engine's (``medh5._core``): construction validates §5.1-§5.3
and ``to_json`` is the canonical serialization, in every language binding alike.
"""

from __future__ import annotations

from medh5._core import (
    BACKGROUND_ID,
    CLOSURES,
    FORMS,
    IGNORE_ID,
    INLINE_REQUIRED_BELOW,
    MAX_CLASS_ID,
    LabelClass,
    LabelSet,
    OntologyCode,
    Relation,
    Skeleton,
    canonical_json,
    check_class_id,
    from_keys,
)

__all__ = [
    "BACKGROUND_ID",
    "CLOSURES",
    "FORMS",
    "IGNORE_ID",
    "INLINE_REQUIRED_BELOW",
    "MAX_CLASS_ID",
    "LabelClass",
    "LabelSet",
    "OntologyCode",
    "Relation",
    "Skeleton",
    "canonical_json",
    "check_class_id",
    "from_keys",
]
