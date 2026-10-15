"""Coding systems as a label set names them and as DICOM designates them.

A class's ontology binding names its system as §5.2 spells it (``SNOMED-CT``,
``RadLex``, ...); DICOM's ``CodingSchemeDesignator`` names the same system
another way (``SCT``, ``RADLEX``, ...).  One table holds the pairs and serves
both directions, so an export and an import cannot disagree about one.  A
system the table does not know has no designator here, and none is invented:
the exporter falls back to a generic concept and says so, and the importer
keeps the designator the file states.

The concepts a DICOM SEG requires and a sample may not state --- a property
category, the laterality modifier --- are here too, so every code the
converters write comes from one place.
"""

from __future__ import annotations

from typing import NamedTuple


class Concept(NamedTuple):
    """A coded concept as DICOM writes it: value, designator and meaning."""

    code: str
    designator: str
    meaning: str


SYSTEMS: tuple[tuple[str, str], ...] = (
    ("SNOMED-CT", "SCT"),
    ("RadLex", "RADLEX"),
    ("FMA", "FMA"),
    ("UBERON", "UBERON"),
    ("LOINC", "LN"),
    ("ICD-10", "I10"),
    ("DCM", "DCM"),
)
"""``(system, designator)``: a §5.2 system and its DICOM designator."""

_BY_SYSTEM = {system.casefold(): designator for system, designator in SYSTEMS}
_BY_DESIGNATOR = {designator.casefold(): system for system, designator in SYSTEMS}


def designator(system: str | None) -> str | None:
    """The DICOM designator of *system*, or ``None`` when the table does not
    know it.

    A designator given where a system is expected is its own (``"SCT"`` is
    SNOMED-CT's), so a binding an earlier import stored as ``SCT`` exports as
    SNOMED-CT's.  Case is not significant.
    """
    if not system:
        return None
    key = system.strip().casefold()
    if key in _BY_SYSTEM:
        return _BY_SYSTEM[key]
    if key in _BY_DESIGNATOR:
        return _BY_SYSTEM[_BY_DESIGNATOR[key].casefold()]
    return None


def system(designator_: str | None) -> str | None:
    """The §5.2 system a DICOM designator names, or ``None`` when the table
    does not know it."""
    if not designator_:
        return None
    return _BY_DESIGNATOR.get(designator_.strip().casefold())


def same_concept(
    first: tuple[str | None, str | None], second: tuple[str | None, str | None]
) -> bool:
    """Whether two ``(system or designator, code)`` pairs name one concept.

    Systems compare through the table, so ``("SCT", "7771000")`` and
    ``("SNOMED-CT", "7771000")`` are one; a system the table does not know
    compares as written.
    """

    def normal(pair: tuple[str | None, str | None]) -> tuple[str, str]:
        scheme, code = pair
        named = designator(scheme)
        return (named if named is not None else str(scheme or "")), str(code or "")

    return normal(first) == normal(second)


GENERIC = Concept("91723000", "SCT", "Anatomical Structure")
"""What a SEG segment's property type or category is when the sample states
none: DICOM requires both and gives no way to leave either out."""

CATEGORIES: dict[str, Concept] = {
    "organ": GENERIC,
    "vessel": GENERIC,
    "bone": GENERIC,
    "lesion": Concept("49755003", "SCT", "Morphologically Abnormal Structure"),
    "device": Concept("260787004", "SCT", "Physical object"),
    "region": Concept("309825002", "SCT", "Spatial and Relational Concept"),
}
"""A §5.2 ``category`` → its SEG property category (CID 7150)."""

CATEGORY_OF: dict[tuple[str, str], str] = {
    (concept.designator, concept.code): category
    for category, concept in CATEGORIES.items()
    if concept is not GENERIC
}
"""A property category → the one §5.2 category it stands for, where there is
one: "Anatomical Structure" is an organ's, a vessel's and a bone's alike."""

LATERALITY: dict[str, Concept] = {
    "left": Concept("7771000", "SCT", "Left"),
    "right": Concept("24028007", "SCT", "Right"),
    "bilateral": Concept("51440002", "SCT", "Right and left"),
}
"""A §5.2 ``laterality`` → its property type modifier (CID 244)."""

LATERALITY_OF: dict[tuple[str, str], str] = {
    (concept.designator, concept.code): side for side, concept in LATERALITY.items()
}

ARTIFICIAL_INTELLIGENCE = Concept("123110", "DCM", "Artificial Intelligence")
"""The algorithm family of a segment a ``predict`` activity produced."""
