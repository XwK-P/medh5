"""The sample document: everything in ``/meta`` (spec §2.4).

The 1.0 rule for where a fact lives is exact, and this module is one half of it:

* **arrays and per-object facts live in HDF5**, as attributes on the object they
  describe;
* **documents live in** ``/meta``.

Nothing is mirrored.  0.x split metadata between typed attributes and a JSON
``extra`` blob and duplicated some values in both; mirrors drift, and a reader
then has to decide which copy to believe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py

from medh5._hdf5 import str_dtype
from medh5.curation.identity import (
    Cohort,
    Deidentification,
    Identity,
    SplitClaim,
    splits_from_json,
)
from medh5.curation.provenance import Provenance
from medh5.curation.quality import QualityRecord, quality_from_json, quality_to_json
from medh5.curation.timeline import Timeline
from medh5.errors import MEDH5SchemaError
from medh5.labels.labelset import LabelSet

SCHEMA_PATH = Path(__file__).parent / "schemas" / "medh5-sample-1.0.schema.json"

META_DATASET = "meta"


@lru_cache(maxsize=1)
def schema() -> dict[str, Any]:
    """The bundled sample-document JSON Schema."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def schema_available() -> bool:
    """Whether ``jsonschema`` is installed, so E005 can actually be checked."""
    try:
        import jsonschema  # noqa: F401
    except ImportError:
        return False
    return True


def validate_against_schema(doc: Mapping[str, Any]) -> list[str]:
    """Validate a document, returning human-readable messages (empty when valid).

    Returns an empty list when ``jsonschema`` is absent; callers that need to
    know whether the check actually ran ask :func:`schema_available` first, so a
    missing optional dependency never masquerades as a clean bill of health.
    """
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - exercised only without the extra
        return []
    validator = jsonschema.Draft202012Validator(schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path) or '<root>'}: {err.message}"
        for err in sorted(validator.iter_errors(dict(doc)), key=lambda e: list(e.path))
    ]


@dataclass(slots=True)
class SampleDocument:
    """Typed access to ``/meta``."""

    identity: Identity
    timepoints: Timeline
    cohort: Cohort = field(default_factory=Cohort)
    label_set: LabelSet | None = None
    provenance: Provenance = field(default_factory=Provenance)
    quality: dict[str, QualityRecord] = field(default_factory=dict)
    splits: tuple[SplitClaim, ...] = ()
    acquisition: dict[str, dict[str, Any]] = field(default_factory=dict)
    deidentification: Deidentification | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # -- serialization -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "identity": self.identity.to_json(),
            "timepoints": self.timepoints.to_json(),
        }
        cohort = self.cohort.to_json()
        if cohort:
            doc["cohort"] = cohort
        if self.label_set is not None:
            doc["label_set"] = self.label_set.to_json()
        if self.provenance:
            doc["provenance"] = self.provenance.to_json()
        if self.quality:
            doc["quality"] = quality_to_json(self.quality)
        if self.splits:
            doc["splits"] = [s.to_json() for s in self.splits]
        if self.acquisition:
            doc["acquisition"] = dict(self.acquisition)
        if self.deidentification is not None:
            doc["deidentification"] = self.deidentification.to_json()
        if self.extra:
            doc["extra"] = dict(self.extra)
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SampleDocument:
        try:
            identity = Identity.from_json(doc["identity"])
            timepoints = Timeline.from_json(doc["timepoints"])
        except KeyError as exc:
            raise MEDH5SchemaError(
                f"sample document is missing required member {exc.args[0]!r}"
            ) from None
        return cls(
            identity=identity,
            timepoints=timepoints,
            cohort=Cohort.from_json(doc.get("cohort")),
            label_set=LabelSet.from_json(doc.get("label_set")),
            provenance=Provenance.from_json(doc.get("provenance")),
            quality=quality_from_json(doc.get("quality")),
            splits=splits_from_json(doc.get("splits")),
            acquisition=dict(doc.get("acquisition") or {}),
            deidentification=Deidentification.from_json(doc.get("deidentification")),
            extra=dict(doc.get("extra") or {}),
        )

    def dumps(self, *, indent: int | None = None) -> str:
        """Serialize to the JSON string stored in ``/meta``."""
        return json.dumps(self.to_json(), ensure_ascii=False, indent=indent)

    @classmethod
    def loads(cls, payload: str | bytes) -> SampleDocument:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MEDH5SchemaError(f"`meta` is not valid JSON: {exc}") from exc
        if not isinstance(doc, dict):
            raise MEDH5SchemaError("`meta` must hold a JSON object")
        return cls.from_json(doc)

    def check_schema(self) -> list[str]:
        return validate_against_schema(self.to_json())

    # -- convenience -------------------------------------------------------

    @property
    def subject_id(self) -> str:
        return self.identity.subject_id

    @property
    def group_id(self) -> str:
        return self.cohort.grouping_key(self.identity.subject_id)

    def quality_of(self, key: str | None) -> QualityRecord | None:
        return self.quality.get(key) if key else None

    def summary(self) -> dict[str, Any]:
        """Compact description for ``medh5 info``."""
        return {
            "sample_id": self.identity.sample_id,
            "subject_id": self.identity.subject_id,
            "timepoints": [
                {
                    "id": t.id,
                    "index": t.index,
                    "label": t.label,
                    "days_from_baseline": t.days_from_baseline,
                }
                for t in self.timepoints
            ],
            "label_set": (
                {
                    "id": self.label_set.id,
                    "version": self.label_set.version,
                    "classes": len(self.label_set),
                    "form": self.label_set.form,
                }
                if self.label_set
                else None
            ),
            "agents": len(self.provenance.agents),
            "activities": len(self.provenance.activities),
            "quality_records": len(self.quality),
            "deidentified": self.deidentification is not None,
        }


# --------------------------------------------------------------------------
# HDF5 carriage
# --------------------------------------------------------------------------


def write_document(root: h5py.Group, document: SampleDocument) -> h5py.Dataset:
    """Write the sample document to ``<root>/meta``.

    Stored as a **scalar variable-length UTF-8 string**, always.  HDF5 filters do
    not apply to variable-length data --- it lives in the global heap --- so a
    compressed ``meta`` is not expressible; a vocabulary large enough for that to
    matter uses ``form: "ref"`` (spec §5.1) instead.
    """
    if META_DATASET in root:
        del root[META_DATASET]
    return root.create_dataset(META_DATASET, data=document.dumps(), dtype=str_dtype())


def read_document_text(root: h5py.Group) -> str:
    """Read the raw ``/meta`` payload as text."""
    if META_DATASET not in root:
        raise MEDH5SchemaError(f"{root.name}: required dataset `meta` is absent")
    raw = root[META_DATASET][()]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def read_document(root: h5py.Group) -> SampleDocument:
    """Read and parse ``<root>/meta``."""
    return SampleDocument.loads(read_document_text(root))


def new_document(
    sample_id: str,
    subject_id: str | None = None,
    *,
    timepoints: Timeline | Sequence[str] | None = None,
    **identity_fields: Any,
) -> SampleDocument:
    """Start a document with the two required members filled in.

    ``subject_id`` defaults to ``sample_id``: a single-timepoint sample keyed by
    patient is the common case, and defaulting keeps the required field from
    becoming boilerplate people paste wrongly.
    """
    if timepoints is None:
        timeline = Timeline.single()
    elif isinstance(timepoints, Timeline):
        timeline = timepoints
    else:
        from medh5.curation.timeline import Timepoint

        timeline = Timeline(
            [Timepoint(id=tp, index=i) for i, tp in enumerate(timepoints)]
        )
    return SampleDocument(
        identity=Identity(
            sample_id=sample_id,
            subject_id=subject_id or sample_id,
            **identity_fields,
        ),
        timepoints=timeline,
    )


__all__ = [
    "META_DATASET",
    "SCHEMA_PATH",
    "SampleDocument",
    "new_document",
    "read_document",
    "read_document_text",
    "schema",
    "schema_available",
    "validate_against_schema",
    "write_document",
]
