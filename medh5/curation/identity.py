"""Identity, cohort, splits and de-identification (spec §11.4, §12).

``subject_id`` prevents the most common evaluation error in medical AI --- the
same patient in train and test --- and because a sample never spans subjects,
assigning whole files to partitions is subject-safe with no further bookkeeping.

Per-occasion identifiers (``study_uid``, ``series_uids``, dates, ages) live on
the *timepoint*, not here, because a sample may have several of each.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.errors import MEDH5ValidationError

SEX_VALUES = ("F", "M", "O", "unknown")
LATERALITY_VALUES = ("left", "right", "bilateral")
PARTITIONS = ("train", "val", "test", "holdout", "unassigned")


@dataclass(frozen=True, slots=True)
class Identity:
    """Who and what the sample is about (spec §12.1)."""

    sample_id: str
    subject_id: str
    sex: str | None = None
    laterality: str | None = None
    bodypart: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id or not self.subject_id:
            raise MEDH5ValidationError(
                "identity requires both sample_id and subject_id", code="E005"
            )
        if self.sex is not None and self.sex not in SEX_VALUES:
            raise MEDH5ValidationError(
                f"sex {self.sex!r} must be one of {list(SEX_VALUES)}", code="E005"
            )
        if self.laterality is not None and self.laterality not in LATERALITY_VALUES:
            raise MEDH5ValidationError(
                f"laterality {self.laterality!r} must be one of "
                f"{list(LATERALITY_VALUES)}",
                code="E005",
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "sample_id": self.sample_id,
            "subject_id": self.subject_id,
        }
        for key in ("sex", "laterality", "bodypart"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Identity:
        known = {"sample_id", "subject_id", "sex", "laterality", "bodypart"}
        return cls(
            sample_id=str(doc["sample_id"]),
            subject_id=str(doc["subject_id"]),
            sex=doc.get("sex"),
            laterality=doc.get("laterality"),
            bodypart=doc.get("bodypart"),
            extra={k: v for k, v in doc.items() if k not in known},
        )


@dataclass(frozen=True, slots=True)
class Cohort:
    """Where the sample came from, and how to group it for splitting (§12.2)."""

    dataset_id: str | None = None
    site_id: str | None = None
    scanner_id: str | None = None
    group_id: str | None = None
    acquisition_protocol: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def grouping_key(self, subject_id: str) -> str:
        """``group_id`` if set, else ``subject_id`` (§12.2)."""
        return self.group_id or subject_id

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "dataset_id",
            "site_id",
            "scanner_id",
            "group_id",
            "acquisition_protocol",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Cohort:
        if not doc:
            return cls()
        known = {
            "dataset_id",
            "site_id",
            "scanner_id",
            "group_id",
            "acquisition_protocol",
        }
        return cls(
            dataset_id=doc.get("dataset_id"),
            site_id=doc.get("site_id"),
            scanner_id=doc.get("scanner_id"),
            group_id=doc.get("group_id"),
            acquisition_protocol=doc.get("acquisition_protocol"),
            extra={k: v for k, v in doc.items() if k not in known},
        )


@dataclass(frozen=True, slots=True)
class SplitClaim:
    """A *claim* of split membership, not an authority (spec §12.3).

    The dataset manifest is authoritative; ``manifest_sha256`` is what lets a
    reader notice that an in-file claim predates the current split instead of
    training on a stale partition.
    """

    set_id: str
    partition: str
    fold: int | None = None
    assigned_by: str | None = None
    assigned_at: str | None = None
    manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.partition not in PARTITIONS:
            raise MEDH5ValidationError(
                f"partition {self.partition!r} must be one of {list(PARTITIONS)}",
                code="E005",
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"set_id": self.set_id, "partition": self.partition}
        for key in ("fold", "assigned_by", "assigned_at", "manifest_sha256"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> SplitClaim:
        return cls(
            set_id=str(doc["set_id"]),
            partition=str(doc["partition"]),
            fold=doc.get("fold"),
            assigned_by=doc.get("assigned_by"),
            assigned_at=doc.get("assigned_at"),
            manifest_sha256=doc.get("manifest_sha256"),
        )


@dataclass(frozen=True, slots=True)
class Deidentification:
    """What was done to remove identifiers (spec §11.4).

    A file *without* this record must be treated as potentially identifying ---
    absence is not evidence of de-identification.
    """

    method: str
    profile: str | None = None
    date_shift_days: int | None = None
    id_mapping: str | None = None
    performed_by: str | None = None
    date: str | None = None
    burned_in_annotation_checked: bool | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"method": self.method}
        for key in (
            "profile",
            "date_shift_days",
            "id_mapping",
            "performed_by",
            "date",
            "burned_in_annotation_checked",
        ):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> Deidentification | None:
        if not doc:
            return None
        known = {
            "method",
            "profile",
            "date_shift_days",
            "id_mapping",
            "performed_by",
            "date",
            "burned_in_annotation_checked",
        }
        return cls(
            method=str(doc["method"]),
            profile=doc.get("profile"),
            date_shift_days=doc.get("date_shift_days"),
            id_mapping=doc.get("id_mapping"),
            performed_by=doc.get("performed_by"),
            date=doc.get("date"),
            burned_in_annotation_checked=doc.get("burned_in_annotation_checked"),
            extra={k: v for k, v in doc.items() if k not in known},
        )


def splits_from_json(
    docs: Sequence[Mapping[str, Any]] | None,
) -> tuple[SplitClaim, ...]:
    if not docs:
        return ()
    return tuple(SplitClaim.from_json(d) for d in docs)


__all__ = [
    "LATERALITY_VALUES",
    "PARTITIONS",
    "SEX_VALUES",
    "Cohort",
    "Deidentification",
    "Identity",
    "SplitClaim",
    "splits_from_json",
]
