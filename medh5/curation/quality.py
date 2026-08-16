"""Quality records (spec §11.2).

Status changes are **activities**, not fields with private history: the audit
trail is the provenance graph, so a quality record says what is true now and
the graph says how it got that way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.errors import MEDH5ValidationError

QUALITY_STATUS = (
    "draft",
    "submitted",
    "reviewed",
    "approved",
    "rejected",
    "deprecated",
)

ISSUE_SEVERITY = ("info", "warning", "error")


@dataclass(frozen=True, slots=True)
class Agreement:
    """One inter-rater or against-reference agreement measurement."""

    metric: str
    value: float
    against: str | None = None
    per_class: Mapping[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metric": self.metric, "value": float(self.value)}
        if self.against is not None:
            out["against"] = self.against
        if self.per_class:
            out["per_class"] = {str(k): float(v) for k, v in self.per_class.items()}
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Agreement:
        return cls(
            metric=str(doc["metric"]),
            value=float(doc["value"]),
            against=doc.get("against"),
            per_class={
                str(k): float(v) for k, v in (doc.get("per_class") or {}).items()
            },
        )


@dataclass(frozen=True, slots=True)
class Issue:
    """A known defect in an annotation, recorded rather than silently tolerated."""

    code: str
    severity: str = "info"
    class_ids: tuple[int, ...] = ()
    note: str | None = None

    def __post_init__(self) -> None:
        if self.severity not in ISSUE_SEVERITY:
            raise MEDH5ValidationError(
                f"issue severity {self.severity!r} must be one of "
                f"{list(ISSUE_SEVERITY)}"
            )
        object.__setattr__(self, "class_ids", tuple(int(c) for c in self.class_ids))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "severity": self.severity}
        if self.class_ids:
            out["class_ids"] = list(self.class_ids)
        if self.note is not None:
            out["note"] = self.note
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Issue:
        return cls(
            code=str(doc["code"]),
            severity=str(doc.get("severity", "info")),
            class_ids=tuple(doc.get("class_ids") or ()),
            note=doc.get("note"),
        )


@dataclass(frozen=True, slots=True)
class QualityRecord:
    """What is known about an annotation's trustworthiness."""

    status: str
    confidence: float | None = None
    reviewed_by: tuple[str, ...] = ()
    agreement: tuple[Agreement, ...] = ()
    issues: tuple[Issue, ...] = ()
    edit_effort_s: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in QUALITY_STATUS:
            raise MEDH5ValidationError(
                f"quality status {self.status!r} must be one of {list(QUALITY_STATUS)}"
            )
        object.__setattr__(self, "reviewed_by", tuple(self.reviewed_by))
        object.__setattr__(self, "agreement", tuple(self.agreement))
        object.__setattr__(self, "issues", tuple(self.issues))

    @property
    def is_usable(self) -> bool:
        """Whether the record marks data fit for training or evaluation."""
        return self.status in ("reviewed", "approved")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.confidence is not None:
            out["confidence"] = float(self.confidence)
        if self.reviewed_by:
            out["reviewed_by"] = list(self.reviewed_by)
        if self.agreement:
            out["agreement"] = [a.to_json() for a in self.agreement]
        if self.issues:
            out["issues"] = [i.to_json() for i in self.issues]
        if self.edit_effort_s is not None:
            out["edit_effort_s"] = float(self.edit_effort_s)
        out.update(self.extra)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> QualityRecord:
        known = {
            "status",
            "confidence",
            "reviewed_by",
            "agreement",
            "issues",
            "edit_effort_s",
        }
        return cls(
            status=str(doc["status"]),
            confidence=doc.get("confidence"),
            reviewed_by=tuple(doc.get("reviewed_by") or ()),
            agreement=tuple(Agreement.from_json(a) for a in doc.get("agreement") or ()),
            issues=tuple(Issue.from_json(i) for i in doc.get("issues") or ()),
            edit_effort_s=doc.get("edit_effort_s"),
            extra={k: v for k, v in doc.items() if k not in known},
        )


def quality_from_json(doc: Mapping[str, Any] | None) -> dict[str, QualityRecord]:
    if not doc:
        return {}
    return {str(k): QualityRecord.from_json(v) for k, v in doc.items()}


def quality_to_json(records: Mapping[str, QualityRecord]) -> dict[str, Any]:
    return {k: v.to_json() for k, v in records.items()}


def dice_agreement(
    per_class: Mapping[int, float], against: str | None = None
) -> Agreement:
    """Build a mean-Dice :class:`Agreement` from per-class values."""
    values: Sequence[float] = list(per_class.values())
    mean = float(sum(values) / len(values)) if values else 0.0
    return Agreement(
        metric="dice",
        value=mean,
        against=against,
        per_class={str(k): float(v) for k, v in per_class.items()},
    )


__all__ = [
    "ISSUE_SEVERITY",
    "QUALITY_STATUS",
    "Agreement",
    "Issue",
    "QualityRecord",
    "dice_agreement",
    "quality_from_json",
    "quality_to_json",
]
