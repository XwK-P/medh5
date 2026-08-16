"""Cohort-level checks: what is wrong *between* files, not inside one.

``medh5 validate`` answers "is this file legal".  Every question that matters
for training is about the cohort: do all these files mean the same thing by
class 3, was this split computed before or after the cohort last changed, is
the same subject in train and test, is a class annotated in a tenth of the
files and absent from the rest.  None of those is visible from inside a single
sample, and each of them silently corrupts a result.

Findings carry the same shape as the validator's --- a code, a severity, a
location and a message --- but the codes are cohort codes (``C1xx``) and belong
to this tool, not to the format.  A file is not non-conforming because the
cohort around it is inconsistent.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5.dataset.manifest import Entry, Manifest

SEVERITIES = ("error", "warning", "info")

CHECK_CODES = {
    "C101": "the cohort uses more than one label set",
    "C102": "a class id means different things in different label sets",
    "C103": "a sample declares no label set",
    "C201": "a split claim's manifest digest is not this manifest's",
    "C202": "one subject's samples claim different partitions of one split",
    "C203": "a sample carries no split claim",
    "C301": "a class is examined in only part of the cohort",
    "C302": "a class appears in no sample",
    "C401": "a file changed after the manifest was written",
    "C402": "a file in the manifest no longer exists",
    "C501": "the cohort mixes de-identified and non-de-identified samples",
}
"""Cohort codes.  Distinct from the format's E/W table on purpose (§15.2)."""


@dataclass(slots=True)
class Finding:
    code: str
    severity: str
    message: str
    where: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "where": list(self.where),
            "summary": CHECK_CODES.get(self.code, ""),
        }

    def __str__(self) -> str:
        head = f"{self.severity.upper():7s} {self.code} {self.message}"
        if not self.where:
            return head
        shown = ", ".join(self.where[:3])
        more = "" if len(self.where) <= 3 else f" (+{len(self.where) - 3} more)"
        return f"{head}\n          {shown}{more}"


@dataclass(slots=True)
class CohortReport:
    manifest_sha256: str
    samples: int
    findings: list[Finding] = field(default_factory=list)
    coverage: dict[int, dict[str, int]] = field(default_factory=dict)

    def add(
        self, code: str, severity: str, message: str, where: Sequence[str] = ()
    ) -> Finding:
        finding = Finding(code, severity, message, tuple(where))
        self.findings.append(finding)
        return finding

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "samples": self.samples,
            "ok": self.ok,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "findings": [f.to_json() for f in self.findings],
            "coverage": {str(k): v for k, v in sorted(self.coverage.items())},
        }

    def format(self) -> str:
        head = (
            f"{self.samples} samples: {'OK' if self.ok else 'FAILED'} "
            f"({len(self.errors)} errors, {len(self.warnings)} warnings)"
        )
        return "\n".join([head, *(f"  {f}" for f in self.findings)])


def check(
    manifest: Manifest,
    *,
    set_id: str | None = None,
    deep: bool = False,
) -> CohortReport:
    """Every cohort-level check, over a manifest.

    ``deep`` re-reads each file's ``content_id`` rather than trusting size and
    mtime --- slower, and the only way to be sure a file is the one that was
    scanned.
    """
    report = CohortReport(manifest_sha256=manifest.sha256(), samples=len(manifest))
    _vocabulary(manifest, report)
    _splits(manifest, report, set_id=set_id)
    _coverage(manifest, report)
    _freshness(manifest, report, deep=deep)
    _deidentification(manifest, report)
    return report


def _vocabulary(manifest: Manifest, report: CohortReport) -> None:
    """One cohort, one meaning per class id."""
    by_digest: dict[str, list[Entry]] = {}
    missing: list[str] = []
    for entry in manifest:
        if entry.label_set_digest is None:
            missing.append(entry.path)
            continue
        by_digest.setdefault(entry.label_set_digest, []).append(entry)
    if missing:
        report.add(
            "C103",
            "warning",
            f"{len(missing)} sample(s) declare no label set, so their class ids "
            "mean whatever the reader assumes",
            missing,
        )
    if len(by_digest) > 1:
        names = sorted({e.label_set_id or "?" for es in by_digest.values() for e in es})
        report.add(
            "C101",
            "warning",
            f"{len(by_digest)} distinct label sets in one cohort ({', '.join(names)})",
            [es[0].path for es in by_digest.values()],
        )
        # Same id, different key, is the failure mode that silently mislabels a
        # whole training run --- so it is an error, not the warning above.
        meanings: dict[int, set[str]] = {}
        for entries in by_digest.values():
            for entry in entries:
                for class_id in entry.class_ids:
                    meanings.setdefault(class_id, set()).add(
                        f"{entry.label_set_id}@{entry.label_set_version}"
                    )
        clashing = sorted(c for c, sets in meanings.items() if len(sets) > 1)
        if clashing:
            report.add(
                "C102",
                "error",
                f"class id(s) {clashing} appear in more than one label set; "
                "check they mean the same thing before training on the union",
                [e.path for es in by_digest.values() for e in es[:1]],
            )


def _splits(manifest: Manifest, report: CohortReport, *, set_id: str | None) -> None:
    """A claim is only worth something if it is about this cohort (§12.3)."""
    digest = manifest.sha256()
    stale: list[str] = []
    unclaimed: list[str] = []
    by_subject: dict[str, dict[str, set[str]]] = {}
    for entry in manifest:
        claims = [
            c for c in entry.splits if set_id is None or c.get("set_id") == set_id
        ]
        if not claims:
            unclaimed.append(entry.path)
            continue
        for claim in claims:
            recorded = claim.get("manifest_sha256")
            if recorded is not None and recorded != digest:
                stale.append(entry.path)
            bucket = by_subject.setdefault(entry.subject_id, {})
            bucket.setdefault(str(claim.get("set_id")), set()).add(
                str(claim.get("partition"))
            )
    if stale:
        report.add(
            "C201",
            "error",
            f"{len(stale)} sample(s) claim a split computed against a different "
            "manifest --- re-split, or the partitions are not the ones in use",
            stale,
        )
    if unclaimed and len(unclaimed) != len(manifest):
        report.add(
            "C203",
            "warning",
            f"{len(unclaimed)} of {len(manifest)} sample(s) carry no split claim",
            unclaimed,
        )
    leaking = sorted(
        subject
        for subject, sets in by_subject.items()
        for partitions in sets.values()
        if len(partitions) > 1
    )
    if leaking:
        report.add(
            "C202",
            "error",
            f"{len(leaking)} subject(s) appear in more than one partition of the "
            "same split --- this leaks anatomy between train and test",
            leaking,
        )


def _coverage(manifest: Manifest, report: CohortReport) -> None:
    """Which classes were examined where (§11.3), and where they were not."""
    coverage: dict[int, dict[str, int]] = {}
    every = sorted(
        {c for e in manifest for c in (*e.class_ids, *e.annotated_class_ids)}
    )
    for class_id in every:
        examined = sum(1 for e in manifest if e.examined(class_id))
        present = sum(1 for e in manifest if e.has_class(class_id))
        coverage[class_id] = {
            "examined_in": examined,
            "present_in": present,
            "of": len(manifest),
        }
    report.coverage = coverage
    partial = [c for c, v in coverage.items() if 0 < v["examined_in"] < len(manifest)]
    if partial:
        report.add(
            "C301",
            "warning",
            f"class(es) {partial} are examined in only part of the cohort; the "
            "unexamined samples are not negative examples of them (§11.3)",
        )
    absent = [c for c, v in coverage.items() if v["present_in"] == 0]
    if absent:
        report.add(
            "C302",
            "info",
            f"class(es) {absent} are named but occur in no sample",
        )


def _freshness(manifest: Manifest, report: CohortReport, *, deep: bool) -> None:
    missing = [e.path for e in manifest if not os.path.exists(e.path)]
    if missing:
        report.add(
            "C402", "error", f"{len(missing)} manifest file(s) no longer exist", missing
        )
    changed = [p for p in manifest.stale() if p not in set(missing)]
    if deep:
        changed = sorted(set(changed) | set(_changed_content(manifest)))
    if changed:
        report.add(
            "C401",
            "warning" if not deep else "error",
            f"{len(changed)} file(s) changed after the manifest was written; "
            "re-scan before trusting a split made from it",
            changed,
        )


def _changed_content(manifest: Manifest) -> list[str]:
    """Paths whose ``content_id`` no longer matches --- the real check."""
    from medh5.collection import Collection, open_any

    out: list[str] = []
    for entry in manifest:
        if entry.content_id is None or not os.path.exists(entry.path):
            continue
        try:
            with open_any(entry.path, key=entry.key) as opened:
                if isinstance(opened, Collection):
                    # No key: the entry names a shard, not a sample in it.
                    out.append(entry.path)
                    continue
                current = opened.content_id
        except Exception:  # noqa: BLE001 - unreadable is "changed" for this check
            out.append(entry.path)
            continue
        if current != entry.content_id:
            out.append(entry.path)
    return out


def _deidentification(manifest: Manifest, report: CohortReport) -> None:
    """A cohort that is only partly de-identified is not de-identified (§11.4)."""
    identified = [e.path for e in manifest if not e.deidentified]
    if identified and len(identified) != len(manifest):
        report.add(
            "C501",
            "warning",
            f"{len(identified)} of {len(manifest)} sample(s) carry no "
            "de-identification record; absence is not evidence of it",
            identified,
        )


__all__ = ["CHECK_CODES", "CohortReport", "Finding", "check"]
