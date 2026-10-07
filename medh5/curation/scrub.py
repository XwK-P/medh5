"""``medh5 scrub`` --- finding identifiers in a container, and attesting to it.

**What this cannot do, stated first, because the attestation depends on it.**
Scrubbing a MEDH5 file inspects metadata.  It does not look at voxels, so it
cannot see text burned into a scanned document, a face reconstructible from a
head CT, or an accession number photographed onto a film.  A file this tool
declares clean may still be identifying, and the record it writes says exactly
what was and was not checked (§11.4) rather than "de-identified".

**What it does do.**  The format never requires an identifier, but a converter
can carry one in anywhere a string is allowed: an ``extra`` namespace holding
raw DICOM tags, an ``acquisition`` block copied wholesale, a real
``FrameOfReferenceUID``, an unshifted study date, a ``subject_id`` that is
somebody's record number.  Each of those has a rule here, each finding names
the rule and the location, and ``--apply`` acts only on the ones that can be
acted on without guessing.

**Where it looks: everywhere a string can be.**  Every string in ``/meta``,
every object name, every attribute and every string dataset in the file ---
including the unknown ones ``amend`` carries through.  Fields are exempted by
name, in :class:`_Sweep`, with the reason, rather than examined by being
remembered: each earlier version kept a list of places to look, and each audit
found string-bearing places the list had missed.  The scan and ``--apply`` are
one traversal, so a finding the scan calls actionable is, by construction, one
the clean acts on.

**Pseudonymising UIDs rather than deleting them.** A UID is how two files agree
they describe the same frame of reference (§3.4); deleting it breaks
registration, so it is replaced by a stable hash of itself.  Unsalted by
default, which keeps a cohort joinable across independent runs on different
machines; ``salt`` makes the mapping unguessable at the cost of having to keep
the salt to reproduce it --- and only a salted run may record
``id_mapping: external``, because an unsalted hash is recoverable by anyone who
already holds the original UIDs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medh5 import _core

PROFILES: tuple[str, ...] = _core.SCRUB_PROFILES

IDENTIFYING_KEYS: frozenset[str] = _core.SCRUB_IDENTIFYING_KEYS
"""Attributes to remove, from the DICOM PS3.15 E.1 basic profile.

Matched case-, space- and underscore-insensitively, so ``PatientName``,
``patient_name`` and ``Patient Name`` are one key --- in ``/meta`` mappings and
as the names of HDF5 attributes alike.
"""

QUASI_IDENTIFYING_KEYS: frozenset[str] = _core.SCRUB_QUASI_IDENTIFYING_KEYS
"""Attributes that identify *in combination* --- and that some pipelines need.

``PatientWeight`` and ``PatientSize`` are inputs to a PET SUV calculation, so
removing them by default would break quantitative imaging to buy privacy the
caller may already have obtained another way.  They are reported for a human
decision under ``basic`` and removed under ``strict``.
"""

DATE_KEYS: frozenset[str] = _core.SCRUB_DATE_KEYS

UID_KEYS: frozenset[str] = _core.SCRUB_UID_KEYS
"""Keys that hold a UID.

A UID-shaped *value* is pseudonymised wherever it appears, which is the real
mechanism.  This set catches the other case: a UID key whose value is not
UID-shaped.  That cannot be pseudonymised safely --- a stable pseudonym needs
something recognisable to hash --- so it is reported for a person rather than
guessed at.
"""

MAX_DEPTH: int = _core.SCRUB_MAX_DEPTH
"""How deep the walk goes before it reports rather than descends.

Real metadata does not nest this far; a payload that does is reported as
unexaminable instead of skipped, because a silent stop in a tool that writes an
attestation is the worst thing this module could do.
"""

PSEUDONYM_PREFIX: str = _core.SCRUB_PSEUDONYM_PREFIX
"""What :func:`pseudonymise` produces, and therefore what the rules skip.

A rule that fires on its own output makes the tool non-idempotent: the second
run reports the same location, and a pipeline gate built on the exit code can
never go green however many times it is run.
"""

PATH_REMOVED: str = _core.SCRUB_PATH_REMOVED
"""What ``--profile strict`` leaves where a filesystem path was."""

AGE_LIMIT: float = _core.SCRUB_AGE_LIMIT
"""HIPAA Safe Harbor aggregates every age over 89 into one category, "90 or
older".  ``strict`` records such an age as 90 --- the category --- so 90 itself is
the aggregated value, and only a larger one identifies."""

INTERNAL_REFERENCES: tuple[str, ...] = _core.SCRUB_INTERNAL_REFERENCES
"""Provenance references to objects in this file, which are not paths."""

FREE_TEXT: int = _core.SCRUB_FREE_TEXT
"""Characters above which a string cannot be reviewed by a rule, only by a person."""

UNFIXABLE_LOCATIONS: tuple[str, ...] = _core.SCRUB_UNFIXABLE_LOCATIONS
"""Findings ``--apply`` never acts on by itself, however hard the profile looks.

The two ids are how every other file, manifest and split claim refers to this
sample (§12.1).  Blanking one because it reads as a person name would leave a
file nothing can join to --- so these are reported for a human to re-mint, or
replaced by stable pseudonyms when the caller asks for exactly that
(``pseudonymise_ids``).  They stay **non-actionable**, because ``actionable``
means "``--apply`` will fix this" and the re-scan check holds it to that.
"""

STRICT_RULES: tuple[str, ...] = _core.SCRUB_STRICT_RULES
"""Rules that only ``--profile strict`` acts on."""

IDENTITY_RULES: tuple[str, ...] = _core.SCRUB_IDENTITY_RULES
"""Findings about what the sample is *called*.

No ``--apply`` rewrites them unless asked to (``pseudonymise_ids``), and under
``--profile strict`` an open one --- or any finding on
:data:`UNFIXABLE_LOCATIONS` --- fails the run: a file cannot be attested
de-identified while its own name for the subject is the subject's record
number.
"""


@dataclass(slots=True)
class Finding:
    """One place an identifier may live."""

    rule: str
    where: str
    detail: str
    value: str | None = None
    actionable: bool = False
    fixable: bool = field(default=True, repr=False)
    """Whether ``--apply`` may ever act here.  An id or a vocabulary entry is
    reported and left to a person at every profile."""

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.scrub_finding_json(self)
        return result

    def __str__(self) -> str:
        result: str = _core.scrub_finding_line(self)
        return result


@dataclass(slots=True)
class ScrubReport:
    """What was found, what was changed, and what was never looked at."""

    path: str
    profile: str = "basic"
    findings: list[Finding] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    applied: bool = False
    remaining: list[Finding] = field(default_factory=list)
    """What a re-scan of the *written* file still finds (``apply`` only).

    An attestation is a claim about a file, so this module makes the claim and
    then checks it against the thing it just wrote.  ``--apply`` used to report
    findings it had left in place and exit 0 regardless.
    """
    uid_map: dict[str, str] = field(default_factory=dict)
    """Every original value this run replaced with a pseudonym --- UIDs, and the
    sample and subject ids under ``pseudonymise_ids``.  It is the key to the
    pseudonyms: keep it with the salt, not with the data."""
    not_checked: tuple[str, ...] = _core.SCRUB_NOT_CHECKED

    def add(
        self,
        rule: str,
        where: str,
        detail: str,
        value: Any = None,
        *,
        actionable: bool = False,
        fixable: bool = True,
    ) -> Finding:
        finding = Finding(
            **_core.scrub_finding(
                rule,
                where,
                detail,
                None if value is None else str(value),
                actionable=actionable,
                fixable=fixable,
            )
        )
        self.findings.append(finding)
        return finding

    @property
    def actionable(self) -> list[Finding]:
        return [f for f in self.findings if f.actionable]

    @property
    def needs_review(self) -> list[Finding]:
        return [f for f in self.findings if not f.actionable]

    @property
    def clean(self) -> bool:
        return not self.findings

    @property
    def remaining_actionable(self) -> list[Finding]:
        """Actionable findings a re-scan of the written file still reports."""
        return [f for f in self.remaining if f.actionable]

    @property
    def open_identity(self) -> list[Finding]:
        """Findings on what the sample is *called*, still open after this run.

        The ids, and a file named after one.  Under ``--profile strict`` each
        of these fails the run; see :data:`IDENTITY_RULES`.
        """
        left = self.remaining if self.applied else self.findings
        return [left[i] for i in _core.scrub_report_open_identity(self)]

    @property
    def ok(self) -> bool:
        """Whether this run leaves nothing a further ``--apply`` could fix.

        For a scan, that is "found nothing at all"; for an apply, "the file it
        wrote has no actionable findings left" --- and, under ``strict``, "and
        its ids are not identifiers".  Both are what a pipeline gate needs, and
        both are checked rather than assumed.
        """
        result: bool = _core.scrub_report_ok(self)
        return result

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = _core.scrub_report_json(self)
        return result

    def format(self) -> str:
        result: str = _core.scrub_report_format(self)
        return result

    def _update(self, fields: Mapping[str, Any]) -> ScrubReport:
        """Take the engine's report, keeping this object (and its lists)."""
        self.path = fields["path"]
        self.profile = fields["profile"]
        self.findings[:] = [Finding(**f) for f in fields["findings"]]
        self.actions[:] = fields["actions"]
        self.applied = fields["applied"]
        self.remaining[:] = [Finding(**f) for f in fields["remaining"]]
        self.uid_map.clear()
        self.uid_map.update(fields["uid_map"])
        self.not_checked = tuple(fields["not_checked"])
        return self

    @classmethod
    def _from_fields(cls, fields: Mapping[str, Any]) -> ScrubReport:
        return cls(path=fields["path"])._update(fields)


def pseudonymise(uid: str, salt: str = "") -> str:
    """A stable pseudonym for a UID: same input, same output, everywhere."""
    result: str = _core.scrub_pseudonymise(uid, salt)
    return result


def scan_document(document: Any, report: ScrubReport) -> ScrubReport:
    """Every rule, over every string in one sample document.  Reads only ``/meta``."""
    return report._update(_core.scrub_scan_document(document, report))


def scan(path: str | os.PathLike[str], *, profile: str = "basic") -> ScrubReport:
    """Find identifiers in one file.  Changes nothing."""
    return ScrubReport._from_fields(_core.scrub_scan(os.fspath(path), profile=profile))


def apply(
    path: str | os.PathLike[str],
    *,
    profile: str = "basic",
    salt: str = "",
    date_shift_days: int | None = None,
    performed_by: str | None = None,
    pseudonymise_ids: bool = False,
) -> ScrubReport:
    """Act on the actionable findings and write the §11.4 attestation.

    ``date_shift_days`` shifts every date the rules found rather than deleting
    it, which preserves the intervals a longitudinal study is about.  Without
    it, dates are removed --- an interval that cannot be trusted is worse than
    no interval.

    ``pseudonymise_ids`` replaces ``sample_id`` and ``subject_id`` --- and a
    ``cohort.group_id`` equal to either --- with salted stable pseudonyms, so
    every file of one subject still agrees.  It needs a *salt*: those ids are
    usually record numbers, and an unsalted hash of a record number is reversed
    by hashing every record number.
    """
    return ScrubReport._from_fields(
        _core.scrub_apply(
            os.fspath(path),
            profile=profile,
            salt=salt,
            date_shift_days=date_shift_days,
            performed_by=performed_by,
            pseudonymise_ids=pseudonymise_ids,
        )
    )


def scrub_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    apply_changes: bool = False,
    **options: Any,
) -> list[ScrubReport]:
    return [
        apply(path, **options)
        if apply_changes
        else scan(path, profile=options.get("profile", "basic"))
        for path in paths
    ]


__all__ = [
    "DATE_KEYS",
    "IDENTIFYING_KEYS",
    "IDENTITY_RULES",
    "MAX_DEPTH",
    "STRICT_RULES",
    "UNFIXABLE_LOCATIONS",
    "QUASI_IDENTIFYING_KEYS",
    "UID_KEYS",
    "PROFILES",
    "PSEUDONYM_PREFIX",
    "Finding",
    "ScrubReport",
    "apply",
    "pseudonymise",
    "scan",
    "scan_document",
    "scrub_paths",
]
