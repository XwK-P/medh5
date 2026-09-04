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
somebody's initials.  Each of those has a rule here, each finding names the
rule and the location, and ``--apply`` acts only on the ones that can be acted
on without guessing.

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

import hashlib
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from medh5._hdf5 import repack
from medh5.curation.identity import Cohort
from medh5.curation.timeline import Timeline, Timepoint
from medh5.errors import MEDH5ValidationError
from medh5.sample import frame_references

PROFILES = ("basic", "strict")

IDENTIFYING_KEYS = frozenset(
    {
        # --- the person -------------------------------------------------
        "patientname",
        "patientid",
        "patientbirthdate",
        "patientbirthtime",
        "patientaddress",
        "patienttelephonenumbers",
        "patienttelecominformation",
        "otherpatientids",
        "otherpatientidssequence",
        "otherpatientnames",
        "patientmothersbirthname",
        "patientinsuranceplancodesequence",
        "patientreligiouspreference",
        "patientinstitutionresidence",
        "currentpatientlocation",
        "countryofresidence",
        "regionofresidence",
        "militaryrank",
        "branchofservice",
        "occupation",
        "medicalrecordlocator",
        "issuerofpatientid",
        "patient_name",
        "patient_id",
        "mrn",
        "nhs_number",
        "ssn",
        # --- free text that routinely carries names ---------------------
        "patientcomments",
        "additionalpatienthistory",
        "medicalalerts",
        "allergies",
        "admittingdiagnosesdescription",
        "specialneeds",
        # --- staff ------------------------------------------------------
        "referringphysicianname",
        "referringphysicianaddress",
        "referringphysiciantelephonenumbers",
        "consultingphysicianname",
        "performingphysicianname",
        "operatorsname",
        "physiciansofrecord",
        "physicianreadingstudy",
        "namesofintendedrecipientsofresults",
        "requestingphysician",
        "responsibleperson",
        "verifyingobservername",
        "contentcreatorname",
        "personname",
        "reviewername",
        "scheduledperformingphysicianname",
        # --- the visit --------------------------------------------------
        "accessionnumber",
        "studyid",
        "admissionid",
        "issuerofadmissionid",
        "performedprocedurestepid",
        "performedprocedurestepdescription",
        "requestedprocedureid",
        "scheduledprocedurestepid",
        "scheduledstudylocation",
        "requestattributessequence",
        # --- the site and the equipment ---------------------------------
        "institutionname",
        "institutionaddress",
        "institutionaldepartmentname",
        "institutioncodesequence",
        "stationname",
        "deviceserialnumber",
        "plateid",
        "cassetteid",
        "detectorid",
        "gantryid",
        "generatorid",
        # --- clinical trials --------------------------------------------
        "clinicaltrialsubjectid",
        "clinicaltrialsubjectreadingid",
        "clinicaltrialsitename",
        "clinicaltrialsiteid",
        "clinicaltrialsponsorname",
        "clinicaltrialprotocolid",
        "clinicaltrialprotocolname",
        "clinicaltrialtimepointid",
    }
)
"""Attributes to remove, from the DICOM PS3.15 E.1 basic profile.

Matched case-, space- and underscore-insensitively, so ``PatientName``,
``patient_name`` and ``Patient Name`` are one key.
"""

QUASI_IDENTIFYING_KEYS = frozenset(
    {
        "patientage",
        "patientweight",
        "patientsize",
        "patientsex",
        "patientsexneutered",
        "ethnicgroup",
        "patientspeciesdescription",
        "patientstate",
        "patientbreeddescription",
        "pregnancystatus",
        "smokingstatus",
        "lastmenstrualdate",
    }
)
"""Attributes that identify *in combination* --- and that some pipelines need.

``PatientWeight`` and ``PatientSize`` are inputs to a PET SUV calculation, so
removing them by default would break quantitative imaging to buy privacy the
caller may already have obtained another way.  They are reported for a human
decision under ``basic`` and removed under ``strict``.
"""

DATE_KEYS = frozenset(
    {
        "studydate",
        "seriesdate",
        "acquisitiondate",
        "acquisitiondatetime",
        "contentdate",
        "instancecreationdate",
        "patientbirthdate",
        "admittingdate",
        "scheduledproceduredate",
        "scheduledprocedurestepstartdate",
        "performedprocedurestepstartdate",
        "lastmenstrualdate",
        "study_date",
        "acquisition_date",
        "birth_date",
        "date_of_birth",
    }
)

UID_KEYS = frozenset(
    {
        "studyinstanceuid",
        "seriesinstanceuid",
        "sopinstanceuid",
        "frameofreferenceuid",
        "mediastoragesopinstanceuid",
        "referencedsopinstanceuid",
        "irradiationeventuid",
        "concatenationuid",
        "storagemediafilesetuid",
        "study_uid",
        "series_uid",
        "frame_uid",
    }
)
"""Keys that hold a UID.

A UID-shaped *value* is pseudonymised wherever it appears, which is the real
mechanism.  This set catches the other case: a UID key whose value is not
UID-shaped.  That cannot be pseudonymised safely --- a stable pseudonym needs
something recognisable to hash --- so it is reported for a person rather than
guessed at.
"""

MAX_DEPTH = 8
"""How deep the walk goes before it reports rather than descends.

Real metadata does not nest this far; a payload that does is reported as
unexaminable instead of skipped, because a silent stop in a tool that writes an
attestation is the worst thing this module could do.
"""

PSEUDONYM_PREFIX = "pseudo:"
"""What :func:`pseudonymise` produces, and therefore what the rules skip.

A rule that fires on its own output makes the tool non-idempotent: the second
run reports the same location, and a pipeline gate built on the exit code can
never go green however many times it is run.
"""

PERSON_NAME = re.compile(r"^[A-Za-z' -]+\^[A-Za-z' -]+")
"""DICOM PN form ``Family^Given``: unambiguous enough to act on."""

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DICOM_DATE = re.compile(r"^\d{8}$")
DICOM_UID = re.compile(r"^\d+(\.\d+){3,}$")
"""A dotted-numeric OID.  ``pseudo:...`` and UUIDs do not match, by design."""

FREE_TEXT = 200
"""Characters above which a string cannot be reviewed by a rule, only by a person."""


@dataclass(slots=True)
class Finding:
    """One place an identifier may live."""

    rule: str
    where: str
    detail: str
    value: str | None = None
    actionable: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "where": self.where,
            "detail": self.detail,
            "value": self.value,
            "actionable": self.actionable,
        }

    def __str__(self) -> str:
        mark = "*" if self.actionable else " "
        shown = f" = {self.value!r}" if self.value is not None else ""
        return f"{mark} {self.rule:14s} {self.where}{shown}\n    {self.detail}"


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
    not_checked: tuple[str, ...] = (
        "pixel data (burned-in text, identifiable anatomy)",
        "free text whose meaning this tool cannot judge",
    )

    def add(
        self,
        rule: str,
        where: str,
        detail: str,
        value: Any = None,
        *,
        actionable: bool = False,
    ) -> Finding:
        finding = Finding(
            rule=rule,
            where=where,
            detail=detail,
            value=None if value is None else _preview(value),
            actionable=actionable,
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
    def ok(self) -> bool:
        """Whether this run leaves nothing a further ``--apply`` could fix.

        For a scan, that is "found nothing at all"; for an apply, "the file it
        wrote has no actionable findings left".  Both are what a pipeline gate
        needs, and both are checked rather than assumed.
        """
        if not self.applied:
            return self.clean
        return not self.remaining_actionable

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "profile": self.profile,
            "applied": self.applied,
            "clean": self.clean,
            "ok": self.ok,
            "findings": [f.to_json() for f in self.findings],
            "actions": list(self.actions),
            "remaining": [f.to_json() for f in self.remaining],
            "uid_map": dict(self.uid_map),
            "not_checked": list(self.not_checked),
        }

    def format(self) -> str:
        head = (
            f"{self.path}: {len(self.findings)} finding(s) "
            f"({len(self.actionable)} actionable, "
            f"{len(self.needs_review)} for review)"
        )
        lines = [head, *(str(f) for f in self.findings)]
        if self.actions:
            lines.append(f"  applied: {len(self.actions)} change(s)")
        if self.applied:
            left = self.remaining_actionable
            lines.append(
                f"  re-scanned after applying: {len(self.remaining)} finding(s) "
                f"remain, {len(left)} actionable"
            )
            lines.extend(f"  REMAINS {f.where}" for f in left)
        lines.append("  NOT checked: " + "; ".join(self.not_checked))
        return "\n".join(lines)


def _preview(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _normalise(key: str) -> str:
    return key.replace("_", "").replace(" ", "").lower()


def pseudonymise(uid: str, salt: str = "") -> str:
    """A stable pseudonym for a UID: same input, same output, everywhere."""
    digest = hashlib.sha256(f"{salt}{uid}".encode()).hexdigest()
    return f"{PSEUDONYM_PREFIX}{digest[:32]}"


def scan_document(document: Any, report: ScrubReport) -> ScrubReport:
    """Every rule, over one sample document.  Reads nothing but ``/meta``."""
    identity = document.identity
    if PERSON_NAME.match(identity.subject_id):
        report.add(
            "person_name",
            "identity.subject_id",
            "reads as a DICOM person name, not a pseudonym (§11.4)",
            identity.subject_id,
        )
    for name in ("sample_id", "subject_id"):
        value = getattr(identity, name)
        if ISO_DATE.search(value) or DICOM_DATE.match(value):
            report.add(
                "date",
                f"identity.{name}",
                "contains what looks like a date",
                value,
            )
    _scan_mapping(dict(identity.extra), "identity.extra", report)
    _scan_mapping(document.cohort.to_json(), "cohort", report)

    shifted = document.deidentification is not None and (
        document.deidentification.date_shift_days is not None
    )
    for index, timepoint in enumerate(document.timepoints):
        where = f"timepoints[{index}]"
        if timepoint.study_uid and DICOM_UID.match(timepoint.study_uid):
            report.add(
                "uid",
                f"{where}.study_uid",
                "a real DICOM UID; SHOULD be a pseudonym (§11.4)",
                timepoint.study_uid,
                actionable=True,
            )
        for value in (timepoint.series_uids or {}).values():
            if DICOM_UID.match(str(value)):
                report.add(
                    "uid",
                    f"{where}.series_uids",
                    "a real DICOM UID; SHOULD be a pseudonym (§11.4)",
                    value,
                    actionable=True,
                )
        if timepoint.date and not shifted:
            report.add(
                "date",
                f"{where}.date",
                "a date with no recorded shift; either shift the cohort "
                "consistently or drop it (§11.4)",
                timepoint.date,
                actionable=True,
            )

    for image_id, params in document.acquisition.items():
        _scan_mapping(params, f"acquisition.{image_id}", report, dates_shifted=shifted)
    for namespace, payload in document.extra.items():
        _scan_mapping(payload, f"extra.{namespace}", report, dates_shifted=shifted)

    for agent in document.provenance.agents:
        pseudonymous = str(agent.name).startswith(PSEUDONYM_PREFIX)
        if getattr(agent, "type", None) == "person" and agent.name and not pseudonymous:
            report.add(
                "staff_name",
                f"provenance.agents[{agent.id}]",
                "names a person; identifying for them, though not for the "
                "subject --- review against your governance",
                agent.name,
            )
        for name in ("role", "qualification"):
            value = getattr(agent, name, None)
            if isinstance(value, str) and value:
                _scan_text(value, f"provenance.agents[{agent.id}].{name}", report)

    # An importer stores what it was told to store: `OperatorsName` lands in an
    # activity's `params` as naturally as in `acquisition`, and a tool string
    # carries a workstation name.  Scanning `/meta` and skipping three of its
    # string-bearing sections made the attestation a claim about where this
    # module happened to look.
    for activity in document.provenance.activities:
        where = f"provenance.activities[{activity.id}]"
        if activity.params:
            _scan_mapping(dict(activity.params), f"{where}.params", report)
        if activity.tool:
            _scan_text(str(activity.tool), f"{where}.tool", report)

    for key, record in document.quality.items():
        for index, issue in enumerate(getattr(record, "issues", ()) or ()):
            if issue.note:
                _scan_text(
                    str(issue.note), f"quality.{key}.issues[{index}].note", report
                )
    return report


def _scan_mapping(
    payload: Any,
    where: str,
    report: ScrubReport,
    depth: int = 0,
    *,
    dates_shifted: bool = False,
) -> None:
    """Walk arbitrary JSON, applying the key and value rules as it goes."""
    if depth > MAX_DEPTH:
        # Reported, not skipped.  A silent stop would leave the file carrying
        # an attestation over a structure nothing ever looked at.
        report.add(
            "too_deep",
            where,
            f"nested more than {MAX_DEPTH} levels; this tool did not inspect "
            "it, and a person must",
        )
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            path = f"{where}.{key}"
            normal = _normalise(str(key))
            if normal in IDENTIFYING_KEYS:
                report.add(
                    "identifier",
                    path,
                    "an identifying DICOM attribute; writers MUST NOT copy tags "
                    "wholesale (§11.4)",
                    value,
                    actionable=True,
                )
                continue
            if normal in UID_KEYS and not DICOM_UID.match(str(value)):
                report.add(
                    "uid",
                    path,
                    "a UID attribute whose value is not a UID; it cannot be "
                    "pseudonymised safely, so a person must decide",
                    value,
                )
                continue
            if normal in QUASI_IDENTIFYING_KEYS:
                report.add(
                    "quasi_identifier",
                    path,
                    "identifying in combination with others, and sometimes "
                    "needed (PatientWeight drives a PET SUV); removed under "
                    "--profile strict, your decision under basic",
                    value,
                )
                continue
            if normal in DATE_KEYS:
                report.add(
                    "date",
                    path,
                    "a date attribute copied from the source",
                    value,
                    # A file that already records a shift has had its dates
                    # handled; flagging them again would make `scrub` as a
                    # pipeline gate fail forever on its own output.
                    actionable=not dates_shifted,
                )
                continue
            _scan_mapping(value, path, report, depth + 1, dates_shifted=dates_shifted)
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _scan_mapping(
                value,
                f"{where}[{index}]",
                report,
                depth + 1,
                dates_shifted=dates_shifted,
            )
        return
    if isinstance(payload, str):
        _scan_text(payload, where, report)


def _scan_text(value: str, where: str, report: ScrubReport) -> None:
    if PERSON_NAME.match(value):
        report.add(
            "person_name", where, "reads as a DICOM person name", value, actionable=True
        )
    elif DICOM_UID.match(value):
        report.add(
            "uid",
            where,
            "a real DICOM UID; SHOULD be a pseudonym (§11.4)",
            value,
            actionable=True,
        )
    elif ISO_DATE.search(value) or DICOM_DATE.match(value):
        report.add("date", where, "contains what looks like a date", value)
    elif len(value) > FREE_TEXT:
        report.add(
            "free_text",
            where,
            f"{len(value)} characters of free text --- no rule can judge this; "
            "a person must",
        )


UNFIXABLE_LOCATIONS = ("identity.sample_id", "identity.subject_id")
"""Findings ``--apply`` must never act on, however hard the profile looks.

The two ids are how every other file, manifest and split claim refers to this
sample (§12.1).  Blanking one because it reads as a person name would leave a
file nothing can join to --- so these are reported for a human to re-mint, and
they stay **non-actionable**, because ``actionable`` means "``--apply`` will
fix this" and the re-scan check below holds it to that.
"""

STRICT_RULES = ("free_text", "staff_name", "person_name", "quasi_identifier")
"""Rules that only ``--profile strict`` acts on."""


def _scan_frames(frames: Mapping[str, Sequence[str]], report: ScrubReport) -> None:
    """Frame-of-reference UIDs, wherever they are named (§3.4).

    Grids are not the only place one appears: a world-space annotation names
    one and a transform names two.  Reporting only the grids made a partial
    scrub look complete.
    """
    for uid, where in frames.items():
        if not DICOM_UID.match(uid):
            continue
        for location in where:
            report.add(
                "uid",
                location,
                "a real FrameOfReferenceUID; SHOULD be pseudonymised (§3.4)",
                uid,
                actionable=True,
            )


def _escalate(report: ScrubReport) -> ScrubReport:
    """Mark the strict-profile rules actionable, except where nothing can act."""
    if report.profile != "strict":
        return report
    for finding in report.findings:
        if finding.rule in STRICT_RULES and finding.where not in UNFIXABLE_LOCATIONS:
            finding.actionable = True
    return report


def scan(path: str | os.PathLike[str], *, profile: str = "basic") -> ScrubReport:
    """Find identifiers in one file.  Changes nothing."""
    import medh5

    if profile not in PROFILES:
        raise MEDH5ValidationError(
            f"unknown profile {profile!r}; expected one of {list(PROFILES)}"
        )
    report = ScrubReport(path=os.fspath(path), profile=profile)
    with medh5.open(path) as sample:
        scan_document(sample.document, report)
        _scan_frames(frame_references(sample.root), report)
    return _escalate(report)


def apply(
    path: str | os.PathLike[str],
    *,
    profile: str = "basic",
    salt: str = "",
    date_shift_days: int | None = None,
    performed_by: str | None = None,
) -> ScrubReport:
    """Act on the actionable findings and write the §11.4 attestation.

    ``date_shift_days`` shifts every date the rules found rather than deleting
    it, which preserves the intervals a longitudinal study is about.  Without
    it, dates are removed --- an interval that cannot be trusted is worse than
    no interval.
    """
    import medh5

    report = scan(path, profile=profile)
    with medh5.amend(path) as writer:
        document = writer.document
        removed: list[str] = []
        already = document.deidentification
        shifted_before = already is not None and already.date_shift_days is not None
        if already is not None and shifted_before:
            # Shifting twice makes the recorded offset a lie, and a scrub that
            # is not idempotent cannot be run from a pipeline.
            date_shift_days = already.date_shift_days
            report.actions.append(
                f"dates were left alone: already shifted by {date_shift_days} days"
            )

        strict = profile == "strict"

        def clean(payload: Any, where: str) -> tuple[Any, list[str]]:
            """One set of options for every section, so none is scrubbed harder."""
            return _clean(
                payload,
                where,
                report,
                salt=salt,
                date_shift_days=None if shifted_before else date_shift_days,
                keep_dates=shifted_before,
                strict=strict,
            )

        for namespace in sorted(document.extra):
            cleaned, actions = clean(document.extra[namespace], f"extra.{namespace}")
            document.extra[namespace] = cleaned
            removed.extend(actions)
        for image_id in sorted(document.acquisition):
            cleaned, actions = clean(
                document.acquisition[image_id], f"acquisition.{image_id}"
            )
            document.acquisition[image_id] = cleaned
            removed.extend(actions)

        # `identity.extra` and `cohort` are scanned, so they are cleaned.  A
        # converter puts raw tags wherever the document allows a string, and
        # `scan` reports every one of them as actionable; acting on two
        # sections and attesting over all of them is the overclaim §11.4
        # exists to prevent.  `sample_id` and `subject_id` are left alone
        # deliberately --- see `UNFIXABLE_LOCATIONS`.
        cleaned_extra, actions = clean(dict(document.identity.extra), "identity.extra")
        document.identity = replace(document.identity, extra=cleaned_extra)
        removed.extend(actions)
        cleaned_cohort, actions = clean(document.cohort.to_json(), "cohort")
        document.cohort = Cohort.from_json(cleaned_cohort)
        removed.extend(actions)

        # The performer is declared *before* the provenance sweep, so the name
        # this run adds gets the same treatment as the names it found: under
        # `strict` a person becomes a stable pseudonym.  Adding it afterwards
        # left `--by RAD-07` as an un-pseudonymised staff name that the tool's
        # own re-scan then reported, so a strict apply could never exit 0.
        agent = (
            writer.person(performed_by)
            if performed_by
            else writer.software("medh5", medh5.__version__)
        )
        removed.extend(_clean_provenance(document, report, clean, strict, salt))
        removed.extend(_clean_quality(document, report, clean, strict))

        # One mapping, applied to every reference at once.  Pseudonymising the
        # grids alone would leave the real UID on the transforms and annotations
        # and break the frame graph joining them --- see `remap_frame_uids`.
        frame_map = {
            uid: pseudonymise(uid, salt)
            for uid in writer.frame_uids()
            if DICOM_UID.match(uid)
        }
        report.uid_map.update(frame_map)
        for location in writer.remap_frame_uids(frame_map):
            removed.append(f"{location} -> pseudonymised")

        # A Timepoint is frozen, so the timeline is rebuilt rather than edited:
        # the invariants §3.7 puts in `Timeline.check` are re-run on the result.
        rebuilt: list[Timepoint] = []
        for timepoint in document.timepoints:
            changes: dict[str, Any] = {}
            uid = timepoint.study_uid
            if uid and DICOM_UID.match(str(uid)):
                replacement = pseudonymise(str(uid), salt)
                report.uid_map[str(uid)] = replacement
                changes["study_uid"] = replacement
                removed.append(f"timepoint {timepoint.id}.study_uid -> {replacement}")
            series = dict(timepoint.series_uids or {})
            if any(DICOM_UID.match(str(v)) for v in series.values()):
                mapped: dict[str, str] = {}
                for key, value in series.items():
                    replacement = (
                        pseudonymise(str(value), salt)
                        if DICOM_UID.match(str(value))
                        else str(value)
                    )
                    report.uid_map[str(value)] = replacement
                    mapped[key] = replacement
                changes["series_uids"] = mapped
                removed.append(f"timepoint {timepoint.id}.series_uids pseudonymised")
            if timepoint.date and not shifted_before:
                moved = _shift(str(timepoint.date), date_shift_days)
                changes["date"] = moved
                removed.append(
                    f"timepoint {timepoint.id}.date "
                    + ("shifted" if moved else "removed")
                )
            rebuilt.append(replace(timepoint, **changes) if changes else timepoint)
        document.timepoints = Timeline(rebuilt)

        writer.deidentification(
            method="medh5-scrub",
            profile=(
                f"medh5 scrub {profile}: container metadata only; "
                + (
                    "quasi-identifiers removed"
                    if profile == "strict"
                    else "quasi-identifiers retained for review"
                )
            ),
            date_shift_days=date_shift_days,
            # `external` only when a salt exists to be held externally.  An
            # unsalted hash is derivable by anyone holding the original UIDs, so
            # claiming a protected mapping would be the overclaim this module
            # is written to avoid.
            id_mapping="external" if salt and report.uid_map else "none",
            performed_by=agent.id,
            # Not a claim that no burned-in text exists: a claim that this tool
            # did not look, which is what §11.4's field is for.
            burned_in_annotation_checked=False,
        )
        # Re-scan the cleaned document, *after* the record is written, and put
        # the result in the activity.  The record has to exist first or the
        # re-scan reports the shifted dates it explains, and the count would
        # describe a document nobody will ever read.  What this run achieved is
        # then in the file itself, where a reader auditing the attestation
        # looks --- rather than only in a report that is not shipped with it.
        left = _rescan(writer, profile)
        writer.activity(
            "deidentify",
            agent=agent,
            tool=f"medh5 scrub --profile {profile}",
            params={
                "findings": len(report.findings),
                "changes": len(removed),
                "remaining": len(left.findings),
                "remaining_actionable": len(left.actionable),
            },
        )
        report.actions = removed
        report.applied = True
    # The amend copied every object and *then* rewrote the pseudonymised
    # attributes, and HDF5 does not reclaim what it supersedes --- so the file
    # it just produced still contains the original `frame_uid` in freed space,
    # recoverable with `strings` while every API read returns the pseudonym.
    # For a de-identification tool that is not a wasted-bytes problem, so the
    # output is compacted before it is handed back.  Digests and `content_id`
    # are unaffected: this rewrites storage, not content (§13.1).
    repack(path)
    # The claim is checked against the file, not against the intention: the
    # rules are re-run over what was written, and `ok` --- and therefore the
    # CLI's exit code --- follows the result.  The attestation above counts the
    # deidentify activity this run added, which the re-scan below also sees, so
    # the two differ by nothing that matters and neither is guesswork.
    report.remaining = scan(path, profile=profile).findings
    return report


def _rescan(writer: Any, profile: str) -> ScrubReport:
    """Run every rule over a writer's state, without going back to disk.

    The same rules as :func:`scan`, sourced from the document and frame
    references the amend is about to commit --- so the count recorded in the
    attestation describes the file that attestation ships in.
    """
    report = ScrubReport(path="<amend>", profile=profile)
    scan_document(writer.document, report)
    _scan_frames(writer.frame_uids(), report)
    return _escalate(report)


def _clean_provenance(
    document: Any,
    report: ScrubReport,
    clean: Any,
    strict: bool,
    salt: str,
) -> list[str]:
    """Clean the provenance graph: activity params and tools, and, under
    ``strict``, the names of the people in it.

    A converter stores what it was given --- ``OperatorsName`` reaches an
    activity's ``params`` exactly as it reaches ``acquisition`` --- and an
    agent of type ``person`` names a member of staff.  §11.4 says a name here
    **MUST NOT** be a direct identifier, so under ``strict`` it becomes a
    stable pseudonym rather than being deleted: the graph still says two
    activities were done by the same person, which is what the graph is for.
    """
    from medh5.curation.provenance import Agent

    actions: list[str] = []
    provenance = document.provenance
    for activity in list(provenance.activities):
        where = f"provenance.activities[{activity.id}]"
        changes: dict[str, Any] = {}
        if activity.params:
            cleaned, sub = clean(dict(activity.params), f"{where}.params")
            if sub:
                changes["params"] = cleaned
                actions.extend(sub)
        if activity.tool and PERSON_NAME.match(str(activity.tool)):
            changes["tool"] = ""
            actions.append(f"{where}.tool removed (person name)")
        if changes:
            provenance.add_activity(replace(activity, **changes), replace=True)

    if not strict:
        return actions
    for agent in list(provenance.agents):
        changes = {}
        if getattr(agent, "type", None) == "person" and agent.name:
            changes["name"] = pseudonymise(agent.name, salt)
            actions.append(f"provenance.agents[{agent.id}].name pseudonymised")
        for field_name in ("role", "qualification"):
            value = getattr(agent, field_name, None)
            if isinstance(value, str) and len(value) > FREE_TEXT:
                changes[field_name] = None
                actions.append(
                    f"provenance.agents[{agent.id}].{field_name} removed (free text)"
                )
        if changes:
            assert isinstance(agent, Agent)
            provenance.add_agent(replace(agent, **changes), replace=True)
    return actions


def _clean_quality(
    document: Any, report: ScrubReport, clean: Any, strict: bool
) -> list[str]:
    """Clean quality issue notes --- free text a rater typed, and scanned."""
    actions: list[str] = []
    for key, record in list(document.quality.items()):
        issues = list(getattr(record, "issues", ()) or ())
        changed = False
        for index, issue in enumerate(issues):
            note = issue.note
            if not note:
                continue
            where = f"quality.{key}.issues[{index}].note"
            if PERSON_NAME.match(str(note)):
                issues[index] = replace(issue, note=None)
                actions.append(f"{where} removed (person name)")
                changed = True
            elif strict and len(str(note)) > FREE_TEXT:
                issues[index] = replace(issue, note=None)
                actions.append(f"{where} removed (free text, strict)")
                changed = True
        if changed:
            document.quality[key] = replace(record, issues=tuple(issues))
    return actions


def _clean(
    payload: Any,
    where: str,
    report: ScrubReport,
    *,
    salt: str,
    date_shift_days: int | None,
    keep_dates: bool = False,
    strict: bool = False,
    depth: int = 0,
) -> tuple[Any, list[str]]:
    """Rebuild a JSON payload without its identifiers, listing what changed."""
    actions: list[str] = []
    if depth > MAX_DEPTH:
        # Left intact and *reported*, to match the scan: this tool did not
        # look here, so it must not imply that it did.
        actions.append(f"{where} left untouched: nested deeper than {MAX_DEPTH}")
        return payload, actions
    if isinstance(payload, Mapping):
        out: dict[str, Any] = {}
        for key, value in payload.items():
            path = f"{where}.{key}"
            normal = _normalise(str(key))
            if normal in IDENTIFYING_KEYS:
                actions.append(f"{path} removed")
                continue
            if normal in QUASI_IDENTIFYING_KEYS:
                if strict:
                    actions.append(f"{path} removed (quasi-identifier, strict)")
                    continue
                out[key] = value
                continue
            if normal in UID_KEYS and not DICOM_UID.match(str(value)):
                out[key] = value
                continue
            if normal in DATE_KEYS:
                if keep_dates:
                    out[key] = value
                    continue
                shifted = _shift(str(value), date_shift_days)
                if shifted is None:
                    actions.append(f"{path} removed")
                    continue
                out[key] = shifted
                actions.append(f"{path} shifted")
                continue
            cleaned, sub = _clean(
                value,
                path,
                report,
                salt=salt,
                date_shift_days=date_shift_days,
                keep_dates=keep_dates,
                strict=strict,
                depth=depth + 1,
            )
            out[key] = cleaned
            actions.extend(sub)
        return out, actions
    if isinstance(payload, list):
        cleaned_list = []
        for index, value in enumerate(payload):
            cleaned, sub = _clean(
                value,
                f"{where}[{index}]",
                report,
                salt=salt,
                date_shift_days=date_shift_days,
                keep_dates=keep_dates,
                strict=strict,
                depth=depth + 1,
            )
            cleaned_list.append(cleaned)
            actions.extend(sub)
        return cleaned_list, actions
    if isinstance(payload, str):
        if PERSON_NAME.match(payload):
            actions.append(f"{where} removed (person name)")
            return "", actions
        if DICOM_UID.match(payload):
            replacement = pseudonymise(payload, salt)
            report.uid_map[payload] = replacement
            actions.append(f"{where} pseudonymised")
            return replacement, actions
        if strict and len(payload) > FREE_TEXT:
            # `scan` marks free text actionable under `strict` because no rule
            # can judge it; the only action available is removal, and leaving
            # it while calling the finding actionable is the mismatch between
            # claim and file that the re-scan check exists to catch.
            actions.append(f"{where} removed (free text, strict)")
            return "", actions
    return payload, actions


def _shift(value: str, days: int | None) -> str | None:
    """Shift a date by *days*, or return ``None`` meaning "drop it"."""
    if days is None:
        return None
    from datetime import date, timedelta

    text = value.strip()
    try:
        if DICOM_DATE.match(text):
            parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
            return (parsed + timedelta(days=days)).strftime("%Y%m%d")
        match = ISO_DATE.search(text)
        if match:
            parsed = date.fromisoformat(match.group())
            return text.replace(
                match.group(), (parsed + timedelta(days=days)).isoformat()
            )
    except ValueError:
        return None
    return None


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
