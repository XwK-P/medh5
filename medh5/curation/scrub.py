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

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from medh5._hdf5 import encode_attr, repack, str_dtype
from medh5.curation.identity import ID_SOURCE, PSEUDONYM_SOURCE
from medh5.errors import MEDH5ValidationError
from medh5.sample import FRAME_ATTRS, frame_references

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
``patient_name`` and ``Patient Name`` are one key --- in ``/meta`` mappings and
as the names of HDF5 attributes alike.
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

PERSON_NAME = re.compile(r"^[^\W\d_][^\^=\d]*\^[^\W\d_]")
"""DICOM PN form ``Family^Given``, in any script: unambiguous enough to act on.

The letters are Unicode letters.  The ASCII-only pattern this replaced let
``Müller^Hans`` and ``山田^太郎`` through as ordinary text.
"""

UID_TOKEN = re.compile(r"(?<![\d.])\d+(?:\.\d+){3,}(?![\d.])")
"""A dotted-numeric UID *inside* a longer string, e.g. ``dicom:<SeriesInstanceUID>``."""

PATH_REMOVED = "<path removed>"
"""What ``--profile strict`` leaves where a filesystem path was."""

AGE_LIMIT = 90.0
"""HIPAA Safe Harbor aggregates every age over 89 into one category, "90 or
older".  ``strict`` records such an age as 90 --- the category --- so 90 itself is
the aggregated value, and only a larger one identifies."""

INTERNAL_REFERENCES = ("annotations/", "images/", "grids/", "transforms/", "index/")
"""Provenance references to objects in this file, which are not paths."""

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]+:)")
"""A reference's ``scheme:`` prefix; two characters at least, so ``C:`` is a drive."""

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
DICOM_DATE = re.compile(r"^\d{8}$")
DICOM_UID = re.compile(r"^\d+(\.\d+){3,}$")
"""A dotted-numeric OID.  ``pseudo:...`` and UUIDs do not match, by design."""

FREE_TEXT = 200
"""Characters above which a string cannot be reviewed by a rule, only by a person."""

UNFIXABLE_LOCATIONS = ("identity.sample_id", "identity.subject_id")
"""Findings ``--apply`` never acts on by itself, however hard the profile looks.

The two ids are how every other file, manifest and split claim refers to this
sample (§12.1).  Blanking one because it reads as a person name would leave a
file nothing can join to --- so these are reported for a human to re-mint, or
replaced by stable pseudonyms when the caller asks for exactly that
(``pseudonymise_ids``).  They stay **non-actionable**, because ``actionable``
means "``--apply`` will fix this" and the re-scan check holds it to that.
"""

STRICT_RULES = (
    "free_text",
    "staff_name",
    "person_name",
    "quasi_identifier",
    "age",
    "organization_name",
)
"""Rules that only ``--profile strict`` acts on."""

IDENTITY_RULES = ("identity", "file_name")
"""Findings about what the sample is *called*.

No ``--apply`` rewrites them unless asked to (``pseudonymise_ids``), and under
``--profile strict`` an open one --- or any finding on
:data:`UNFIXABLE_LOCATIONS` --- fails the run: a file cannot be attested
de-identified while its own name for the subject is the subject's record
number.
"""

_NAME = "an identifier other records refer to, so --apply leaves it for a person"
"""Why a finding on an id is reported and never rewritten."""

_VOCABULARY = (
    "part of a label vocabulary shared across the cohort and pinned by its "
    "digest, so --apply leaves it for a person"
)

_UID_DETAIL = "a real DICOM UID; SHOULD be a pseudonym (§11.4)"

_DROP = object()
"""What a rule returns, when cleaning, for a mapping entry that must go."""

_ROOT_MANAGED_ATTRS = frozenset(
    {
        "medh5_version",
        "medh5_kind",
        "medh5_profiles",
        "created",
        "generator",
        "digest_algo",
        "content_id",
    }
)
"""Root attributes ``commit`` rewrites from the amended state, so nothing a
source file put there survives --- and ``created`` is the writer's own clock,
which is a date every file has and none of it is about the subject."""

_REFERENCE_ATTRS = frozenset(
    {
        "grid",
        "grid_levels",
        "timepoint",
        "timepoints",
        "prov",
        "quality",
        "metrics",
        "derived_from",
        "ignore_mask",
        "valid_mask",
        "skeleton",
        "correspondence",
        "from_grid",
        "to_grid",
        "field_grid",
        "cp_grid",
        "inverse_id",
        "components",
    }
)
"""Attributes whose value is the id of another object, timepoint, activity or
record.  They are reviewed like the ids they name and never rewritten: renaming
one end of a reference breaks it, and the file then fails E101/E409/E601."""


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
    reported and left to a person at every profile; see :data:`_NAME`."""

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


def _names_the_sample(finding: Finding) -> bool:
    return finding.rule in IDENTITY_RULES or finding.where in UNFIXABLE_LOCATIONS


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
        fixable: bool = True,
    ) -> Finding:
        finding = Finding(
            rule=rule,
            where=where,
            detail=detail,
            value=None if value is None else _preview(value),
            actionable=actionable and fixable,
            fixable=fixable,
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
        return [f for f in left if _names_the_sample(f)]

    @property
    def ok(self) -> bool:
        """Whether this run leaves nothing a further ``--apply`` could fix.

        For a scan, that is "found nothing at all"; for an apply, "the file it
        wrote has no actionable findings left" --- and, under ``strict``, "and
        its ids are not identifiers".  Both are what a pipeline gate needs, and
        both are checked rather than assumed.
        """
        if not self.applied:
            return self.clean
        if self.remaining_actionable:
            return False
        return not (self.profile == "strict" and self.open_identity)

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
            if self.profile == "strict":
                lines.extend(
                    f"  IDENTITY {f.where}: re-mint it, or re-run --apply with "
                    "--pseudonymise-ids --salt ..."
                    for f in self.open_identity
                )
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


def _split_reference(value: str) -> tuple[str, str]:
    """``("scheme:", rest)`` for a provenance reference, or ``("", value)``."""
    match = _SCHEME.match(value)
    if match is None:
        return "", value
    return match.group(1), value[match.end() :]


def _is_path(rest: str) -> bool:
    return ("/" in rest or "\\" in rest) and not rest.startswith(INTERNAL_REFERENCES)


def _basename(rest: str) -> str:
    return re.split(r"[\\/]", rest.rstrip("\\/"))[-1]


def _embedded_json(value: str) -> Any:
    """A JSON object or array stored as a string, decoded; ``None`` otherwise.

    Converters that cannot write a mapping write its JSON instead --- into an
    HDF5 attribute, or a box's ``attributes`` column.  Scanned as one long
    string, ``{"PatientName": "Doe^Jane"}`` matches no rule; decoded, it is the
    same mapping every key rule already handles.
    """
    text = value.strip()
    if len(text) < 2 or text[0] not in "{[":
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _decode(value: Any) -> Any:
    """An HDF5 attribute or element as Python: strings as :class:`str`."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode(value[()])
        if value.dtype.kind in "SUO":
            return [_decode(v) for v in value.ravel()]
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _string_dtype(dtype: np.dtype[Any]) -> bool:
    return h5py.check_string_dtype(dtype) is not None


class _Sweep:
    """One walk over every string in a sample, for scanning and for cleaning.

    Every rule reports a finding and, when ``writing``, returns what the slot
    should hold instead.  The scan and ``--apply`` used to be two walks over two
    hand-kept lists of locations, and they drifted: the scan called things
    actionable that the clean never visited, and neither visited what a
    converter wrote outside the list.  One walk makes "actionable" mean what it
    says.

    ``review`` marks a subtree whose findings are reported and never acted on
    --- an id something else refers to, a shared vocabulary --- and says why.
    """

    __slots__ = (
        "actions",
        "date_shift_days",
        "dates_shifted",
        "report",
        "salt",
        "strict",
        "uid_map",
        "writing",
    )

    def __init__(
        self,
        report: ScrubReport,
        *,
        writing: bool = False,
        strict: bool = False,
        salt: str = "",
        date_shift_days: int | None = None,
        uid_map: dict[str, str] | None = None,
    ) -> None:
        self.report = report
        self.writing = writing
        self.strict = strict
        self.salt = salt
        self.date_shift_days = date_shift_days
        self.dates_shifted = False
        self.uid_map = report.uid_map if uid_map is None else uid_map
        self.actions: list[str] = []

    # -- the primitives ----------------------------------------------------

    def find(
        self,
        rule: str,
        where: str,
        detail: str,
        value: Any = None,
        *,
        actionable: bool = False,
        review: str | None = None,
    ) -> None:
        if review is not None:
            self.report.add(rule, where, f"{detail} --- {review}", value, fixable=False)
        else:
            self.report.add(rule, where, detail, value, actionable=actionable)

    def acts(self, review: str | None) -> bool:
        return self.writing and review is None

    def replace(self, old: Any, new: Any, review: str | None, action: str) -> Any:
        if not self.acts(review):
            return old
        self.actions.append(action)
        return new

    def swap(self, old: str, new: str, review: str | None, action: str) -> str:
        """:meth:`replace`, for a string slot."""
        return new if self.replace(old, new, review, action) is new else old

    def pseudonym(self, uid: str) -> str:
        replacement = pseudonymise(uid, self.salt)
        if self.writing:
            self.uid_map[uid] = replacement
        return replacement

    # -- arbitrary JSON ----------------------------------------------------

    def json(
        self, payload: Any, where: str, *, depth: int = 0, review: str | None = None
    ) -> Any:
        """Walk arbitrary JSON, applying the key and value rules as it goes."""
        if depth > MAX_DEPTH:
            # Reported, not skipped.  A silent stop would leave the file carrying
            # an attestation over a structure nothing ever looked at.
            self.find(
                "too_deep",
                where,
                f"nested more than {MAX_DEPTH} levels; this tool did not inspect "
                "it, and a person must",
            )
            if self.writing:
                self.actions.append(
                    f"{where} left untouched: nested deeper than {MAX_DEPTH}"
                )
            return payload
        if isinstance(payload, Mapping):
            out: dict[Any, Any] = {}
            for key, value in payload.items():
                path = f"{where}.{key}"
                kept = self.entry(str(key), value, path, depth=depth, review=review)
                if kept is _DROP:
                    continue
                name = self.key(key, path, review)
                if name is _DROP:
                    continue
                out[name] = kept
            return out
        if isinstance(payload, (list, tuple)):
            return [
                self.json(value, f"{where}[{index}]", depth=depth + 1, review=review)
                for index, value in enumerate(payload)
            ]
        if isinstance(payload, str):
            return self.text(payload, where, depth=depth, review=review)
        return payload

    def entry(
        self, key: str, value: Any, path: str, *, depth: int, review: str | None
    ) -> Any:
        """One mapping entry: the rules keyed on its name, then its value."""
        normal = _normalise(key)
        if normal in IDENTIFYING_KEYS:
            self.find(
                "identifier",
                path,
                "an identifying DICOM attribute; writers MUST NOT copy tags "
                "wholesale (§11.4)",
                value,
                actionable=True,
                review=review,
            )
            return self.replace(value, _DROP, review, f"{path} removed")
        if normal in UID_KEYS and not DICOM_UID.match(str(value)):
            self.find(
                "uid",
                path,
                "a UID attribute whose value is not a UID; it cannot be "
                "pseudonymised safely, so a person must decide",
                value,
                review=review,
            )
            return value
        if normal in QUASI_IDENTIFYING_KEYS:
            self.find(
                "quasi_identifier",
                path,
                "identifying in combination with others, and sometimes needed "
                "(PatientWeight drives a PET SUV); removed under --profile "
                "strict, your decision under basic",
                value,
                review=review,
            )
            if self.strict:
                return self.replace(
                    value, _DROP, review, f"{path} removed (quasi-identifier, strict)"
                )
            return value
        if normal in DATE_KEYS:
            self.find(
                "date",
                path,
                "a date attribute copied from the source",
                value,
                # A file that already records a shift has had its dates
                # handled; flagging them again would make `scrub` as a
                # pipeline gate fail forever on its own output.
                actionable=not self.dates_shifted,
                review=review,
            )
            if self.dates_shifted or not self.acts(review):
                return value
            moved = _shift(str(value), self.date_shift_days)
            if moved is None:
                self.actions.append(f"{path} removed")
                return _DROP
            self.actions.append(f"{path} shifted")
            return moved
        return self.json(value, path, depth=depth + 1, review=review)

    def key(self, key: Any, path: str, review: str | None) -> Any:
        """A mapping *key* that is itself a name or a UID --- ``{"Doe^Jane": ...}``."""
        if not isinstance(key, str) or key.startswith(PSEUDONYM_PREFIX):
            return key
        if PERSON_NAME.match(key):
            self.find(
                "person_name",
                path,
                "a key that reads as a DICOM person name",
                key,
                actionable=True,
                review=review,
            )
            return self.replace(key, _DROP, review, f"{path} removed (person name)")
        if DICOM_UID.match(key):
            self.find(
                "uid",
                path,
                f"a key that is {_UID_DETAIL}",
                key,
                actionable=True,
                review=review,
            )
            return self.replace(
                key, self.pseudonym(key), review, f"{path} key pseudonymised"
            )
        if ISO_DATE.search(key) or DICOM_DATE.match(key):
            self.find("date", path, "a key that looks like a date", key, review=review)
        return key

    def text(
        self,
        value: str,
        where: str,
        *,
        depth: int = 0,
        review: str | None = None,
        removed: str = "",
    ) -> str:
        """The value rules, for one string.

        *removed* is what a removal leaves: empty, except where the format
        requires a value --- an agent's name --- and a stable pseudonym keeps
        the slot valid and the graph joined.
        """
        parsed = _embedded_json(value)
        if parsed is not None:
            cleaned = self.json(parsed, where, depth=depth + 1, review=review)
            if self.acts(review) and cleaned != parsed:
                return json.dumps(cleaned, sort_keys=True)
            return value
        if value.startswith(PSEUDONYM_PREFIX):
            return value
        if PERSON_NAME.match(value):
            self.find(
                "person_name",
                where,
                "reads as a DICOM person name",
                value,
                actionable=True,
                review=review,
            )
            return self.swap(value, removed, review, f"{where} removed (person name)")
        if DICOM_UID.match(value):
            self.find("uid", where, _UID_DETAIL, value, actionable=True, review=review)
            return self.swap(
                value, self.pseudonym(value), review, f"{where} pseudonymised"
            )
        if ISO_DATE.search(value) or DICOM_DATE.match(value):
            self.find(
                "date", where, "contains what looks like a date", value, review=review
            )
            return value
        if len(value) > FREE_TEXT:
            self.find(
                "free_text",
                where,
                f"{len(value)} characters of free text --- no rule can judge "
                "this; a person must",
                review=review,
            )
            if self.strict:
                # `scan` marks free text actionable under `strict` because no
                # rule can judge it; the only action available is removal.
                return self.swap(
                    value, removed, review, f"{where} removed (free text, strict)"
                )
        return value

    def name(self, value: Any, where: str) -> None:
        """An id something else refers to: every rule, and never a rewrite."""
        if isinstance(value, str):
            self.text(value, where, review=_NAME)

    def reference(self, value: str, where: str) -> str:
        """One provenance ``inputs``/``outputs`` entry: UIDs, paths, then text.

        Left alone, ``dicom:<SeriesInstanceUID>`` put the real UID beside the
        pseudonym that replaced it everywhere else, mapping one straight back
        to the other --- and a source path names the export directory, which
        routinely names the patient or the site.
        """
        scheme, rest = _split_reference(value)
        cleaned = value
        if UID_TOKEN.search(value):
            self.find(
                "uid",
                where,
                "a real DICOM UID inside a provenance reference; the same UID is "
                "pseudonymised where it is a field, and left here it maps the "
                "pseudonym straight back (§11.4)",
                value,
                actionable=True,
            )
            if self.writing:
                cleaned = UID_TOKEN.sub(lambda m: self.pseudonym(m.group()), cleaned)
                self.actions.append(f"{where}: UID pseudonymised")
        if _is_path(rest):
            self.find(
                "path",
                where,
                "a filesystem path; export directories routinely name the "
                "patient or the site. --apply keeps only the file name, and "
                "--profile strict removes the path",
                value,
                actionable=True,
            )
            scheme, rest = _split_reference(cleaned)
            # A name in the file name is still a name.
            base = self.text(_basename(rest), where)
            if self.writing:
                cleaned = scheme + (PATH_REMOVED if self.strict else base)
                self.actions.append(
                    f"{where}: path " + ("removed" if self.strict else "reduced")
                )
        elif not UID_TOKEN.search(value) and rest:
            cleaned = scheme + self.text(rest, where)
        return cleaned

    # -- the sample document -----------------------------------------------

    def document(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """Every string in ``/meta``; the cleaned document when writing.

        Sections have rules of their own where the format gives a string a
        meaning --- a timepoint's UIDs and date, an agent's name --- and fall
        through to the generic rules everywhere else, so a field is examined
        because it is there and not because it was remembered.  The fields
        skipped are listed where they are skipped, with the reason.
        """
        record = doc.get("deidentification") or {}
        self.dates_shifted = record.get("date_shift_days") is not None
        out = dict(doc)
        out["identity"] = self.identity(doc["identity"], doc)
        if "cohort" in doc:
            out["cohort"] = self.json(doc["cohort"], "cohort")
        out["timepoints"] = [
            self.timepoint(tp, index) for index, tp in enumerate(doc["timepoints"])
        ]
        for section in ("acquisition", "extra"):
            if section in doc:
                entries = {}
                for key, value in doc[section].items():
                    self.name(key, f"{section}.{key}")
                    entries[key] = self.json(value, f"{section}.{key}")
                out[section] = entries
        if "provenance" in doc:
            out["provenance"] = self.provenance(doc["provenance"])
        if "splits" in doc:
            out["splits"] = [
                self.split(claim, index) for index, claim in enumerate(doc["splits"])
            ]
        if "quality" in doc:
            records = {}
            for key, record_json in doc["quality"].items():
                self.name(key, f"quality.{key}")
                records[key] = self.json(record_json, f"quality.{key}")
            out["quality"] = records
        if "label_set" in doc:
            self.json(doc["label_set"], "label_set", review=_VOCABULARY)
        # `deidentification` is not scanned: it is the attestation, and
        # `--apply` replaces it whole.
        return out

    def identity(
        self, identity: Mapping[str, Any], doc: Mapping[str, Any]
    ) -> dict[str, Any]:
        out = dict(identity)
        for name in ("sample_id", "subject_id"):
            self.identity_id(identity, name, doc)
        for key, value in identity.items():
            # `sex` and `laterality` are closed vocabularies the typed layer
            # checks; `id_source` is this module's own record.
            if key in ("sample_id", "subject_id", "sex", "laterality", ID_SOURCE):
                continue
            where = (
                "identity.bodypart" if key == "bodypart" else f"identity.extra.{key}"
            )
            kept = self.entry(str(key), value, where, depth=0, review=None)
            renamed = self.key(key, where, None) if kept is not _DROP else _DROP
            if kept is _DROP or renamed is _DROP:
                out.pop(key, None)
            elif renamed != key:
                del out[key]
                out[str(renamed)] = kept
            else:
                out[key] = kept
        return out

    def identity_id(
        self, identity: Mapping[str, Any], name: str, doc: Mapping[str, Any]
    ) -> None:
        """Whether one of the sample's own ids is an identifier (§11.4).

        Never actionable by default --- every manifest, split claim and join
        names the sample by these --- but never silent either: they are the one
        place a DICOM import is guaranteed to have put the record number.
        """
        value = str(identity[name])
        if value.startswith(PSEUDONYM_PREFIX):
            return
        where = f"identity.{name}"
        remint = "re-mint it, or re-run --apply with --pseudonymise-ids"
        if PERSON_NAME.match(value):
            self.find(
                "person_name",
                where,
                f"reads as a DICOM person name, not a pseudonym (§11.4); {remint}",
                value,
            )
        if ISO_DATE.search(value) or DICOM_DATE.match(value):
            self.find(
                "date", where, f"contains what looks like a date; {remint}", value
            )
        recorded = identity.get(ID_SOURCE)
        source = recorded if isinstance(recorded, Mapping) else {}
        origin = str(source.get(name, ""))
        if origin.startswith("dicom:"):
            self.find(
                "identity",
                where,
                f"copied from DICOM {origin.removeprefix('dicom:')}, a direct "
                f"identifier (§11.4); {remint}",
                value,
            )
        elif UID_TOKEN.search(value):
            self.find(
                "identity",
                where,
                f"contains a real DICOM UID; {remint}",
                value,
            )
        elif not source and _imported_from_dicom(doc):
            self.find(
                "identity",
                where,
                "this sample was imported from DICOM, which keys a sample by "
                "PatientID, and records no source for this id; check it is not "
                f"the record number and record where it came from in "
                f"identity.{ID_SOURCE}, or {remint}",
                value,
            )

    def timepoint(self, tp: Mapping[str, Any], index: int) -> dict[str, Any]:
        where = f"timepoints[{index}]"
        out = dict(tp)
        for key, value in tp.items():
            path = f"{where}.{key}"
            if key in ("index", "days_from_baseline"):
                continue  # numbers
            if key == "id":
                self.name(value, path)
            elif key == "study_uid":
                out[key] = self.uid(value, path)
            elif key == "series_uids":
                out[key] = {
                    image: self.uid(uid, f"{path}.{image}")
                    for image, uid in dict(value).items()
                }
            elif key == "date":
                if not value or self.dates_shifted:
                    continue
                self.find(
                    "date",
                    path,
                    "a date with no recorded shift; either shift the cohort "
                    "consistently or drop it (§11.4)",
                    value,
                    actionable=True,
                )
                if self.writing:
                    moved = _shift(str(value), self.date_shift_days)
                    if moved is None:
                        del out[key]
                    else:
                        out[key] = moved
                    self.actions.append(
                        f"{path} " + ("shifted" if moved else "removed")
                    )
            elif key == "subject_age_years":
                if value is None or float(value) <= AGE_LIMIT:
                    continue
                self.find(
                    "age",
                    path,
                    "an age over 89 identifies on its own under HIPAA Safe "
                    "Harbor, which aggregates them as 90 or older; --profile "
                    "strict records it as 90",
                    value,
                )
                if self.writing and self.strict:
                    out[key] = AGE_LIMIT
                    self.actions.append(f"{path} recorded as 90 (strict)")
            else:
                # `label`, `description`, and whatever a later minor adds.
                out[key] = self.json(value, path)
        return out

    def uid(self, value: Any, where: str) -> Any:
        """A field that holds a UID: pseudonymised when it is a real one."""
        text = str(value)
        if DICOM_UID.match(text):
            self.find("uid", where, _UID_DETAIL, text, actionable=True)
            return self.replace(
                value, self.pseudonym(text), None, f"{where} pseudonymised"
            )
        return self.json(value, where)

    def provenance(self, prov: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(prov)
        out["agents"] = [self.agent(agent) for agent in prov.get("agents", ())]
        out["activities"] = [
            self.activity(activity) for activity in prov.get("activities", ())
        ]
        return out

    def agent(self, agent: Mapping[str, Any]) -> dict[str, Any]:
        """An agent's name, by what the agent is; then everything else it holds.

        A person's name is identifying for them, though not for the subject, and
        an organization is often the site that imaged the subject.  §11.4 says a
        name here **MUST NOT** be a direct identifier, so under ``strict`` each
        becomes a stable pseudonym rather than being deleted: the graph still
        says two activities were done by the same person, which is what the
        graph is for.
        """
        where = f"provenance.agents[{agent.get('id')}]"
        out = dict(agent)
        kind = agent.get("type")
        name = str(agent.get("name") or "")
        pseudonymous = name.startswith(PSEUDONYM_PREFIX)
        if name and not pseudonymous and kind in ("person", "organization"):
            if kind == "person":
                self.find(
                    "staff_name",
                    where,
                    "names a person; identifying for them, though not for the "
                    "subject --- review against your governance",
                    name,
                )
            else:
                self.find(
                    "organization_name",
                    where,
                    "names an organization; if it is the site that imaged the "
                    "subject it identifies them in combination (InstitutionName "
                    "is removed by the DICOM basic profile) --- pseudonymised "
                    "under --profile strict",
                    name,
                )
            if self.writing and self.strict:
                out["name"] = pseudonymise(name, self.salt)
                self.actions.append(f"{where}.name pseudonymised")
        elif name:
            # Software, a model, a device: the format requires a name, so one
            # that reads as a person's becomes a pseudonym rather than blank.
            out["name"] = self.text(
                name, f"{where}.name", removed=pseudonymise(name, self.salt)
            )
        for key, value in agent.items():
            # The id and the organization it belongs to are ids (reviewed
            # where they are declared); `type` is a closed vocabulary.
            if key in ("id", "type", "name", "organization"):
                continue
            out[key] = self.json(value, f"{where}.{key}")
        self.name(agent.get("id"), f"{where}.id")
        return out

    def activity(self, activity: Mapping[str, Any]) -> dict[str, Any]:
        where = f"provenance.activities[{activity.get('id')}]"
        out = dict(activity)
        for key, value in activity.items():
            # `started` and `ended` say when a tool ran, not when the subject
            # was seen, and the provenance graph is for auditing that; `agent`
            # names an agent, reviewed where the agent is declared.
            if key in ("id", "type", "agent", "started", "ended"):
                continue
            path = f"{where}.{key}"
            if key in ("inputs", "outputs"):
                out[key] = [
                    self.reference(str(entry), f"{path}[{index}]")
                    for index, entry in enumerate(value)
                ]
            else:
                # `tool`, `params`, and whatever a later minor adds.
                out[key] = self.json(value, path)
        self.name(activity.get("id"), f"{where}.id")
        return out

    def split(self, claim: Mapping[str, Any], index: int) -> dict[str, Any]:
        where = f"splits[{index}]"
        out = dict(claim)
        for key, value in claim.items():
            # A closed vocabulary, a number, the tool's clock and a digest.
            if key in ("partition", "fold", "assigned_at", "manifest_sha256"):
                continue
            if key == "set_id":
                self.name(value, f"{where}.set_id")
            else:
                out[key] = self.json(value, f"{where}.{key}")
        return out

    # -- the HDF5 file -----------------------------------------------------

    def hdf5(self, root: h5py.Group) -> None:
        """Every object name, attribute and string dataset outside ``/meta``.

        ``/meta`` is the document, examined above.  Object names are ids,
        reviewed and never renamed.  Attributes are a mapping from name to
        value, so the key rules apply to them exactly as to a JSON mapping --- a
        converter that stored ``PatientName`` as an attribute is caught by the
        same rule as one that put it in ``extra``.
        """
        framed = {group: set(attrs) for group, attrs in FRAME_ATTRS}
        self.attributes(root, "root", skip=_ROOT_MANAGED_ATTRS)
        names: list[str] = []
        root.visit(names.append)
        for name in names:
            node = root[name]
            where = name.replace("/", ".")
            self.name(name.rsplit("/", 1)[-1], where)
            parts = name.split("/")
            # The frame graph is pseudonymised as one mapping, with the grids,
            # annotations and transforms that name it (`_scan_frames`).
            skip = framed.get(parts[0], set()) if len(parts) == 2 else set()
            self.attributes(node, where, skip=skip)
            if isinstance(node, h5py.Dataset) and name != "meta":
                self.dataset(node, where)

    def attributes(
        self, node: h5py.HLObject, where: str, *, skip: frozenset[str] | set[str]
    ) -> None:
        attrs = node.attrs
        for key in list(attrs.keys()):
            if key in skip:
                continue
            path = f"{where}.{key}"
            try:
                raw = attrs[key]
            except (OSError, TypeError, ValueError):
                self.find(
                    "unreadable",
                    path,
                    "an attribute this tool cannot decode; a person must look at it",
                )
                continue
            value = _decode(raw)
            if key in _REFERENCE_ATTRS:
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        self.name(item, f"{path}[{index}]")
                else:
                    self.name(value, path)
                continue
            kept = self.entry(key, value, path, depth=0, review=None)
            if not self.writing:
                continue
            if kept is _DROP:
                del attrs[key]
            elif kept is not value and kept != value:
                attrs[key] = (
                    np.array(kept, dtype=str_dtype())
                    if isinstance(kept, list) and all(isinstance(v, str) for v in kept)
                    else encode_attr(kept)
                )

    def dataset(self, dataset: h5py.Dataset, where: str) -> None:
        dtype = dataset.dtype
        if dtype.names is not None and any(
            _string_dtype(dtype.fields[n][0]) for n in dtype.names
        ):
            self.find(
                "unreadable",
                where,
                "a compound dataset with string fields; this tool does not decode "
                "compound types, so a person must look at it",
            )
            return
        if not _string_dtype(dtype):
            return
        try:
            raw = dataset[...]
        except (OSError, TypeError, ValueError):
            self.find(
                "unreadable",
                where,
                "a string dataset this tool cannot read; a person must look at it",
            )
            return
        flat = np.asarray(raw).ravel() if np.ndim(raw) else np.asarray([raw])
        values = [_decode(v) for v in flat]
        kept = [
            self.text(str(v), f"{where}[{index}]", depth=0)
            for index, v in enumerate(values)
        ]
        if self.writing and kept != values:
            _rewrite_strings(dataset, kept)


def _rewrite_strings(dataset: h5py.Dataset, values: list[str]) -> None:
    """Write cleaned strings back, keeping the dataset's whole filter pipeline.

    In place wherever the values fit, which keeps every filter, the chunking
    and the fill value by construction.  A fixed-length dataset whose width is
    shorter than a pseudonym is widened instead: recreated with the same kind
    of string, a copy of its creation property list --- so shuffle, Fletcher32
    and any codec come across, not only the compression --- and its attributes.
    ``commit`` then restamps its digest from the new bytes.
    """
    info = h5py.check_string_dtype(dataset.dtype)
    if info is None or info.length is None:
        dataset[...] = np.array(values, dtype=object).reshape(dataset.shape)
        return
    encoding = info.encoding
    try:
        encoded = [v.encode(encoding) for v in values]
    except UnicodeEncodeError:
        encoding = "utf-8"
        encoded = [v.encode(encoding) for v in values]
    width = max([info.length, *(len(b) for b in encoded)])
    if width == info.length and encoding == info.encoding:
        dataset[...] = np.array(encoded, dtype=dataset.dtype).reshape(dataset.shape)
        return
    _widen(dataset, h5py.string_dtype(encoding, width), encoded)


def _widen(dataset: h5py.Dataset, dtype: Any, encoded: list[bytes]) -> None:
    """Recreate *dataset* with a wider string type and its own creation plist.

    Built under a temporary name and moved into place, so a pipeline HDF5
    cannot rebuild for the new type leaves the original untouched.
    """
    from h5py import h5d, h5t

    parent = dataset.parent
    name = dataset.name.rsplit("/", 1)[-1]
    staging = f"{name}.scrub-{uuid.uuid4().hex[:8]}"
    dcpl = dataset.id.get_create_plist()
    if dataset.chunks is not None:
        # The copied layout records the old element size beside the chunk
        # shape; setting the shape again lets HDF5 size it for the new type.
        dcpl.set_chunk(dataset.chunks)
    created = h5d.create(
        parent.id,
        staging.encode(),
        h5t.py_create(dtype, logical=True),
        dataset.id.get_space(),
        dcpl=dcpl,
    )
    replacement = h5py.Dataset(created)
    replacement[...] = np.array(encoded, dtype=dtype).reshape(dataset.shape)
    for key in dataset.attrs:
        replacement.attrs[key] = dataset.attrs[key]
    del parent[name]
    parent.move(staging, name)


def _imported_from_dicom(doc: Mapping[str, Any]) -> bool:
    for activity in (doc.get("provenance") or {}).get("activities", ()):
        if "from-dicom" in str(activity.get("tool") or ""):
            return True
        if any(str(v).startswith("dicom:") for v in activity.get("inputs") or ()):
            return True
    return False


def scan_document(document: Any, report: ScrubReport) -> ScrubReport:
    """Every rule, over every string in one sample document.  Reads only ``/meta``."""
    _Sweep(report).document(document.to_json())
    return report


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


def _scan_file_name(
    path: str | os.PathLike[str], identity: Any, report: ScrubReport
) -> None:
    """A file named after an id that is itself an identifier.

    The DICOM importer names each file of a multi-subject import after its
    PatientID.  The name is outside the bytes the attestation covers, and it is
    the first thing anyone receiving the file sees.
    """
    name_of_file = Path(os.fspath(path)).name
    stems = _stems(path)
    if any(UID_TOKEN.search(stem) for stem in stems):
        # The DICOM importer names a study-scoped sample after its
        # StudyInstanceUID when there is no PatientID to name it by.
        report.add(
            "file_name",
            "file name",
            "the file name contains a real DICOM UID; rename it before sharing "
            "--- medh5 does not rename files",
            name_of_file,
        )
        return
    flagged = {f.where for f in report.findings}
    for name in ("sample_id", "subject_id"):
        value = str(getattr(identity, name))
        if value in stems and f"identity.{name}" in flagged:
            report.add(
                "file_name",
                "file name",
                f"the file is named after identity.{name}, which is reported "
                "above; rename it before sharing --- medh5 does not rename files",
                name_of_file,
            )
            return


def _stems(path: str | os.PathLike[str]) -> set[str]:
    """What a file name says a sample is called: without its suffix, and up to
    the first dot --- the default the writer derives an id from."""
    name = Path(os.fspath(path)).name
    return {Path(name).stem, name.split(".", 1)[0]}


def _escalate(report: ScrubReport) -> ScrubReport:
    """Mark the strict-profile rules actionable, except where nothing can act."""
    if report.profile != "strict":
        return report
    for finding in report.findings:
        if (
            finding.rule in STRICT_RULES
            and finding.fixable
            and finding.where not in UNFIXABLE_LOCATIONS
        ):
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
        sweep = _Sweep(report)
        sweep.document(sample.document.to_json())
        _scan_frames(frame_references(sample.root), report)
        sweep.hdf5(sample.root)
        _scan_file_name(path, sample.document.identity, report)
    return _escalate(report)


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
    import medh5
    from medh5.document import SampleDocument

    if pseudonymise_ids and not salt:
        raise MEDH5ValidationError(
            "pseudonymise_ids needs a salt: a sample or subject id is usually a "
            "record number, and an unsalted hash of one is reversed by hashing "
            "every record number; pass --salt and keep it apart from the data"
        )
    report = scan(path, profile=profile)
    strict = profile == "strict"
    replaced: dict[str, str] = {}
    with medh5.amend(path) as writer:
        already = writer.document.deidentification
        shifted_before = already is not None and already.date_shift_days is not None
        notes: list[str] = []
        if already is not None and shifted_before:
            # Shifting twice makes the recorded offset a lie, and a scrub that
            # is not idempotent cannot be run from a pipeline.
            date_shift_days = already.date_shift_days
            notes.append(
                f"dates were left alone: already shifted by {date_shift_days} days"
            )

        # The performer is declared *before* the sweep, so the name this run
        # adds gets the same treatment as the names it found: under `strict` a
        # person becomes a stable pseudonym.  Adding it afterwards left `--by
        # RAD-07` as an un-pseudonymised staff name that the tool's own re-scan
        # then reported, so a strict apply could never exit 0.
        agent = (
            writer.person(performed_by)
            if performed_by
            else writer.software("medh5", medh5.__version__)
        )
        sweep = _Sweep(
            ScrubReport(path="<apply>", profile=profile),
            writing=True,
            strict=strict,
            salt=salt,
            date_shift_days=date_shift_days,
            uid_map=report.uid_map,
        )
        cleaned = sweep.document(writer.document.to_json())
        if pseudonymise_ids:
            replaced = _pseudonymise_ids(cleaned, salt, report, sweep.actions)
        writer.document = SampleDocument.from_json(cleaned)
        sweep.hdf5(writer.handle)

        # One mapping, applied to every reference at once.  Pseudonymising the
        # grids alone would leave the real UID on the transforms and annotations
        # and break the frame graph joining them --- see `remap_frame_uids`.
        frame_map = {
            uid: pseudonymise(uid, salt)
            for uid in writer.frame_uids()
            if DICOM_UID.match(uid)
        }
        report.uid_map.update(frame_map)
        removed = [*notes, *sweep.actions]
        removed.extend(
            f"{location} -> pseudonymised"
            for location in writer.remap_frame_uids(frame_map)
        )

        writer.deidentification(
            method="medh5-scrub",
            profile=(
                f"medh5 scrub {profile}: container metadata only, voxel data not "
                "examined; "
                + (
                    "quasi-identifiers removed"
                    if strict
                    else "quasi-identifiers retained for review"
                )
                + (
                    "; sample and subject ids pseudonymised"
                    if pseudonymise_ids
                    else "; sample and subject ids retained"
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
        # Re-scan the cleaned sample, *after* the record is written, and put the
        # result in the activity.  The record has to exist first or the re-scan
        # reports the shifted dates it explains, and the count would describe a
        # document nobody will ever read.  What this run achieved is then in the
        # file itself, where a reader auditing the attestation looks --- rather
        # than only in a report that is not shipped with it.
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
    # The amend copied every object and *then* rewrote what it cleaned, and
    # HDF5 does not reclaim what it supersedes --- so the file it just produced
    # still contains the original `frame_uid`, the old `/meta` and every
    # rewritten attribute in freed space, recoverable with `strings` while every
    # API read returns the pseudonym.  For a de-identification tool that is not
    # a wasted-bytes problem, so the output is compacted before it is handed
    # back.  Digests and `content_id` are unaffected: this rewrites storage, not
    # content (§13.1).
    repack(path)
    # The claim is checked against the file, not against the intention: the
    # rules are re-run over what was written, and `ok` --- and therefore the
    # CLI's exit code --- follows the result.
    report.remaining = scan(path, profile=profile).findings
    stems = _stems(path)
    for name, old in replaced.items():
        if old in stems:
            report.remaining.append(
                Finding(
                    rule="file_name",
                    where="file name",
                    detail=f"the file is still named after the {name} this run "
                    "replaced; rename it before sharing --- medh5 does not "
                    "rename files, and a later scan cannot tell the name was an "
                    "id",
                    value=Path(os.fspath(path)).name,
                )
            )
            break
    return report


def _pseudonymise_ids(
    doc: dict[str, Any], salt: str, report: ScrubReport, actions: list[str]
) -> dict[str, str]:
    """Replace the sample's ids with salted pseudonyms; return what was replaced.

    ``cohort.group_id`` defaults to the subject id (§12.2) and is often set to
    it explicitly, so a group id equal to a replaced id is replaced with it ---
    every file of one subject, scrubbed with one salt, still groups together.
    """
    identity = doc["identity"]
    replaced: dict[str, str] = {}
    mapping: dict[str, str] = {}
    for name in ("sample_id", "subject_id"):
        value = str(identity[name])
        if value.startswith(PSEUDONYM_PREFIX):
            continue
        pseudonym = pseudonymise(value, salt)
        identity[name] = pseudonym
        report.uid_map[value] = pseudonym
        mapping[value] = pseudonym
        replaced[name] = value
        actions.append(f"identity.{name} pseudonymised")
    if not replaced:
        return replaced
    recorded = identity.get(ID_SOURCE)
    source = dict(recorded) if isinstance(recorded, Mapping) else {}
    source.update({name: PSEUDONYM_SOURCE for name in replaced})
    identity[ID_SOURCE] = source
    cohort = doc.get("cohort")
    if isinstance(cohort, dict) and cohort.get("group_id") in mapping:
        cohort["group_id"] = mapping[cohort["group_id"]]
        actions.append("cohort.group_id pseudonymised with the id it equalled")
    return replaced


def _rescan(writer: Any, profile: str) -> ScrubReport:
    """Run every rule over a writer's state, without going back to disk.

    The same rules as :func:`scan`, sourced from the document, the frame
    references and the objects the amend is about to commit --- so the count
    recorded in the attestation describes the file that attestation ships in.
    """
    report = ScrubReport(path="<amend>", profile=profile)
    sweep = _Sweep(report)
    sweep.document(writer.document.to_json())
    _scan_frames(writer.frame_uids(), report)
    sweep.hdf5(writer.handle)
    return _escalate(report)


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
