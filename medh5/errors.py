"""Exception hierarchy and the normative diagnostic code table (spec §15.2).

Diagnostic codes are **stable API**: a code's meaning never changes, codes are
never reused, and third-party validators are expected to emit the same code for
the same defect.  The table below is the single source of truth --- the
validator (:mod:`medh5.validate`), the conformance corpus and the documentation
all read it rather than repeating literals.

Codes are grouped by domain::

    E0xx  container      E1xx  geometry     E2xx  images
    E3xx  label set      E4xx  annotations  E5xx  transforms
    E6xx  curation       E7xx  integrity    W9xx  warnings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["error", "warning"]

Domain = Literal[
    "container",
    "geometry",
    "images",
    "labels",
    "annotations",
    "transforms",
    "curation",
    "integrity",
]


class MEDH5Error(Exception):
    """Base exception for every error raised by this package."""


class MEDH5FileError(MEDH5Error, OSError):
    """A file cannot be opened, is not HDF5, or is structurally unreadable."""


class MEDH5VersionError(MEDH5Error):
    """The file declares a ``medh5_version`` major this reader does not implement."""


class MEDH5SchemaError(MEDH5Error):
    """The sample document is absent, is not JSON, or violates its JSON Schema."""


class MEDH5ValidationError(MEDH5Error, ValueError):
    """Input rejected by a writer, or a validation failure raised as an exception.

    Carries the diagnostic ``code`` when one applies, so callers can branch on a
    stable identifier instead of on message text.
    """

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message if code is None else f"[{code}] {message}")
        self.code = code
        self.message = message


class MEDH5IntegrityError(MEDH5Error):
    """A digest or ``content_id`` does not match the data it covers."""


@dataclass(frozen=True, slots=True)
class Code:
    """One diagnostic code."""

    code: str
    severity: Severity
    domain: Domain
    summary: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code} ({self.severity}): {self.summary}"


def _c(code: str, domain: Domain, summary: str) -> Code:
    return Code(code, "warning" if code.startswith("W") else "error", domain, summary)


_TABLE: tuple[Code, ...] = (
    # --- E0xx container -----------------------------------------------------
    _c("E001", "container", "missing or empty `medh5_version`"),
    _c("E002", "container", "unsupported major version"),
    _c("E003", "container", "identifier does not match [A-Za-z0-9_.-]{1,128}"),
    _c("E004", "container", "`meta` is absent or is not valid UTF-8 JSON"),
    _c("E005", "container", "`meta` fails the sample-document JSON Schema"),
    _c("E006", "container", "missing or unknown `medh5_kind`"),
    _c("E007", "container", "missing `medh5_profiles`, or an unknown profile"),
    _c("E008", "container", "a group required by the layout is absent"),
    _c("E009", "container", "declared profile's requirements are not met"),
    _c(
        "E010",
        "container",
        "sample root in a `collection` lacks its own `content_id`",
    ),
    # --- E1xx geometry ------------------------------------------------------
    _c("E101", "geometry", "referenced grid does not exist"),
    _c("E102", "geometry", "`direction` is not orthonormal to 1e-4"),
    _c("E103", "geometry", "spatial axes are not contiguous and trailing"),
    _c("E104", "geometry", "`spacing` is not strictly positive"),
    _c("E105", "geometry", "multiscale level geometry is inconsistent with level 0"),
    _c("E106", "geometry", "grid has no `timepoint` in a multi-timepoint sample"),
    _c("E107", "geometry", "grid `timepoint` names an undeclared timepoint"),
    _c("E108", "geometry", "`timepoints` is empty, or `index` is not dense/increasing"),
    _c("E109", "geometry", "required grid attribute is missing or has the wrong rank"),
    _c("E110", "geometry", "`axis_kinds` is invalid for the declared dimensionality"),
    _c("E111", "geometry", "`grids` contains no grid"),
    # --- E2xx images --------------------------------------------------------
    _c("E201", "images", "`images` contains no image"),
    _c("E202", "images", "image shape does not equal its grid's `shape`"),
    _c("E203", "images", "unknown `value_type`"),
    _c("E204", "images", "`channel_names` length does not match the channel extent"),
    _c("E205", "images", "required image attribute is missing"),
    # --- E3xx label set -----------------------------------------------------
    _c("E301", "labels", "a declared profile requires a label set and none is present"),
    _c("E302", "labels", "duplicate class id or key in the label set"),
    _c("E303", "labels", "a reserved class id (0 or 65535) was assigned to a class"),
    _c("E304", "labels", "the class hierarchy contains a cycle"),
    _c(
        "E305",
        "labels",
        "`form: ref` label set lacks `uri`/`sha256`, or is unresolvable",
    ),
    _c(
        "E306",
        "labels",
        "class entry is missing a required field or has a bad id range",
    ),
    # --- E4xx annotations ---------------------------------------------------
    _c("E401", "annotations", "unknown annotation `kind`"),
    _c("E402", "annotations", "class id is not present in the label set"),
    _c("E403", "annotations", "`annotated_class_ids` is not a subset of `class_ids`"),
    _c("E404", "annotations", "encoding invariant violated"),
    _c("E405", "annotations", "dataset shape is inconsistent with the grid"),
    _c("E406", "annotations", "box `lo` exceeds `hi`"),
    _c("E407", "annotations", "`rotations` is not a proper rotation matrix"),
    _c("E408", "annotations", "`mask_offsets` are not monotonically increasing"),
    _c("E409", "annotations", "`timepoints` references an undeclared timepoint"),
    _c("E410", "annotations", "a dataset required by the annotation `kind` is absent"),
    _c(
        "E411",
        "annotations",
        "dataset dtype is not permitted for the annotation `kind`",
    ),
    _c("E412", "annotations", "required annotation attribute is missing"),
    _c("E413", "annotations", "annotation references an object that does not exist"),
    _c("E414", "annotations", "`space` is invalid for the annotation's grid or frame"),
    # --- E5xx transforms ----------------------------------------------------
    _c("E501", "transforms", "composite frame chain is broken"),
    _c("E502", "transforms", "unknown transform `kind`"),
    _c("E503", "transforms", "displacement field grid is not in `from_frame`"),
    _c("E504", "transforms", "affine last row is not [0 … 0 1]"),
    _c("E505", "transforms", "`inverse_id` is not mutually consistent"),
    # --- E6xx curation ------------------------------------------------------
    _c("E601", "curation", "`prov` names an activity that does not exist"),
    _c("E602", "curation", "`quality` names a record that does not exist"),
    _c("E603", "curation", "unknown agent or activity type"),
    _c("E604", "curation", "timestamp is not RFC 3339"),
    _c("E605", "curation", "activity names an agent that does not exist"),
    # --- E7xx integrity -----------------------------------------------------
    _c("E701", "integrity", "object digest does not match its content"),
    _c("E702", "integrity", "`content_id` does not match the file"),
    _c("E703", "integrity", "malformed digest string"),
    # --- W9xx warnings ------------------------------------------------------
    _c("W901", "integrity", "no object digests are present"),
    _c("W902", "container", "bulk dataset is unchunked or uncompressed"),
    _c("W903", "curation", "no `deidentification` record"),
    _c("W904", "annotations", "partial coverage without an ignore region"),
    _c("W905", "integrity", "`index/` entry is stale for its source annotation"),
    _c("W906", "curation", "conflicting split claims for one `set_id`"),
    _c("W907", "images", "float storage where int16 + rescale would be lossless"),
    _c(
        "W908", "annotations", "`layers` count is far from the greedy-colouring optimum"
    ),
    _c("W909", "annotations", "one `instance_id` carries two different class ids"),
    _c("W910", "geometry", "grids in different timepoints share a `frame_uid`"),
    _c(
        "W911",
        "geometry",
        "multi-timepoint sample has no transform relating timepoints",
    ),
    _c("W912", "labels", "class id used by an annotation has no ontology binding"),
)

CODES: dict[str, Code] = {c.code: c for c in _TABLE}
"""Every diagnostic code, keyed by code string."""


def code(name: str) -> Code:
    """Look up a diagnostic code, raising :class:`KeyError` for unknown codes."""
    try:
        return CODES[name]
    except KeyError:  # pragma: no cover - guards typos in rule definitions
        raise KeyError(f"unknown diagnostic code {name!r}") from None


def codes_for(domain: Domain) -> tuple[Code, ...]:
    """Every code in one domain, in table order."""
    return tuple(c for c in _TABLE if c.domain == domain)


__all__ = [
    "CODES",
    "Code",
    "Domain",
    "MEDH5Error",
    "MEDH5FileError",
    "MEDH5IntegrityError",
    "MEDH5SchemaError",
    "MEDH5ValidationError",
    "MEDH5VersionError",
    "Severity",
    "code",
    "codes_for",
]
