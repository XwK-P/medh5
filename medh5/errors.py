"""Exception hierarchy and the normative diagnostic code table (spec §15.2).

Diagnostic codes are **stable API**: a code's meaning never changes, codes are
never reused, and third-party validators are expected to emit the same code for
the same defect.  The table lives in the format engine
(``crates/medh5/data/codes.json``) --- the one source the validator, the
conformance corpus, every frontend and the documentation all read --- and
:data:`CODES` is that table.

Codes are grouped by domain::

    E0xx  container      E1xx  geometry     E2xx  images
    E3xx  label set      E4xx  annotations  E5xx  transforms
    E6xx  curation       E7xx  integrity    E8xx  clinical (1.1)
    W9xx  warnings

The table is allocated in numbered *revisions* (§15.2): each code names the
format version whose rules it reports (:attr:`Code.since`) and the revision
that allocated it (:attr:`Code.revision`), and :data:`CODES_REVISION` is the
table's own.  A validator pinned to a revision
(``validate_file(..., codes_revision=1)``) reports no later code.

The package's own exceptions derive from :class:`MEDH5Error`, and two of them
from a built-in as well: :class:`MEDH5FileError` is an ``OSError`` and
:class:`MEDH5ValidationError` a ``ValueError``.  Not every error raised is one
of them.  An argument of the wrong type or value raises the built-in of that
meaning (``TypeError``, ``ValueError``, ``KeyError``, ``IndexError``), and a
read or write that HDF5 fails --- a damaged chunk, say --- a plain
``OSError``: catch ``OSError`` as well as :class:`MEDH5Error` to catch both.
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
    "clinical",
]


class MEDH5Error(Exception):
    """Base of the package's own exceptions: a file, version, schema,
    validation or integrity error.

    Not every error raised derives from it: a wrong argument raises the
    built-in of that meaning, and a read or write that HDF5 fails a plain
    ``OSError`` (see the module's documentation).
    """


class MEDH5FileError(MEDH5Error, OSError):
    """A file cannot be opened, is not HDF5, or is structurally unreadable."""


class MEDH5VersionError(MEDH5Error):
    """The file's ``medh5_version`` is one this operation cannot work with.

    Raised when a file declares a major this reader does not implement, or
    declares no version at all (a 0.x file, which ``medh5 migrate`` converts);
    and when ``amend``, ``recompress`` or ``pack`` is given a file --- or a
    member --- of a later minor version, which this engine reads only as a
    projection and so cannot rewrite without losing what it does not know
    (1.1 §2.3).
    """


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
    since: str = "1.0"
    """The format version whose rules the code reports."""
    revision: int = 1
    """The revision of the table that allocated the code (§15.2)."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code} ({self.severity}): {self.summary}"


def _load_table() -> tuple[int, tuple[Code, ...]]:
    """The table the engine embeds (``crates/medh5/data/codes.json``), and
    its revision."""
    import json

    from medh5._core import codes_table

    table = json.loads(codes_table())
    return int(table["revision"]), tuple(
        Code(
            c["code"],
            c["severity"],
            c["domain"],
            c["summary"],
            c["since"],
            c["revision"],
        )
        for c in table["codes"]
    )


CODES_REVISION: int
"""The revision of the code table this engine implements (§15.2)."""
CODES_REVISION, _TABLE = _load_table()

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
    "CODES_REVISION",
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
