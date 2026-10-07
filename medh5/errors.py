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


def _load_table() -> tuple[Code, ...]:
    """The table the engine embeds (``crates/medh5/data/codes.json``)."""
    import json

    from medh5._core import codes_table

    return tuple(
        Code(c["code"], c["severity"], c["domain"], c["summary"])
        for c in json.loads(codes_table())["codes"]
    )


_TABLE: tuple[Code, ...] = _load_table()

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
