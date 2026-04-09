"""Custom exception hierarchy for medh5."""

from __future__ import annotations


class MEDH5Error(Exception):
    """Base exception for all medh5 errors."""


class MEDH5ValidationError(MEDH5Error, ValueError):
    """Raised when input data fails validation checks."""


class MEDH5FileError(MEDH5Error, OSError):
    """Raised when a ``.medh5`` file cannot be read or is corrupt."""


class MEDH5SchemaError(MEDH5Error):
    """Raised when a file's schema version is unsupported."""
