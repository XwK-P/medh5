"""Validation (spec §15).

Four levels, each a superset of the last:

``structural``
    layout, required attributes, dtypes, shapes, identifier syntax, JSON Schema
``semantic``
    cross-references resolve, geometry consistency, class ids in the label set,
    encoding invariants, profile requirements
``integrity``
    per-object digests, ``content_id``, index ``source_digest`` currency
``strict``
    all of the above, with warnings promoted to failures

A validation pass never raises on a bad file --- it reports.  Curation needs to
see everything wrong with a file at once, and a validator that stops at the
first problem turns one review cycle into ten.

The rules are the format engine's (``medh5`` crate, ``validate`` module): the
command line, this package and the Rust SDK run the same code, so a file one
of them accepts the others accept.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from medh5 import _core
from medh5.validate.report import (
    LEVEL_ORDER,
    LEVELS,
    Diagnostic,
    Level,
    Report,
    merge,
)


def validate_file(
    path: str | os.PathLike[str],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> Report:
    """Validate one ``.medh5`` sample or ``.medh5c`` collection file.

    A file that cannot be opened is reported (E001), not raised; an unknown
    *level* raises ``ValueError``.
    """
    return Report.from_json(
        _core.validate_file(os.fspath(path), level=level, profiles=profiles)
    )


def validate_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> list[Report]:
    """Validate many files, one report each."""
    return [
        Report.from_json(doc)
        for doc in _core.validate_paths(
            [os.fspath(p) for p in paths], level=level, profiles=profiles
        )
    ]


def validate_root(
    sample: Any,
    *,
    path: str | None = None,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
    errors_only: bool = False,
) -> Report:
    """Validate an open :class:`~medh5.sample.Sample` --- a file, or a member
    of a collection.

    ``errors_only`` skips the warning-only checks that read bulk data, which
    is what the writer's commit gate asks for.
    """
    return Report.from_json(
        _core.validate_root(
            sample,
            path=path,
            level=level,
            profiles=profiles,
            errors_only=errors_only,
        )
    )


def rules_for(level: Level) -> tuple[str, ...]:
    """The rules a pass at *level* runs, in order."""
    return tuple(_core.rules_for(level))


__all__ = [
    "LEVELS",
    "LEVEL_ORDER",
    "Diagnostic",
    "Level",
    "Report",
    "merge",
    "rules_for",
    "validate_file",
    "validate_paths",
    "validate_root",
]
