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
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import h5py

from medh5._hdf5 import as_str_tuple, open_h5
from medh5.errors import MEDH5FileError
from medh5.validate.report import (
    LEVELS,
    Diagnostic,
    Level,
    Report,
    merge,
)
from medh5.validate.rules import Context, rules_for


def validate_root(
    root: h5py.Group,
    *,
    path: str = "<memory>",
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
    attr_names: Any = None,
) -> Report:
    """Validate an open sample root."""
    if level not in LEVELS:
        raise ValueError(
            f"unknown validation level {level!r}; expected one of {LEVELS}"
        )
    declared = (
        tuple(profiles)
        if profiles is not None
        else (
            as_str_tuple(root.attrs["medh5_profiles"])
            if "medh5_profiles" in root.attrs
            else ()
        )
    )
    ctx = Context(root=root, path=path, level=level, profiles=declared)
    if attr_names is not None:
        ctx.notes["attr_names"] = attr_names
    report = Report(path=path, level=level, profiles=declared)
    for rule in rules_for(level):
        report.extend(rule(ctx))
    report.checked = {
        "rules": [rule.__name__ for rule in rules_for(level)],
        "schema_checked": ctx.notes.get("schema_checked", False),
    }
    report.diagnostics.sort(key=lambda d: (d.severity != "error", d.code, d.location))
    return report


def validate_file(
    path: str | os.PathLike[str],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> Report:
    """Validate one ``.medh5`` file."""
    text_path = os.fspath(path)
    try:
        handle = open_h5(text_path, "r")
    except MEDH5FileError as exc:
        report = Report(path=text_path, level=level)
        report.add(
            Diagnostic(
                code="E001",
                location="/",
                message=str(exc),
                severity="error",
                level=level,
            )
        )
        return report
    with handle:
        attr_names = None
        if level in ("integrity", "strict"):
            from medh5.sample import Sample

            try:
                attr_names = Sample(handle, path=text_path).attr_name_map()
            except Exception:  # noqa: BLE001 - a broken file still gets a report
                attr_names = None
        return validate_root(
            handle,
            path=text_path,
            level=level,
            profiles=profiles,
            attr_names=attr_names,
        )


def validate_paths(
    paths: Sequence[str | os.PathLike[str]],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> list[Report]:
    """Validate many files, one report each."""
    return [validate_file(p, level=level, profiles=profiles) for p in paths]


__all__ = [
    "LEVELS",
    "Context",
    "Diagnostic",
    "Level",
    "Report",
    "merge",
    "validate_file",
    "validate_paths",
    "validate_root",
]
