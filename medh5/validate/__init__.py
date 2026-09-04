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
from medh5.validate.rules import COLLECTION_RULES, Context, rules_for


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
        # A rule that cannot read what it is checking must still produce a
        # diagnostic.  `validate` is the tool you point at a file of unknown
        # provenance, and on a file whose bytes were corrupted past the header
        # h5py raises from inside the traversal or the decompressor -- so the
        # command exited with a traceback, printed nothing on stdout, and
        # `--json` emitted no JSON at all.  Failing to read an object is a
        # finding about the file, not a crash of the tool; the remaining rules
        # still run, so one unreadable object does not hide everything else.
        try:
            report.extend(rule(ctx))
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            report.add(
                Diagnostic(
                    code="E001",
                    location="/",
                    message=(
                        f"{rule.__name__} could not read the file: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    severity="error",
                    level=level,
                )
            )
    report.checked = {
        "rules": [rule.__name__ for rule in rules_for(level)],
        "schema_checked": ctx.notes.get("schema_checked", False),
    }
    report.diagnostics.sort(key=lambda d: (d.severity != "error", d.code, d.location))
    return report


def validate_collection(
    root: h5py.Group,
    *,
    path: str = "<memory>",
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> Report:
    """Validate a ``collection`` root and every sample in it (spec §2.2).

    Members are validated with the ordinary sample rules and nothing else ---
    a sample root inside a shard is a sample root, and a validator that treated
    it as a lesser thing would let packed data drift from unpacked data.
    Locations are prefixed with ``/samples/<key>`` so a diagnostic still names
    exactly one object in exactly one file.
    """
    from medh5.collection import SAMPLES_GROUP

    header = Report(path=path, level=level)
    ctx = Context(root=root, path=path, level=level)
    for rule in COLLECTION_RULES:
        header.extend(rule(ctx))
    members: list[Report] = []
    node = root.get(SAMPLES_GROUP)
    if node is not None:
        for key in sorted(node):
            members.append(
                validate_root(
                    node[key],
                    path=f"/{SAMPLES_GROUP}/{key}",
                    level=level,
                    profiles=profiles,
                    attr_names=_attr_names(node[key], level),
                )
            )
    combined = merge(members, path=path) if members else Report(path=path, level=level)
    combined.level = level
    combined.diagnostics = header.diagnostics + combined.diagnostics
    combined.checked = {"samples": [r.path for r in members]}
    combined.diagnostics.sort(key=lambda d: (d.severity != "error", d.code, d.location))
    return combined


def _attr_names(root: h5py.Group, level: Level) -> Any:
    """The digest attribute map, needed only where integrity is checked."""
    if level not in ("integrity", "strict"):
        return None
    from medh5.sample import Sample

    try:
        return Sample(root).attr_name_map()
    except Exception:
        return None


def validate_file(
    path: str | os.PathLike[str],
    *,
    level: Level = "semantic",
    profiles: Sequence[str] | None = None,
) -> Report:
    """Validate one ``.medh5`` sample or ``.medh5c`` collection file."""
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
        from medh5.collection import is_collection

        # Everything from here reads the file, and on a file corrupted past the
        # header any of it can raise out of h5py -- deciding whether the root is
        # a collection, reading `medh5_profiles`, walking the members.  The rule
        # loop guards itself; this guards the scaffolding around it, so the
        # command always returns a report rather than a traceback.
        try:
            if is_collection(handle):
                return validate_collection(
                    handle, path=text_path, level=level, profiles=profiles
                )
            return validate_root(
                handle,
                path=text_path,
                level=level,
                profiles=profiles,
                attr_names=_attr_names(handle, level),
            )
        except (OSError, RuntimeError, KeyError, ValueError) as exc:
            report = Report(path=text_path, level=level)
            report.add(
                Diagnostic(
                    code="E001",
                    location="/",
                    message=(
                        f"the file could not be read: {type(exc).__name__}: {exc}"
                    ),
                    severity="error",
                    level=level,
                )
            )
            return report


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
    "validate_collection",
    "validate_file",
    "validate_paths",
    "validate_root",
]
