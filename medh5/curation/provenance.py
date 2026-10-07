"""Provenance: a two-node W3C PROV-lite graph (spec §11.1).

Agents do things; activities are the things done.  Objects point at an activity
through their ``prov`` attribute.  Two node types are enough to describe the
workflow that actually dominates real curation --- a model pre-annotates, a
human corrects, a second human reviews --- which a "review status" field cannot
describe at all, because it records *that* something was reviewed without
recording what produced the thing being reviewed.
"""

from __future__ import annotations

import re

from medh5._core import (
    ACTIVITY_FIELDS,
    ACTIVITY_TYPES,
    AGENT_FIELDS,
    AGENT_TYPES,
    Activity,
    Agent,
    Provenance,
    check_timestamp,
)

RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
)
"""The timestamp form §11.1 requires; ``check_timestamp`` is the rule."""

__all__ = [
    "ACTIVITY_FIELDS",
    "ACTIVITY_TYPES",
    "AGENT_FIELDS",
    "AGENT_TYPES",
    "RFC3339",
    "Activity",
    "Agent",
    "Provenance",
    "check_timestamp",
]
