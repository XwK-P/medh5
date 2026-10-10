"""Curation records: who produced what, how good it is, and who it is about.

Spec §3.7 (timepoints), §11 (provenance, quality, de-identification) and §12
(identity, cohorts, splits).  These are the *documents* of the sample document
(§2.4); nothing here writes an HDF5 attribute.  The record classes are the
format engine's; the tools that work on them --- agreement between raters
(:mod:`~medh5.curation.agreement`), the identifier sweep
(:mod:`~medh5.curation.scrub`) and the split audit
(:mod:`~medh5.curation.splits`) --- are modules of this package.

**Identity** (§11.4, §12).  ``subject_id`` prevents the most common evaluation
error in medical AI --- the same patient in train and test --- and because a
sample never spans subjects, assigning whole files to partitions is
subject-safe with no further bookkeeping.  Per-occasion identifiers
(``study_uid``, ``series_uids``, dates, ages) live on the *timepoint*, because a
sample may have several of each.

**Provenance** (§11.1) is a two-node W3C PROV-lite graph: agents do things;
activities are the things done, and objects point at an activity through their
``prov`` attribute.  That is enough to describe the workflow that dominates real
curation --- a model pre-annotates, a human corrects, a second human reviews ---
which a "review status" field cannot, because it records *that* something was
reviewed without recording what produced the thing being reviewed.

**Quality** (§11.2).  Status changes are activities, not fields with private
history: a quality record says what is true now and the provenance graph says
how it got that way.

**Timepoints** (§3.7) are the sample's observation occasions, declared in
``/meta -> timepoints``; every grid names one, and images, annotations and
transforms inherit theirs from it.  ``days_from_baseline`` rather than ``date``
is what models should consume: the interval survives de-identification date
shifting, and it is the clinically load-bearing quantity.

**Tracking** (§7.4, §11.3) is a join on ``instance_id``, not a structure: an
``instance_id`` names one physical object within the sample, reused by every
annotation that describes it.  Absence is not a measurement --- a lesion missing
from a follow-up is *resolved* only if the annotator committed to looking for
its class there (``annotated_class_ids``), so absence resolves to ``present``,
``resolved`` or ``unexamined``, never to a silent zero.  A track carries one
class: two class ids under one instance id is reported (W909), not resolved.
``Sample.tracks()`` runs the join.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from medh5 import _core
from medh5._core import (
    ACTIVITY_FIELDS,
    ACTIVITY_TYPES,
    AGENT_FIELDS,
    AGENT_TYPES,
    ID_SOURCE,
    ISSUE_SEVERITY,
    LATERALITY_VALUES,
    PARTITIONS,
    PSEUDONYM_SOURCE,
    QUALITY_FIELDS,
    QUALITY_STATUS,
    SEX_VALUES,
    TIMEPOINT_FIELDS,
    Activity,
    Agent,
    Agreement,
    Cohort,
    Deidentification,
    Identity,
    Issue,
    Observation,
    Provenance,
    QualityRecord,
    SplitClaim,
    Timeline,
    Timepoint,
    Track,
    Tracking,
    carries_instance_ids,
    check_timestamp,
    dice_agreement,
    quality_from_json,
    quality_to_json,
    splits_from_json,
)
from medh5.curation.agreement import (
    InstanceAgreement,
    VoxelAgreement,
    compare,
    compare_instances,
    compare_voxel,
)
from medh5.curation.splits import SplitAudit, audit_splits

RFC3339 = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$"
)
"""The timestamp form §11.1 requires; ``check_timestamp`` is the rule."""

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")
"""An ISO-8601 calendar date prefix, as ``date`` must start."""

PRESENT: str = _core.PRESENT
RESOLVED: str = _core.RESOLVED
UNEXAMINED: str = _core.UNEXAMINED
STATES: tuple[str, ...] = _core.STATES

Sequence.register(Timeline)
Mapping.register(Tracking)


def build_tracks(
    sample: Any, class_key: int | str | None = None, *, measure: bool = True
) -> Any:
    """Join ``instance_id`` across a sample's timepoints (``Sample.tracks``)."""
    return sample.tracks(class_key, measure=measure)


__all__ = [
    "ACTIVITY_FIELDS",
    "ACTIVITY_TYPES",
    "AGENT_FIELDS",
    "AGENT_TYPES",
    "DATE_PATTERN",
    "ID_SOURCE",
    "ISSUE_SEVERITY",
    "LATERALITY_VALUES",
    "PARTITIONS",
    "PRESENT",
    "PSEUDONYM_SOURCE",
    "QUALITY_FIELDS",
    "QUALITY_STATUS",
    "RESOLVED",
    "RFC3339",
    "SEX_VALUES",
    "STATES",
    "TIMEPOINT_FIELDS",
    "UNEXAMINED",
    "Activity",
    "Agent",
    "Agreement",
    "Cohort",
    "Deidentification",
    "Identity",
    "InstanceAgreement",
    "Issue",
    "Observation",
    "Provenance",
    "QualityRecord",
    "SplitAudit",
    "SplitClaim",
    "Timeline",
    "Timepoint",
    "Track",
    "Tracking",
    "VoxelAgreement",
    "audit_splits",
    "build_tracks",
    "carries_instance_ids",
    "check_timestamp",
    "compare",
    "compare_instances",
    "compare_voxel",
    "dice_agreement",
    "quality_from_json",
    "quality_to_json",
    "splits_from_json",
]
