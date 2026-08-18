"""Curation records: who produced what, how good it is, and who it is about.

Spec §3.7 (timepoints), §11 (provenance, quality, de-identification) and §12
(identity, cohorts, splits).  These are the *documents* of the sample document
(§2.4); nothing here writes an HDF5 attribute.
"""

from __future__ import annotations

from medh5.curation.agreement import (
    InstanceAgreement,
    VoxelAgreement,
    compare,
    compare_instances,
    compare_voxel,
)
from medh5.curation.identity import Cohort, Deidentification, Identity, SplitClaim
from medh5.curation.provenance import (
    ACTIVITY_TYPES,
    AGENT_TYPES,
    Activity,
    Agent,
    Provenance,
)
from medh5.curation.quality import (
    QUALITY_STATUS,
    Agreement,
    Issue,
    QualityRecord,
)
from medh5.curation.splits import SplitAudit, audit_splits
from medh5.curation.timeline import Timeline, Timepoint
from medh5.curation.tracking import Observation, Track, Tracking, build_tracks

__all__ = [
    "ACTIVITY_TYPES",
    "AGENT_TYPES",
    "QUALITY_STATUS",
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
    "compare",
    "compare_instances",
    "compare_voxel",
]
