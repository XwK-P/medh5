"""medh5 --- a self-describing HDF5 container for one medical imaging sample.

A **sample** is one subject at one or more timepoints, with every image,
annotation, transform and curation record about them in a single file.  See
``docs/spec/medh5-1.0.md`` for the normative format specification.

.. code-block:: python

    import medh5

    with medh5.open("case_0001.medh5") as s:
        s.identity.subject_id
        s.at("tp1").images["CT_tp1"].read(physical=True)
        s.annotations["organs"].dense(["liver", "spleen"])

0.x files are read by ``medh5 migrate`` and nothing else: 1.0 ships a reader for
the old layout, not an implementation of it.
"""

from __future__ import annotations

from medh5.__about__ import __format_version__, __version__
from medh5.annotations.base import Annotation, Instance, VoxelAnnotation
from medh5.collection import Collection, open_collection, pack, unpack
from medh5.curation.agreement import compare as compare_annotations
from medh5.curation.identity import Cohort, Deidentification, Identity, SplitClaim
from medh5.curation.provenance import Activity, Agent, Provenance
from medh5.curation.quality import Agreement, Issue, QualityRecord
from medh5.curation.splits import SplitAudit, audit_splits
from medh5.curation.timeline import Timeline, Timepoint
from medh5.curation.tracking import Observation, Track, Tracking
from medh5.document import SampleDocument
from medh5.errors import (
    CODES,
    MEDH5Error,
    MEDH5FileError,
    MEDH5IntegrityError,
    MEDH5SchemaError,
    MEDH5ValidationError,
    MEDH5VersionError,
)
from medh5.geometry.grid import Grid
from medh5.image import Image
from medh5.labels.labelset import LabelClass, LabelSet
from medh5.sample import (
    FORMAT_VERSION,
    PROFILES,
    Sample,
    SampleWriter,
    amend,
    create,
    open_sample,
)
from medh5.sampling import (
    Patch,
    PatchSampler,
    TimepointPair,
    TimepointPairSampler,
    grid_patches,
)

open = open_sample  # noqa: A001 - `medh5.open` is the documented entry point

__all__ = [
    "CODES",
    "FORMAT_VERSION",
    "PROFILES",
    "Activity",
    "Agent",
    "Agreement",
    "Annotation",
    "Cohort",
    "Collection",
    "Deidentification",
    "Grid",
    "Identity",
    "Image",
    "Instance",
    "Issue",
    "LabelClass",
    "LabelSet",
    "MEDH5Error",
    "MEDH5FileError",
    "MEDH5IntegrityError",
    "MEDH5SchemaError",
    "MEDH5ValidationError",
    "MEDH5VersionError",
    "Observation",
    "Patch",
    "PatchSampler",
    "Provenance",
    "QualityRecord",
    "Sample",
    "SampleDocument",
    "SampleWriter",
    "SplitAudit",
    "SplitClaim",
    "Timeline",
    "Timepoint",
    "TimepointPair",
    "TimepointPairSampler",
    "Track",
    "Tracking",
    "VoxelAnnotation",
    "__format_version__",
    "__version__",
    "amend",
    "audit_splits",
    "compare_annotations",
    "create",
    "grid_patches",
    "open",
    "open_collection",
    "open_sample",
    "pack",
    "unpack",
]
