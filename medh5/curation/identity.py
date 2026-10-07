"""Identity, cohort, splits and de-identification (spec §11.4, §12).

``subject_id`` prevents the most common evaluation error in medical AI --- the
same patient in train and test --- and because a sample never spans subjects,
assigning whole files to partitions is subject-safe with no further bookkeeping.

Per-occasion identifiers (``study_uid``, ``series_uids``, dates, ages) live on
the *timepoint*, not here, because a sample may have several of each.
"""

from __future__ import annotations

from medh5._core import (
    ID_SOURCE,
    LATERALITY_VALUES,
    PARTITIONS,
    PSEUDONYM_SOURCE,
    SEX_VALUES,
    Cohort,
    Deidentification,
    Identity,
    SplitClaim,
    splits_from_json,
)

__all__ = [
    "ID_SOURCE",
    "LATERALITY_VALUES",
    "PARTITIONS",
    "PSEUDONYM_SOURCE",
    "SEX_VALUES",
    "Cohort",
    "Deidentification",
    "Identity",
    "SplitClaim",
    "splits_from_json",
]
