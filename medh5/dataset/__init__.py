"""Cohort tools: manifests, splits, streaming statistics, cross-file checks.

Everything above the single file.  A sample is self-describing (that is the
format's whole premise), but a *cohort* has properties no file can carry: which
label set everyone agrees on, which subject is in which partition, what the
intensity distribution is, whether a class was examined everywhere or only in
the twenty files somebody got around to.  This package computes those from
metadata alone, so the answers cost milliseconds rather than a pass over every
voxel in the study.
"""

from __future__ import annotations

from medh5.dataset.check import CHECK_CODES, CohortReport, Finding, check
from medh5.dataset.manifest import Entry, Manifest, entries_for, find, scan
from medh5.dataset.split import Assignment, Split, make_splits, write_claims
from medh5.dataset.stats import ClassStats, DatasetStats, Moments, compute_stats

__all__ = [
    "CHECK_CODES",
    "Assignment",
    "ClassStats",
    "CohortReport",
    "DatasetStats",
    "Entry",
    "Finding",
    "Manifest",
    "Moments",
    "Split",
    "check",
    "compute_stats",
    "entries_for",
    "find",
    "make_splits",
    "scan",
    "write_claims",
]
