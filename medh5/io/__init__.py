"""Converters (plan §7).

Every converter is lazy: ``import medh5`` never pulls in nibabel, pydicom,
SimpleITK or highdicom, and a missing one produces a message naming the extra
to install rather than an ``ImportError`` from three frames down.

Each returns a :class:`~medh5.io.report.ConversionReport` recording what the
source did not determine --- the encoding chosen, the class ids minted, the
half-voxel box convention changed, the timepoint order inferred.  Those are the
steps that are invisible in the output and expensive to discover later.
"""

from __future__ import annotations

from typing import Any

from medh5.io.report import ConversionReport, Note, merge_reports

_LAZY: dict[str, str] = {
    "from_nifti": "medh5.io.nifti",
    "to_nifti": "medh5.io.nifti",
    "read_nifti": "medh5.io.nifti",
    "import_seg_nifti": "medh5.io.nifti",
    "convert_world": "medh5.io.nifti",
    "from_dicom": "medh5.io.dicom",
    "scan_dicom": "medh5.io.dicom",
    "Series": "medh5.io.dicom",
    "from_dicom_seg": "medh5.io.dicom_seg",
    "to_dicom_seg": "medh5.io.dicom_seg",
    "from_rtstruct": "medh5.io.rtstruct",
    "to_rtstruct": "medh5.io.rtstruct",
    "from_nnunetv2": "medh5.io.nnunetv2",
    "to_nnunetv2": "medh5.io.nnunetv2",
    # The module is `legacy`, not `migrate`: importing `medh5.io.legacy` would
    # bind the *module* to that name on the package and shadow the function
    # below, so `medh5.io.legacy(...)` would mean two different things
    # depending on what had been imported first.
    "migrate": "medh5.io.legacy",
    "migrate_paths": "medh5.io.legacy",
    "build_label_set": "medh5.io.legacy",
    "group_by_subject": "medh5.io.grouping",
    "SubjectGroup": "medh5.io.grouping",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = ["ConversionReport", "Note", "merge_reports", *sorted(_LAZY)]
