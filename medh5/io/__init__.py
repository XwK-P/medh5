"""Format converters between ``.medh5`` and external medical imaging formats.

Submodules:

- :mod:`medh5.io.nifti` — NIfTI ⇄ medh5 (requires ``nibabel``).
- :mod:`medh5.io.dicom` — DICOM series → medh5 (requires ``pydicom``).

These submodules are import-guarded: importing :mod:`medh5.io` itself does
**not** trigger heavy optional dependencies. The functions exposed here are
imported lazily on first call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from medh5.io.dicom import from_dicom
    from medh5.io.nifti import from_nifti, import_seg_nifti, to_nifti

__all__ = ["from_dicom", "from_nifti", "import_seg_nifti", "to_nifti"]


def __getattr__(name: str) -> Any:
    if name in ("from_nifti", "import_seg_nifti", "to_nifti"):
        from medh5.io import nifti as _nifti

        return getattr(_nifti, name)
    if name == "from_dicom":
        from medh5.io import dicom as _dicom

        return _dicom.from_dicom
    raise AttributeError(f"module 'medh5.io' has no attribute {name!r}")
