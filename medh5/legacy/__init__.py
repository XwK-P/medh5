"""medh5 — HDF5 + Blosc2 multi-array format for ML workloads."""

from medh5.legacy._shared import open_shared
from medh5.legacy.chunks import optimize_chunks
from medh5.legacy.core import (
    MEDH5File,
    MEDH5Sample,
    ValidationReport,
    validate_bboxes,
)
from medh5.legacy.exceptions import (
    MEDH5Error,
    MEDH5FileError,
    MEDH5SchemaError,
    MEDH5ValidationError,
)
from medh5.legacy.integrity import VerifyResult
from medh5.legacy.meta import SampleMeta, SpatialMeta
from medh5.legacy.review import ReviewStatus

__version__ = "0.6.0"

__all__ = [
    "MEDH5Error",
    "MEDH5File",
    "MEDH5FileError",
    "MEDH5Sample",
    "MEDH5SchemaError",
    "MEDH5ValidationError",
    "ReviewStatus",
    "SampleMeta",
    "SpatialMeta",
    "ValidationReport",
    "VerifyResult",
    "__version__",
    "open_shared",
    "optimize_chunks",
    "validate_bboxes",
]
