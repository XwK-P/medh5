"""medh5 — HDF5 + Blosc2 multi-array format for ML workloads."""

from medh5.chunks import optimize_chunks
from medh5.core import MEDH5File, MEDH5Sample
from medh5.exceptions import (
    MEDH5Error,
    MEDH5FileError,
    MEDH5SchemaError,
    MEDH5ValidationError,
)
from medh5.meta import SampleMeta, SpatialMeta

__all__ = [
    "MEDH5File",
    "MEDH5Sample",
    "MEDH5Error",
    "MEDH5FileError",
    "MEDH5SchemaError",
    "MEDH5ValidationError",
    "SampleMeta",
    "SpatialMeta",
    "optimize_chunks",
]
