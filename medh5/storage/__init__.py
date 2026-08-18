"""Storage-layer concerns: chunking, codecs and derived sampling indices (spec §14)."""

from __future__ import annotations

from medh5.storage.chunking import optimize_chunks, spatial_chunk_for
from medh5.storage.codecs import (
    PROFILES,
    CodecProfile,
    dataset_kwargs,
    describe_filters,
    resolve_profile,
)

__all__ = [
    "PROFILES",
    "CodecProfile",
    "dataset_kwargs",
    "describe_filters",
    "optimize_chunks",
    "resolve_profile",
    "spatial_chunk_for",
]
