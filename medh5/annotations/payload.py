"""The intermediate an annotation encoder produces before anything touches HDF5.

Keeping encoders pure --- arrays in, arrays out --- is what lets the transcoding
matrix be tested without a file, and what lets the writer decide chunking and
codecs in one place instead of in every encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt


@dataclass(slots=True)
class AnnotationPayload:
    """Datasets and kind-specific attributes for one encoded annotation."""

    kind: str
    datasets: dict[str, npt.NDArray[Any]] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    stacked_axes: int = 0
    """Leading axes of ``data`` that must get chunk extent 1 (spec §14.1)."""

    class_ids: tuple[int, ...] = ()

    @property
    def data(self) -> npt.NDArray[Any]:
        return self.datasets["data"]

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.datasets.values())

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "datasets": {
                name: {"shape": list(a.shape), "dtype": a.dtype.str}
                for name, a in self.datasets.items()
            },
            "nbytes": self.nbytes,
        }


__all__ = ["AnnotationPayload"]
