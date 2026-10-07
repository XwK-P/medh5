"""``layers``: overlapping classes packed into the fewest exclusive planes (§7.2)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations.base import VoxelAnnotation

encode_layers = _core.encode_layers


class LayersAnnotation(VoxelAnnotation):
    """Classes coloured onto layers so no two that overlap share one."""

    __slots__ = ()

    @property
    def data(self) -> Any:
        return self._handle.dataset("data")

    @property
    def layer_class_ids(self) -> npt.NDArray[np.uint16]:
        found: npt.NDArray[np.uint16] = self._handle.layer_class_ids
        return found

    @property
    def n_layers(self) -> int:
        return int(self._handle.n_layers)

    @property
    def layer_of(self) -> dict[int, int]:
        """class id -> the layer it is painted on."""
        found: dict[int, int] = self._handle.layer_of
        return found

    def layer_classes(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(layer) for layer in self._handle.layer_classes())

    def read_layer(
        self, layer: int, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[Any]:
        found: npt.NDArray[Any] = self._handle.read_layer(int(layer), roi)
        return found

    def ignore_mask(self, roi: Sequence[slice] | None = None) -> npt.NDArray[np.bool_]:
        """The in-band ignore region (the reserved id on any layer)."""
        found: npt.NDArray[np.bool_] = self._handle.ignore_mask(roi)
        return found


__all__ = ["LayersAnnotation", "encode_layers"]
