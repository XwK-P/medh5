"""Choosing a voxel encoding by measurement (spec §7.6).

The measurement and the cost model are the format engine's; the writer, the
command line and this module call the same functions.
"""

from __future__ import annotations

from medh5 import _core

LOCALIZED_BBOX_FRACTION: float = _core.LOCALIZED_BBOX_FRACTION
SPARSE_FILL: float = _core.SPARSE_FILL

OverlapStats = _core.OverlapStats
CostModel = _core.CostModel

analyse = _core.analyse
greedy_colour = _core.greedy_colour
layers_from_colouring = _core.layers_from_colouring
label_dtype_size = _core.label_dtype_size
cost_model = _core.cost_model
select_encoding = _core.select_encoding

__all__ = [
    "LOCALIZED_BBOX_FRACTION",
    "SPARSE_FILL",
    "CostModel",
    "OverlapStats",
    "analyse",
    "cost_model",
    "greedy_colour",
    "label_dtype_size",
    "layers_from_colouring",
    "select_encoding",
]
