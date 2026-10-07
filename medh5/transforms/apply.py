"""Evaluating fields: interpolation, Jacobians, TRE (spec §10.4--§10.6).

The numerics are the format engine's --- linear and cubic interpolation match
SciPy's ``map_coordinates``, and the Jacobian matches NumPy's gradient ---
so every frontend evaluates a stored field identically.
"""

from __future__ import annotations

from medh5 import _core

EXTRAPOLATIONS: tuple[str, ...] = _core.EXTRAPOLATIONS

inside_extent = _core.inside_extent
refuse_outside = _core.refuse_outside
linear_sample = _core.linear_sample
cubic_sample = _core.cubic_sample
sample_field = _core.sample_field
linear_part = _core.linear_part
to_world_vectors = _core.to_world_vectors
jacobian_determinant = _core.jacobian_determinant
folding_fraction = _core.folding_fraction
target_registration_error = _core.target_registration_error

__all__ = [
    "EXTRAPOLATIONS",
    "cubic_sample",
    "folding_fraction",
    "inside_extent",
    "jacobian_determinant",
    "linear_part",
    "linear_sample",
    "refuse_outside",
    "sample_field",
    "target_registration_error",
    "to_world_vectors",
]
