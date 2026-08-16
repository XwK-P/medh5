"""Registration transforms (spec §10)."""

from __future__ import annotations

from medh5.transforms.affine import (
    AffineTransform,
    IdentityTransform,
    encode_affine,
    encode_identity,
)
from medh5.transforms.apply import (
    folding_fraction,
    jacobian_determinant,
    linear_sample,
    sample_field,
    target_registration_error,
    to_world_vectors,
)
from medh5.transforms.base import (
    EXTRAPOLATIONS,
    INTERPOLATIONS,
    SPEC_TRANSFORM_ATTRS,
    TRANSFORM_KINDS,
    VECTOR_SPACES,
    Transform,
    TransformHeader,
    frame_graph,
    open_transform,
    read_transforms,
)
from medh5.transforms.bspline import BSplineTransform, encode_bspline
from medh5.transforms.composite import CompositeTransform, encode_composite
from medh5.transforms.displacement import DisplacementTransform, encode_displacement
from medh5.transforms.resolve import (
    ChainTransform,
    InverseTransform,
    frames_of_timepoint,
    resolve_between,
)

__all__ = [
    "EXTRAPOLATIONS",
    "INTERPOLATIONS",
    "SPEC_TRANSFORM_ATTRS",
    "TRANSFORM_KINDS",
    "VECTOR_SPACES",
    "AffineTransform",
    "BSplineTransform",
    "ChainTransform",
    "CompositeTransform",
    "DisplacementTransform",
    "IdentityTransform",
    "InverseTransform",
    "Transform",
    "TransformHeader",
    "encode_affine",
    "encode_bspline",
    "encode_composite",
    "encode_displacement",
    "encode_identity",
    "folding_fraction",
    "frame_graph",
    "frames_of_timepoint",
    "jacobian_determinant",
    "linear_sample",
    "open_transform",
    "read_transforms",
    "resolve_between",
    "sample_field",
    "target_registration_error",
    "to_world_vectors",
]
