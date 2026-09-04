"""MONAI interoperability (implementation plan §2.3).

``to_metatensor`` hands MONAI a tensor with the **correct affine**, so
``Spacingd``, ``Orientationd`` and ``SaveImaged`` work unmodified.  That is the
whole job, and the only part of it that is subtle is which world convention the
affine is expressed in.

MEDH5 grids declare their ``coord_system`` (§3.1) --- usually ``LPS``, the DICOM
convention.  MONAI's transforms read `MetaKeys.SPACE`, so the honest thing is to
pass the affine through unchanged and *label* it, rather than silently
converting to RAS and hoping the consumer agrees.  Callers who need RAS ask for
it explicitly with ``space="RAS"``, and the conversion is a single sign flip on
the first two world axes --- applied to the affine, never to the voxels, because
flipping the array would make the tensor disagree with the file it came from.

The affine and metadata are built by :func:`affine_for` and :func:`meta_dict`,
which need neither MONAI nor torch, so the geometry is testable without either.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.geometry.grid import Grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.sample import Sample

SPACES = ("LPS", "RAS")

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
"""World-axis sign flip; ``RAS = _LPS_TO_RAS @ LPS`` for 3-D affines."""


def available() -> bool:
    """Whether MONAI can be imported.

    Both branches are reachable and both are measured -- the blanket
    ``pragma: no cover`` this used to carry excluded even the ``return True``
    that runs wherever the extra is installed, which is the same "covered by a
    test that never executed" failure the MONAI CI job exists to prevent.
    """
    try:
        import monai  # noqa: F401
    except ImportError:
        return False
    return True


def require_monai() -> Any:
    """Import MONAI, or explain how to install it."""
    try:
        import monai
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "MONAI is required for medh5.monai. Install it with: "
            "pip install 'medh5[monai]'"
        ) from exc
    return monai


def convert_affine(
    affine: npt.ArrayLike, *, source: str, target: str
) -> npt.NDArray[np.float64]:
    """Re-express an index→world affine in another world convention.

    Only ``LPS`` and ``RAS`` are handled, and only in 3-D: those are the two
    conventions medical tooling actually disagrees about, and inventing a
    mapping for anything else would be guessing.
    """
    matrix = np.asarray(affine, dtype=np.float64)
    for name in (source, target):
        if name not in SPACES:
            raise MEDH5ValidationError(
                f"cannot convert world convention {name!r}; known: {list(SPACES)}"
            )
    if source == target:
        return matrix
    if matrix.shape != (4, 4):
        raise MEDH5ValidationError(
            f"LPS↔RAS conversion is defined for 3-D affines; got {matrix.shape}"
        )
    return np.asarray(_LPS_TO_RAS @ matrix, dtype=np.float64)


def affine_for(
    sample: Sample, image_id: str, *, space: str | None = None, level: int = 0
) -> npt.NDArray[np.float64]:
    """The index→world affine of one image, optionally re-expressed."""
    grid = _grid_for(sample, image_id, level)
    if space is None or space == grid.coord_system:
        return grid.affine
    return convert_affine(grid.affine, source=grid.coord_system, target=space)


def _grid_for(sample: Sample, image_id: str, level: int) -> Grid:
    image = sample.images[image_id]
    grid: Grid = (
        image.level(level).grid if level and image.is_multiscale else image.grid
    )
    return grid


def meta_dict(
    sample: Sample,
    image_id: str,
    *,
    space: str | None = None,
    level: int = 0,
) -> dict[str, Any]:
    """The metadata MONAI carries alongside a tensor.

    Keys follow MONAI's own names where it has them (``affine``,
    ``original_affine``, ``spatial_shape``, ``space``) so its transforms find
    what they expect, plus the MEDH5 identifiers a writer needs to put the
    result back where it came from.
    """
    image = sample.images[image_id]
    grid = _grid_for(sample, image_id, level)
    affine = affine_for(sample, image_id, space=space, level=level)
    return {
        "affine": affine,
        "original_affine": affine,
        "spatial_shape": np.asarray(grid.spatial_shape, dtype=np.int64),
        "space": space or grid.coord_system,
        "original_channel_dim": "no_channel",
        "medh5": {
            "path": sample.path,
            "sample_id": sample.identity.sample_id,
            "subject_id": sample.identity.subject_id,
            "image_id": image_id,
            "grid_id": grid.grid_id,
            "timepoint": grid.timepoint,
            "frame_uid": grid.frame_uid,
            "modality": image.modality,
            "value_units": image.value_units,
            "coord_system": grid.coord_system,
            "units": grid.units,
        },
    }


def to_metatensor(
    sample: Sample,
    image_id: str,
    *,
    roi: Sequence[slice] | None = None,
    physical: bool = True,
    space: str | None = None,
    level: int = 0,
    dtype: npt.DTypeLike = np.float32,
) -> Any:
    """One image (or ROI of it) as a MONAI ``MetaTensor``.

    ``physical=True`` applies the rescale, because a MONAI pipeline that
    windows on HU and receives raw stored counts produces images that look
    plausible and are wrong (§4.2).

    An ROI shifts the origin: the affine returned maps *the ROI's* index
    coordinates to world, so a crop stays in the right place.  Returning the
    full-volume affine with a cropped array is the classic way a saved
    prediction lands centimetres from the anatomy it describes.
    """
    require_monai()

    image = sample.images[image_id]
    # The affine and spatial_shape in `meta` describe *this* level, so the
    # voxels have to come from it too: pairing full-resolution data with a
    # downsampled affine puts every resampled batch and saved prediction in the
    # wrong place, and nothing about the MetaTensor says so.
    source = image.level(level) if level else image
    array = np.ascontiguousarray(source.read(roi, physical=physical, dtype=dtype))
    meta = meta_dict(sample, image_id, space=space, level=level)
    if roi is not None:
        meta["affine"] = _shift_origin(meta["affine"], roi)
        meta["spatial_shape"] = np.asarray(array.shape, dtype=np.int64)
        meta["medh5"]["roi"] = [[s.start, s.stop] for s in roi]

    return _metatensor(array, meta)


def _metatensor(array: npt.NDArray[Any], meta: dict[str, Any]) -> Any:
    """``MetaTensor`` from an array and its meta, without the redundant affine.

    ``meta`` already carries ``affine``; passing it again as ``affine=`` makes
    MONAI warn that it is overwriting the one in the meta with the identical
    value, so every caller saw a warning telling them their geometry was wrong
    when it was not.
    """
    import torch
    from monai.data import MetaTensor

    return MetaTensor(torch.from_numpy(np.ascontiguousarray(array)), meta=meta)


def _shift_origin(
    affine: npt.NDArray[np.float64], roi: Sequence[slice]
) -> npt.NDArray[np.float64]:
    """Move the affine's origin to the ROI's first voxel."""
    out = np.array(affine, dtype=np.float64, copy=True)
    n = out.shape[0] - 1
    start = np.zeros(n, dtype=np.float64)
    for axis, sl in enumerate(roi[-n:] if len(roi) > n else roi):
        start[axis] = float(sl.start or 0)
    out[:n, n] = out[:n, n] + out[:n, :n] @ start
    return out


def from_metatensor(tensor: Any) -> tuple[npt.NDArray[Any], dict[str, Any]]:
    """A ``MetaTensor`` back to ``(array, geometry)`` a writer can consume.

    The geometry comes out as ``spacing``/``origin``/``direction`` in the
    tensor's own declared space, so a caller can hand it straight to
    ``SampleWriter.add_grid`` without re-deriving anything.
    """
    from medh5.geometry.affine import decompose_affine

    array = np.asarray(
        tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
    )
    meta = dict(getattr(tensor, "meta", {}) or {})
    affine = np.asarray(meta.get("affine", np.eye(array.ndim + 1)), dtype=np.float64)
    spacing, origin, direction = decompose_affine(affine)
    return array, {
        "spacing": [float(v) for v in spacing],
        "origin": [float(v) for v in origin],
        "direction": [[float(v) for v in row] for row in direction],
        "coord_system": str(meta.get("space", "LPS")),
        "medh5": meta.get("medh5", {}),
    }


def to_dict(
    sample: Sample,
    images: Sequence[str] | None = None,
    annotations: Sequence[str] = (),
    *,
    space: str | None = None,
    physical: bool = True,
) -> dict[str, Any]:
    """A MONAI dictionary-transform item: ``{image_id: MetaTensor, ...}``.

    Annotations come through as ``int64`` label tensors carrying **their own
    grid's** affine, which is what ``Spacingd(keys=["CT", "organs"],
    mode=["bilinear", "nearest"])`` needs in order to resample both
    consistently.  Class ids keep their §5.3 values and the ignore id stays
    ``65535``.

    The annotation's grid is not assumed to be the first image's.  A sample
    holding CT and PET on different grids --- the case this format exists for ---
    has annotations bound to whichever grid they were drawn on, and handing one
    of them another image's affine silently places the ground truth somewhere
    the anatomy is not.  Where an annotation's grid cannot be resolved this
    refuses rather than substituting an identity affine: an invented affine is
    indistinguishable from a measured one downstream (§3.3).
    """
    require_monai()

    wanted = list(images) if images is not None else sorted(sample.images)
    item: dict[str, Any] = {}
    for image_id in wanted:
        item[image_id] = to_metatensor(sample, image_id, physical=physical, space=space)
    for ann_id in annotations:
        ann = sample.annotations[ann_id]
        # `int64`, the dtype torch indexes and one-hots with.  `int16` wrapped
        # every class id above 32767 -- §5.3 gives them the full `uint16` range
        # -- and turned the ignore id 65535 into -1 without anything saying so.
        # The ignore id comes through as 65535; mask it out with `ignore_index`
        # rather than relying on a sign.
        planes = np.asarray(ann.labelmap(), dtype=np.int64)
        item[ann_id] = _metatensor(planes, _annotation_meta(sample, ann, space=space))
    return item


def _annotation_meta(
    sample: Sample, ann: Any, *, space: str | None = None
) -> dict[str, Any]:
    """Meta for an annotation, built from the grid the annotation is bound to."""
    grid = ann.grid
    affine = (
        grid.affine
        if space is None or space == grid.coord_system
        else convert_affine(grid.affine, source=grid.coord_system, target=space)
    )
    return {
        "affine": affine,
        "original_affine": affine,
        "spatial_shape": np.asarray(grid.spatial_shape, dtype=np.int64),
        "space": space or grid.coord_system,
        "original_channel_dim": "no_channel",
        "medh5": {
            "path": sample.path,
            "sample_id": sample.identity.sample_id,
            "subject_id": sample.identity.subject_id,
            "annotation_id": ann.ann_id,
            "grid_id": ann.grid_id,
            "coord_system": grid.coord_system,
            "units": grid.units,
        },
    }


__all__ = [
    "SPACES",
    "affine_for",
    "available",
    "convert_affine",
    "from_metatensor",
    "meta_dict",
    "require_monai",
    "to_dict",
    "to_metatensor",
]
