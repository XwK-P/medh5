"""Patient coordinates and frames of reference, for the DICOM converters.

DICOM places everything --- a SEG's frames, an RTSTRUCT's contours, a source
image's first pixel --- in *patient coordinates*: millimetres, LPS, in a frame
of reference.  A grid's world coordinates are in its own units and convention
(§3.2, §3.5), so every DICOM converter maps between the two, and compares
frames of reference before it does (§3.4).  The SEG and RTSTRUCT converters
each kept a copy of the mapping; this is the one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.geometry import COORD_SYSTEMS, MM_PER_UNIT


def patient_mapping(grid: Any, what: str) -> tuple[npt.NDArray[np.float64], float]:
    """The world-axis signs and the millimetres per grid unit relating
    *grid*'s world coordinates to DICOM patient coordinates (LPS, mm):
    ``patient = signs * scale * world`` and ``world = signs * patient /
    scale``, the signs being their own inverse.

    A grid in another convention, or in pixels, has no patient coordinates,
    and is refused (E414); *what* names what is in patient coordinates, for
    the message.
    """
    if grid.coord_system not in COORD_SYSTEMS:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} is in {grid.coord_system!r}; {what} are "
            "patient coordinates (LPS), which only an LPS or RAS grid relates to",
            code="E414",
        )
    scale = MM_PER_UNIT.get(grid.units)
    if scale is None:
        raise MEDH5ValidationError(
            f"grid {grid.grid_id!r} is in {grid.units!r}, uncalibrated, which "
            "relates to no patient coordinate in millimetres",
            code="E414",
        )
    flip = 1.0 if grid.coord_system == "LPS" else -1.0
    return np.array([flip, flip, 1.0]), scale


def frames_agree(
    seg_frame: str | None, grid_frame: str | None, salt: str | None = None
) -> bool | None:
    """Whether a SEG's frame of reference is a grid's; ``None`` when either
    states none, so there is nothing to compare.

    A grid of a sample ``medh5 scrub`` pseudonymised holds its frame's
    pseudonym, which the SEG's own UID never equals; with the scrub's *salt*
    the SEG's is compared as the scrub would have written it --- an exact
    correspondence, not an assumption.
    """
    if not seg_frame or not grid_frame:
        return None
    if seg_frame == grid_frame:
        return True
    if salt is not None:
        from medh5.curation.scrub import pseudonymise

        return pseudonymise(seg_frame, salt) == grid_frame
    return False


def _require_same_frame(
    frame: str | None, grid: Any, salt: str | None, *, source: str = "the SEG"
) -> None:
    """Refuse a grid whose known frame of reference is not *frame* (§3.4).

    Equal numbers in two frames are no correspondence: a grid with another
    known frame and the same geometry received every voxel, from an explicit
    ``grid=`` without a word and from automatic matching on a recorded guess.
    Relating two frames is registration, which the importers do not apply.
    Where either side states no frame, there is nothing to compare, and
    placement rests on geometry, as it always has; the import records that
    as a guess.  *source* names what was drawn in *frame*, for the message.
    """
    if frames_agree(frame, grid.frame_uid, salt) is not False:
        return
    hint = _salt_hint(grid.frame_uid, salt, "The grid's", f"{source}'s frame")
    raise MEDH5ValidationError(
        f"{source} was drawn in frame of reference {frame!r} and grid "
        f"{grid.grid_id!r} is in {grid.frame_uid!r}: equal geometry in two "
        "frames is no correspondence, and placing it there needs a "
        f"registration, which this importer does not apply.{hint}",
        code="E414",
    )


def _salt_hint(frame: str | None, salt: str | None, whose: str, compared: str) -> str:
    """What a frame-of-reference refusal adds when *frame* --- *whose* frame
    --- is a pseudonym ``medh5 scrub`` wrote and no *salt* was given: the
    scrub's salt compares *compared* as the scrub wrote it.  Empty otherwise.
    """
    from medh5.curation.scrub import PSEUDONYM_PREFIX

    if salt is not None or not str(frame).startswith(PSEUDONYM_PREFIX):
        return ""
    return (
        f" {whose} frame is a `medh5 scrub` pseudonym: pass the scrub's salt "
        "as frame_salt= (`--frame-salt` on the command line) to compare "
        f"{compared} as the scrub wrote it."
    )


def _one_frame(datasets: Sequence[Any]) -> str:
    """The frame of reference every source image shares: an RTSTRUCT's
    contours are in one, and only the first image's was read."""
    frames = {str(getattr(d, "FrameOfReferenceUID", "") or "") for d in datasets}
    if "" in frames or len(frames) != 1:
        stated = sorted(f or "none" for f in frames)
        raise MEDH5ValidationError(
            f"the source images state the frames of reference {stated}; an "
            "RTSTRUCT's contours are in one, which every source image must state",
            code="E414",
        )
    return frames.pop()


__all__ = ["frames_agree", "patient_mapping"]
