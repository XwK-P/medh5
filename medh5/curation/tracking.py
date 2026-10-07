"""Longitudinal joins on ``instance_id`` (spec §7.4, §11.3).

Tracking is a **join, not a structure**.  An ``instance_id`` names one physical
object within the sample, reused by every annotation that describes it, so the
lesion a radiologist followed across four visits is recovered by grouping
objects on that column --- no track table, no correspondence graph, nothing that
can disagree with the annotations it indexes.

What the join adds is the part that is easy to get wrong:

**Absence is not a measurement.**  A lesion missing from a follow-up annotation
is *resolved* only if the annotator committed to looking for its class there.
That commitment is ``annotated_class_ids`` (§11.3), so absence resolves to one
of three states --- ``present``, ``resolved`` and ``unexamined`` --- and never
to a silent zero.

**A track carries one class.**  Two class ids under one instance id is almost
always a tracking mistake rather than a reclassification, and it is reported
(W909) rather than resolved by a rule the file cannot justify.

The join is the format engine's; ``Sample.tracks()`` runs it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from medh5 import _core

PRESENT: str = _core.PRESENT
RESOLVED: str = _core.RESOLVED
UNEXAMINED: str = _core.UNEXAMINED
STATES: tuple[str, ...] = _core.STATES

Observation = _core.Observation
Track = _core.Track
Tracking = _core.Tracking
Mapping.register(Tracking)

carries_instance_ids = _core.carries_instance_ids


def build_tracks(
    sample: Any, class_key: int | str | None = None, *, measure: bool = True
) -> Any:
    """Join ``instance_id`` across a sample's timepoints (``Sample.tracks``)."""
    return sample.tracks(class_key, measure=measure)


__all__ = [
    "PRESENT",
    "RESOLVED",
    "STATES",
    "UNEXAMINED",
    "Observation",
    "Track",
    "Tracking",
    "build_tracks",
    "carries_instance_ids",
]
