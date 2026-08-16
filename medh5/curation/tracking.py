"""Longitudinal joins on ``instance_id`` (spec §7.4, §11.3).

Tracking is a **join, not a structure**.  An ``instance_id`` names one physical
object within the sample, reused by every annotation that describes it, so the
lesion a radiologist followed across four visits is recovered by grouping
objects on that column --- no track table, no correspondence graph, nothing that
can disagree with the annotations it indexes.

What this module adds on top of the join is the part that is easy to get wrong:

**Absence is not a measurement.**  A lesion missing from a follow-up annotation
is *resolved* only if the annotator committed to looking for its class there.
That commitment is ``annotated_class_ids`` (§11.3), so absence resolves to one
of three states --- ``present``, ``resolved`` and ``unexamined`` --- and never
to a silent zero.  A growth curve that treats "not assessed" as volume 0 reports
a complete response that never happened, which is exactly the error the coverage
contract exists to prevent.

**A track carries one class.**  Two class ids under one instance id is almost
always a tracking mistake rather than a reclassification, and it is reported
(W909) rather than resolved by a rule the file cannot justify.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from medh5.geometry.affine import box_to_slices, voxel_volume
from medh5.geometry.grid import Grid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.annotations.base import Annotation
    from medh5.sample import Sample

PRESENT = "present"
RESOLVED = "resolved"
UNEXAMINED = "unexamined"
STATES = (PRESENT, RESOLVED, UNEXAMINED)


@dataclass(frozen=True, slots=True)
class Observation:
    """One object seen once: a row of one annotation."""

    timepoint: str
    annotation: str
    index: int
    instance_id: int
    class_id: int
    box: npt.NDArray[np.float32]
    voxel_count: int | None = None
    volume: float | None = None
    """Physical volume in the grid's ``units**S``, or ``None`` when unmeasurable."""
    units: str | None = None
    score: float | None = None
    grid: str | None = None

    @property
    def centroid(self) -> npt.NDArray[np.float64]:
        """Box centre, in whatever space the annotation stores its boxes."""
        out: npt.NDArray[np.float64] = np.asarray(self.box, dtype=np.float64).mean(
            axis=1
        )
        return out

    @property
    def extent(self) -> npt.NDArray[np.float64]:
        return (
            np.asarray(self.box, dtype=np.float64)[:, 1]
            - np.asarray(self.box, dtype=np.float64)[:, 0]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "timepoint": self.timepoint,
            "annotation": self.annotation,
            "index": self.index,
            "class_id": self.class_id,
            "box": np.asarray(self.box, dtype=float).tolist(),
            "voxel_count": self.voxel_count,
            "volume": self.volume,
            "units": self.units,
            "score": self.score,
        }

    def __repr__(self) -> str:
        volume = "-" if self.volume is None else f"{self.volume:.4g}"
        return (
            f"Observation({self.annotation}[{self.index}] @{self.timepoint}, "
            f"class={self.class_id}, volume={volume})"
        )


@dataclass(frozen=True, slots=True)
class Track:
    """Every observation of one physical object, ordered by timepoint."""

    instance_id: int
    class_ids: tuple[int, ...]
    observations: tuple[Observation, ...]
    class_key: str | None = None

    @property
    def class_id(self) -> int:
        """The object's class.  See :attr:`has_class_conflict` first."""
        return self.class_ids[0]

    @property
    def has_class_conflict(self) -> bool:
        """Whether this id carries more than one class id (W909)."""
        return len(self.class_ids) > 1

    @property
    def timepoints(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(obs.timepoint for obs in self.observations))

    def at(self, timepoint: str) -> Observation | None:
        for obs in self.observations:
            if obs.timepoint == timepoint:
                return obs
        return None

    def volume(self, timepoint: str) -> float | None:
        obs = self.at(timepoint)
        return None if obs is None else obs.volume

    @property
    def volumes(self) -> dict[str, float | None]:
        return {obs.timepoint: obs.volume for obs in self.observations}

    def relative_change(self, first: str, second: str) -> float | None:
        """``(v2 - v1) / v1`` between two timepoints, or ``None`` if unmeasured.

        ``None`` when either volume is missing --- including when the object was
        not observed.  A caller wanting "disappeared" to mean −1 must first
        establish that the class was examined at *second* (:meth:`state_at`);
        this method will not make that judgement on its own.
        """
        before = self.volume(first)
        after = self.volume(second)
        if before is None or after is None or before <= 0.0:
            return None
        return (after - before) / before

    def to_json(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "class_ids": list(self.class_ids),
            "class_key": self.class_key,
            "timepoints": list(self.timepoints),
            "observations": [o.to_json() for o in self.observations],
        }

    def __len__(self) -> int:
        return len(self.observations)

    def __iter__(self) -> Iterator[Observation]:
        return iter(self.observations)

    def __repr__(self) -> str:
        return (
            f"Track({self.instance_id}, class={self.class_key or self.class_id}, "
            f"seen at {list(self.timepoints)})"
        )


class Tracking(Mapping[int, Track]):
    """The result of joining objects on ``instance_id`` across a sample."""

    __slots__ = ("_tracks", "coverage", "timepoints")

    def __init__(
        self,
        tracks: Mapping[int, Track],
        timepoints: Sequence[str],
        coverage: Mapping[str, frozenset[int]],
    ) -> None:
        self._tracks = dict(sorted(tracks.items()))
        self.timepoints = tuple(timepoints)
        self.coverage = dict(coverage)
        """``timepoint -> class ids the annotators committed to finding`` (§11.3)."""

    def __getitem__(self, instance_id: int) -> Track:
        return self._tracks[instance_id]

    def __iter__(self) -> Iterator[int]:
        return iter(self._tracks)

    def __len__(self) -> int:
        return len(self._tracks)

    def __repr__(self) -> str:
        return f"Tracking({len(self)} tracks over {len(self.timepoints)} timepoints)"

    # -- the three-state answer -------------------------------------------

    def state_at(self, instance_id: int, timepoint: str) -> str:
        """``present``, ``resolved`` or ``unexamined`` (§7.4, §11.3).

        ``resolved`` is claimed only where the object's class is in that
        timepoint's ``annotated_class_ids``.  Everywhere else the honest answer
        is ``unexamined``: nobody looked, so the absence measures nothing.
        """
        track = self._tracks[instance_id]
        if track.at(timepoint) is not None:
            return PRESENT
        examined = self.coverage.get(timepoint, frozenset())
        return RESOLVED if any(c in examined for c in track.class_ids) else UNEXAMINED

    def states(self, instance_id: int) -> dict[str, str]:
        return {tp: self.state_at(instance_id, tp) for tp in self.timepoints}

    def is_new(self, instance_id: int) -> bool:
        """Absent-then-present: not seen at baseline, seen later."""
        states = self.states(instance_id)
        if not self.timepoints:
            return False
        return states[self.timepoints[0]] == RESOLVED and PRESENT in states.values()

    def is_resolved(self, instance_id: int) -> bool:
        """Present-then-gone, where the later visit did look for it."""
        states = self.states(instance_id)
        if len(self.timepoints) < 2:  # noqa: PLR2004 - a series needs two visits
            return False
        return states[self.timepoints[0]] == PRESENT and (
            states[self.timepoints[-1]] == RESOLVED
        )

    def is_persistent(self, instance_id: int) -> bool:
        return all(state == PRESENT for state in self.states(instance_id).values())

    # -- diagnostics -------------------------------------------------------

    def class_conflicts(self) -> dict[int, tuple[int, ...]]:
        """Instance ids carrying more than one class id (W909).

        Sample-scoped on purpose: the interesting conflict is *between* the
        annotations of two visits, where a lesion silently becomes a cyst.
        """
        return {
            instance_id: track.class_ids
            for instance_id, track in self._tracks.items()
            if track.has_class_conflict
        }

    def unexamined(self) -> dict[str, tuple[int, ...]]:
        """``timepoint -> instance ids whose class nobody committed to finding``."""
        out: dict[str, tuple[int, ...]] = {}
        for tp in self.timepoints:
            ids = tuple(i for i in self._tracks if self.state_at(i, tp) == UNEXAMINED)
            if ids:
                out[tp] = ids
        return out

    # -- reporting ---------------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "timepoints": list(self.timepoints),
            "coverage": {tp: sorted(ids) for tp, ids in sorted(self.coverage.items())},
            "tracks": [
                {**track.to_json(), "states": self.states(instance_id)}
                for instance_id, track in self._tracks.items()
            ],
            "class_conflicts": {
                str(k): list(v) for k, v in sorted(self.class_conflicts().items())
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "tracks": len(self),
            "timepoints": list(self.timepoints),
            "new": sorted(i for i in self._tracks if self.is_new(i)),
            "resolved": sorted(i for i in self._tracks if self.is_resolved(i)),
            "persistent": sorted(i for i in self._tracks if self.is_persistent(i)),
            "class_conflicts": sorted(self.class_conflicts()),
        }


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------


def carries_instance_ids(ann: Annotation) -> bool:
    """Whether an annotation declares object identity a join can trust.

    A ``boxes`` annotation without an ``instance_ids`` dataset numbers its rows
    positionally.  Joining on a row number would fabricate correspondences ---
    "object 3 at baseline" and "object 3 at follow-up" have nothing to do with
    one another --- so such annotations are skipped rather than guessed at.
    """
    if ann.kind == "instances":
        return True
    return "instance_ids" in ann.group


def build_tracks(
    sample: Sample,
    class_key: int | str | None = None,
    *,
    measure: bool = True,
) -> Tracking:
    """Join every instance-carrying annotation in *sample* on ``instance_id``."""
    wanted: int | None = None
    tracks: dict[int, list[Observation]] = {}
    coverage: dict[str, set[int]] = {tp: set() for tp in sample.timepoints.ids}
    for ann in sample.annotations.values():
        if not carries_instance_ids(ann):
            continue
        if class_key is not None:
            wanted = ann.resolve_class(class_key)
        timepoints = ann.timepoints or ("",)
        for tp in timepoints:
            coverage.setdefault(tp, set()).update(ann.annotated_class_ids)
        grid = _grid_of(sample, ann)
        for obj in _objects(ann):
            if wanted is not None and obj.class_id != wanted:
                continue
            volume, count = _measure(obj, grid, ann) if measure else (None, None)
            for tp in timepoints:
                tracks.setdefault(obj.instance_id, []).append(
                    Observation(
                        timepoint=tp,
                        annotation=ann.ann_id,
                        index=obj.index,
                        instance_id=obj.instance_id,
                        class_id=obj.class_id,
                        box=obj.box,
                        voxel_count=count,
                        volume=volume,
                        units=None if grid is None else grid.units,
                        score=obj.score,
                        grid=ann.grid_id,
                    )
                )
    order = {tp: i for i, tp in enumerate(sample.timepoints.ids)}
    built: dict[int, Track] = {}
    for instance_id, observations in tracks.items():
        observations.sort(key=lambda o: (order.get(o.timepoint, 1 << 30), o.annotation))
        class_ids = tuple(sorted({o.class_id for o in observations}))
        built[instance_id] = Track(
            instance_id=instance_id,
            class_ids=class_ids,
            observations=tuple(observations),
            class_key=_class_key(sample, class_ids[0]),
        )
    return Tracking(
        built,
        sample.timepoints.ids,
        {tp: frozenset(ids) for tp, ids in coverage.items()},
    )


def _class_key(sample: Sample, class_id: int) -> str | None:
    label_set = sample.label_set
    if label_set is None or class_id not in label_set:
        return None
    return label_set[class_id].key


def _grid_of(sample: Sample, ann: Annotation) -> Grid | None:
    gid = ann.grid_id
    return sample.grids.get(gid) if gid else None


def _objects(ann: Annotation) -> Iterator[Any]:
    """Objects of an annotation, whichever kind carries them.

    ``instances`` exposes them through a method and the §8 kinds through
    iteration; both yield :class:`~medh5.annotations.base.Instance`.
    """
    node: Any = ann
    source = node.instances() if ann.kind == "instances" else iter(node)
    yield from source


def _measure(
    obj: Any, grid: Grid | None, ann: Annotation
) -> tuple[float | None, int | None]:
    """Physical volume and voxel count of one object, where they are knowable.

    A mask gives an exact voxel count; a box alone gives the box's volume, which
    is an over-estimate of the object and is reported as such by leaving
    ``voxel_count`` unset.  In ``world`` space the box is already physical, so
    no spacing is applied --- multiplying by spacing there would scale a
    millimetre measurement by millimetres.
    """
    mask = getattr(obj, "mask", None)
    space = getattr(ann, "space", None) or "index"
    if mask is not None:
        count = int(np.count_nonzero(mask))
        if grid is None:
            return None, count
        return count * voxel_volume(grid.spacing), count
    box = np.asarray(obj.box, dtype=np.float64)
    if box.ndim != 2 or box.shape[1] != 2:  # noqa: PLR2004 - (S, 2) boxes only
        return None, None
    extent = box[:, 1] - box[:, 0]
    if space == "world":
        return float(np.prod(extent)), None
    if grid is None:
        return None, None
    if getattr(ann, "kind", "") == "instances":
        # An instances box without a mask still measures whole voxels.
        counted = int(
            np.prod([s.stop - s.start for s in box_to_slices(box)], dtype=np.int64)
        )
        return counted * voxel_volume(grid.spacing), None
    return float(np.prod(extent * np.asarray(grid.spacing, dtype=np.float64))), None


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
