"""``Sample`` and ``SampleWriter`` --- where the pieces become a file.

A sample is **one subject at one or more timepoints**.  Reading is lazy and
timepoint-aware; writing is a builder that validates as it goes and commits
atomically (spec §14.4).

A :class:`Sample` is a read-only view over the format engine's reader.  It is
memoised the way the file is immutable to it: ``amend`` is copy-on-write and
replaces the inode, so an open sample never sees an edit --- reopen to read
one.

The write model: ``create`` writes to a sibling temporary file and atomically
replaces the target on ``commit``, so a reader never sees a half-written
sample and a crash leaves the previous file intact.  ``amend`` is
copy-on-write: it builds a new file from the old one and replaces it, because
HDF5 does not reclaim space on delete --- and anything holding the old file
open keeps reading the old inode.  Every ``add_*`` validates as it goes, and
``commit`` refuses to write a file the validator would reject.  The builder is
the format engine's ``SampleWriter``::

    with medh5.create("case.medh5", sample_id="case-001") as w:
        w.add_grid("ct", shape=(64, 64, 32), spacing=(0.8, 0.8, 2.0))
        w.add_image("CT", volume, grid="ct", modality="CT")
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5.annotations import Annotation, open_annotation
from medh5.curation import Cohort, Identity, Timeline, Timepoint, Tracking
from medh5.document import SampleDocument
from medh5.errors import MEDH5ValidationError
from medh5.geometry import Grid
from medh5.image import Image
from medh5.labels import LabelSet
from medh5.storage import SamplingIndex
from medh5.transforms import Transform, wrap_transform

FORMAT_VERSION: str = _core.FORMAT_VERSION
PROFILES: tuple[str, ...] = _core.PROFILES

ROOT_DIGEST_ATTRS: tuple[str, ...] = _core.ROOT_DIGEST_ATTRS
"""Root attributes covered by ``content_id``.

``created`` and ``generator`` are deliberately excluded: two byte-identical
samples written an hour apart must share a ``content_id``, or it is not a
content address and cannot be used as a cache or dedup key (spec §13.2).
"""

MANAGED_ROOT_ATTRS: tuple[str, ...] = _core.MANAGED_ROOT_ATTRS
"""Root attributes ``commit`` writes itself; an amend does not copy them."""

STANDARD_GROUPS: tuple[str, ...] = _core.STANDARD_GROUPS

SampleWriter = _core.SampleWriter
create = _core.create
amend = _core.amend


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------


class _Collection(Mapping[str, Any]):
    """A read-only mapping with a helpful ``KeyError``."""

    __slots__ = ("_items", "_what")

    def __init__(self, what: str, items: Mapping[str, Any]) -> None:
        self._what = what
        self._items = dict(items)

    def __getitem__(self, key: str) -> Any:
        try:
            return self._items[key]
        except KeyError:
            raise KeyError(
                f"no {self._what} {key!r}; available: {sorted(self._items)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"<{self._what}s: {sorted(self._items)}>"


class ImageCollection(_Collection):
    """The sample's images, with timepoint- and modality-aware views."""

    __slots__ = ()

    def by_timepoint(self, timepoint_id: str) -> tuple[str, ...]:
        return tuple(
            name
            for name, image in self._items.items()
            if image.timepoint == timepoint_id
        )

    def by_modality(self, modality: str) -> tuple[str, ...]:
        return tuple(
            name for name, image in self._items.items() if image.modality == modality
        )

    def on_grid(self, grid_id: str) -> tuple[str, ...]:
        return tuple(
            name for name, image in self._items.items() if image.grid_id == grid_id
        )


class AnnotationCollection(_Collection):
    """The sample's annotations, with task- and timepoint-aware views."""

    __slots__ = ()

    def by_task(self, task: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if a.task == task)

    def by_kind(self, kind: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if a.kind == kind)

    def by_timepoint(self, timepoint_id: str) -> tuple[Annotation, ...]:
        return tuple(a for a in self._items.values() if timepoint_id in a.timepoints)

    def spanning(self) -> tuple[Annotation, ...]:
        """Annotations covering more than one timepoint --- change labels."""
        return tuple(a for a in self._items.values() if len(a.timepoints) > 1)


class TimepointView:
    """Everything in the sample that belongs to one timepoint."""

    __slots__ = ("_sample", "timepoint")

    def __init__(self, sample: Sample, timepoint: Timepoint) -> None:
        self._sample = sample
        self.timepoint = timepoint

    @property
    def id(self) -> str:
        return str(self.timepoint.id)

    @property
    def grids(self) -> dict[str, Grid]:
        return {
            gid: g
            for gid, g in self._sample.grids.items()
            if g.timepoint == self.timepoint.id
        }

    @property
    def images(self) -> ImageCollection:
        names = self._sample.images.by_timepoint(self.timepoint.id)
        return ImageCollection("image", {n: self._sample.images[n] for n in names})

    @property
    def annotations(self) -> AnnotationCollection:
        found = self._sample.annotations.by_timepoint(self.timepoint.id)
        return AnnotationCollection("annotation", {a.ann_id: a for a in found})

    def __repr__(self) -> str:
        return (
            f"TimepointView({self.timepoint.id!r}, "
            f"{len(self.images)} images, {len(self.annotations)} annotations)"
        )


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


class Sample:
    """A read-only view of one sample root (a file, or a collection member)."""

    __slots__ = (
        "_annotations",
        "_document",
        "_grids",
        "_handle",
        "_images",
        "_index",
        "_resolved",
        "_transforms",
        "path",
    )

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.path: str | None = handle.path
        self._document: SampleDocument | None = None
        self._grids: dict[str, Grid] | None = None
        self._images: ImageCollection | None = None
        self._annotations: AnnotationCollection | None = None
        self._transforms: _Collection | None = None
        self._index: dict[str, SamplingIndex] | None = None
        self._resolved: dict[tuple[str, str], Transform | None] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Sample:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the file.  Safe to call twice."""
        self._handle.close()

    @property
    def is_open(self) -> bool:
        return bool(self._handle.is_open)

    def __repr__(self) -> str:
        return str(self._handle.repr())

    @property
    def root(self) -> Any:
        """The sample's root group, for inspection (attributes, members)."""
        return self._handle.root

    # -- document ----------------------------------------------------------

    @property
    def document(self) -> SampleDocument:
        if self._document is None:
            self._document = self._handle.document()
        return self._document

    @property
    def identity(self) -> Identity:
        found: Identity = self.document.identity
        return found

    @property
    def cohort(self) -> Cohort:
        found: Cohort = self.document.cohort
        return found

    @property
    def timepoints(self) -> Timeline:
        found: Timeline = self.document.timepoints
        return found

    @property
    def label_set(self) -> LabelSet | None:
        found: LabelSet | None = self.document.label_set
        return found

    @property
    def version(self) -> str:
        return str(self._handle.version)

    @property
    def kind(self) -> str:
        return str(self._handle.kind)

    @property
    def profiles(self) -> frozenset[str]:
        return frozenset(self._handle.profiles)

    @property
    def content_id(self) -> str | None:
        found: str | None = self._handle.content_id
        return found

    # -- objects -----------------------------------------------------------

    @property
    def grids(self) -> dict[str, Grid]:
        """The sample's grids, with §3.7's implicit timepoint resolved.

        A grid MUST name its timepoint only when the sample declares more than
        one (§3.7 rule 2); with exactly one declared, the grid belongs to that
        one, and every timepoint-aware reader takes its answer from here.
        """
        if self._grids is None:
            self._grids = dict(self._handle.grids())
        return self._grids

    @property
    def reference_grid(self) -> Grid:
        """``grids/ref`` when present, else the grid of the first image (§3.2)."""
        found: Grid = self._handle.reference_grid()
        return found

    @property
    def images(self) -> ImageCollection:
        if self._images is None:
            self._images = ImageCollection(
                "image",
                {
                    name: Image(self._handle.image(name))
                    for name in sorted(self._handle.image_ids())
                },
            )
        return self._images

    @property
    def annotations(self) -> AnnotationCollection:
        if self._annotations is None:
            self._annotations = AnnotationCollection(
                "annotation",
                {
                    name: open_annotation(self._handle.annotation(name))
                    for name in sorted(self._handle.annotation_ids())
                },
            )
        return self._annotations

    def ignore_region(
        self, ann_id: str, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.bool_]:
        """The §7.7 ignore region of a voxel annotation, under any encoding.

        ``labelmap`` and ``layers`` may keep it in band; every encoding may
        instead name a sibling `mask` annotation in ``header.ignore_mask``.
        This reads both.  All ``False`` where the annotation declares no region.
        """
        found: npt.NDArray[np.bool_] = self._handle.ignore_region(ann_id, roi)
        return found

    def valid_region(
        self, image_id: str, roi: Sequence[slice] | None = None
    ) -> npt.NDArray[np.bool_]:
        """Where an image holds data (§4.4): its ``valid_mask``, or everywhere."""
        found: npt.NDArray[np.bool_] = self._handle.valid_region(image_id, roi)
        return found

    @property
    def index(self) -> dict[str, SamplingIndex]:
        if self._index is None:
            self._index = {
                name: self._handle.index(name) for name in self._handle.index_ids()
            }
        return self._index

    @property
    def fresh_indices(self) -> frozenset[str]:
        """Index entries whose ``source_digest`` still matches their source
        (§13.3).  A stale entry is ignored, not trusted."""
        return frozenset(self._handle.fresh_indices())

    @property
    def transforms(self) -> _Collection:
        if self._transforms is None:
            self._transforms = _Collection(
                "transform",
                {
                    name: wrap_transform(self._handle.transform(name))
                    for name in self._handle.transform_ids()
                },
            )
        return self._transforms

    def transform_between(self, source: str, target: str) -> Transform | None:
        """The transform relating two timepoints, grids or frames (spec §10).

        A key is read as a timepoint first, then as a grid, then as a frame
        uid; one that is none of the three raises ``KeyError``.  Returns
        ``None`` when the two already share a frame --- or when no transform
        relates them: geometry is never invented.
        """
        found = self._handle.transform_between(source, target)
        return None if found is None else wrap_transform(found)

    def _frames_for(self, key: str) -> tuple[str, ...]:
        """The frame uids a key names, read as :meth:`transform_between` reads
        it: a timepoint (every grid of the visit), then a grid, then a frame."""
        return tuple(self._handle.frames_for(key))

    def resolve_frames(self, from_frame: str, to_frame: str) -> Transform | None:
        """The transform relating two frame uids, resolved once per handle.

        A paired dataset asks the same question for every item; the answer
        (and the object carrying it) is the same each time.
        """
        key = (from_frame, to_frame)
        if key not in self._resolved:
            found = self._handle.resolve_frames(from_frame, to_frame)
            self._resolved[key] = None if found is None else wrap_transform(found)
        return self._resolved[key]

    # -- timepoints --------------------------------------------------------

    def at(self, timepoint: str | int) -> TimepointView:
        """A timepoint-scoped view of the whole sample."""
        return TimepointView(self, self.timepoints[timepoint])

    @property
    def is_longitudinal(self) -> bool:
        return bool(self.timepoints.is_longitudinal)

    def tracks(
        self, class_key: int | str | None = None, *, measure: bool = True
    ) -> Tracking:
        """Join ``instance_id`` across timepoints --- the tracking operation (§7.4).

        Whether an absence means *resolved* or *unexamined* is answered by
        ``annotated_class_ids`` (§11.3), which is why the result carries the
        coverage it needs.
        """
        found: Tracking = self._handle.tracks(class_key, measure=measure)
        return found

    # -- integrity ---------------------------------------------------------

    def attr_name_map(self) -> dict[str, tuple[str, ...]]:
        """Object path -> spec-defined attribute names, for ``content_id``."""
        found: dict[str, tuple[str, ...]] = self._handle.attr_name_map()
        return found

    def verify(self, partial: Sequence[str] | None = None) -> Any:
        from medh5.integrity import VerifyResult

        return VerifyResult(
            **self._handle.verify(None if partial is None else list(partial))
        )

    def compute_content_id(self) -> str:
        return str(self._handle.compute_content_id())

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        found: dict[str, Any] = self._handle.summary()
        return found


FRAME_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grids", ("frame_uid",)),
    ("annotations", ("frame_uid",)),
    ("transforms", ("from_frame", "to_frame")),
)


def annotation_id(reference: str) -> str:
    """An annotation id from a reference that may be written as a path.

    §6.2 says ``derived_from`` holds annotation ids; the RTSTRUCT importer
    wrote ``annotations/<id>`` for a release, so both spellings name one thing.
    """
    return str(_core.annotation_id(reference))


def open_sample(path: str | os.PathLike[str], mode: str = "r") -> Sample:
    """Open a ``.medh5`` sample file, read-only.

    ``mode`` exists for the callers that spelled out ``"r"``; edits go through
    :func:`amend`, which is copy-on-write.
    """
    if mode != "r":
        raise MEDH5ValidationError(
            f"open() is read-only; use create() or amend() to write, not {mode!r}"
        )
    return Sample(_core.open_sample(os.fspath(path)))


__all__ = [
    "FORMAT_VERSION",
    "FRAME_ATTRS",
    "MANAGED_ROOT_ATTRS",
    "PROFILES",
    "ROOT_DIGEST_ATTRS",
    "STANDARD_GROUPS",
    "AnnotationCollection",
    "ImageCollection",
    "Sample",
    "SampleWriter",
    "TimepointView",
    "amend",
    "annotation_id",
    "create",
    "open_sample",
]
