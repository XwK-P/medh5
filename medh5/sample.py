"""``Sample`` and ``SampleWriter`` --- where the pieces become a file.

A sample is **one subject at one or more timepoints**.  Reading is lazy and
timepoint-aware; writing is a builder that validates as it goes and commits
atomically.

The write model is spec §14.4: create writes to a sibling temporary file and
``os.replace``s it, so a reader never sees a half-written sample and a crash
leaves the previous file intact.  Amend is copy-on-write by default, because
HDF5 does not reclaim space on ``del`` --- repeated in-place add/remove
monotonically bloats a file and fragments its chunk index.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any

import h5py

from medh5._hdf5 import (
    as_str,
    as_str_tuple,
    open_h5,
)
from medh5.annotations.base import (
    SPEC_ANNOTATION_ATTRS,
    Annotation,
    open_annotation,
)
from medh5.curation.identity import Cohort, Identity
from medh5.curation.timeline import Timeline, Timepoint
from medh5.curation.tracking import Tracking
from medh5.document import (
    SampleDocument,
    read_document,
)
from medh5.errors import (
    MEDH5Error,
    MEDH5FileError,
    MEDH5ValidationError,
    MEDH5VersionError,
)
from medh5.geometry.grid import SPEC_GRID_ATTRS, Grid, read_grids
from medh5.image import SPEC_IMAGE_ATTRS, Image
from medh5.labels.labelset import LabelSet
from medh5.storage.index import (
    SamplingIndex,
    read_indices,
)
from medh5.transforms.base import (
    SPEC_TRANSFORM_ATTRS,
    Transform,
    read_transforms,
)
from medh5.transforms.resolve import frames_of_timepoint, resolve_between

FORMAT_VERSION = "1.0"
PROFILES = (
    "core",
    "seg",
    "det",
    "cls",
    "reg",
    "curation",
    "multiscale",
    "training",
    "longitudinal",
)

ROOT_DIGEST_ATTRS = ("medh5_version", "medh5_kind", "medh5_profiles")
"""Root attributes covered by ``content_id``.

``created`` and ``generator`` are deliberately excluded: two byte-identical
samples written an hour apart must share a ``content_id``, or it is not a
content address and cannot be used as a cache or dedup key (spec §13.2).
"""


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
        return self.timepoint.id

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
    """A read-only view of one sample root."""

    __slots__ = (
        "_annotations",
        "_document",
        "_fresh_indices",
        "_grids",
        "_handle",
        "_images",
        "_index",
        "_owns_handle",
        "_resolved",
        "_transforms",
        "path",
        "root",
    )

    def __init__(
        self,
        root: h5py.Group,
        *,
        handle: h5py.File | None = None,
        owns_handle: bool = False,
        path: str | None = None,
    ) -> None:
        self.root = root
        self._handle = handle
        self._owns_handle = owns_handle
        self.path = path
        self._document: SampleDocument | None = None
        self._grids: dict[str, Grid] | None = None
        self._images: ImageCollection | None = None
        self._annotations: AnnotationCollection | None = None
        self._transforms: _Collection | None = None
        self._index: dict[str, SamplingIndex] | None = None
        self._fresh_indices: frozenset[str] | None = None
        self._resolved: dict[tuple[str, str], Transform | None] = {}

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> Sample:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_handle and self._handle is not None:
            self._handle.close()
            self._handle = None

    def __repr__(self) -> str:
        return (
            f"Sample({self.identity.sample_id!r}, {len(self.timepoints)} timepoints, "
            f"{len(self.images)} images, {len(self.annotations)} annotations)"
        )

    # -- document ----------------------------------------------------------

    @property
    def document(self) -> SampleDocument:
        if self._document is None:
            self._document = read_document(self.root)
        return self._document

    @property
    def identity(self) -> Identity:
        return self.document.identity

    @property
    def cohort(self) -> Cohort:
        return self.document.cohort

    @property
    def timepoints(self) -> Timeline:
        return self.document.timepoints

    @property
    def label_set(self) -> LabelSet | None:
        return self.document.label_set

    @property
    def version(self) -> str:
        return as_str(self.root.attrs["medh5_version"])

    @property
    def kind(self) -> str:
        return as_str(self.root.attrs.get("medh5_kind", "sample"))

    @property
    def profiles(self) -> frozenset[str]:
        attrs = self.root.attrs
        if "medh5_profiles" not in attrs:
            return frozenset({"core"})
        return frozenset(as_str_tuple(attrs["medh5_profiles"]))

    @property
    def content_id(self) -> str | None:
        value = self.root.attrs.get("content_id")
        return as_str(value) if value is not None else None

    # -- objects -----------------------------------------------------------

    @property
    def grids(self) -> dict[str, Grid]:
        """The sample's grids, with §3.7's implicit timepoint resolved.

        A grid **MUST** name its timepoint only when the sample declares more
        than one (§3.7 rule 2); with exactly one declared, the attribute is
        optional and the grid belongs to that one.  Every timepoint-aware reader
        --- ``Image.timepoint``, ``Annotation.timepoints``, ``at()``,
        ``tracks()``, the frame lookup behind ``transform_between`` --- takes
        its answer from the grid, so the resolution happens here, once.  Left
        unresolved, a single-visit file whose converter omitted the attribute
        (every NIfTI and nnU-Net import did) had an empty ``at("tp0")``, blank
        ``medh5 timeline`` columns, and a ``tracks()`` that called every lesion
        *unexamined* at the only visit there was.  The writer's own view is the
        file's attributes, untouched.
        """
        if self._grids is None:
            self._grids = self._with_implicit_timepoint(read_grids(self.root))
        return self._grids

    def _with_implicit_timepoint(self, grids: dict[str, Grid]) -> dict[str, Grid]:
        try:
            declared = self.timepoints.ids
        except MEDH5Error:
            # A document that does not parse is its own diagnostic; the grids
            # are still readable as stored.
            return grids
        if len(declared) != 1:
            return grids
        only = declared[0]
        return {
            gid: replace(grid, timepoint=only) if grid.timepoint is None else grid
            for gid, grid in grids.items()
        }

    @property
    def reference_grid(self) -> Grid:
        """``grids/ref`` when present, else the grid of the first image (§3.2)."""
        grids = self.grids
        if "ref" in grids:
            return grids["ref"]
        names = sorted(self.images)
        if names:
            first: Image = self.images[names[0]]
            return first.grid
        if not grids:
            raise MEDH5ValidationError(
                f"{self.path or '<memory>'} declares no grids, so it has no "
                "reference grid",
                code="E111",
            )
        return grids[sorted(grids)[0]]

    @property
    def images(self) -> ImageCollection:
        if self._images is None:
            node = self.root.get("images")
            items = (
                {name: Image(name, node[name], self.grids) for name in sorted(node)}
                if node is not None
                else {}
            )
            self._images = ImageCollection("image", items)
        return self._images

    @property
    def annotations(self) -> AnnotationCollection:
        if self._annotations is None:
            node = self.root.get("annotations")
            items: dict[str, Annotation] = {}
            if node is not None:
                for name in sorted(node):
                    items[name] = open_annotation(
                        name, node[name], self.grids, self.label_set
                    )
            self._annotations = AnnotationCollection("annotation", items)
        return self._annotations

    @property
    def index(self) -> dict[str, SamplingIndex]:
        if self._index is None:
            self._index = read_indices(self.root)
        return self._index

    @property
    def fresh_indices(self) -> frozenset[str]:
        """Index entries whose ``source_digest`` still matches their source (§13.3).

        A stale entry is not a file error --- readers must ignore it and fall
        back to the annotation itself, because counts and coordinates for the
        mask as it was before somebody edited it are wrong rather than merely
        old.  Computed once per handle: a reader cannot edit the file it holds
        open, and re-digesting every annotation per patch draw would cost more
        than the index saves.
        """
        if self._fresh_indices is None:
            from medh5.integrity.verify import stale_index_entries

            stale = set(stale_index_entries(self.root))
            self._fresh_indices = frozenset(
                name for name in self.index if name not in stale
            )
        return self._fresh_indices

    @property
    def transforms(self) -> _Collection:
        """The file's transforms, read once.

        Memoized like ``grids``/``images``/``annotations``, and for the same
        reason: a ``Sample`` is a read-only view, and ``amend`` is copy-on-write
        --- it replaces the inode, so an open handle never sees an edit.
        Rebuilding this per access re-opened every transform group on every
        ``transform_between``, which is once per pair per training item.
        """
        if self._transforms is None:
            self._transforms = _Collection(
                "transform", read_transforms(self.root, self.grids)
            )
        return self._transforms

    def transform_between(self, source: str, target: str) -> Transform | None:
        """The transform relating two timepoints or two frames (spec §10).

        Resolution walks the frame graph, not transform names: a file may relate
        baseline to follow-up with one affine, a composite, or an affine plus a
        deformable refinement, and a consumer should not have to know which.
        Returns ``None`` when the two already share a frame --- nothing to apply.

        **A key is read as a timepoint first, then as a grid, then as a frame
        uid.**  Uniqueness is scoped to the group (§2.3), so a grid MAY be named
        after the visit it belongs to and a conforming file can have both; where
        it does, the timepoint reading wins and the answer covers *every* frame
        of that visit rather than the one grid.  That is only ambiguous when a
        visit spans several frames --- a CT and a PET, say --- but there it is
        the difference between a registration this pair owns and one belonging
        to another modality.  Pass ``grids[gid].frame_uid`` to ask about one
        grid's frame specifically; a frame uid is matched last, so it answers
        for that frame alone.
        """
        pairs = [
            (a, b) for a in self._frames_for(source) for b in self._frames_for(target)
        ]
        for a, b in pairs:
            if a == b:
                return None
            found = self.resolve_frames(a, b)
            if found is not None:
                return found
        return None

    def resolve_frames(self, from_frame: str, to_frame: str) -> Transform | None:
        """The transform relating two frame uids, resolved once per handle.

        No name is interpreted: the arguments are frame uids, which is what a
        loader holding two grids wants to ask about (§10.2).  The answer is
        memoised because a ``Sample`` is a read-only view --- ``amend`` replaces
        the inode --- and a paired dataset asks the same question once per
        training item.  An ambiguous pair raises every time; refusals are not
        cached.
        """
        if from_frame == to_frame:
            return None
        key = (from_frame, to_frame)
        if key not in self._resolved:
            self._resolved[key] = resolve_between(
                dict(self.transforms), from_frame, to_frame
            )
        return self._resolved[key]

    def _frames_for(self, key: str) -> tuple[str, ...]:
        """Frames named by a timepoint id, a grid id, or a frame uid itself."""
        if key in self.timepoints.ids:
            return frames_of_timepoint(self.grids, key)
        if key in self.grids:
            frame = self.grids[key].frame_uid
            return (frame,) if frame else ()
        return (key,)

    # -- timepoints --------------------------------------------------------

    def at(self, timepoint: str | int) -> TimepointView:
        """A timepoint-scoped view of the whole sample."""
        return TimepointView(self, self.timepoints[timepoint])

    @property
    def is_longitudinal(self) -> bool:
        return self.timepoints.is_longitudinal

    def tracks(
        self, class_key: int | str | None = None, *, measure: bool = True
    ) -> Tracking:
        """Join ``instance_id`` across timepoints --- the tracking operation (§7.4).

        A lesion that persisted appears under several timepoints; one that
        vanished appears under fewer than the sample declares.  Whether that
        absence means *resolved* or *unexamined* is answered by
        ``annotated_class_ids`` (§11.3), which is why the result is a
        :class:`~medh5.curation.tracking.Tracking` rather than a plain dict:
        the coverage it needs to answer that question travels with the join.
        """
        from medh5.curation.tracking import build_tracks

        return build_tracks(self, class_key, measure=measure)

    # -- integrity ---------------------------------------------------------

    def attr_name_map(self) -> dict[str, tuple[str, ...]]:
        """Object path -> spec-defined attribute names, for ``content_id``."""
        return attr_name_map_of(self.root)

    def verify(self, partial: Sequence[str] | None = None) -> Any:
        from medh5.integrity.verify import verify_root

        return verify_root(self.root, self.attr_name_map(), partial=partial)

    def compute_content_id(self) -> str:
        from medh5.integrity.digest import compute_content_id

        algo = (
            as_str(self.root.attrs["digest_algo"])
            if "digest_algo" in self.root.attrs
            else "sha256"
        )
        return compute_content_id(self.root, self.attr_name_map(), algo=algo)

    # -- reporting ---------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "kind": self.kind,
            "profiles": sorted(self.profiles),
            "content_id": self.content_id,
            **self.document.summary(),
            "grids": [g.summary() for g in self.grids.values()],
            "images": [i.summary() for i in self.images.values()],
            "annotations": [a.summary() for a in self.annotations.values()],
            "transforms": [t.summary() for t in self.transforms.values()],
            "index": sorted(self.index),
        }


FRAME_ATTRS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("grids", ("frame_uid",)),
    ("annotations", ("frame_uid",)),
    ("transforms", ("from_frame", "to_frame")),
)


def frame_references(root: h5py.Group) -> dict[str, tuple[str, ...]]:
    """Every frame-of-reference UID in a file, and the attributes naming it.

    A frame UID is a *shared* identifier rather than a property of one object:
    grids declare it, a world-space annotation names the frame its coordinates
    are in, and a transform names two of them (§3.4, §10.2).  Anything that
    rewrites one has to rewrite all of them --- see
    :meth:`SampleWriter.remap_frame_uids` for why half a rename is worse than
    none.
    """
    out: dict[str, list[str]] = {}
    for group, attrs in FRAME_ATTRS:
        if group not in root:
            continue
        for name in sorted(root[group]):
            node = root[group][name]
            for attr in attrs:
                raw = node.attrs.get(attr)
                uid = as_str(raw) if raw is not None else ""
                if uid:
                    out.setdefault(uid, []).append(f"{group}.{name}.{attr}")
    return {uid: tuple(where) for uid, where in sorted(out.items())}


def attr_name_map_of(root: h5py.Group) -> dict[str, tuple[str, ...]]:
    """Object path -> spec-defined attribute names, for ``content_id`` (§13.2).

    Derived from the file, and shared by the reader and the writer, because the
    two have to agree by construction rather than by maintenance.  The writer
    used to build this from its own in-memory caches, so every object class it
    forgot to restore in ``_inherit`` produced a ``content_id`` computed over
    fewer objects than a reader would later find --- an amend that verified on
    the way out and reported E702 on the way back in.  Four caches needed that
    restore before a fifth was noticed; asking the file removes the class.
    """
    out: dict[str, tuple[str, ...]] = {"": ROOT_DIGEST_ATTRS}
    for group, attrs in (
        ("grids", SPEC_GRID_ATTRS),
        ("images", SPEC_IMAGE_ATTRS),
        ("annotations", SPEC_ANNOTATION_ATTRS),
        ("transforms", SPEC_TRANSFORM_ATTRS),
    ):
        if group not in root:
            continue
        # The digest attribute is what the map is used to compute; hashing it
        # into its own input is the one attribute that can never be included.
        names = tuple(a for a in attrs if a != "digest")
        for name in root[group]:
            out[f"{group}/{name}"] = names
    return out


def annotation_id(reference: str) -> str:
    """An annotation id from a reference that may be written as a path.

    §6.2 says ``derived_from`` holds annotation ids; the RTSTRUCT importer wrote
    ``annotations/<id>`` for a release, so both spellings name one thing.
    """
    return reference.removeprefix("annotations/")


def require_major(handle: h5py.Group, path: str | os.PathLike[str]) -> str:
    """The file's ``medh5_version``, or a refusal of a major this reader lacks.

    One check for every door --- ``open``, ``open_collection``, ``open_any``,
    ``pack`` --- because a 2.0 shard used to open through the collection door
    while the sample door refused it (§2.1, §16).
    """
    version = handle.attrs.get("medh5_version")
    if version is None:
        raise MEDH5VersionError(
            f"{os.fspath(path)!r} declares no `medh5_version`; a 0.x file must be "
            "converted with `medh5 migrate`"
        )
    text = as_str(version)
    if text.split(".", 1)[0] != FORMAT_VERSION.split(".", 1)[0]:
        raise MEDH5VersionError(
            f"{os.fspath(path)!r} is MEDH5 {text}; this reader implements "
            f"{FORMAT_VERSION}"
        )
    return text


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def open_sample(path: str | os.PathLike[str], mode: str = "r") -> Sample:
    """Open a ``.medh5`` sample file, read-only.

    ``mode`` exists for the callers that spelled out ``"r"``.  ``"r+"`` used to
    be accepted too, and bought nothing: a :class:`Sample` has no mutating
    method, and §14.4's edits go through :func:`amend`, which is copy-on-write.
    """
    if mode != "r":
        raise MEDH5ValidationError(
            f"open() is read-only; use create() or amend() to write, not {mode!r}"
        )
    handle = open_h5(path, mode)
    try:
        require_major(handle, path)
        kind = as_str(handle.attrs.get("medh5_kind", "sample"))
        if kind != "sample":
            raise MEDH5FileError(
                f"{os.fspath(path)!r} is a {kind!r}; open it with open_collection()"
            )
    except BaseException:
        handle.close()
        raise
    return Sample(handle, handle=handle, owns_handle=True, path=os.fspath(path))


# The writer lives beside the reader, not inside it: `sample.py` was 2 400
# lines holding both, and every contributor read the whole to touch either.
# Imported last so `writer.py` can import this module's names while it is
# still initialising (the two are a pair by design, not a cycle by accident).
from medh5.writer import SampleWriter, amend, create  # noqa: E402

__all__ = [
    "FORMAT_VERSION",
    "PROFILES",
    "ROOT_DIGEST_ATTRS",
    "AnnotationCollection",
    "ImageCollection",
    "Sample",
    "SampleWriter",
    "TimepointView",
    "amend",
    "annotation_id",
    "create",
    "open_sample",
    "require_major",
]
