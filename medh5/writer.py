"""``SampleWriter`` --- where the pieces become a file.

Writing is a builder that validates as it goes and commits atomically.  The
write model is spec §14.4: create writes to a sibling temporary file and
``os.replace``s it, so a reader never sees a half-written sample and a crash
leaves the previous file intact.  Amend is copy-on-write by default, because
HDF5 does not reclaim space on ``del`` --- repeated in-place add/remove
monotonically bloats a file and fragments its chunk index.

The reader is :mod:`medh5.sample`; every name here is also importable from
there, and from ``medh5``, as it always was.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from datetime import datetime, timezone
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5 import __about__
from medh5._hdf5 import (
    as_str,
    as_str_tuple,
    atomic_h5,
    copy_object,
    copy_root_attrs,
    copy_unknown,
    encode_attr,
    open_h5,
    set_attrs,
    validate_id,
)
from medh5.annotations.base import (
    VOXEL_KINDS,
    AnnotationHeader,
    VoxelAnnotation,
    open_annotation,
)
from medh5.annotations.classification import encode_classification
from medh5.annotations.geometric import (
    Polygon,
    check_slice_index,
    check_space,
    encode_boxes,
    encode_contours,
    encode_keypoints,
    encode_mesh,
    encode_obb,
    encode_points,
)
from medh5.annotations.voxel import (
    IN_BAND_IGNORE_KINDS,
    AnnotationPayload,
    InstanceInput,
    OverlapStats,
    encode_instances,
    encode_mask,
    encode_masks,
    encode_probmap,
    normalize_masks,
    select_encoding,
)
from medh5.curation.identity import Cohort, Deidentification, Identity, SplitClaim
from medh5.curation.provenance import Activity, Agent
from medh5.curation.quality import QualityRecord
from medh5.curation.timeline import Timeline, Timepoint
from medh5.document import (
    SampleDocument,
    read_document,
    write_document,
)
from medh5.errors import (
    MEDH5ValidationError,
)
from medh5.geometry.grid import Grid, read_grids, write_grid
from medh5.geometry.multiscale import Pyramid, check_pyramid, pyramid_factors
from medh5.image import check_value_type
from medh5.labels.labelset import LabelSet
from medh5.sample import (
    FORMAT_VERSION,
    FRAME_ATTRS,
    PROFILES,
    annotation_id,
    attr_name_map_of,
    frame_references,
)
from medh5.storage.chunking import optimize_chunks
from medh5.storage.codecs import dataset_kwargs, resolve_profile
from medh5.storage.index import (
    DEFAULT_MAX_COORDS,
    DEFAULT_OCCUPANCY_FACTOR,
    build_index,
    write_index,
)
from medh5.transforms.affine import encode_affine, encode_identity
from medh5.transforms.base import (
    TRANSFORM_KINDS,
    TransformHeader,
    check_transform_id,
)
from medh5.transforms.bspline import encode_bspline
from medh5.transforms.composite import encode_composite
from medh5.transforms.displacement import encode_displacement

_MANAGED_ROOT_ATTRS = (
    "medh5_version",
    "medh5_kind",
    "medh5_profiles",
    "created",
    "generator",
    "digest_algo",
    "content_id",
)
"""Root attributes ``commit`` writes itself, so an amend must not copy them.

Everything else on the root is copied through: §16 permits a minor version to
add root attributes, and a 1.0 tool amending such a file must preserve what it
does not understand rather than silently dropping it.
"""

_STANDARD_GROUPS = ("meta", "grids", "images", "annotations", "transforms", "index")


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


class SampleWriter:
    """Builder for one sample.  Every ``add_*`` validates immediately."""

    __slots__ = (
        "_annotation_kinds",
        "_committed",
        "_declared_profiles",
        "_default_timeline",
        "_document",
        "_file",
        "_grids",
        "_image_multiscale",
        "_images",
        "_stack",
        "_transform_frames",
        "codec",
        "path",
    )

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        sample_id: str | None = None,
        subject_id: str | None = None,
        codec: str = "balanced",
        profiles: Sequence[str] = (),
        source: h5py.Group | None = None,
    ) -> None:
        self.path = os.fspath(path)
        self.codec = resolve_profile(codec).name
        self._committed = False
        self._stack = ExitStack()
        self._file = self._stack.enter_context(atomic_h5(self.path))
        try:
            self._grids: dict[str, Grid] = {}
            self._images: dict[str, tuple[str, str]] = {}
            self._image_multiscale: dict[str, bool] = {}
            self._annotation_kinds: dict[str, str] = {}
            self._transform_frames: dict[str, tuple[str, str]] = {}
            self._declared_profiles = set(profiles)
            self._default_timeline = True
            default_id = sample_id or os.path.basename(self.path).split(".")[0]
            self._document = SampleDocument(
                identity=Identity(
                    sample_id=default_id, subject_id=subject_id or default_id
                ),
                timepoints=Timeline.single(),
            )
            for name in ("grids", "images", "annotations"):
                self._file.create_group(name)
            if source is not None:
                self._inherit(source)
        except BaseException:
            # A writer whose `__init__` raised is never returned, so nothing
            # will ever call `abort()` or `commit()` on it and the `with`
            # statement that would have was never entered.  The sibling
            # `.<name>.medh5.tmp-<pid>` and its open HDF5 handle then survive
            # for as long as the traceback does --- which is until the exception
            # object is dropped, and in a notebook that is the rest of the
            # session.  `amend` on a file with an unreadable `/meta` or a grid
            # that fails `Grid.check` is the way in.
            self.abort()
            raise

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> SampleWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *rest: object) -> None:
        if exc_type is not None:
            self.abort()
            return
        if not self._committed:
            self.commit()

    def abort(self) -> None:
        """Discard the in-progress file, leaving any existing one untouched."""
        self._stack.__exit__(RuntimeError, RuntimeError("aborted"), None)
        self._committed = True

    @property
    def document(self) -> SampleDocument:
        return self._document

    def _inherit(self, source: h5py.Group) -> None:
        """Copy an existing sample into this writer --- the copy-on-write amend."""
        self._document = read_document(source)
        self._default_timeline = False
        for name in ("grids", "images", "annotations", "transforms", "index"):
            if name not in source:
                continue
            if name in self._file:
                del self._file[name]
            copy_object(source, name, self._file)
        self._grids = read_grids(self._file)
        for name in self._file.get("images", {}):
            node = self._file["images"][name]
            self._images[name] = (
                as_str(node.attrs.get("grid", "")),
                as_str(node.attrs.get("modality", "OT")),
            )
            self._image_multiscale[name] = isinstance(node, h5py.Group)
        for name in self._file.get("annotations", {}):
            self._annotation_kinds[name] = as_str(
                self._file["annotations"][name].attrs.get("kind", "mask")
            )
        for name in self._file.get("transforms", {}):
            node = self._file["transforms"][name]
            self._transform_frames[name] = (
                as_str(node.attrs.get("from_frame", "")),
                as_str(node.attrs.get("to_frame", "")),
            )
        copy_unknown(source, self._file, _STANDARD_GROUPS)
        # Root attributes too, not just objects.  §16 lets a minor version add
        # attributes and requires readers to ignore ones they do not recognise,
        # so dropping them here would make a 1.0 amend quietly destroy what a
        # 1.1 writer put on the root -- every other level already survives.
        # The six `commit` manages are skipped because it rewrites them from the
        # amended state; `content_id` is skipped because it is restamped there too.
        copy_root_attrs(source, self._file, skip=_MANAGED_ROOT_ATTRS)
        # Profiles are deliberately *not* inherited.  They are a derived fact,
        # and an amend that removes what justified one (dropping a stale index,
        # say) must stop claiming it --- a flag that can disagree with the data
        # is a bug generator.

    # -- document ----------------------------------------------------------

    def identity(self, **fields: Any) -> Identity:
        """Set the sample's identity (``sample_id`` and ``subject_id`` required)."""
        merged = {**self._document.identity.to_json(), **fields}
        self._document.identity = Identity.from_json(merged)
        return self._document.identity

    def cohort(self, **fields: Any) -> Cohort:
        merged = {**self._document.cohort.to_json(), **fields}
        self._document.cohort = Cohort.from_json(merged)
        return self._document.cohort

    def add_timepoint(self, timepoint_id: str, **fields: Any) -> Timepoint:
        """Declare a timepoint.

        A writer starts with an implicit single ``tp0`` so a cross-sectional
        sample needs no ceremony; the first explicit ``add_timepoint`` replaces
        it rather than appending to it.
        """
        existing = [] if self._default_timeline else list(self._document.timepoints)
        self._default_timeline = False
        index = fields.pop("index", len(existing))
        tp = Timepoint(id=timepoint_id, index=index, **fields)
        self._document.timepoints = Timeline([*existing, tp])
        return tp

    def label_set(self, label_set: LabelSet) -> LabelSet:
        self._document.label_set = label_set
        return label_set

    def person(self, name: str, agent_id: str | None = None, **fields: Any) -> Agent:
        return self._agent("person", name, agent_id, **fields)

    def software(self, name: str, version: str | None = None, **fields: Any) -> Agent:
        return self._agent("software", name, None, version=version, **fields)

    def organization(self, name: str, **fields: Any) -> Agent:
        return self._agent("organization", name, None, **fields)

    def _agent(
        self, agent_type: str, name: str, agent_id: str | None = None, **fields: Any
    ) -> Agent:
        """Declare an agent; an explicit id must be free, an automatic one is.

        Automatic ids are ``<type initial><n>``, which is exactly the id a
        caller who named one node explicitly is likely to have used, so the
        counter skips every taken id rather than colliding with it.  An
        explicit duplicate is refused by :meth:`Provenance.add_agent`.
        """
        prov = self._document.provenance
        if agent_id is None:
            n = len(prov.agents) + 1
            while prov.has_agent(f"{agent_type[0]}{n}"):
                n += 1
            agent_id = f"{agent_type[0]}{n}"
        return prov.add_agent(Agent(id=agent_id, type=agent_type, name=name, **fields))

    def activity(
        self,
        activity_type: str,
        *,
        agent: Agent | str | None = None,
        activity_id: str | None = None,
        **fields: Any,
    ) -> Activity:
        """Record an activity.  See :meth:`_agent` for how ids are minted."""
        prov = self._document.provenance
        if activity_id is None:
            n = len(prov.activities) + 1
            while prov.has_activity(f"act_{activity_type}_{n}"):
                n += 1
            activity_id = f"act_{activity_type}_{n}"
        agent_id = agent.id if isinstance(agent, Agent) else agent
        return prov.add_activity(
            Activity(id=activity_id, type=activity_type, agent=agent_id, **fields)
        )

    def set_quality(self, key: str, **fields: Any) -> QualityRecord:
        record = QualityRecord.from_json({"status": "draft", **fields})
        self._document.quality[key] = record
        return record

    def split(self, **fields: Any) -> SplitClaim:
        """Record a split claim, replacing any earlier claim for the same set.

        Replacing rather than appending: two claims for one ``set_id`` is
        exactly the W906 conflict §12.3 defines, so a writer that appends on a
        re-split manufactures the defect the validator exists to catch.  Claims
        for *other* sets are left alone --- a sample may belong to several.
        """
        claim = SplitClaim.from_json(fields)
        kept = tuple(s for s in self._document.splits if s.set_id != claim.set_id)
        self._document.splits = (*kept, claim)
        return claim

    def deidentification(self, **fields: Any) -> Deidentification:
        record = Deidentification.from_json(fields)
        assert record is not None
        self._document.deidentification = record
        return record

    def acquisition(self, image_id: str, **params: Any) -> dict[str, Any]:
        self._document.acquisition.setdefault(image_id, {}).update(params)
        return self._document.acquisition[image_id]

    def extra(self, namespace: str, value: Any) -> None:
        self._document.extra[namespace] = value

    # -- grids -------------------------------------------------------------

    def add_grid(
        self,
        grid_id: str,
        *,
        shape: Sequence[int],
        spacing: Sequence[float],
        origin: Sequence[float] | None = None,
        direction: npt.ArrayLike | None = None,
        axis_names: Sequence[str] | None = None,
        axis_kinds: Sequence[str] | None = None,
        coord_system: str = "LPS",
        units: str = "mm",
        timepoint: str | None = None,
        frame_uid: str | None = None,
        patch_hint: Sequence[int] | None = None,
        chunk_hint: Sequence[int] | None = None,
        time_values: Sequence[float] | None = None,
        time_units: str | None = None,
    ) -> Grid:
        """Declare a grid.  Geometry lives here and nowhere else."""
        validate_id(grid_id, what="grid id")
        if grid_id in self._grids:
            raise MEDH5ValidationError(f"grid {grid_id!r} is already declared")
        shape_t = tuple(int(s) for s in shape)
        if axis_kinds is None:
            axis_kinds = _default_axis_kinds(len(shape_t))
        if axis_names is None:
            axis_names = _default_axis_names(tuple(axis_kinds))
        if "time" in tuple(axis_kinds) and time_values is None:
            raise MEDH5ValidationError(
                f"grid {grid_id!r} declares a time axis but no time_values; §3.2 "
                "requires one acquisition time per frame",
                code="E109",
            )
        n_spatial = tuple(axis_kinds).count("spatial")
        grid = Grid(
            grid_id=grid_id,
            shape=shape_t,
            axis_names=tuple(axis_names),
            axis_kinds=tuple(axis_kinds),
            spacing=tuple(float(v) for v in spacing),
            origin=tuple(origin) if origin is not None else (0.0,) * n_spatial,
            direction=np.asarray(direction, dtype=np.float64)
            if direction is not None
            else np.eye(n_spatial),
            coord_system=coord_system,
            units=units,
            timepoint=timepoint,
            frame_uid=frame_uid,
            time_values=tuple(time_values) if time_values is not None else None,
            time_units=time_units,
            patch_hint=tuple(patch_hint) if patch_hint is not None else None,
            chunk_hint=tuple(chunk_hint) if chunk_hint is not None else None,
        )
        write_grid(self._file["grids"], grid)
        self._grids[grid_id] = grid
        return grid

    @property
    def grids(self) -> dict[str, Grid]:
        """Grids declared so far, so a pass over them does not need internals."""
        return dict(self._grids)

    def remap_frame_uids(self, mapping: Mapping[str, str]) -> tuple[str, ...]:
        """Rename frames of reference everywhere they are named (§3.4).

        The frame UID is the one geometric field that is an *identifier* rather
        than a measurement, so a de-identification pass has to be able to
        pseudonymise it without rewriting the geometry around it.  It is renamed
        here for the whole file at once, and deliberately not per object: grids
        declare a frame, world-space annotations name one, and transforms name
        two, so renaming it on the grids alone leaves the original UID on the
        other two --- in a file that has just been certified de-identified ---
        *and* disconnects the grids from the transforms that relate them, which
        makes ``transform_between`` answer ``None`` and every longitudinal
        loader quietly drop the pair.  Half a rename is worse than none: it
        loses the registration and keeps the identifier.

        Returns the attributes it rewrote.  UIDs absent from *mapping* are left
        alone, so a caller may pseudonymise the real ones and keep the rest.
        """
        changed: list[str] = []
        for group, attrs in FRAME_ATTRS:
            if group not in self._file:
                continue
            for name in sorted(self._file[group]):
                node = self._file[group][name]
                for attr in attrs:
                    raw = node.attrs.get(attr)
                    current = as_str(raw) if raw is not None else ""
                    replacement = mapping.get(current)
                    if replacement is None or replacement == current:
                        continue
                    node.attrs[attr] = encode_attr(replacement)
                    changed.append(f"{group}.{name}.{attr}")
        if changed:
            # Both caches are keyed on the strings that just moved.
            self._grids = read_grids(self._file)
            for name in self._transform_frames:
                node = self._file["transforms"][name]
                self._transform_frames[name] = (
                    as_str(node.attrs.get("from_frame", "")),
                    as_str(node.attrs.get("to_frame", "")),
                )
        return tuple(changed)

    def frame_uids(self) -> dict[str, tuple[str, ...]]:
        """Every frame UID in the file so far, and the attributes naming it."""
        return frame_references(self._file)

    def _grid(self, grid_id: str) -> Grid:
        try:
            return self._grids[grid_id]
        except KeyError:
            raise MEDH5ValidationError(
                f"grid {grid_id!r} is not declared; declare it before referencing it",
                code="E101",
            ) from None

    # -- images ------------------------------------------------------------

    def add_image(
        self,
        image_id: str,
        data: npt.ArrayLike,
        *,
        grid: str,
        modality: str,
        value_type: str = "intensity",
        value_units: str | None = None,
        channel_names: Sequence[str] | None = None,
        rescale_slope: float | None = None,
        rescale_intercept: float | None = None,
        window_center: Sequence[float] | None = None,
        window_width: Sequence[float] | None = None,
        valid_mask: str | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> h5py.Dataset:
        """Write one image, chunked for its grid's patch hint."""
        validate_id(image_id, what="image id")
        if image_id in self._images:
            raise MEDH5ValidationError(f"image {image_id!r} already exists")
        check_value_type(value_type)
        target = self._grid(grid)
        array = np.asarray(data)
        if array.shape != target.shape:
            raise MEDH5ValidationError(
                f"image {image_id!r} has shape {array.shape}; grid {grid!r} "
                f"declares {target.shape}",
                code="E202",
            )
        if channel_names is not None:
            axis = target.channel_axis
            if axis is None or len(channel_names) != target.shape[axis]:
                raise MEDH5ValidationError(
                    f"image {image_id!r}: channel_names does not match the channel "
                    f"axis",
                    code="E204",
                )
        chunks = self._chunks_for(target, array.dtype.itemsize)
        node = self._file["images"].create_dataset(
            image_id,
            data=array,
            **dataset_kwargs(
                array.shape,
                array.dtype,
                profile=codec or self.codec,
                role="image",
                chunks=chunks,
            ),
        )
        set_attrs(
            node,
            {
                "grid": grid,
                "modality": modality,
                "value_type": value_type,
                "value_units": value_units,
                "channel_names": list(channel_names) if channel_names else None,
                "rescale_slope": rescale_slope,
                "rescale_intercept": rescale_intercept,
                "window_center": list(window_center) if window_center else None,
                "window_width": list(window_width) if window_width else None,
                "valid_mask": valid_mask,
                "prov": _prov_id(prov),
            },
        )
        self._images[image_id] = (grid, modality)
        self._image_multiscale[image_id] = False
        return node

    def add_pyramid(
        self,
        image_id: str,
        levels: Sequence[npt.ArrayLike],
        *,
        grid_levels: Sequence[str],
        modality: str,
        value_type: str = "intensity",
        downsample_method: str = "mean",
        value_units: str | None = None,
        rescale_slope: float | None = None,
        rescale_intercept: float | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> h5py.Group:
        """Write a multiscale image (spec §4.3).

        Each level has its own grid, and the derivation rule --- ``spacing * f``
        with the ``(f-1)/2`` half-voxel origin shift --- is checked here rather
        than left to the validator, because a pyramid whose levels are offset by
        half a low-resolution voxel looks correct in every viewer and silently
        misplaces every prediction mapped back to level 0.
        """
        validate_id(image_id, what="image id")
        if image_id in self._images:
            raise MEDH5ValidationError(f"image {image_id!r} already exists")
        check_value_type(value_type)
        if len(levels) != len(grid_levels):
            raise MEDH5ValidationError(
                f"image {image_id!r}: {len(levels)} arrays for "
                f"{len(grid_levels)} grid levels",
                code="E105",
            )
        grids = [self._grid(gid) for gid in grid_levels]
        factors = pyramid_factors(grids[0], grids)
        problems = check_pyramid(grids[0], grids, factors)
        if problems:
            raise MEDH5ValidationError(
                f"image {image_id!r}: inconsistent pyramid geometry: "
                + "; ".join(problems),
                code="E105",
            )
        group = self._file["images"].create_group(image_id)
        for level, (array_like, target) in enumerate(zip(levels, grids, strict=True)):
            array = np.asarray(array_like)
            if array.shape != target.shape:
                raise MEDH5ValidationError(
                    f"image {image_id!r} level {level} has shape {array.shape}; "
                    f"grid {target.grid_id!r} declares {target.shape}",
                    code="E202",
                )
            group.create_dataset(
                str(level),
                data=array,
                **dataset_kwargs(
                    array.shape,
                    array.dtype,
                    profile=codec or self.codec,
                    role="image",
                    chunks=self._chunks_for(target, array.dtype.itemsize),
                ),
            )
        pyramid = Pyramid(
            levels=len(levels),
            downsample_factors=factors,
            downsample_method=downsample_method,
            grid_levels=tuple(grid_levels),
        )
        set_attrs(
            group,
            {
                "grid": grid_levels[0],
                "modality": modality,
                "value_type": value_type,
                "value_units": value_units,
                "rescale_slope": rescale_slope,
                "rescale_intercept": rescale_intercept,
                "prov": _prov_id(prov),
                **pyramid.attrs(),
            },
        )
        self._images[image_id] = (grid_levels[0], modality)
        self._image_multiscale[image_id] = True
        return group

    def _chunks_for(
        self, grid: Grid, itemsize: int, *, leading: int = 0
    ) -> tuple[int, ...]:
        if grid.chunk_hint is not None and leading == 0:
            return grid.chunk_hint
        return optimize_chunks(
            (1,) * leading + grid.shape,
            grid.axis_kinds,
            grid.patch_hint,
            itemsize=itemsize,
            leading=leading,
        )

    # -- annotations -------------------------------------------------------

    def add_segmentation(
        self,
        ann_id: str,
        *,
        grid: str,
        masks: Mapping[Any, npt.NDArray[Any]] | None = None,
        probabilities: Mapping[Any, npt.NDArray[Any]] | None = None,
        instances: Sequence[InstanceInput] | None = None,
        encoding: str = "auto",
        threshold: float | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        ignore: npt.NDArray[np.bool_] | None = None,
        ignore_mask: str | None = None,
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "segmentation",
        codec: str | None = None,
    ) -> tuple[str, OverlapStats | None]:
        """Write a voxel annotation, choosing the encoding by measurement.

        Class keys may be ids or label-set keys, which is why the mapping's key
        type is left open: ``Mapping`` is invariant in its key, so a narrower
        annotation would reject the ``dict[int, ...]`` callers naturally build.

        Returns the chosen ``kind`` and the overlap statistics behind the choice.
        ``annotated_classes`` is the coverage contract: ``"all_given"`` claims only
        the classes supplied, ``"all"`` claims the whole label set, and an explicit
        sequence claims exactly what the annotator looked for.  ``threshold``
        declares the probability at or above which a voxel contains a class
        (§7.5); it applies to ``probabilities=`` and nothing else.

        ``ignore`` is a boolean array over the grid marking voxels nobody
        examined, which must not be read as background for any annotated class
        (§7.7).  Where it ends up depends on the encoding written, and the
        caller does not have to know which:

        * ``labelmap`` and ``layers`` hold it in band, as the reserved id
          ``65535`` in the data; ``ignore_mask()`` on the reader returns it.
        * ``bitmask``, ``instances`` and ``probmap`` cannot hold a reserved id,
          so the region is written as a sibling ``mask`` annotation named
          ``<ann_id>_ignore`` --- same grid, same ``prov``, ``task="other"`` ---
          and this annotation's ``ignore_mask`` attribute names it.

        Either way ``has_ignore_region`` is true on the result and the region
        reads back equal to the array given.  With ``encoding="auto"`` the
        region is part of the measurement: an in-band ignore forces ``uint16``
        planes, so the cost model sees it before choosing.

        ``ignore_mask`` names a ``mask`` annotation the caller wrote (or will
        write) themselves, for the case where one region serves several
        annotations.  Passing both is refused: two sources of the same fact
        cannot be kept in agreement.
        """
        target = self._grid(grid)
        if threshold is not None and probabilities is None:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r}: threshold= applies to probabilities= only",
                code="E404",
            )
        if ignore is not None and ignore_mask is not None:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r}: pass ignore= (an array this writer stores) "
                "or ignore_mask= (a `mask` annotation you wrote), not both",
                code="E404",
            )
        ignore_array: npt.NDArray[np.bool_] | None = None
        if ignore is not None:
            ignore_array = np.asarray(ignore, dtype=bool)
            if ignore_array.shape != target.spatial_shape:
                raise MEDH5ValidationError(
                    f"annotation {ann_id!r}: ignore has shape {ignore_array.shape}; "
                    f"grid {grid!r} spatial shape is {target.spatial_shape}",
                    code="E405",
                )
        payload, stats, class_ids = self._encode_segmentation(
            masks,
            probabilities,
            instances,
            target,
            encoding,
            ignore_array,
            self._named_classes(annotated_classes),
            threshold=threshold,
        )
        sibling: str | None = None
        if ignore_array is not None and payload.kind not in IN_BAND_IGNORE_KINDS:
            # §7.7: the region lives in a separate `mask` annotation.  Both ids
            # are checked before either is written, so a refusal leaves nothing
            # half-done in the file.
            sibling = f"{ann_id}_ignore"
            validate_id(sibling, what="annotation id")
            for taken in (ann_id, sibling):
                if taken in self._file["annotations"]:
                    raise MEDH5ValidationError(f"annotation {taken!r} already exists")
            ignore_mask = sibling
        annotated = self._resolve_annotated(annotated_classes, class_ids)
        header = AnnotationHeader(
            kind=payload.kind,
            task=task,
            grid=grid,
            timepoints=tuple(timepoints) if timepoints is not None else None,
            class_ids=class_ids,
            annotated_class_ids=annotated,
            closure=closure,
            ignore_mask=ignore_mask,
            prov=_prov_id(prov),
            quality=self._quality_key(ann_id, quality),
            derived_from=tuple(derived_from),
            extra=dict(payload.attrs),
        )
        self._write_annotation(ann_id, header, payload, target, codec)
        if sibling is not None and ignore_array is not None:
            self.add_mask(sibling, ignore_array, grid=grid, prov=prov, codec=codec)
        return payload.kind, stats

    def _encode_segmentation(
        self,
        masks: Mapping[Any, npt.NDArray[Any]] | None,
        probabilities: Mapping[Any, npt.NDArray[Any]] | None,
        instances: Sequence[InstanceInput] | None,
        grid: Grid,
        encoding: str,
        ignore: npt.NDArray[np.bool_] | None,
        examined: tuple[int, ...] = (),
        *,
        threshold: float | None = None,
    ) -> tuple[AnnotationPayload, OverlapStats | None, tuple[int, ...]]:
        """Encode the payload, keeping every *examined* class expressible.

        A class the annotator searched for and did not find has to survive into
        ``class_ids``, or the file cannot tell "verified absent" from "never
        looked for" (spec §11.3).  Empty classes therefore reach the encoders
        rather than being dropped for having no voxels.
        """
        if instances is not None:
            payload = encode_instances(
                instances, grid.spatial_shape, class_ids=examined or None
            )
            return payload, None, payload.class_ids
        if probabilities is not None:
            resolved = {self._class_id(k): v for k, v in probabilities.items()}
            # Examined-but-absent classes get an all-zero plane, exactly as the
            # mask branch below gives them an empty mask.  Without it a partial
            # probability map could not state coverage at all: `class_ids` held
            # only the planes supplied while `annotated_class_ids` named more,
            # and `AnnotationHeader` rejected the write with E403.
            for class_id in examined:
                resolved.setdefault(
                    class_id, np.zeros(grid.spatial_shape, dtype=np.float32)
                )
            payload = encode_probmap(resolved, grid.spatial_shape, threshold=threshold)
            return payload, None, payload.class_ids
        if masks is None:
            raise MEDH5ValidationError(
                "add_segmentation needs one of masks=, probabilities= or instances="
            )
        given = {self._class_id(k): v for k, v in masks.items()}
        for class_id in examined:
            given.setdefault(class_id, np.zeros(grid.spatial_shape, dtype=bool))
        resolved_masks, shape = normalize_masks(given, grid.spatial_shape)
        kind, stats = select_encoding(
            resolved_masks,
            shape,
            prefer=None if encoding == "auto" else encoding,
            ignore=ignore is not None,
        )
        kwargs: dict[str, Any] = {}
        if ignore is not None and kind in IN_BAND_IGNORE_KINDS:
            kwargs["ignore"] = ignore
        # Any other kind gets the region as a sibling `mask`, written by
        # `add_segmentation` once the payload's kind is known (§7.7).
        payload = encode_masks(resolved_masks, kind, shape, **kwargs)
        return payload, stats, payload.class_ids

    def add_mask(
        self,
        ann_id: str,
        mask: npt.NDArray[Any],
        *,
        grid: str,
        task: str = "other",
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> None:
        """Write a boolean ``mask`` annotation (FOV, ignore region)."""
        target = self._grid(grid)
        array = np.asarray(mask, dtype=bool)
        if array.shape != target.spatial_shape:
            raise MEDH5ValidationError(
                f"mask {ann_id!r} has shape {array.shape}; grid {grid!r} spatial "
                f"shape is {target.spatial_shape}",
                code="E405",
            )
        payload = encode_mask(array)
        header = AnnotationHeader(
            kind="mask", task=task, grid=grid, prov=_prov_id(prov)
        )
        self._write_annotation(ann_id, header, payload, target, codec)

    def _write_annotation(
        self,
        ann_id: str,
        header: AnnotationHeader,
        payload: AnnotationPayload,
        grid: Grid | None,
        codec: str | None,
    ) -> h5py.Group:
        validate_id(ann_id, what="annotation id")
        node = self._file["annotations"]
        if ann_id in node:
            raise MEDH5ValidationError(f"annotation {ann_id!r} already exists")
        group = node.create_group(ann_id)
        for name, array in payload.datasets.items():
            chunks = None
            if grid is not None and name == "data" and array.ndim >= grid.n_spatial:
                proposed = self._chunks_for(
                    grid, array.dtype.itemsize, leading=payload.stacked_axes
                )[-array.ndim :]
                if len(proposed) == array.ndim:
                    chunks = tuple(
                        min(int(c), int(s))
                        for c, s in zip(proposed, array.shape, strict=True)
                    )
            group.create_dataset(
                name,
                data=array,
                **dataset_kwargs(
                    array.shape,
                    array.dtype,
                    profile=codec or self.codec,
                    role="label",
                    chunks=chunks,
                ),
            )
        set_attrs(group, header.attrs())
        self._annotation_kinds[ann_id] = header.kind
        return group

    def _class_id(self, key: int | str) -> int:
        if isinstance(key, (int, np.integer)):
            return int(key)
        label_set = self._document.label_set
        if label_set is None:
            raise MEDH5ValidationError(
                f"cannot resolve class name {key!r}: declare a label set first"
            )
        return label_set[key].id

    def _named_classes(self, annotated: str | Sequence[int | str]) -> tuple[int, ...]:
        """The classes an explicit ``annotated_classes=`` names, resolved to ids.

        ``"all"`` names the whole label set, and has to do so *here* as well as in
        :meth:`_resolve_annotated`: this is the list ``_encode_segmentation``
        injects zero masks for, so a class that was searched for and not found
        reaches ``class_ids`` instead of being dropped for having no voxels.
        Returning ``()`` for it made ``"all"`` a synonym for ``"all_given"`` and
        silently destroyed every "examined and absent" negative in the file.
        """
        if annotated == "all":
            return self._label_set_ids()
        if isinstance(annotated, str):
            return ()
        return tuple(self._class_id(k) for k in annotated)

    def _label_set_ids(self) -> tuple[int, ...]:
        label_set = self._document.label_set
        if label_set is None:
            raise MEDH5ValidationError(
                "annotated_classes='all' needs a declared label set"
            )
        return tuple(label_set.ids)

    def _resolve_annotated(
        self, annotated: str | Sequence[int | str], class_ids: Sequence[int]
    ) -> tuple[int, ...]:
        if annotated == "all_given":
            return tuple(class_ids)
        if annotated == "all":
            # Not intersected with `class_ids`: "all" claims the whole label set,
            # which is the entire difference between it and "all_given".  The
            # intersection made the two identical and turned every examined-but-
            # absent class back into "never looked for" (§11.3).
            return self._label_set_ids()
        return tuple(self._class_id(k) for k in annotated)

    def _quality_key(
        self, ann_id: str, quality: str | Mapping[str, Any] | None
    ) -> str | None:
        if quality is None:
            return None
        if isinstance(quality, str):
            return quality
        self._document.quality[ann_id] = QualityRecord.from_json(
            {"status": "draft", **quality}
        )
        return ann_id

    def remove_annotation(self, ann_id: str) -> None:
        """Drop an annotation, and any index entry derived from it."""
        node = self._file["annotations"]
        if ann_id not in node:
            raise MEDH5ValidationError(f"no annotation {ann_id!r} to remove")
        del node[ann_id]
        self._annotation_kinds.pop(ann_id, None)
        if "index" in self._file and ann_id in self._file["index"]:
            del self._file["index"][ann_id]

    def transcode_annotation(
        self, ann_id: str, to_kind: str, *, codec: str | None = None
    ) -> str:
        """Re-encode a voxel annotation in place, preserving its header.

        Lossless for every class and voxel (spec §7.6), which is what makes the
        encoding a storage decision: a cohort can be re-encoded for a different
        access pattern without anyone re-deriving the ground truth.
        """
        from medh5.annotations.voxel.transcode import transcode as _transcode

        node = self._file["annotations"]
        if ann_id not in node:
            raise MEDH5ValidationError(f"no annotation {ann_id!r} to transcode")
        group = node[ann_id]
        header = AnnotationHeader.read(group)
        annotation = open_annotation(
            ann_id, group, self._grids, self._document.label_set
        )
        if not isinstance(annotation, VoxelAnnotation):
            raise MEDH5ValidationError(
                f"annotation {ann_id!r} of kind {header.kind!r} is not a voxel encoding"
            )
        if header.kind == to_kind:
            return to_kind
        payload = _transcode(annotation, to_kind)
        grid = self._grid(header.grid or "")
        self.remove_annotation(ann_id)
        header.kind = to_kind
        # The header's `class_ids` is the encoding order (§6.2), and the new
        # payload sets its own -- `probmap` in particular carries plane order
        # *only* there, and its encoder always emits ascending.  Keeping the old
        # header over the new payload therefore mislabels every plane whenever
        # the source order was not ascending, which §6.2 permits and §7.3 makes
        # explicit for `bitmask` via `bit_class_ids`.  The result read back with
        # each class's mask under a different class's name, per-class voxel
        # counts unchanged, and nothing to see in the validator.
        header.class_ids = tuple(payload.class_ids)
        header.annotated_class_ids = tuple(
            c for c in header.annotated_class_ids if c in set(payload.class_ids)
        )
        header.extra = {**dict(header.extra), **payload.attrs}
        self._write_annotation(ann_id, header, payload, grid, codec)
        return to_kind

    # -- geometric and classification annotations (§8, §9) -----------------

    def _add_object_annotation(
        self,
        ann_id: str,
        payload: AnnotationPayload,
        *,
        grid: str | None,
        space: str | None,
        frame_uid: str | None,
        annotated_classes: str | Sequence[int | str],
        closure: str,
        timepoints: Sequence[str] | None,
        prov: Activity | str | None,
        quality: str | Mapping[str, Any] | None,
        derived_from: Sequence[str],
        task: str,
        codec: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> h5py.Group:
        """Shared path for every §8/§9 kind: resolve the space, then write.

        ``space = None`` is the classification case: a label is *about* a scope
        unit rather than located in one, so it has no coordinates to place.
        """
        target = self._grid(grid) if grid is not None else None
        if space is not None:
            check_space(space)
        if space == "index" and target is None:
            raise MEDH5ValidationError(
                f"annotation {ann_id!r}: space='index' names a grid's coordinates, "
                "so `grid` is required",
                code="E412",
            )
        resolved_frame = frame_uid
        if space == "world":
            if resolved_frame is None and target is not None:
                resolved_frame = target.frame_uid
            if resolved_frame is None:
                raise MEDH5ValidationError(
                    f"annotation {ann_id!r}: space='world' names a physical frame, so "
                    "`frame_uid` is required (directly or via the grid)",
                    code="E412",
                )
        if target is not None and target.units == "px" and space != "index":
            raise MEDH5ValidationError(
                f"annotation {ann_id!r}: grid {target.grid_id!r} is uncalibrated "
                "(units='px'), so geometric annotations on it must use space='index'",
                code="E414",
            )
        # For §8/§9 kinds the `class_ids` attribute is a *declaration*, not an
        # index into storage the way a layer table or a bitplane order is.  A
        # class that was searched for and not found therefore belongs in it:
        # otherwise `annotated_class_ids` could not be a subset (E403), and §9's
        # "looked for and not found is a negative" would be inexpressible.
        annotated = self._resolve_annotated(annotated_classes, payload.class_ids)
        declared = tuple(sorted(set(payload.class_ids) | set(annotated)))
        header = AnnotationHeader(
            kind=payload.kind,
            task=task,
            grid=grid,
            timepoints=tuple(timepoints) if timepoints is not None else None,
            space=space,
            frame_uid=resolved_frame,
            class_ids=declared,
            annotated_class_ids=annotated,
            closure=closure,
            prov=_prov_id(prov),
            quality=self._quality_key(ann_id, quality),
            derived_from=tuple(derived_from),
            extra={**dict(payload.attrs), **dict(extra or {})},
        )
        return self._write_annotation(ann_id, header, payload, target, codec)

    def add_boxes(
        self,
        ann_id: str,
        boxes: npt.ArrayLike,
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        attributes: Sequence[Mapping[str, Any]] | None = None,
        slice_index: Sequence[int] | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "detection",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write axis-aligned boxes as ``(N, S, 2)`` float `[lo, hi]` (spec §8.2)."""
        payload = encode_boxes(
            boxes,
            [self._class_id(c) for c in class_ids],
            instance_ids=instance_ids,
            scores=scores,
            attributes=attributes,
            slice_index=slice_index,
        )
        if slice_index is not None and space == "index" and grid is not None:
            # `encode_boxes` checks the shape but has no grid, so the plane's
            # *range* is checked here, where one is named. Writing a plane the
            # grid does not have used to be accepted and clamped to the nearest
            # edge on the way out, silently moving the box to another plane.
            known = self._grids.get(grid)
            problem = (
                None
                if known is None
                else check_slice_index(
                    slice_index,
                    len(payload.datasets["boxes"]),
                    boxes=np.asarray(payload.datasets["boxes"], dtype=np.float64),
                    shape=known.spatial_shape,
                )
            )
            if problem:
                raise MEDH5ValidationError(f"{ann_id!r}: {problem}", code="E405")
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_obb(
        self,
        ann_id: str,
        centers: npt.ArrayLike,
        sizes: npt.ArrayLike,
        rotations: npt.ArrayLike,
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        attributes: Sequence[Mapping[str, Any]] | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "detection",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write oriented boxes: centre, full edge lengths, rotation matrix (§8.3)."""
        payload = encode_obb(
            centers,
            sizes,
            rotations,
            [self._class_id(c) for c in class_ids],
            instance_ids=instance_ids,
            scores=scores,
            attributes=attributes,
        )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_keypoints(
        self,
        ann_id: str,
        points: npt.ArrayLike,
        keypoint_classes: Sequence[int | str],
        class_ids: Sequence[int | str],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        visibility: npt.ArrayLike | None = None,
        instance_ids: Sequence[int] | None = None,
        scores: Sequence[float] | None = None,
        skeleton: str | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "detection",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write ``(N, K, S)`` keypoints with per-slot classes (spec §8.4)."""
        if skeleton is not None:
            label_set = self._document.label_set
            if label_set is None or not any(
                sk.id == skeleton for sk in label_set.skeletons
            ):
                raise MEDH5ValidationError(
                    f"annotation {ann_id!r} names skeleton {skeleton!r}, which the "
                    "label set does not declare",
                    code="E413",
                )
        payload = encode_keypoints(
            points,
            [self._class_id(c) for c in keypoint_classes],
            [self._class_id(c) for c in class_ids],
            visibility=visibility,
            instance_ids=instance_ids,
            scores=scores,
            skeleton=skeleton,
        )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_points(
        self,
        ann_id: str,
        points: npt.ArrayLike,
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        class_ids: Sequence[int | str] | None = None,
        names: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        correspondence: str | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "detection",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write a point set: landmarks, seeds, or half of a correspondence (§8.5)."""
        payload = encode_points(
            points,
            class_ids=(
                [self._class_id(c) for c in class_ids]
                if class_ids is not None
                else None
            ),
            names=names,
            weights=weights,
            correspondence=correspondence,
        )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_contours(
        self,
        ann_id: str,
        polygons: Sequence[Polygon],
        *,
        grid: str | None = None,
        space: str = "index",
        frame_uid: str | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "segmentation",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write planar polygons (spec §8.6) --- the RTSTRUCT-shaped annotation.

        No polygons at all is a verified negative when ``annotated_classes``
        names what was looked for, exactly as an empty box set is (§9).
        """
        known = self._grids.get(grid) if grid is not None else None
        payload = encode_contours(
            polygons, ndim=None if known is None else known.n_spatial
        )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_mesh(
        self,
        ann_id: str,
        vertices: npt.ArrayLike,
        faces: npt.ArrayLike,
        *,
        grid: str | None = None,
        space: str = "world",
        frame_uid: str | None = None,
        normals: npt.ArrayLike | None = None,
        vertex_class_ids: Sequence[int | str] | None = None,
        mesh_offsets: Sequence[int] | None = None,
        mesh_class_ids: Sequence[int | str] | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        task: str = "segmentation",
        codec: str | None = None,
    ) -> h5py.Group:
        """Write a triangle surface mesh (spec §8.7)."""
        payload = encode_mesh(
            vertices,
            faces,
            normals=normals,
            vertex_class_ids=(
                [self._class_id(c) for c in vertex_class_ids]
                if vertex_class_ids is not None
                else None
            ),
            mesh_offsets=mesh_offsets,
            mesh_class_ids=(
                [self._class_id(c) for c in mesh_class_ids]
                if mesh_class_ids is not None
                else None
            ),
        )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=space,
            frame_uid=frame_uid,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task=task,
            codec=codec,
        )

    def add_classification(
        self,
        ann_id: str,
        labels: Mapping[int | str, float],
        *,
        scope: str = "sample",
        multilabel: bool = True,
        scope_ids: Sequence[int] | None = None,
        schemes: Sequence[str] | None = None,
        scheme_values: Sequence[str] | None = None,
        grid: str | None = None,
        annotated_classes: str | Sequence[int | str] = "all_given",
        closure: str = "explicit",
        timepoints: Sequence[str] | None = None,
        prov: Activity | str | None = None,
        quality: str | Mapping[str, Any] | None = None,
        derived_from: Sequence[str] = (),
        codec: str | None = None,
    ) -> h5py.Group:
        """Write a classification annotation (spec §9).

        A **change label** is this call with ``scope="sample"`` and an explicit
        ``timepoints`` list naming the visits compared --- the format adds no
        ``change`` kind, because what makes a change label well defined is that
        the compared timepoints are named rather than implied.
        """
        payload = encode_classification(
            {self._class_id(k): v for k, v in labels.items()},
            scope=scope,
            multilabel=multilabel,
            scope_ids=scope_ids,
            schemes=schemes,
            scheme_values=scheme_values,
        )
        if scope == "timepoint" and scope_ids is not None:
            declared = len(self._document.timepoints)
            unknown = sorted({int(v) for v in scope_ids if not 0 <= int(v) < declared})
            if unknown:
                raise MEDH5ValidationError(
                    f"annotation {ann_id!r}: scope='timepoint' scope_ids {unknown} "
                    f"are not timepoint indices (0..{declared - 1})",
                    code="E409",
                )
        return self._add_object_annotation(
            ann_id,
            payload,
            grid=grid,
            space=None,
            frame_uid=None,
            annotated_classes=annotated_classes,
            closure=closure,
            timepoints=timepoints,
            prov=prov,
            quality=quality,
            derived_from=derived_from,
            task="classification",
            codec=codec,
        )

    # -- transforms (§10) --------------------------------------------------

    def add_transform(
        self,
        transform_id: str,
        *,
        kind: str,
        from_frame: str,
        to_frame: str,
        matrix: npt.ArrayLike | None = None,
        field: npt.ArrayLike | None = None,
        control_points: npt.ArrayLike | None = None,
        components: Sequence[str] | None = None,
        field_grid: str | None = None,
        cp_grid: str | None = None,
        vector_space: str = "world",
        interpolation: str = "linear",
        extrapolation: str = "zero",
        order: int = 3,
        units: str = "mm",
        from_grid: str | None = None,
        to_grid: str | None = None,
        invertible: bool | None = None,
        inverse_id: str | None = None,
        metrics: str | Mapping[str, Any] | None = None,
        prov: Activity | str | None = None,
        codec: str | None = None,
    ) -> h5py.Group:
        """Write a transform mapping points from *from_frame* to *to_frame* (§10).

        The direction is fixed and unswitchable: ``x_M = T(x_F)``, the ITK
        convention.  To warp a moving image onto a fixed grid, evaluate T at each
        fixed-grid point.  There is no attribute to select the other convention,
        because ambiguity here is the leading cause of silently mirrored results.
        """
        check_transform_id(transform_id)
        if from_frame == to_frame:
            raise MEDH5ValidationError(
                f"transform {transform_id!r} maps {from_frame!r} to itself; grids that "
                "share a frame need no transform (§3.4)",
                code="E502",
            )
        payload = self._encode_transform(
            transform_id,
            kind,
            matrix=matrix,
            field=field,
            control_points=control_points,
            components=components,
            field_grid=field_grid,
            cp_grid=cp_grid,
            vector_space=vector_space,
            interpolation=interpolation,
            extrapolation=extrapolation,
            order=order,
            from_frame=from_frame,
        )
        header = TransformHeader(
            kind=kind,
            from_frame=from_frame,
            to_frame=to_frame,
            units=units,
            from_grid=from_grid,
            to_grid=to_grid,
            invertible=invertible,
            inverse_id=inverse_id,
            prov=_prov_id(prov),
            metrics=self._quality_key(transform_id, metrics),
            extra=dict(payload.attrs),
        )
        node = self._file.require_group("transforms")
        if transform_id in node:
            raise MEDH5ValidationError(f"transform {transform_id!r} already exists")
        group = node.create_group(transform_id)
        for name, array in payload.datasets.items():
            chunks = None
            if name in ("field", "control_points") and field_grid is not None:
                grid = self._grids.get(field_grid)
                if grid is not None and array.ndim == grid.n_spatial + 1:
                    chunks = (
                        1,
                        *self._chunks_for(grid, array.dtype.itemsize)[
                            -grid.n_spatial :
                        ],
                    )
                    chunks = tuple(
                        min(int(c), int(s))
                        for c, s in zip(chunks, array.shape, strict=True)
                    )
            group.create_dataset(
                name,
                data=array,
                **dataset_kwargs(
                    array.shape,
                    array.dtype,
                    profile=codec or self.codec,
                    role="label",
                    chunks=chunks,
                ),
            )
        set_attrs(group, header.attrs())
        self._transform_frames[transform_id] = (from_frame, to_frame)
        return group

    def _encode_transform(
        self,
        transform_id: str,
        kind: str,
        *,
        matrix: npt.ArrayLike | None,
        field: npt.ArrayLike | None,
        control_points: npt.ArrayLike | None,
        components: Sequence[str] | None,
        field_grid: str | None,
        cp_grid: str | None,
        vector_space: str,
        interpolation: str,
        extrapolation: str,
        order: int,
        from_frame: str,
    ) -> AnnotationPayload:
        if kind == "identity":
            return encode_identity()
        if kind == "affine":
            if matrix is None:
                raise MEDH5ValidationError(
                    f"transform {transform_id!r}: kind 'affine' needs `matrix`",
                    code="E502",
                )
            return encode_affine(matrix)
        if kind == "displacement":
            if field is None or field_grid is None:
                raise MEDH5ValidationError(
                    f"transform {transform_id!r}: kind 'displacement' needs `field` "
                    "and `field_grid`",
                    code="E503",
                )
            self._check_field_frame(transform_id, field_grid, from_frame)
            return encode_displacement(
                field,
                field_grid=field_grid,
                vector_space=vector_space,
                interpolation=interpolation,
                extrapolation=extrapolation,
            )
        if kind == "bspline":
            if control_points is None or cp_grid is None:
                raise MEDH5ValidationError(
                    f"transform {transform_id!r}: kind 'bspline' needs "
                    "`control_points` and `cp_grid`",
                    code="E503",
                )
            self._check_field_frame(transform_id, cp_grid, from_frame)
            return encode_bspline(
                control_points, cp_grid=cp_grid, order=order, vector_space=vector_space
            )
        if kind == "composite":
            if components is None:
                raise MEDH5ValidationError(
                    f"transform {transform_id!r}: kind 'composite' needs `components`",
                    code="E501",
                )
            missing = [c for c in components if c not in self._transform_frames]
            if missing:
                raise MEDH5ValidationError(
                    f"transform {transform_id!r} names components {missing} that do "
                    "not exist yet; declare them first",
                    code="E501",
                )
            return encode_composite(components)
        raise MEDH5ValidationError(
            f"unknown transform kind {kind!r}; expected one of {list(TRANSFORM_KINDS)}",
            code="E502",
        )

    def _check_field_frame(
        self, transform_id: str, grid_id: str, from_frame: str
    ) -> None:
        """A field is sampled at the points being mapped, so it lives in the source."""
        grid = self._grid(grid_id)
        if grid.frame_uid is not None and grid.frame_uid != from_frame:
            raise MEDH5ValidationError(
                f"transform {transform_id!r}: grid {grid_id!r} is in frame "
                f"{grid.frame_uid!r} but the transform starts in {from_frame!r}; the "
                "field must be sampled in the source frame",
                code="E503",
            )

    # -- index -------------------------------------------------------------

    def build_index(
        self,
        ann_ids: Sequence[str] | None = None,
        *,
        max_coords: int = DEFAULT_MAX_COORDS,
        occupancy: int | None = DEFAULT_OCCUPANCY_FACTOR,
        seed: int = 0,
    ) -> tuple[str, ...]:
        """Build sampling indices for the named voxel annotations (spec §14.3)."""
        from medh5.integrity.digest import group_digest

        names = (
            list(ann_ids)
            if ann_ids is not None
            else [
                name
                for name, kind in self._annotation_kinds.items()
                if kind in VOXEL_KINDS and kind != "mask"
            ]
        )
        built: list[str] = []
        for name in names:
            group = self._file["annotations"][name]
            annotation = open_annotation(
                name, group, self._grids, self._document.label_set
            )
            if not isinstance(annotation, VoxelAnnotation):
                continue
            payload = build_index(
                annotation,
                max_coords=max_coords,
                occupancy=occupancy,
                seed=seed,
                source_digest=group_digest(group, root=self._file),
            )
            write_index(self._file, payload)
            built.append(name)
        return tuple(built)

    # -- commit ------------------------------------------------------------

    def infer_profiles(self) -> frozenset[str]:
        """Profiles this sample actually satisfies, unioned with declared ones."""
        kinds = set(self._annotation_kinds.values())
        found = {"core"}
        doc = self._document
        if kinds & {"labelmap", "layers", "bitmask", "instances", "probmap"} and (
            doc.label_set
        ):
            found.add("seg")
        if doc.label_set and self._has_task("detection"):
            found.add("det")
        if "classification" in kinds and doc.label_set:
            found.add("cls")
        if self._transform_frames:
            found.add("reg")
        annotated = set(self._annotation_kinds)
        if (
            doc.provenance
            and annotated
            and all("quality" in self._file["annotations"][a].attrs for a in annotated)
        ):
            found.add("curation")
        if self._images and all(self._image_multiscale.get(i) for i in self._images):
            found.add("multiscale")
        if "index" in self._file and len(self._file["index"]):
            found.add("training")
        if doc.timepoints.is_longitudinal and all(
            g.timepoint for g in self._grids.values()
        ):
            found.add("longitudinal")
        unknown = (self._declared_profiles | found) - set(PROFILES)
        if unknown:
            raise MEDH5ValidationError(
                f"unknown profile(s) {sorted(unknown)}", code="E007"
            )
        return frozenset(found | self._declared_profiles)

    def _has_task(self, task: str) -> bool:
        """Whether any annotation declares this task.

        `det` is about the *task* an annotation serves, not merely about it being
        a §8 kind: contours and a surface mesh are geometry in service of
        segmentation (§6.3), and claiming a detection profile for them would make
        the profile mean nothing.
        """
        node = self._file["annotations"]
        return any(as_str(node[name].attrs.get("task", "")) == task for name in node)

    def _check_references(self) -> None:
        doc = self._document
        declared = set(doc.timepoints.ids)
        multi = len(declared) > 1
        for grid in self._grids.values():
            if grid.timepoint is None:
                if multi:
                    raise MEDH5ValidationError(
                        f"grid {grid.grid_id!r} has no `timepoint`, but the sample "
                        f"declares {len(declared)} timepoints",
                        code="E106",
                    )
            elif grid.timepoint not in declared:
                raise MEDH5ValidationError(
                    f"grid {grid.grid_id!r} names undeclared timepoint "
                    f"{grid.timepoint!r}",
                    code="E107",
                )
        label_set = doc.label_set
        for name in self._annotation_kinds:
            node = self._file["annotations"][name]
            if label_set is not None and "class_ids" in node.attrs:
                missing = label_set.missing(
                    int(c) for c in np.atleast_1d(node.attrs["class_ids"])
                )
                if missing:
                    raise MEDH5ValidationError(
                        f"annotation {name!r} uses class ids {list(missing)} that "
                        f"are not in label set {label_set.id!r}",
                        code="E402",
                    )
            prov = node.attrs.get("prov")
            if prov is not None and not doc.provenance.has_activity(as_str(prov)):
                raise MEDH5ValidationError(
                    f"annotation {name!r} names activity {as_str(prov)!r}, which is "
                    "not in the provenance graph",
                    code="E601",
                )
            quality = node.attrs.get("quality")
            if quality is not None and as_str(quality) not in doc.quality:
                raise MEDH5ValidationError(
                    f"annotation {name!r} names quality record {as_str(quality)!r}, "
                    "which does not exist",
                    code="E602",
                )
            tps = node.attrs.get("timepoints")
            if tps is not None:
                for tp in as_str_tuple(tps):
                    if tp not in declared:
                        raise MEDH5ValidationError(
                            f"annotation {name!r} names undeclared timepoint {tp!r}",
                            code="E409",
                        )
            ignore_mask = node.attrs.get("ignore_mask")
            if ignore_mask is not None:
                self._check_mask_reference(
                    f"annotation {name!r}", "ignore_mask", as_str(ignore_mask), node
                )
            derived = node.attrs.get("derived_from")
            if derived is not None:
                for reference in as_str_tuple(derived):
                    if annotation_id(reference) not in self._annotation_kinds:
                        raise MEDH5ValidationError(
                            f"annotation {name!r}: derived_from names "
                            f"{reference!r}, which does not exist",
                            code="E413",
                        )
        for image_id in self._images:
            image_node = self._file["images"][image_id]
            prov = image_node.attrs.get("prov")
            if prov is not None and not doc.provenance.has_activity(as_str(prov)):
                raise MEDH5ValidationError(
                    f"image {image_id!r} names activity {as_str(prov)!r}, which is "
                    "not in the provenance graph",
                    code="E601",
                )
            valid_mask = image_node.attrs.get("valid_mask")
            if valid_mask is not None:
                self._check_mask_reference(
                    f"image {image_id!r}", "valid_mask", as_str(valid_mask), image_node
                )
        for name in self._transform_frames:
            attrs = self._file["transforms"][name].attrs
            prov = attrs.get("prov")
            if prov is not None and not doc.provenance.has_activity(as_str(prov)):
                raise MEDH5ValidationError(
                    f"transform {name!r} names activity {as_str(prov)!r}, which is "
                    "not in the provenance graph",
                    code="E601",
                )
            metrics = attrs.get("metrics")
            if metrics is not None and as_str(metrics) not in doc.quality:
                raise MEDH5ValidationError(
                    f"transform {name!r} names quality record {as_str(metrics)!r}, "
                    "which does not exist",
                    code="E602",
                )
        dangling = doc.provenance.dangling_agent_refs()
        if dangling:
            pairs = ", ".join(f"{a} -> {g}" for a, g in dangling)
            raise MEDH5ValidationError(
                f"activities name undeclared agents: {pairs}", code="E605"
            )

    def _check_mask_reference(
        self, owner: str, attr: str, target: str, node: h5py.HLObject
    ) -> None:
        """``ignore_mask`` and ``valid_mask`` name a `mask` on the owner's grid."""
        kind = self._annotation_kinds.get(target)
        if kind is None:
            raise MEDH5ValidationError(
                f"{owner}: {attr} names annotation {target!r}, which does not exist",
                code="E413",
            )
        if kind != "mask":
            raise MEDH5ValidationError(
                f"{owner}: {attr} names {target!r}, whose kind is {kind!r}; §4.4 and "
                "§7.7 require a `mask` annotation",
                code="E413",
            )
        mine = node.attrs.get("grid")
        theirs = self._file["annotations"][target].attrs.get("grid")
        if mine is not None and theirs is not None and as_str(mine) != as_str(theirs):
            raise MEDH5ValidationError(
                f"{owner}: {attr} names {target!r} on grid {as_str(theirs)!r}, but "
                f"this object is on grid {as_str(mine)!r}",
                code="E413",
            )

    def commit(self, *, digests: bool = True) -> str | None:
        """Validate, write ``/meta``, stamp digests and atomically replace.

        ``digests=False`` stamps only the datasets that do not already carry a
        digest --- on an amend, the ones this writer added, since the rest came
        through the copy with theirs intact --- and **always** computes
        ``content_id``.  The root address is what makes a file citable (§13.2),
        and it used to be dropped entirely under this flag, so an amend that
        skipped digests for speed turned an addressed file into an unaddressed
        one and every cache keyed on it missed.
        """
        from medh5.integrity.digest import compute_content_id, stamp_digests

        if self._committed:
            return None
        if not self._images:
            raise MEDH5ValidationError(
                "a sample must contain at least one image", code="E201"
            )
        self._check_references()
        errors = self._document.check_schema()
        if errors:
            raise MEDH5ValidationError(
                "sample document fails its JSON Schema: " + "; ".join(errors[:5]),
                code="E005",
            )
        write_document(self._file, self._document)
        profiles = self.infer_profiles()
        set_attrs(
            self._file,
            {
                "medh5_version": FORMAT_VERSION,
                "medh5_kind": "sample",
                "medh5_profiles": sorted(profiles),
                "created": _utcnow(),
                "generator": f"medh5 {__about__.__version__}",
                "digest_algo": "sha256",
            },
        )
        stamp_digests(self._file, only_missing=not digests)
        content_id = compute_content_id(self._file, attr_name_map_of(self._file))
        self._file.attrs["content_id"] = content_id
        self._committed = True
        self._stack.close()
        return content_id


def _default_axis_kinds(ndim: int) -> tuple[str, ...]:
    if ndim in (2, 3):
        return ("spatial",) * ndim
    raise MEDH5ValidationError(
        f"a {ndim}-D grid needs explicit axis_kinds: only 2-D and 3-D have an "
        "unambiguous default"
    )


def _default_axis_names(axis_kinds: Sequence[str]) -> tuple[str, ...]:
    spatial = ("z", "y", "x")[-tuple(axis_kinds).count("spatial") :]
    out: list[str] = []
    cursor = 0
    for kind in axis_kinds:
        if kind == "spatial":
            out.append(spatial[cursor])
            cursor += 1
        else:
            out.append(kind[0])
    return tuple(out)


def _prov_id(prov: Activity | str | None) -> str | None:
    if prov is None:
        return None
    return prov.id if isinstance(prov, Activity) else prov


def create(
    path: str | os.PathLike[str],
    *,
    sample_id: str | None = None,
    subject_id: str | None = None,
    codec: str = "balanced",
    profiles: Sequence[str] = (),
) -> SampleWriter:
    """Create a new sample.  Use as a context manager; exit commits."""
    return SampleWriter(
        path,
        sample_id=sample_id,
        subject_id=subject_id,
        codec=codec,
        profiles=profiles,
    )


def amend(path: str | os.PathLike[str], *, codec: str = "balanced") -> SampleWriter:
    """Copy-on-write amend: build a new file from the old and replace it (§14.4).

    Unknown objects --- including ones written by a future minor version --- are
    copied through untouched, so amending never silently drops what this reader
    does not understand.
    """
    source = open_h5(path, "r")
    try:
        writer = SampleWriter(path, codec=codec, source=source)
    finally:
        source.close()
    return writer


__all__ = ["SampleWriter", "amend", "create"]
