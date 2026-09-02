"""PyTorch datasets over MEDH5 samples (implementation plan §2.3).

Three datasets, one contract: every item is a ``dict`` with ``images``,
``label`` and ``meta``.  Keeping the shape uniform is what lets a training
script swap whole-volume for patch-based sampling by changing one constructor.

Two things here are load-bearing:

* **Reads are single-call.**  Every array comes out of one ``dataset[roi]`` and
  never ``dataset[...][roi]``, because the latter materialises the whole volume
  first --- 40× slower for a 64³ ROI out of a 160³ bitplane (§14.5).  The API
  offers no way to write the slow form.
* **Files stay open per worker, keyed by PID.**  See :mod:`medh5.torch.handles`.

Labels come back in one of three formats.  ``onehot`` is ``(C, *patch)`` float
planes in the caller's class order; ``labelmap`` is a single ``uint16`` plane
with ties broken by that same order; ``instances`` returns the objects
overlapping the patch.  The class order is always the one the caller asked for,
never the file's storage order, so channel *i* means the same structure across
a cohort whose files were written by different tools.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.sample import Sample, open_sample
from medh5.sampling import (
    PairReport,
    Patch,
    PatchSampler,
    TimepointPair,
    TimepointPairSampler,
    grid_patches,
)
from medh5.torch._compat import dataset_base, require_torch, to_tensor
from medh5.torch.handles import open_cached
from medh5.transforms.resolve import resolve_between

LABEL_FORMATS = ("onehot", "labelmap", "instances", "none")
ALIGNMENTS = ("none", "transform")

PathLike = str | os.PathLike[str]


def _check_label_format(name: str) -> str:
    if name not in LABEL_FORMATS:
        raise MEDH5ValidationError(
            f"unknown label_format {name!r}; expected one of {list(LABEL_FORMATS)}"
        )
    return name


_DatasetBase = dataset_base()


class _Base(_DatasetBase):  # type: ignore[misc,valid-type]
    """Shared reading logic, over ``torch.utils.data.Dataset`` when it exists."""

    def __init__(
        self,
        paths: Sequence[PathLike],
        *,
        images: Sequence[str] | None = None,
        annotations: Mapping[str, Sequence[int | str]] | None = None,
        label_format: str = "onehot",
        physical: bool = True,
        dtype: npt.DTypeLike = np.float32,
        timepoint: str | None = None,
    ) -> None:
        require_torch()
        self.paths = [str(p) for p in paths]
        if not self.paths:
            raise MEDH5ValidationError("a dataset needs at least one file")
        self.images = None if images is None else tuple(images)
        self.annotations = (
            {} if annotations is None else {k: tuple(v) for k, v in annotations.items()}
        )
        self.label_format = _check_label_format(label_format)
        self.physical = bool(physical)
        self.dtype = np.dtype(dtype)
        self.timepoint = timepoint

    # -- reading -----------------------------------------------------------

    def _sample(self, path: str) -> Sample:
        return open_cached(path)

    @staticmethod
    @contextmanager
    def _scan(path: str) -> Iterator[Sample]:
        """A short-lived open, for reading metadata while building a plan.

        Deliberately not ``open_cached``.  ``CACHE`` is process-global and
        shared by every dataset in the worker, so walking a 10 000-file cohort
        through its 32-entry LRU evicts whatever training was about to reuse and
        leaves it holding the tail of the cohort instead --- entries chosen by
        the order the plan was built in rather than by what is being read.  The
        pass itself is unavoidable: ``__len__`` is not knowable without one
        metadata read per file.
        """
        sample = open_sample(path)
        try:
            yield sample
        finally:
            sample.close()

    def _image_ids(self, sample: Sample) -> tuple[str, ...]:
        if self.images is not None:
            missing = [i for i in self.images if i not in sample.images]
            if missing:
                raise MEDH5ValidationError(
                    f"{sample.path}: no image(s) {missing}; present: "
                    f"{sorted(sample.images)}"
                )
            return self.images
        if self.timepoint is not None:
            return tuple(sorted(sample.at(self.timepoint).images))
        return tuple(sorted(sample.images))

    @staticmethod
    def _single_grid(sample: Sample, members: Mapping[str, str]) -> str:
        """The one grid *members* share, or a refusal to guess across grids.

        A patch is a window in a single grid's index space.  Reading a second
        grid with those same slices assumes both index the same anatomy at the
        same spacing, which nothing in the format guarantees, and a smaller grid
        silently returns a truncated array because the padding was computed for
        the first.  Converters refuse rather than resample (§3); so does the
        loader --- narrowing the selection or resampling is the caller's call
        to make deliberately.
        """
        grids = sorted(set(members.values()))
        if len(grids) > 1:
            listed = ", ".join(f"{k} on {v!r}" for k, v in sorted(members.items()))
            raise MEDH5ValidationError(
                f"{sample.path}: a patch is a window in one grid, but this "
                f"selection spans {len(grids)} of them ({listed}); restrict "
                "`images=`/`annotations=` to one grid, scope the dataset with "
                "`timepoint=`, or resample the volumes onto a common grid first"
            )
        return grids[0]

    def _check_patch_grid(self, sample: Sample, patch: Patch) -> None:
        """Everything a patch window is read out of must sit on the window's grid.

        The window itself is a member of the comparison, not just the objects
        being read.  Checking only the objects passes whenever they happen to
        agree with each other --- a single image on grid B read with a window
        drawn on annotation grid A looks like one grid and is silently
        misregistered, and truncated wherever B is the smaller of the two.
        """
        members = {
            f"image {name!r}": sample.images[name].grid_id
            for name in self._image_ids(sample)
        }
        if patch.grid_id is not None:
            members["the patch window"] = patch.grid_id
        if self.label_format != "none":
            for name in self.annotations:
                if name not in sample.annotations:
                    continue
                grid_id = sample.annotations[name].grid_id
                if grid_id is not None:
                    members[f"annotation {name!r}"] = grid_id
        if members:
            self._single_grid(sample, members)

    def _read_images(
        self, sample: Sample, patch: Patch | None
    ) -> dict[str, npt.NDArray[Any]]:
        roi = None if patch is None else patch.slices
        out: dict[str, npt.NDArray[Any]] = {}
        for image_id in self._image_ids(sample):
            image = sample.images[image_id]
            array = image.read(roi, physical=self.physical, dtype=self.dtype)
            out[image_id] = array if patch is None else patch.apply_padding(array)
        return out

    def _read_labels(self, sample: Sample, patch: Patch | None) -> dict[str, Any]:
        if self.label_format == "none" or not self.annotations:
            return {}
        roi = None if patch is None else list(patch.slices)
        out: dict[str, Any] = {}
        for ann_id, classes in self.annotations.items():
            if ann_id not in sample.annotations:
                raise MEDH5ValidationError(
                    f"{sample.path}: no annotation {ann_id!r}; present: "
                    f"{sorted(sample.annotations)}"
                )
            ann = sample.annotations[ann_id]
            wanted = list(classes) if classes else None
            if self.label_format == "instances":
                out[ann_id] = self._instances_in(ann, patch)
                continue
            if self.label_format == "labelmap":
                array = np.asarray(ann.labelmap(roi=roi, priority=wanted))
            else:
                array = np.asarray(ann.dense(wanted, roi=roi), dtype=self.dtype)
            out[ann_id] = array if patch is None else patch.apply_padding(array)
        return out

    @staticmethod
    def _instances_in(ann: Any, patch: Patch | None) -> list[dict[str, Any]]:
        """Objects overlapping the patch, with boxes in patch coordinates."""
        objects = []
        for obj in ann.instances():
            box = np.asarray(obj.box, dtype=np.float64)
            if patch is not None:
                offset = np.asarray([s.start for s in patch.slices], dtype=np.float64)
                extent = np.asarray(
                    [s.stop - s.start for s in patch.slices], dtype=np.float64
                )
                local = box - offset[:, None]
                if np.any(local[:, 1] < 0) or np.any(local[:, 0] > extent):
                    continue
                box = local
            objects.append(
                {
                    "instance_id": obj.instance_id,
                    "class_id": obj.class_id,
                    "box": box.astype(np.float32),
                    "score": obj.score,
                }
            )
        return objects

    def _meta(self, sample: Sample, patch: Patch | None) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "path": sample.path,
            "sample_id": sample.identity.sample_id,
            "subject_id": sample.identity.subject_id,
        }
        if patch is not None:
            meta["patch"] = patch.to_json()
        return meta

    def _item(
        self,
        sample: Sample,
        patch: Patch | None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if patch is not None:
            self._check_patch_grid(sample, patch)
        images = self._read_images(sample, patch)
        item: dict[str, Any] = {
            "images": {k: to_tensor(v) for k, v in images.items()},
            "meta": {**self._meta(sample, patch), **(extra or {})},
        }
        labels = self._read_labels(sample, patch)
        if labels:
            item["label"] = {
                k: (v if isinstance(v, list) else to_tensor(v))
                for k, v in labels.items()
            }
        return item


class VolumeDataset(_Base):
    """One item per file: the whole volume, eagerly read.

    For validation and inference on volumes that fit in memory.  Training on
    3-D volumes almost always wants :class:`PatchDataset` instead.
    """

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._sample(self.paths[index])
        return self._item(sample, None)


class PatchDataset(_Base):
    """``samples_per_volume`` patches from each file, drawn by a sampler.

    Length is ``len(paths) * samples_per_volume`` so that an epoch is a
    well-defined number of steps; *which* patches those are is redrawn every
    epoch, seeded per item so a run is reproducible and workers do not
    duplicate each other's draws.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        sampler: PatchSampler,
        *,
        samples_per_volume: int = 1,
        annotation: str | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, **kwargs)
        self.sampler = sampler
        self.samples_per_volume = int(samples_per_volume)
        if self.samples_per_volume < 1:
            raise MEDH5ValidationError("samples_per_volume must be at least 1")
        self.annotation = annotation or (
            next(iter(self.annotations)) if self.annotations else None
        )
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Redraw patches next epoch --- call it from the training loop."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths) * self.samples_per_volume

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index // self.samples_per_volume]
        sample = self._sample(path)
        rng = np.random.default_rng((self.seed, self.epoch, index))
        patch = self.sampler.draw(sample, self.annotation, rng)
        return self._item(sample, patch)


class GridPatchDataset(_Base):
    """Deterministic sliding-window cover of every file --- the inference path."""

    def __init__(
        self,
        paths: Sequence[PathLike],
        patch_size: int | Sequence[int],
        *,
        overlap: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, **kwargs)
        self.patch_size = patch_size
        self.overlap = int(overlap)
        self._plan: list[tuple[str, Patch]] = []
        for path in self.paths:
            with self._scan(path) as sample:
                reference = sample.reference_grid
                shape, grid_id = reference.spatial_shape, reference.grid_id
            self._plan.extend(
                (path, patch)
                for patch in grid_patches(
                    shape, patch_size, overlap=self.overlap, grid_id=grid_id
                )
            )

    def __len__(self) -> int:
        return len(self._plan)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, patch = self._plan[index]
        return self._item(self._sample(path), patch)


class PairedPatchDataset(_Base):
    """Corresponding patches from two visits of one subject (§3.7, §10).

    ``align="transform"`` maps the patch centre through the transform relating
    the two frames, so the two patches cover the same anatomy.  ``align="none"``
    reads the same index window from both, which is what a model that learns
    its own alignment wants.

    A cross-sectional file contributes no pairs.  The count is reported through
    :attr:`report` rather than absorbed: a dataset that silently drops most of
    its files looks exactly like one that is training normally.
    """

    def __init__(
        self,
        paths: Sequence[PathLike],
        sampler: PatchSampler,
        *,
        pair_sampler: TimepointPairSampler | None = None,
        align: str = "transform",
        samples_per_pair: int = 1,
        annotation: str | None = None,
        label: str | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(paths, **kwargs)
        if align not in ALIGNMENTS:
            raise MEDH5ValidationError(
                f"unknown align {align!r}; expected one of {list(ALIGNMENTS)}"
            )
        self.sampler = sampler
        self.pair_sampler = pair_sampler or TimepointPairSampler()
        self.align = align
        self.samples_per_pair = int(samples_per_pair)
        self.annotation = annotation
        self.label = label
        self.seed = int(seed)
        self.epoch = 0
        self.report = PairReport()
        self._plan: list[tuple[str, TimepointPair]] = []
        for path in self.paths:
            with self._scan(path) as sample:
                pairs = self.pair_sampler.pairs(sample)
            self.report.files += 1
            if not pairs:
                self.report.add_skip(path)
                continue
            self._plan.extend((path, pair) for pair in pairs)
        self.report.pairs = len(self._plan)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._plan) * self.samples_per_pair

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, pair = self._plan[index // self.samples_per_pair]
        sample = self._sample(path)
        rng = np.random.default_rng((self.seed, self.epoch, index))
        first = self._patch_for(sample, pair.first, rng)
        second = self._corresponding(sample, pair, first)
        item = {
            "images": {
                pair.first: {
                    k: to_tensor(v)
                    for k, v in self._read_at(sample, pair.first, first).items()
                },
                pair.second: {
                    k: to_tensor(v)
                    for k, v in self._read_at(sample, pair.second, second).items()
                },
            },
            "meta": {
                **self._meta(sample, first),
                "pair": [pair.first, pair.second],
                "interval_days": pair.interval_days,
                "aligned": self.align,
                "patches": {
                    pair.first: first.to_json(),
                    pair.second: second.to_json(),
                },
            },
        }
        label_id = self.label or pair.label
        if label_id and label_id in sample.annotations:
            ann = sample.annotations[label_id]
            item["label"] = {label_id: dict(ann.labels)}
            item["meta"]["label_annotation"] = label_id
        return item

    # -- internals ---------------------------------------------------------

    def _annotation_at(
        self, sample: Sample, timepoint: str, grid: str | None = None
    ) -> str | None:
        """The annotation to draw *timepoint*'s foreground from, if there is one.

        Auto-selection is confined to *grid* --- the space the images being read
        live in --- so a visit holding both a CT and a PET grid picks the
        annotation belonging to the modality that was asked for rather than
        whichever annotation sorts first.
        """
        if self.annotation is not None:
            return self.annotation
        for name in sorted(sample.at(timepoint).annotations):
            annotation = sample.annotations[name]
            if annotation.kind == "classification":
                continue
            if grid is not None and annotation.grid_id != grid:
                continue
            return name
        return None

    def _images_at(self, sample: Sample, timepoint: str) -> list[str]:
        """The images this dataset reads at *timepoint*."""
        view = sample.at(timepoint)
        if self.images is None:
            return sorted(view.images)
        return [name for name in self.images if name in view.images]

    def _window_grid_at(self, sample: Sample, timepoint: str) -> str:
        """The grid every window for *timepoint* is measured in.

        Taken from the data this dataset will actually read at that visit
        rather than from whichever grid sorts first: a visit may legitimately
        hold a CT grid and a PET grid, and pinning the window to the
        alphabetical winner refused the PET pair instead of sampling it.  Both
        ends of a pair go through here, so the window drawn in the first visit
        and the one derived for the second agree about what they are measuring.
        """
        names = self._images_at(sample, timepoint)
        if not names:
            return str(self._grid_at(sample, timepoint).grid_id)
        return self._single_grid(
            sample, {f"image {n!r}": sample.images[n].grid_id for n in names}
        )

    def _patch_for(
        self, sample: Sample, timepoint: str, rng: np.random.Generator
    ) -> Patch:
        """A window in *timepoint*'s own grid, annotated or not.

        The grid is named rather than left to the sampler.  `_annotation_at`
        answers `None` for a visit with no voxel annotation, and `None` on its
        own tells the sampler to go and find one anywhere in the sample --- so a
        pair whose follow-up is annotated and whose baseline is not drew the
        baseline window in the follow-up's grid.  It comes from the images
        being read rather than from the visit's first grid, so a visit holding
        more than one modality is sampled in the one that was asked for.
        """
        grid = self._window_grid_at(sample, timepoint)
        return self.sampler.draw(
            sample, self._annotation_at(sample, timepoint, grid), rng, grid=grid
        )

    def _corresponding(
        self, sample: Sample, pair: TimepointPair, patch: Patch
    ) -> Patch:
        """The window in the second visit covering the same anatomy."""
        target = sample.grids[self._window_grid_at(sample, pair.second)]
        shape = target.spatial_shape
        # The *requested* size, not the clipped extent.  Where the first visit
        # is smaller than the patch its window is short and gets padded back up
        # to `patch.shape`; asking the second visit for the clipped extent would
        # return a full, unpadded array of that smaller size and leave the pair
        # with tensors of different shapes.
        size = patch.shape
        if self.align == "none":
            from medh5.sampling import window_around

            slices, pad = window_around(patch.center, size, shape)
            return Patch(
                slices=slices,
                pad=pad,
                center=patch.center,
                strategy="paired",
                grid_id=target.grid_id,
            )
        center = self._map_center(sample, pair, patch)
        from medh5.sampling import window_around

        slices, pad = window_around(center, size, shape)
        return Patch(
            slices=slices,
            pad=pad,
            center=center,
            strategy="paired",
            grid_id=target.grid_id,
        )

    def _grid_at(self, sample: Sample, timepoint: str) -> Any:
        view = sample.at(timepoint)
        names = sorted(view.grids)
        if not names:
            raise MEDH5ValidationError(
                f"{sample.path}: timepoint {timepoint!r} has no grid"
            )
        return sample.grids[names[0]]

    def _map_center(
        self, sample: Sample, pair: TimepointPair, patch: Patch
    ) -> tuple[int, ...]:
        """Move the centre through the transform relating the two frames (§10.2)."""
        source = sample.grids[patch.grid_id or self._window_grid_at(sample, pair.first)]
        target = sample.grids[self._window_grid_at(sample, pair.second)]
        world = source.index_to_world(np.asarray([patch.center], dtype=np.float64))
        # Resolve between the two *frames*, not the two timepoints and not by
        # name.  A visit may hold a CT grid and a PET grid on different frames,
        # and a timepoint-level question searches every frame of one visit
        # against every frame of the other and returns the first path it finds
        # --- so a CT registration answered "yes" for a PET pair with no
        # registration of its own, and its displacement was then applied to PET
        # coordinates.  Nothing raised; the patches came back the right shape
        # from the wrong place.
        #
        # Asking `transform_between` for the two *grid ids* is not enough to
        # close that.  Grid ids and timepoint ids are separate namespaces (§2.3
        # scopes uniqueness to the group), so a grid may legitimately be named
        # `tp0`, and `Sample._frames_for` reads a key as a timepoint before it
        # reads it as a grid --- which puts the whole visit's frames back in
        # play and restores the bug for exactly the files most likely to hit it.
        # Here the grids are already in hand, so resolve frame to frame and let
        # no name be interpreted at all.
        transform = None
        if source.frame_uid and target.frame_uid:
            transform = resolve_between(
                dict(sample.transforms), source.frame_uid, target.frame_uid
            )
        if transform is None and source.frame_uid != target.frame_uid:
            # A `None` here has two possible meanings: the grids already share
            # a frame (nothing to apply), or no path exists between them.  Only
            # the first makes the coordinates comparable.
            # Treating the second as "no transform needed" feeds source-frame
            # world coordinates straight into an unrelated grid and returns
            # paired patches from different anatomy, which trains quietly.
            raise MEDH5ValidationError(
                f"{sample.path}: align='transform' needs a transform relating "
                f"grid {source.grid_id!r} at {pair.first!r} (frame "
                f"{source.frame_uid!r}) to grid {target.grid_id!r} at "
                f"{pair.second!r} (frame {target.frame_uid!r}), and the file has "
                "none; register those grids, or use align='none' to read the same "
                "index window from both"
            )
        if transform is not None:
            world = transform.transform_points(world)
        index = target.world_to_index(world)[0]
        return tuple(int(round(float(v))) for v in index)

    def _read_at(
        self, sample: Sample, timepoint: str, patch: Patch
    ) -> dict[str, npt.NDArray[Any]]:
        wanted = self._images_at(sample, timepoint)
        if wanted:
            members = {f"image {n!r}": sample.images[n].grid_id for n in wanted}
            if patch.grid_id is not None:
                members["the patch window"] = patch.grid_id
            self._single_grid(sample, members)
        out: dict[str, npt.NDArray[Any]] = {}
        for image_id in wanted:
            array = sample.images[image_id].read(
                patch.slices, physical=self.physical, dtype=self.dtype
            )
            out[image_id] = patch.apply_padding(array)
        return out


__all__ = [
    "ALIGNMENTS",
    "LABEL_FORMATS",
    "GridPatchDataset",
    "PairedPatchDataset",
    "PatchDataset",
    "VolumeDataset",
]
