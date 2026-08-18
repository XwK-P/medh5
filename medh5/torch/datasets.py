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

    def _annotation_at(self, sample: Sample, timepoint: str) -> str | None:
        if self.annotation is not None:
            return self.annotation
        for name in sorted(sample.at(timepoint).annotations):
            if sample.annotations[name].kind != "classification":
                return name
        return None

    def _patch_for(
        self, sample: Sample, timepoint: str, rng: np.random.Generator
    ) -> Patch:
        """A window in *timepoint*'s own grid, annotated or not.

        The grid is named rather than left to the sampler.  `_annotation_at`
        answers `None` for a visit with no voxel annotation, and `None` on its
        own tells the sampler to go and find one anywhere in the sample --- so a
        pair whose follow-up is annotated and whose baseline is not drew the
        baseline window in the follow-up's grid.
        """
        return self.sampler.draw(
            sample,
            self._annotation_at(sample, timepoint),
            rng,
            grid=self._grid_at(sample, timepoint).grid_id,
        )

    def _corresponding(
        self, sample: Sample, pair: TimepointPair, patch: Patch
    ) -> Patch:
        """The window in the second visit covering the same anatomy."""
        target = self._grid_at(sample, pair.second)
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
        source = self._grid_at(sample, pair.first)
        target = self._grid_at(sample, pair.second)
        world = source.index_to_world(np.asarray([patch.center], dtype=np.float64))
        transform = sample.transform_between(pair.first, pair.second)
        if transform is None and source.frame_uid != target.frame_uid:
            # `transform_between` returns None for two different reasons: the
            # grids already share a frame (nothing to apply), or no path exists
            # between them.  Only the first makes the coordinates comparable.
            # Treating the second as "no transform needed" feeds source-frame
            # world coordinates straight into an unrelated grid and returns
            # paired patches from different anatomy, which trains quietly.
            raise MEDH5ValidationError(
                f"{sample.path}: align='transform' needs a transform relating "
                f"{pair.first!r} (frame {source.frame_uid!r}) to {pair.second!r} "
                f"(frame {target.frame_uid!r}), and the file has none; register "
                "the visits, or use align='none' to read the same index window "
                "from both"
            )
        if transform is not None:
            world = transform.transform_points(world)
        index = target.world_to_index(world)[0]
        return tuple(int(round(float(v))) for v in index)

    def _read_at(
        self, sample: Sample, timepoint: str, patch: Patch
    ) -> dict[str, npt.NDArray[Any]]:
        view = sample.at(timepoint)
        wanted = (
            [i for i in self.images if i in view.images]
            if self.images is not None
            else sorted(view.images)
        )
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
