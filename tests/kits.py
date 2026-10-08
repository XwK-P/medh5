"""Small samples the regression tests build.

Five audits found most of the suite's regression tests, and each wrote the
fixtures its findings needed.  They stay distinct because the tests depend on
what distinguishes them --- class keys, ids, frames, shapes --- and each is
defined once, here, under a name for what it builds:

* :class:`Hierarchy` --- a liver and a lesion inside it, on 8×10×10;
* :class:`Flat` --- a liver and a lesion, and a writer open on a CT, 8×10×10;
* :class:`Organs` --- liver, spleen and lesion with overlapping blocks and an
  ignored band, 8×12×12;
* :class:`Numbered` --- classes ``c1`` … ``cn`` on 8×12×12, for subject
  ``subj-1``;
* :class:`Framed` --- classes ``c1`` … ``cn`` on 8×16×16, the grid in a frame of
  reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.labels import LabelClass, LabelSet
from tests.helpers import block


class Hierarchy:
    """A liver and a lesion inside it (``parents``), on an 8×10×10 grid."""

    SHAPE = (8, 10, 10)
    BIG = 2**32 + 7
    """An instance id past ``uint32``."""

    @staticmethod
    def label_set(*extra: LabelClass) -> LabelSet:
        return LabelSet(
            "reg-1.2.1",
            version="1.0.0",
            classes=[
                LabelClass(1, "liver", "Liver"),
                LabelClass(3, "lesion", "Lesion", parents=[1]),
                *extra,
            ],
        )

    @staticmethod
    def mask(origin: tuple[int, int, int] = (2, 2, 2), size: int = 3) -> Any:
        return block(Hierarchy.SHAPE, origin, size)

    @staticmethod
    def image() -> Any:
        rng = np.random.default_rng(1)
        return rng.integers(-1000, 1500, Hierarchy.SHAPE).astype(np.int16)


class Flat:
    """A liver and a lesion, unrelated, and a writer open on one CT (8×10×10)."""

    SHAPE = (8, 10, 10)

    @staticmethod
    def label_set() -> LabelSet:
        return LabelSet(
            "rel-1.3.0",
            version="1.0.0",
            classes=[
                LabelClass(1, "liver", "Liver"),
                LabelClass(3, "lesion", "Lesion"),
            ],
        )

    @staticmethod
    def image() -> Any:
        rng = np.random.default_rng(3)
        return rng.integers(-1000, 1500, Flat.SHAPE).astype(np.int16)

    @staticmethod
    def mask(origin: tuple[int, int, int] = (2, 2, 2), size: int = 3) -> Any:
        return block(Flat.SHAPE, origin, size)

    @staticmethod
    def open_writer(path: Path, *, frames: bool = False) -> Any:
        """A writer with the label set, grid ``g`` at ``tp0`` and image ``CT``."""
        w = medh5.create(path, sample_id=path.stem)
        w.label_set(Flat.label_set())
        w.add_grid(
            "g",
            shape=Flat.SHAPE,
            spacing=(1.0, 1.0, 1.0),
            timepoint="tp0",
            frame_uid="f0" if frames else None,
        )
        w.add_image("CT", Flat.image(), grid="g", modality="CT")
        return w


class Organs:
    """Liver, spleen and lesion; the lesion overlaps the liver (8×12×12)."""

    SHAPE = (8, 12, 12)

    @staticmethod
    def label_set() -> LabelSet:
        return LabelSet(
            "rel-1.4.0",
            version="1.0.0",
            classes=[
                LabelClass(1, "liver", "Liver"),
                LabelClass(2, "spleen", "Spleen"),
                LabelClass(3, "lesion", "Lesion"),
            ],
        )

    @staticmethod
    def blocks() -> dict[int, Any]:
        liver = np.zeros(Organs.SHAPE, dtype=bool)
        liver[1:4, 1:6, 1:6] = True
        lesion = np.zeros(Organs.SHAPE, dtype=bool)
        lesion[2:3, 2:4, 2:4] = True  # inside the liver, so the classes overlap
        return {1: liver, 3: lesion}

    @staticmethod
    def ignore() -> Any:
        """The bottom two slices: nobody looked at them."""
        region = np.zeros(Organs.SHAPE, dtype=bool)
        region[6:, :, :] = True
        return region

    @staticmethod
    def writer(path: Path) -> Any:
        """A writer with the label set, grid ``g`` at ``tp0`` and image ``CT``."""
        w = medh5.create(path, sample_id=path.stem, codec="portable")
        w.label_set(Organs.label_set())
        w.add_grid("g", shape=Organs.SHAPE, spacing=(2.0, 0.8, 0.8), timepoint="tp0")
        w.add_image("CT", np.zeros(Organs.SHAPE, np.int16), grid="g", modality="CT")
        return w


class Numbered:
    """Classes ``c1`` … ``cn`` on 8×12×12; sample ``s1`` of subject ``subj-1``."""

    SHAPE = (8, 12, 12)
    ENCODINGS = ("labelmap", "layers", "bitmask", "instances", "probmap")

    @staticmethod
    def label_set(n: int = 3) -> LabelSet:
        return LabelSet(
            "rel-1.4.1",
            version="1.0.0",
            classes=[LabelClass(i, f"c{i}", f"C{i}") for i in range(1, n + 1)],
        )

    @staticmethod
    def writer(path: Path, *, shape: tuple[int, ...] = SHAPE, **options: Any) -> Any:
        """A writer with grid ``g``, image ``CT`` and three classes."""
        options.setdefault("codec", "portable")
        w = medh5.create(path, sample_id="s1", subject_id="subj-1", **options)
        w.add_grid("g", shape=shape, spacing=(2.0, 1.0, 1.0))
        w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        w.label_set(Numbered.label_set())
        return w

    @staticmethod
    def plain(path: Path, **options: Any) -> Path:
        """The writer's sample, committed with nothing added."""
        with Numbered.writer(path, **options):
            pass
        return path

    @staticmethod
    def two_classes() -> tuple[Any, Any, Any]:
        """Two disjoint classes and an ignore region touching neither."""
        liver = np.zeros(Numbered.SHAPE, bool)
        liver[1:5, 1:6, 1:6] = True
        spleen = np.zeros(Numbered.SHAPE, bool)
        spleen[1:5, 7:11, 7:11] = True
        region = np.zeros(Numbered.SHAPE, bool)
        region[6:, :, :] = True
        return liver, spleen, region

    @staticmethod
    def encoded(path: Path, encoding: str, *, overlap: bool = False) -> Any:
        """One annotation ``seg`` under *encoding*, carrying the ignore region."""
        liver, spleen, region = Numbered.two_classes()
        if overlap:
            region = region.copy()
            region[3:5, 2:4, 2:4] = True  # inside the liver
        with Numbered.writer(path) as w:
            if encoding == "instances":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    instances=[
                        InstanceInput(1, 1, mask=liver),
                        InstanceInput(2, 2, mask=spleen),
                    ],
                    ignore=region,
                )
            elif encoding == "probmap":
                w.add_segmentation(
                    "seg",
                    grid="g",
                    probabilities={1: liver.astype(float), 2: spleen.astype(float)},
                    ignore=region,
                )
            else:
                w.add_segmentation(
                    "seg",
                    grid="g",
                    masks={1: liver, 2: spleen},
                    ignore=region,
                    encoding=encoding,
                )
        return region


class Framed:
    """Classes ``c1`` … ``cn`` on 8×16×16, the grid in frame ``1.2.3.4``."""

    SHAPE = (8, 16, 16)

    @staticmethod
    def label_set(n: int = 3) -> LabelSet:
        return LabelSet(
            "rel-1.4.2",
            version="1.0.0",
            classes=[LabelClass(i, f"c{i}", f"C{i}") for i in range(1, n + 1)],
        )

    @staticmethod
    def writer(path: Path, *, shape: tuple[int, ...] = SHAPE, **options: Any) -> Any:
        """A writer with grid ``g``, image ``CT`` and three classes."""
        options.setdefault("codec", "portable")
        w = medh5.create(path, sample_id=path.stem, subject_id="subj-1", **options)
        w.add_grid("g", shape=shape, spacing=(2.0, 1.0, 1.0), frame_uid="1.2.3.4")
        w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        w.label_set(Framed.label_set())
        return w

    @staticmethod
    def box(lo: float, hi: float) -> list[list[float]]:
        return [[lo, hi]] * 3

    @staticmethod
    def many_classes(
        path: Path, classes: int = 63, shape: tuple[int, ...] = (16, 32, 32)
    ) -> Path:
        """*classes* small classes, segmented and indexed."""
        rng = np.random.default_rng(0)
        masks = {}
        for c in range(1, classes + 1):
            mask = np.zeros(shape, bool)
            z, y, x = (int(rng.integers(0, n - 2)) for n in shape)
            mask[z : z + 2, y : y + 2, x : x + 2] = True
            masks[c] = mask
        w = medh5.create(path, sample_id="s", subject_id="s", codec="portable")
        w.add_grid("g", shape=shape, spacing=(1, 1, 1))
        w.add_image("CT", np.zeros(shape, np.int16), grid="g", modality="CT")
        w.label_set(Framed.label_set(classes))
        w.add_segmentation("organs", grid="g", masks=masks)
        w.build_index()
        w.commit()
        return path
