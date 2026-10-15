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
  reference;
* :class:`History` --- two CT visits with a clinical history (format 1.1),
  8×12×12;
* :class:`SplitCohort` --- a cohort manifest of one-file groups in strata of
  60 %, 30 % and 10 % and three rare ones, with no files behind it;
* :class:`EveryCheckCode` --- a cohort manifest raising every code ``dataset
  check`` reports, with one file behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
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


class History:
    """Two CT visits with a clinical history on one subject clock (format 1.1).

    The 1.1 §9.3 shape, small: a laboratory value before the scan, the
    baseline CT, a report and its revision two days later, the follow-up CT and
    a response at day 90 --- plus whatever a test adds through ``events``,
    ``documents`` and ``links``.  Built by the public writer, so every reader
    test is a writer test too.
    """

    SHAPE = (8, 12, 12)
    RECIST = "org.example.recist"

    @staticmethod
    def label_set() -> LabelSet:
        return LabelSet(
            "history-v1", version="1.0.0", classes=[LabelClass(3, "lesion", "Lesion")]
        )

    @staticmethod
    def events(response: str = "SD") -> list[Any]:
        from medh5.clinical import DAY, HOUR, Event

        def point(event_id: str, kind: str, t: int, avail: int, **f: Any) -> Any:
            status = f.pop("status", "final")
            record = f.pop("record_id", event_id)
            return Event(
                event_id,
                record,
                kind,
                "point",
                status,
                effective_start_us=t,
                available_us=avail,
                **f,
            )

        return [
            point(
                "lab0",
                "observation",
                -48 * HOUR,
                -47 * HOUR,
                code_system="http://loinc.org",
                code="2160-0",
                value_num=1.1,
                unit="mg/dL",
            ),
            point("ct0", "imaging", 0, HOUR, timepoint_id="tp0"),
            point(
                "rep_v1", "document", 0, 4 * HOUR, status="preliminary", record_id="rep"
            ),
            point(
                "rep_v2", "document", 0, 48 * HOUR, status="amended", record_id="rep"
            ),
            point("ct1", "imaging", 90 * DAY, 90 * DAY + HOUR, timepoint_id="tp1"),
            point(
                "recist1",
                "assessment",
                90 * DAY,
                91 * DAY,
                code_system=History.RECIST,
                code="overall_response",
                value_text=response,
            ),
        ]

    @staticmethod
    def links() -> list[Any]:
        from medh5.clinical import Link

        return [
            Link.between(("event", "ct0"), "describes", ("image", "CT_tp0")),
            Link.between(("event", "ct1"), "describes", ("image", "CT_tp1")),
            Link.between(("event", "rep_v1"), "describes", ("document", "rep_text_v1")),
            Link.between(("event", "rep_v2"), "describes", ("document", "rep_text_v2")),
            Link.between(("event", "rep_v2"), "supersedes", ("event", "rep_v1")),
        ]

    @staticmethod
    def documents() -> list[Any]:
        from medh5.clinical import Document

        return [
            Document("rep_text_v1", "Baseline: a 14 mm nodule ≈ 1.4 cm. Preliminary."),
            Document("rep_text_v2", "Baseline: a 14 mm nodule ≈ 1.4 cm. Final."),
        ]

    @staticmethod
    def writer_for(
        path: Path, *, subject_id: str = "P-01", fill: int | None = None
    ) -> Any:
        """A writer holding the label set and the two CT visits, no history;
        *fill* gives every voxel of both one value."""
        rng = np.random.default_rng(len(subject_id))
        w = medh5.create(
            path, sample_id=path.stem, subject_id=subject_id, codec="portable"
        )
        w.label_set(History.label_set())
        for tp, days in (("tp0", 0), ("tp1", 90)):
            w.add_timepoint(tp, days_from_baseline=days)
            w.add_grid(
                f"ct_{tp}",
                shape=History.SHAPE,
                spacing=(2.0, 0.8, 0.8),
                timepoint=tp,
                frame_uid=f"{subject_id}-{tp}",
            )
            volume = rng.integers(-100, 200, History.SHAPE).astype(np.int16)
            if fill is not None:
                volume[...] = fill
            w.add_image(f"CT_{tp}", volume, grid=f"ct_{tp}", modality="CT")
        return w

    @staticmethod
    def write(
        path: Path,
        *,
        subject_id: str = "P-01",
        response: str = "SD",
        events: Sequence[Any] = (),
        documents: Sequence[Any] = (),
        links: Sequence[Any] = (),
        lesion: bool = True,
        clock: Any = None,
        fill: int | None = None,
    ) -> str:
        """Write the sample; returns its ``content_id``."""
        from medh5.clinical import Clock

        with History.writer_for(path, subject_id=subject_id, fill=fill) as w:
            if lesion:
                w.add_segmentation(
                    "lesions_tp0",
                    grid="ct_tp0",
                    instances=[
                        InstanceInput(3, 1, mask=block(History.SHAPE, (2, 1, 6), 3))
                    ],
                    annotated_classes=[3],
                )
            w.set_clock(clock or Clock.relative("clock", "baseline CT acquisition"))
            for event in [*History.events(response), *events]:
                w.add_event(event)
            for document in [*History.documents(), *documents]:
                w.add_document(document)
            for link in [*History.links(), *links]:
                w.add_link(link)
            content_id = w.commit()
        assert content_id is not None
        return str(content_id)


class SplitCohort:
    """A manifest of *n* one-file groups (subjects ``s000``, ``s001``, …) in
    three strata of ``site_id`` holding 60 %, 30 % and 10 % of *n*, and rare
    strata of 3, 5 and 7 groups.  Splits read no file, so none is written.

    *extra* groups join the largest stratum after the others, so every other
    subject keeps its id.
    """

    RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
    RARE = (3, 5, 7)

    @staticmethod
    def manifest(n: int = 100, *, extra: int = 0) -> Any:
        from medh5.dataset import Entry, Manifest

        strata = [("big", n * 6 // 10), ("mid", n * 3 // 10), ("small", n // 10)]
        strata += [(f"rare-{k}", k) for k in SplitCohort.RARE]
        sites = [site for site, count in strata for _ in range(count)]
        sites += ["big"] * extra
        return Manifest(
            entries=[
                Entry(
                    path=f"s{i:03}.medh5",
                    sample_id=f"s{i:03}",
                    subject_id=f"s{i:03}",
                    group_id=f"s{i:03}",
                    site_id=site,
                )
                for i, site in enumerate(sites)
            ]
        )


class EveryCheckCode:
    """A manifest that raises every code ``dataset check`` reports: two label
    sets that disagree on class 1 and a sample with none (C101, C102, C103), a
    stale claim, one subject in two partitions and a sample with no claim
    (C201, C202, C203), class 1 examined in one sample of three and class 2 in
    none (C301, C302), a changed file and a missing one (C401, C402), and one
    sample without a de-identification record (C501).  Only the changed file
    is written.
    """

    @staticmethod
    def manifest(root: Path) -> Any:
        from medh5.dataset import Entry, Manifest

        changed = root / "changed.medh5"
        changed.write_bytes(b"not the bytes scanned")

        def claim(partition: str) -> dict[str, Any]:
            return {"set_id": "s", "partition": partition, "manifest_sha256": "x"}

        return Manifest(
            entries=[
                Entry(
                    path=str(root / "gone.medh5"),
                    sample_id="a",
                    subject_id="p1",
                    group_id="p1",
                    class_ids=(1,),
                    annotated_class_ids=(1,),
                    class_keys={1: "liver"},
                    label_set_id="one",
                    label_set_digest="d1",
                    splits=(claim("train"),),
                    deidentified=True,
                ),
                Entry(
                    path=str(changed),
                    sample_id="b",
                    subject_id="p1",
                    group_id="p1",
                    class_ids=(1, 2),
                    class_keys={1: "kidney", 2: "spleen"},
                    label_set_id="two",
                    label_set_digest="d2",
                    splits=(claim("test"),),
                ),
                Entry(
                    path=str(root / "bare.medh5"),
                    sample_id="c",
                    subject_id="p2",
                    group_id="p2",
                    deidentified=True,
                ),
            ]
        )
