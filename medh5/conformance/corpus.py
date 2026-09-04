"""Corpus cases: what a conforming file looks like, and what a broken one reports."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

import medh5
from medh5._hdf5 import encode_attr, open_h5, str_dtype
from medh5.annotations.geometric import Polygon
from medh5.annotations.voxel import InstanceInput
from medh5.labels.labelset import LabelClass, LabelSet, Skeleton
from medh5.validate import validate_file
from medh5.validate.report import Level

SEED = 20260815

_LS = LabelSet(
    "conformance-v1",
    version="1.0.0",
    classes=[
        LabelClass(1, "liver", "Liver", category="organ", color=(200, 90, 70, 255)),
        LabelClass(2, "spleen", "Spleen", category="organ", color=(190, 60, 60, 255)),
        LabelClass(
            3,
            "lesion",
            "Lesion",
            parents=(1,),
            category="lesion",
            color=(255, 214, 64, 255),
        ),
        LabelClass(4, "vessel", "Vessel", category="vessel", color=(80, 150, 220, 255)),
    ],
)


@dataclass(frozen=True, slots=True)
class Case:
    """One corpus entry."""

    name: str
    description: str
    clause: str
    build: Callable[[Path], None]
    level: Level = "semantic"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    suffix: str = ".medh5"
    """``.medh5c`` for a collection case (§2.1); the corpus runner honours it."""
    mutated: bool = False
    """Built by editing a committed file, so its digests are deliberately stale.

    Mutation is how invalid cases are made --- the writer refuses to produce
    them --- but it leaves ``content_id`` covering the pre-mutation bytes.  The
    flag says so, rather than letting a consumer read "no expected errors" as
    "this file also verifies".
    """

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "clause": self.clause,
            "level": self.level,
            "file_suffix": self.suffix,
            "valid": self.valid,
            "mutated": self.mutated,
            "expect_errors": sorted(self.errors),
            "expect_warnings": sorted(self.warnings),
        }


@dataclass(slots=True)
class CaseResult:
    """What running one case produced."""

    case: Case
    path: str
    got_errors: tuple[str, ...] = ()
    got_warnings: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    error: str | None = None
    details: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.unexpected and self.error is None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.case.name,
            "ok": self.ok,
            "expect_errors": sorted(self.case.errors),
            "got_errors": sorted(self.got_errors),
            "expect_warnings": sorted(self.case.warnings),
            "got_warnings": sorted(self.got_warnings),
            "missing": sorted(self.missing),
            "unexpected": sorted(self.unexpected),
            "error": self.error,
        }


_CASES: list[Case] = []


def case(
    name: str,
    description: str,
    clause: str,
    *,
    level: Level = "semantic",
    errors: Sequence[str] = (),
    warnings: Sequence[str] = (),
    suffix: str = ".medh5",
    mutated: bool = False,
) -> Callable[[Callable[[Path], None]], Callable[[Path], None]]:
    def wrap(fn: Callable[[Path], None]) -> Callable[[Path], None]:
        _CASES.append(
            Case(
                name=name,
                description=description,
                clause=clause,
                build=fn,
                level=level,
                errors=tuple(errors),
                warnings=tuple(warnings),
                suffix=suffix,
                mutated=mutated,
            )
        )
        return fn

    return wrap


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------


def _blocks(
    shape: tuple[int, ...], spec: dict[int, tuple[int, ...]]
) -> dict[int, npt.NDArray[np.bool_]]:
    out: dict[int, npt.NDArray[np.bool_]] = {}
    for class_id, (z, y, x) in spec.items():
        mask = np.zeros(shape, dtype=bool)
        mask[z : z + 6, y : y + 8, x : x + 8] = True
        out[class_id] = mask
    return out


def _base(
    path: Path,
    *,
    shape: tuple[int, ...] = (16, 24, 24),
    timepoints: Sequence[tuple[str, dict[str, Any]]] = (
        ("tp0", {"label": "baseline"}),
    ),
    label_set: LabelSet | None = None,
    deidentify: bool = True,
    codec: str = "portable",
) -> None:
    """A valid, minimal-but-complete sample: one grid, one image, one timepoint."""
    rng = np.random.default_rng(SEED)
    with medh5.create(path, sample_id=path.stem, subject_id="subj-A", codec=codec) as w:
        w.identity(sex="F", bodypart="abdomen")
        w.cohort(dataset_id="conformance", site_id="site-A")
        for tp_id, fields in timepoints:
            w.add_timepoint(tp_id, **fields)
        if label_set is not None:
            w.label_set(label_set)
        tool = w.software("medh5", medh5.__version__)
        act = w.activity("import", agent=tool, tool="conformance corpus")
        w.add_grid(
            "ct",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            origin=(-12.0, -9.6, -9.6),
            timepoint=timepoints[0][0],
            frame_uid="pseudo:frame-100",
            patch_hint=(8, 8, 8),
        )
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
            prov=act,
        )
        if deidentify:
            w.deidentification(method="dicom-psi-profile", date_shift_days=-117)


def _seg_base(
    path: Path,
    *,
    encoding: str = "auto",
    annotated: Any = "all_given",
    classes: dict[int, tuple[int, ...]] | None = None,
    shape: tuple[int, ...] = (16, 24, 24),
    quality: dict[str, Any] | None = None,
    index: bool = False,
    codec: str = "portable",
) -> None:
    """A valid sample carrying one voxel annotation."""
    rng = np.random.default_rng(SEED)
    spec = classes or {1: (2, 2, 2), 2: (2, 12, 2), 3: (4, 4, 4)}
    with medh5.create(path, sample_id=path.stem, subject_id="subj-A", codec=codec) as w:
        w.identity(sex="F", bodypart="abdomen")
        w.add_timepoint("tp0", label="baseline")
        w.label_set(_LS)
        tool = w.software("medh5", medh5.__version__)
        imp = w.activity("import", agent=tool)
        rad = w.person("pseudonym:RAD-07", role="annotator")
        ann = w.activity("annotate", agent=rad, tool="3D Slicer 5.6.2")
        w.add_grid(
            "ct",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            origin=(-12.0, -9.6, -9.6),
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
            patch_hint=(8, 8, 8),
        )
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
            prov=imp,
        )
        w.add_segmentation(
            "organs",
            grid="ct",
            masks=_blocks(shape, spec),
            encoding=encoding,
            annotated_classes=annotated,
            prov=ann,
            quality=quality or {"status": "approved", "confidence": 0.9},
        )
        if index:
            w.build_index(["organs"], max_coords=128)
        w.deidentification(method="dicom-psi-profile", date_shift_days=-117)


def _mutate(path: Path, fn: Callable[[h5py.File], None]) -> None:
    with open_h5(path, "r+") as handle:
        fn(handle)


def _restamp(path: Path) -> None:
    """Re-stamp digests and ``content_id`` after a legitimate authoring edit.

    Used only where the edit is something a writer would do (adding a transform
    the 1.0 API does not expose yet); invalid cases deliberately skip it, which
    is what makes E701/E702 reachable at all.
    """
    from medh5.integrity.digest import compute_content_id, stamp_digests
    from medh5.sample import Sample

    with open_h5(path, "r+") as handle:
        stamp_digests(handle)
        attr_names = Sample(handle, path=str(path)).attr_name_map()
        handle.attrs["content_id"] = encode_attr(compute_content_id(handle, attr_names))


def _set_meta(handle: h5py.File, fn: Callable[[dict[str, Any]], None]) -> None:
    raw = handle["meta"][()]
    doc = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw))
    fn(doc)
    del handle["meta"]
    handle.create_dataset("meta", data=json.dumps(doc), dtype=str_dtype())


# --------------------------------------------------------------------------
# Valid cases
# --------------------------------------------------------------------------


@case("core-minimal", "One grid, one image, one timepoint.", "§2, §3, §4")
def _core_minimal(path: Path) -> None:
    _base(path)


@case(
    "core-two-images-one-grid",
    "Two co-registered images sharing a grid and a frame of reference.",
    "§3.4",
)
def _core_two_images(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0", label="baseline")
        w.add_grid(
            "ct",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_grid(
            "pet",
            shape=(8, 12, 12),
            spacing=(3.0, 1.6, 1.6),
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_image(
            "PET",
            rng.random((8, 12, 12)).astype(np.float32),
            grid="pet",
            modality="PT",
            value_type="quantitative",
            value_units="SUVbw",
        )
        w.deidentification(method="dicom-psi-profile")


@case("core-2d-radiograph", "A 2-D image with two spatial axes.", "§3.6")
def _core_2d(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.add_grid("dx", shape=(64, 64), spacing=(0.2, 0.2), timepoint="tp0")
        w.add_image(
            "DX",
            rng.integers(0, 4095, (64, 64)).astype(np.uint16),
            grid="dx",
            modality="DX",
            value_type="intensity",
        )
        w.deidentification(method="dicom-psi-profile")


@case("core-4d-time", "A 4-D dynamic series with one time axis.", "§3.6")
def _core_4d(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (4, 8, 16, 16)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.add_grid(
            "dce",
            shape=shape,
            spacing=(2.0, 1.0, 1.0),
            axis_kinds=("time", "spatial", "spatial", "spatial"),
            axis_names=("t", "z", "y", "x"),
            time_values=(0.0, 12.0, 24.0, 36.0),
            time_units="s",
            timepoint="tp0",
        )
        w.add_image(
            "DCE",
            rng.random(shape).astype(np.float32),
            grid="dce",
            modality="MR",
            value_type="intensity",
        )
        w.deidentification(method="dicom-psi-profile")


@case("core-rgb-channels", "A channel axis with named channels.", "§3.6, §4.1")
def _core_rgb(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (3, 32, 32)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.add_grid(
            "slide",
            shape=shape,
            spacing=(0.5, 0.5),
            units="um",
            axis_kinds=("channel", "spatial", "spatial"),
            axis_names=("c", "y", "x"),
            timepoint="tp0",
        )
        w.add_image(
            "RGB",
            rng.integers(0, 255, shape).astype(np.uint8),
            grid="slide",
            modality="OT",
            value_type="rgb",
            channel_names=["R", "G", "B"],
        )
        w.deidentification(method="dicom-psi-profile")


@case(
    "seg-labelmap",
    "Mutually exclusive classes stored as one integer volume.",
    "§7.1",
    warnings=["W912"],
)
def _seg_labelmap(path: Path) -> None:
    _seg_base(
        path, encoding="labelmap", classes={1: (2, 2, 2), 2: (2, 12, 2), 4: (9, 2, 12)}
    )


@case(
    "seg-layers",
    "Overlapping classes coloured into layers --- the default encoding.",
    "§7.2",
    warnings=["W912"],
)
def _seg_layers(path: Path) -> None:
    _seg_base(path, encoding="layers")


@case("seg-bitmask", "One bit per class per voxel.", "§7.3", warnings=["W912"])
def _seg_bitmask(path: Path) -> None:
    _seg_base(path, encoding="bitmask")


@case(
    "seg-probmap",
    "Soft ground truth as per-class probabilities.",
    "§7.5",
    warnings=["W912"],
)
def _seg_probmap(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(_LS)
        w.add_grid("ct", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint="tp0")
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_segmentation(
            "soft",
            grid="ct",
            probabilities={1: rng.random(shape), 3: rng.random(shape)},
        )
        w.deidentification(method="dicom-psi-profile")


@case(
    "seg-instances",
    "Per-object boxes and bit-packed crops with sample-scoped instance ids.",
    "§7.4",
    warnings=["W912"],
)
def _seg_instances(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    objects = []
    for i, (z, y, x) in enumerate([(2, 2, 2), (6, 12, 6), (10, 4, 14)]):
        mask = np.zeros(shape, dtype=bool)
        mask[z : z + 4, y : y + 5, x : x + 5] = True
        objects.append(
            InstanceInput(class_id=3, instance_id=i + 1, mask=mask, score=0.9)
        )
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(_LS)
        w.add_grid("ct", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint="tp0")
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_segmentation("lesions", grid="ct", instances=objects)
        w.deidentification(method="dicom-psi-profile")


@case(
    "seg-partial-coverage-with-ignore",
    "Two of four classes annotated, with an explicit ignore region: no W904.",
    "§7.7, §11.3",
    warnings=["W912"],
)
def _seg_partial_ignore(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    ignore = np.zeros(shape, dtype=bool)
    ignore[12:, :, :] = True
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(_LS)
        w.add_grid("ct", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint="tp0")
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_segmentation(
            "organs",
            grid="ct",
            masks=_blocks(shape, {1: (2, 2, 2), 2: (2, 12, 2)}),
            encoding="layers",
            annotated_classes=[1],
            ignore=ignore,
        )
        w.deidentification(method="dicom-psi-profile")


@case(
    "training-index",
    "A sampling index that is current for its annotation.",
    "§14.3",
    level="integrity",
    warnings=["W912"],
)
def _training_index(path: Path) -> None:
    _seg_base(path, index=True)


@case(
    "longitudinal-two-timepoints",
    "Two timepoints, distinct frames, a transform relating them.",
    "§3.7, §7.4",
    warnings=["W912"],
)
def _longitudinal(path: Path) -> None:
    rng = np.random.default_rng(SEED)
    shape, shape_fu = (16, 24, 24), (14, 24, 24)
    with medh5.create(
        path, sample_id=path.stem, subject_id="subj-A", codec="portable"
    ) as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up_3mo", days_from_baseline=92)
        w.label_set(_LS)
        for gid, tp, sh, frame in (
            ("ct_tp0", "tp0", shape, "pseudo:frame-100"),
            ("ct_tp1", "tp1", shape_fu, "pseudo:frame-101"),
        ):
            w.add_grid(
                gid, shape=sh, spacing=(1.5, 0.8, 0.8), timepoint=tp, frame_uid=frame
            )
            w.add_image(
                f"CT_{tp}",
                rng.integers(-1000, 1500, sh).astype(np.int16),
                grid=gid,
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        w.add_segmentation(
            "organs_tp0",
            grid="ct_tp0",
            masks=_blocks(shape, {1: (2, 2, 2), 3: (4, 4, 4)}),
        )
        w.add_segmentation(
            "organs_tp1",
            grid="ct_tp1",
            masks=_blocks(shape_fu, {1: (2, 2, 2), 3: (4, 4, 4)}),
        )
        w.deidentification(method="dicom-psi-profile")

    # A minimal affine transform: phase 4 owns the transform API; the corpus
    # needs only a well-formed object so W911 does not fire on a valid file.
    def add_transform(handle: h5py.File) -> None:
        group = handle.require_group("transforms").create_group("tp0_to_tp1")
        group.create_dataset("matrix", data=np.eye(4, dtype=np.float64))
        for key, value in (
            ("kind", "affine"),
            ("from_frame", "pseudo:frame-100"),
            ("to_frame", "pseudo:frame-101"),
        ):
            group.attrs[key] = encode_attr(value)

    _mutate(path, add_transform)
    _restamp(path)


# --------------------------------------------------------------------------
# Invalid cases --- one per diagnostic code
# --------------------------------------------------------------------------


def _invalid(
    name: str,
    description: str,
    clause: str,
    codes: Sequence[str],
    mutation: Callable[[h5py.File], None],
    *,
    base: Callable[[Path], None] | None = None,
    level: Level = "semantic",
    warnings: Sequence[str] = (),
) -> None:
    def build(path: Path) -> None:
        (base or _base)(path)
        _mutate(path, mutation)

    build.__name__ = f"_build_{name.replace('-', '_')}"
    _CASES.append(
        Case(
            name=name,
            description=description,
            clause=clause,
            build=build,
            level=level,
            errors=tuple(codes),
            warnings=tuple(warnings),
            mutated=True,
        )
    )


def _seg_invalid(
    name: str,
    description: str,
    clause: str,
    codes: Sequence[str],
    mutation: Callable[[h5py.File], None],
    *,
    warnings: Sequence[str] = ("W912",),
    level: Level = "semantic",
) -> None:
    _invalid(
        name,
        description,
        clause,
        codes,
        mutation,
        base=lambda p: _seg_base(p),
        level=level,
        warnings=warnings,
    )


def _register_invalid_cases() -> None:
    _invalid(
        "E001-missing-version",
        "Root without `medh5_version`.",
        "§2.1",
        ["E001"],
        lambda f: f.attrs.__delitem__("medh5_version"),
    )
    _invalid(
        "E002-unsupported-major",
        "A major version this reader must refuse.",
        "§2.1",
        ["E002"],
        lambda f: f.attrs.__setitem__("medh5_version", encode_attr("2.0")),
    )
    _invalid(
        "E003-bad-identifier",
        "An object name with a forbidden character.",
        "§2.3",
        ["E003"],
        lambda f: f["images"].move("CT", "CT scan"),
    )
    _invalid(
        "E004-meta-not-json",
        "`meta` holding text that is not JSON.",
        "§2.4",
        ["E004"],
        _write_garbage_meta,
    )
    _invalid(
        "E005-meta-schema",
        "A document missing a required member.",
        "§2.4",
        ["E005"],
        lambda f: _set_meta(f, lambda d: d.pop("identity", None)),
    )
    _invalid(
        "E006-missing-kind",
        "Root without `medh5_kind`.",
        "§2.1",
        ["E006"],
        lambda f: f.attrs.__delitem__("medh5_kind"),
    )
    _invalid(
        "E007-unknown-profile",
        "A declared profile outside the registry.",
        "§1.3",
        ["E007"],
        lambda f: f.attrs.__setitem__(
            "medh5_profiles", encode_attr(["core", "quantum"])
        ),
    )
    _invalid(
        "E008-missing-grids",
        "The `grids` group removed.",
        "§2.3",
        ["E008", "E101"],
        lambda f: f.__delitem__("grids"),
    )
    _invalid(
        "E101-dangling-grid",
        "An image naming a grid that does not exist.",
        "§3.2",
        ["E101"],
        lambda f: f["images/CT"].attrs.__setitem__("grid", encode_attr("nope")),
    )
    _invalid(
        "E102-non-orthonormal",
        "A `direction` that is not orthonormal.",
        "§3.2",
        ["E102"],
        lambda f: f["grids/ct"].attrs.__setitem__(
            "direction", np.array([[1.0, 0.4, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ),
    )
    _invalid(
        "E103-spatial-not-trailing",
        "Spatial axes that are not the trailing axes.",
        "§3.1",
        ["E103"],
        lambda f: f["grids/ct"].attrs.__setitem__(
            "axis_kinds", encode_attr(["spatial", "spatial", "channel"])
        ),
    )
    _invalid(
        "E104-nonpositive-spacing",
        "A zero voxel spacing.",
        "§3.2",
        ["E104"],
        lambda f: f["grids/ct"].attrs.__setitem__("spacing", np.array([1.5, 0.0, 0.8])),
    )
    _invalid(
        "E110-bad-axis-kinds",
        "One spatial axis and two time axes: both counts are out of range.",
        "§3.1",
        ["E110"],
        lambda f: f["grids/ct"].attrs.__setitem__(
            "axis_kinds", encode_attr(["time", "time", "spatial"])
        ),
    )
    _invalid(
        "E201-no-images",
        "A sample with an empty `images` group.",
        "§4.1",
        ["E201"],
        lambda f: f["images"].__delitem__("CT"),
    )
    _invalid(
        "E202-shape-mismatch",
        "An image whose shape differs from its grid.",
        "§4.1",
        ["E202"],
        lambda f: f["grids/ct"].attrs.__setitem__("shape", np.array([16, 24, 25])),
    )
    _invalid(
        "E203-unknown-value-type",
        "An unregistered `value_type`.",
        "§4.2",
        ["E203"],
        lambda f: f["images/CT"].attrs.__setitem__("value_type", encode_attr("vibes")),
    )
    _invalid(
        "E204-channel-names",
        "`channel_names` on a grid with no channel axis.",
        "§4.1",
        ["E204"],
        lambda f: f["images/CT"].attrs.__setitem__(
            "channel_names", encode_attr(["R", "G", "B"])
        ),
    )
    _invalid(
        "E603-unknown-activity",
        "A provenance activity of an unknown type.",
        "§11.1",
        ["E005", "E603"],
        lambda f: _set_meta(f, _set_activity_type),
    )
    _invalid(
        "E604-bad-timestamp",
        "An activity timestamp that is not RFC 3339.",
        "§11.1",
        ["E604"],
        lambda f: _set_meta(f, _set_bad_timestamp),
    )
    _invalid(
        "E605-dangling-agent",
        "An activity naming an undeclared agent.",
        "§11.1",
        ["E605"],
        lambda f: _set_meta(f, _set_dangling_agent),
    )
    _invalid(
        "W903-no-deidentification",
        "No de-identification record.",
        "§11.4",
        [],
        lambda f: _set_meta(f, lambda d: d.pop("deidentification", None)),
        warnings=["W903"],
    )
    _invalid(
        "W906-conflicting-splits",
        "One split set claimed against two manifests.",
        "§12.3",
        [],
        lambda f: _set_meta(f, _set_conflicting_splits),
        warnings=["W906"],
    )
    _invalid(
        "W907-float-storage",
        "Integer-valued data stored as float32.",
        "§4.2",
        [],
        _store_float_ct,
        warnings=["W907"],
    )

    _seg_invalid(
        "E303-reserved-class-id",
        "An annotation claiming the ignore id.",
        "§5.3",
        ["E303", "E402", "E403", "E404"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "class_ids", np.array([1, 2, 65535], dtype=np.uint16)
        ),
    )
    _seg_invalid(
        "E401-unknown-kind",
        "An annotation of an unregistered kind.",
        "§6.3",
        ["E009", "E401"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "kind", encode_attr("voxelthing")
        ),
    )
    _seg_invalid(
        "E401-reserved-kind",
        "The reserved `rle` kind in a 1.0 file.",
        "§16",
        ["E009", "E401"],
        lambda f: f["annotations/organs"].attrs.__setitem__("kind", encode_attr("rle")),
    )
    _seg_invalid(
        "E402-class-not-in-labelset",
        "A class id absent from the label set.",
        "§5.1",
        ["E402", "E403", "E404"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "class_ids", np.array([1, 2, 99], dtype=np.uint16)
        ),
    )
    _seg_invalid(
        "E403-coverage-superset",
        "`annotated_class_ids` claiming more than `class_ids`.",
        "§6.2",
        ["E403"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "annotated_class_ids", np.array([1, 2, 3, 4], dtype=np.uint16)
        ),
    )
    _seg_invalid(
        "E404-class-in-two-layers",
        "One class assigned to two layers.",
        "§7.2",
        ["E404"],
        _duplicate_layer_class,
    )
    _seg_invalid(
        "E405-data-shape",
        "Annotation data whose spatial shape differs from the grid.",
        "§7.2",
        ["E202", "E405"],
        lambda f: f["grids/ct"].attrs.__setitem__("shape", np.array([16, 24, 25])),
    )
    _seg_invalid(
        "E409-undeclared-timepoint",
        "An annotation naming a timepoint that is not declared.",
        "§3.7",
        ["E409"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "timepoints", encode_attr(["tp9"])
        ),
    )
    _seg_invalid(
        "E410-missing-dataset",
        "A `layers` annotation without `layer_class_ids`.",
        "§7.2",
        ["E410"],
        lambda f: f["annotations/organs"].__delitem__("layer_class_ids"),
    )
    _seg_invalid(
        "E411-bad-dtype",
        "A `layers` volume stored as int32.",
        "§7.2",
        ["E411"],
        _retype_layers,
    )
    _seg_invalid(
        "E601-dangling-prov",
        "An annotation naming an unknown activity.",
        "§11.1",
        ["E601"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "prov", encode_attr("act_nope")
        ),
    )
    _seg_invalid(
        "E602-unknown-quality",
        "An annotation naming an unknown quality record.",
        "§11.2",
        ["E602"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "quality", encode_attr("nope")
        ),
    )
    _seg_invalid(
        "W904-partial-no-ignore",
        "Partial coverage with no ignore region --- `0` is not a negative.",
        "§11.3",
        [],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "annotated_class_ids", np.array([1], dtype=np.uint16)
        ),
        warnings=["W904", "W912"],
    )
    _seg_invalid(
        "E701-digest-mismatch",
        "A dataset edited after its digest was stamped. `content_id` still "
        "matches: it covers the digest *list*, so the mismatch stays local "
        "to the object that changed.",
        "§13.1",
        ["E701"],
        _corrupt_data,
        level="integrity",
    )
    _seg_invalid(
        "E702-content-id-mismatch",
        "A `content_id` that does not match.",
        "§13.2",
        ["E702"],
        lambda f: f.attrs.__setitem__("content_id", encode_attr("sha256:" + "0" * 64)),
        level="integrity",
    )
    _seg_invalid(
        "E703-malformed-digest",
        "A digest string that does not parse.",
        "§13.1",
        ["E702", "E703"],
        lambda f: f["annotations/organs/data"].attrs.__setitem__(
            "digest", encode_attr("not-a-digest")
        ),
        level="integrity",
    )
    _seg_invalid(
        "W901-no-digests",
        "A file carrying no digests at all.",
        "§13.1",
        [],
        _strip_digests,
        level="integrity",
        warnings=["W901", "W912"],
    )
    _seg_invalid(
        "W909-instance-id-two-classes",
        "One instance id carrying two class ids.",
        "§7.4",
        [],
        _instances_two_classes,
        warnings=["W909", "W912"],
    )


def _write_garbage_meta(handle: h5py.File) -> None:
    del handle["meta"]
    handle.create_dataset("meta", data="{not json", dtype=str_dtype())


def _set_activity_type(doc: dict[str, Any]) -> None:
    doc.setdefault("provenance", {}).setdefault("activities", [])
    if not doc["provenance"]["activities"]:
        doc["provenance"]["activities"].append({"id": "a1", "type": "import"})
    doc["provenance"]["activities"][0]["type"] = "vibecheck"


def _set_bad_timestamp(doc: dict[str, Any]) -> None:
    activities = doc.setdefault("provenance", {}).setdefault("activities", [])
    if not activities:
        activities.append({"id": "a1", "type": "import"})
    activities[0]["ended"] = "yesterday"


def _set_dangling_agent(doc: dict[str, Any]) -> None:
    activities = doc.setdefault("provenance", {}).setdefault("activities", [])
    if not activities:
        activities.append({"id": "a1", "type": "import"})
    activities[0]["agent"] = "ghost"


def _set_conflicting_splits(doc: dict[str, Any]) -> None:
    doc["splits"] = [
        {"set_id": "cv5", "partition": "train", "manifest_sha256": "a" * 64},
        {"set_id": "cv5", "partition": "test", "manifest_sha256": "b" * 64},
    ]


def _store_float_ct(handle: h5py.File) -> None:
    values = np.asarray(handle["images/CT"][...]).astype(np.float32)
    attrs = dict(handle["images/CT"].attrs)
    del handle["images/CT"]
    node = handle["images"].create_dataset("CT", data=values)
    for key, value in attrs.items():
        node.attrs[key] = value


def _duplicate_layer_class(handle: h5py.File) -> None:
    table = np.asarray(handle["annotations/organs/layer_class_ids"][...])
    if table.shape[0] < 2:
        raise RuntimeError("case needs at least two layers")
    table[1, 0] = table[0, 0]
    handle["annotations/organs/layer_class_ids"][...] = table


def _retype_layers(handle: h5py.File) -> None:
    group = handle["annotations/organs"]
    values = np.asarray(group["data"][...]).astype(np.int32)
    attrs = dict(group["data"].attrs)
    del group["data"]
    node = group.create_dataset("data", data=values)
    for key, value in attrs.items():
        node.attrs[key] = value


def _corrupt_data(handle: h5py.File) -> None:
    data = handle["annotations/organs/data"]
    block = np.asarray(data[...])
    block[tuple(0 for _ in block.shape)] = 7
    data[...] = block


def _strip_digests(handle: h5py.File) -> None:
    def visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset) and "digest" in obj.attrs:
            del obj.attrs["digest"]

    handle.visititems(visit)
    if "content_id" in handle.attrs:
        del handle.attrs["content_id"]


def _instances_two_classes(handle: h5py.File) -> None:
    del handle["annotations/organs"]
    group = handle["annotations"].create_group("organs")
    group.create_dataset(
        "boxes",
        data=np.array(
            [
                [[1.5, 5.5], [1.5, 5.5], [1.5, 5.5]],
                [[6.5, 9.5], [6.5, 9.5], [6.5, 9.5]],
            ],
            dtype=np.float32,
        ),
    )
    group.create_dataset("class_ids", data=np.array([1, 2], dtype=np.uint16))
    group.create_dataset("instance_ids", data=np.array([1, 1], dtype=np.uint32))
    for key, value in (
        ("kind", "instances"),
        ("task", "segmentation"),
        ("grid", "ct"),
        ("closure", "explicit"),
        ("quality", "organs"),
    ):
        group.attrs[key] = encode_attr(value)
    group.attrs["class_ids"] = np.array([1, 2], dtype=np.uint16)
    group.attrs["annotated_class_ids"] = np.array([1, 2], dtype=np.uint16)


_register_invalid_cases()


def _longitudinal_base(
    path: Path,
    *,
    shared_frame: bool = False,
    drop_timepoint: bool = False,
    bad_timepoint: bool = False,
) -> None:
    """Two timepoints, with the failure modes §3.7 warns about as switches."""
    rng = np.random.default_rng(SEED)
    shape = (12, 16, 16)
    with medh5.create(
        path, sample_id=path.stem, subject_id="subj-A", codec="portable"
    ) as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up", days_from_baseline=92)
        frame1 = "pseudo:frame-100" if shared_frame else "pseudo:frame-101"
        w.add_grid(
            "ct_tp0",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_grid(
            "ct_tp1",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            timepoint="tp1",
            frame_uid=frame1,
        )
        for gid, tp in (("ct_tp0", "tp0"), ("ct_tp1", "tp1")):
            w.add_image(
                f"CT_{tp}",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid=gid,
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        w.deidentification(method="dicom-psi-profile")
    if drop_timepoint:
        _mutate(path, lambda f: f["grids/ct_tp1"].attrs.__delitem__("timepoint"))
    if bad_timepoint:
        _mutate(
            path,
            lambda f: f["grids/ct_tp1"].attrs.__setitem__(
                "timepoint", encode_attr("tp7")
            ),
        )


def _pyramid_base(path: Path, *, break_origin: bool = False) -> None:
    from medh5.geometry.multiscale import derive_level_grid

    rng = np.random.default_rng(SEED)
    shape = (16, 32, 32)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        base = w.add_grid(
            "l0",
            shape=shape,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            timepoint="tp0",
        )
        level1 = derive_level_grid(base, (2, 2, 2), "l1")
        w.add_grid(
            "l1",
            shape=level1.shape,
            spacing=level1.spacing,
            origin=level1.origin,
            direction=level1.direction,
            timepoint="tp0",
        )
        w.add_pyramid(
            "CT",
            [
                rng.random(shape).astype(np.float32),
                rng.random((8, 16, 16)).astype(np.float32),
            ],
            grid_levels=["l0", "l1"],
            modality="CT",
            downsample_method="mean",
        )
        w.deidentification(method="dicom-psi-profile")
    if break_origin:
        _mutate(
            path,
            lambda f: f["grids/l1"].attrs.__setitem__(
                "origin", np.array([0.0, 0.0, 0.0])
            ),
        )


def _over_layered(handle: h5py.File) -> None:
    """Rewrite a 2-layer annotation as 5 layers --- W908's reason to exist."""
    group = handle["annotations/organs"]
    data = np.asarray(group["data"][...])
    table = np.asarray(group["layer_class_ids"][...])
    classes = sorted({int(v) for v in table.reshape(-1) if int(v)})
    shape = data.shape[1:]
    wide = np.zeros((5, *shape), dtype=data.dtype)
    for position, class_id in enumerate(classes):
        merged = np.zeros(shape, dtype=bool)
        for layer in range(data.shape[0]):
            merged |= data[layer] == class_id
        wide[position][merged] = class_id
    new_table = np.zeros((5, 1), dtype=np.uint16)
    for position, class_id in enumerate(classes):
        new_table[position, 0] = class_id
    del group["data"], group["layer_class_ids"]
    group.create_dataset("data", data=wide)
    group.create_dataset("layer_class_ids", data=new_table)


def _register_second_batch() -> None:
    @case("multiscale-pyramid", "A two-level pyramid with consistent geometry.", "§4.3")
    def _valid_pyramid(path: Path) -> None:
        _pyramid_base(path)

    _CASES.append(
        Case(
            name="E105-pyramid-origin",
            description="A pyramid level missing the half-voxel origin shift.",
            clause="§4.3",
            build=lambda p: _pyramid_base(p, break_origin=True),
            errors=("E105",),
        )
    )
    _CASES.append(
        Case(
            name="W911-no-relating-transform",
            description="Two timepoints and nothing relating them.",
            clause="§3.7",
            build=_longitudinal_base,
            warnings=("W911",),
        )
    )
    _CASES.append(
        Case(
            name="W910-shared-frame-across-timepoints",
            description="Two timepoints sharing one frame of reference, which "
            "asserts an alignment nobody computed.",
            clause="§3.4",
            build=lambda p: _longitudinal_base(p, shared_frame=True),
            warnings=("W910", "W911"),
        )
    )
    _CASES.append(
        Case(
            name="E106-grid-without-timepoint",
            description="A grid with no `timepoint` in a multi-timepoint sample.",
            clause="§3.7",
            build=lambda p: _longitudinal_base(p, drop_timepoint=True),
            errors=("E106",),
            warnings=("W911",),
        )
    )
    _CASES.append(
        Case(
            name="E107-undeclared-grid-timepoint",
            description="A grid naming a timepoint the document does not declare.",
            clause="§3.7",
            build=lambda p: _longitudinal_base(p, bad_timepoint=True),
            errors=("E107",),
            warnings=("W911",),
        )
    )

    _invalid(
        "E108-nondense-timepoint-index",
        "Timepoint indices that are not dense from zero.",
        "§3.7",
        ["E108"],
        lambda f: _set_meta(f, _break_timepoint_index),
    )
    _invalid(
        "E111-empty-grids",
        "A `grids` group with no grid in it.",
        "§3.2",
        ["E101", "E111"],
        lambda f: f["grids"].__delitem__("ct"),
    )
    _invalid(
        "E412-missing-coverage",
        "An annotation without `annotated_class_ids`.",
        "§6.2",
        ["E412"],
        lambda f: f["annotations/organs"].attrs.__delitem__("annotated_class_ids"),
        base=lambda p: _seg_base(p),
        warnings=["W904", "W912"],
    )
    _invalid(
        "E301-seg-without-labelset",
        "The `seg` profile declared with no label set.",
        "§5.1",
        ["E301"],
        lambda f: _set_meta(f, lambda d: d.pop("label_set", None)),
        base=lambda p: _seg_base(p),
    )
    # A label set that cannot be constructed stops the document from parsing, so
    # the rules that depend on a parsed document correctly do not run.
    _invalid(
        "E302-duplicate-class-id",
        "Two label-set entries with one id.",
        "§5.2",
        ["E302"],
        lambda f: _set_meta(f, _duplicate_class_id),
        base=lambda p: _seg_base(p),
    )
    _invalid(
        "W908-too-many-layers",
        "Five layers where a greedy colouring needs two.",
        "§7.6",
        [],
        _over_layered,
        base=lambda p: _seg_base(p),
        warnings=["W908", "W912"],
    )
    _invalid(
        "E406-box-lo-gt-hi",
        "An instance box with lo greater than hi.",
        "§8.1",
        ["E406"],
        _flip_box,
        base=_seg_instances_base,
        warnings=["W912"],
    )
    _invalid(
        "E408-nonmonotonic-offsets",
        "Mask offsets that decrease.",
        "§7.4",
        ["E408"],
        _break_offsets,
        base=_seg_instances_base,
        warnings=["W912"],
    )
    _invalid(
        "W905-stale-index",
        "An index whose `source_digest` is out of date.",
        "§13.3",
        [],
        lambda f: f["index/organs"].attrs.__setitem__(
            "source_digest", encode_attr("sha256:" + "0" * 64)
        ),
        base=lambda p: _seg_base(p, index=True),
        level="integrity",
        warnings=["W905", "W912"],
    )


def _break_timepoint_index(doc: dict[str, Any]) -> None:
    doc["timepoints"] = [
        {"id": "tp0", "index": 0},
        {"id": "tp2", "index": 2},
    ]


def _duplicate_class_id(doc: dict[str, Any]) -> None:
    classes = doc["label_set"]["classes"]
    classes.append({**classes[0], "key": "liver_copy"})


def _seg_instances_base(path: Path) -> None:
    _seg_instances(path)


def _flip_box(handle: h5py.File) -> None:
    boxes = np.asarray(handle["annotations/lesions/boxes"][...])
    boxes[0, 0] = boxes[0, 0][::-1]
    handle["annotations/lesions/boxes"][...] = boxes


def _break_offsets(handle: h5py.File) -> None:
    offsets = np.asarray(handle["annotations/lesions/mask_offsets"][...])
    offsets[1], offsets[2] = offsets[2], offsets[1]
    handle["annotations/lesions/mask_offsets"][...] = offsets


_register_second_batch()


def _cycle(doc: dict[str, Any]) -> None:
    by_key = {c["key"]: c for c in doc["label_set"]["classes"]}
    by_key["liver"]["parents"] = [3]
    by_key["lesion"]["parents"] = [1]


def _unknown_parent(doc: dict[str, Any]) -> None:
    doc["label_set"]["classes"][0]["parents"] = [900]


def _ref_without_uri(doc: dict[str, Any]) -> None:
    doc["label_set"] = {
        "id": "external-v1",
        "version": "1.0.0",
        "form": "ref",
        "sha256": "0" * 64,
    }


def _decompress_image(handle: h5py.File) -> None:
    values = np.asarray(handle["images/CT"][...])
    attrs = dict(handle["images/CT"].attrs)
    del handle["images/CT"]
    node = handle["images"].create_dataset("CT", data=values)
    for key, value in attrs.items():
        node.attrs[key] = value


def _register_third_batch() -> None:
    _invalid(
        "E109-missing-grid-attribute",
        "A grid without `spacing`.",
        "§3.2",
        ["E109"],
        lambda f: f["grids/ct"].attrs.__delitem__("spacing"),
    )
    _invalid(
        "E205-missing-image-attribute",
        "An image without `modality`.",
        "§4.1",
        ["E205"],
        lambda f: f["images/CT"].attrs.__delitem__("modality"),
    )
    _invalid(
        "E304-hierarchy-cycle",
        "A class hierarchy with a cycle.",
        "§5.3",
        ["E304"],
        lambda f: _set_meta(f, _cycle),
        base=lambda p: _seg_base(p),
    )
    _invalid(
        "E305-ref-without-uri",
        "A `form: ref` label set with no URI.",
        "§5.1",
        ["E005", "E305"],
        lambda f: _set_meta(f, _ref_without_uri),
        base=lambda p: _seg_base(p),
    )
    _invalid(
        "E306-unknown-parent",
        "A class naming a parent that does not exist.",
        "§5.2",
        ["E306"],
        lambda f: _set_meta(f, _unknown_parent),
        base=lambda p: _seg_base(p),
    )
    _invalid(
        "W902-uncompressed-bulk",
        "A multi-megabyte image stored contiguous and uncompressed.",
        "§14.1",
        [],
        _decompress_image,
        base=lambda p: _base(p, shape=(48, 112, 112)),
        warnings=["W902"],
    )


_register_third_batch()


def _det_base(
    path: Path,
    *,
    space: str = "index",
    units: str = "mm",
    bad_box: bool = False,
    bad_rotation: bool = False,
    skeleton: str | None = None,
    with_obb: bool = True,
    with_keypoints: bool = False,
) -> None:
    """A detection sample: boxes, oriented boxes and optionally keypoints."""
    rng = np.random.default_rng(SEED)
    shape = (16, 24, 24)
    boxes = np.array(
        [
            [[1.5, 7.5], [1.5, 9.5], [1.5, 9.5]],
            [[6.5, 11.5], [10.5, 18.5], [4.5, 12.5]],
        ],
        dtype=np.float32,
    )
    label_set = _LS
    if skeleton is not None:
        label_set = LabelSet(
            _LS.id,
            version=_LS.version,
            classes=list(_LS.classes),
            skeletons=[Skeleton(skeleton, (1, 2), ((1, 2),))],
        )
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(label_set)
        w.add_grid(
            "ct",
            shape=shape,
            spacing=(1.5, 0.8, 0.8),
            units=units,
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_boxes(
            "lesions",
            boxes,
            [3, 3],
            grid="ct",
            space=space,
            instance_ids=[1, 2],
            scores=[0.91, 0.62],
            attributes=[{"reader": "r1"}, {"reader": "r1"}],
        )
        if with_obb:
            angle = np.pi / 5
            rotation = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(angle), -np.sin(angle)],
                    [0.0, np.sin(angle), np.cos(angle)],
                ],
                dtype=np.float32,
            )
            w.add_obb(
                "lesions_obb",
                centers=np.array([[6.0, 8.0, 8.0]], dtype=np.float32),
                sizes=np.array([[4.0, 6.0, 6.0]], dtype=np.float32),
                rotations=rotation[None],
                class_ids=[3],
                grid="ct",
                space=space,
            )
        if with_keypoints:
            w.add_keypoints(
                "landmarks",
                points=np.array([[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]], dtype=np.float32),
                keypoint_classes=[1, 2],
                class_ids=[1],
                grid="ct",
                space=space,
                visibility=np.array([[2, 1]], dtype=np.uint8),
                skeleton=skeleton,
            )
        w.deidentification(method="dicom-psi-profile")
    if bad_box:

        def flip(handle: h5py.File) -> None:
            values = np.asarray(handle["annotations/lesions/boxes"][...])
            values[0, 0] = values[0, 0][::-1]
            handle["annotations/lesions/boxes"][...] = values

        _mutate(path, flip)
    if bad_rotation:

        def skew(handle: h5py.File) -> None:
            values = np.asarray(handle["annotations/lesions_obb/rotations"][...])
            values[0, 0, 1] = 0.7
            handle["annotations/lesions_obb/rotations"][...] = values

        _mutate(path, skew)


def _cls_base(path: Path, *, multilabel: bool = True) -> None:
    """A two-timepoint sample with per-visit staging and a change label."""
    rng = np.random.default_rng(SEED)
    shape = (12, 16, 16)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up", days_from_baseline=92)
        w.label_set(_LS)
        for tp, frame in (("tp0", "pseudo:frame-100"), ("tp1", "pseudo:frame-101")):
            w.add_grid(
                f"ct_{tp}",
                shape=shape,
                spacing=(1.5, 0.8, 0.8),
                timepoint=tp,
                frame_uid=frame,
            )
            w.add_image(
                f"CT_{tp}",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid=f"ct_{tp}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        w.add_classification(
            "staging",
            {3: 1.0, 4: 1.0},
            scope="timepoint",
            scope_ids=[0, 1],
            multilabel=multilabel,
            schemes=["Lung-RADS", "Lung-RADS"],
            scheme_values=["4A", "4B"],
        )
        w.add_classification(
            "response",
            {1: 1.0},
            scope="sample",
            multilabel=False,
            timepoints=["tp0", "tp1"],
        )
        w.deidentification(method="dicom-psi-profile")


def _shape_base(path: Path) -> None:
    """Contours and a surface mesh, the two non-voxel shape representations."""
    rng = np.random.default_rng(SEED)
    shape = (12, 16, 16)
    square = np.array(
        [[4.0, 4.0, 4.0], [4.0, 4.0, 9.0], [4.0, 9.0, 9.0], [4.0, 9.0, 4.0]],
        dtype=np.float32,
    )
    hole = np.array(
        [[4.0, 6.0, 6.0], [4.0, 6.0, 7.0], [4.0, 7.0, 7.0]], dtype=np.float32
    )
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0")
        w.label_set(_LS)
        w.add_grid(
            "ct",
            shape=shape,
            spacing=(1.0, 1.0, 1.0),
            timepoint="tp0",
            frame_uid="pseudo:frame-100",
        )
        w.add_image(
            "CT",
            rng.integers(-1000, 1500, shape).astype(np.int16),
            grid="ct",
            modality="CT",
            value_type="quantitative",
            value_units="HU",
        )
        w.add_contours(
            "rtstruct",
            [
                Polygon(square, class_id=1, plane=(0, 4), role="outer"),
                Polygon(hole, class_id=1, plane=(0, 4), role="hole"),
            ],
            grid="ct",
        )
        w.add_mesh(
            "liver_surface",
            vertices,
            faces,
            grid="ct",
            space="world",
            frame_uid="pseudo:frame-100",
            mesh_class_ids=[1],
        )
        w.add_points(
            "landmarks",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
            grid="ct",
            names=["apex", "carina"],
            weights=[1.0, 0.5],
        )
        w.deidentification(method="dicom-psi-profile")


def _crowd_one_scope_unit(handle: h5py.File) -> None:
    """Put both positives in one scope unit, then forbid more than one."""
    group = handle["annotations/staging"]
    group["scope_ids"][...] = np.zeros_like(np.asarray(group["scope_ids"][...]))
    group.attrs["multilabel"] = np.bool_(False)


def _register_geometric_cases() -> None:
    _CASES.append(
        Case(
            name="det-boxes-obb",
            description="Axis-aligned and oriented boxes with scores and attributes.",
            clause="§8.2, §8.3",
            build=_det_base,
            warnings=("W912",),
        )
    )
    _CASES.append(
        Case(
            name="det-keypoints",
            description="Keypoints with per-slot classes, visibility and a skeleton.",
            clause="§8.4",
            build=lambda p: _det_base(p, with_keypoints=True, skeleton="pair"),
            warnings=("W912",),
        )
    )
    _CASES.append(
        Case(
            name="det-boxes-world",
            description="Boxes stored in world coordinates of a named frame.",
            clause="§8.1",
            build=lambda p: _det_base(p, space="world"),
            warnings=("W912",),
        )
    )
    _CASES.append(
        Case(
            name="shapes-contours-mesh",
            description="Planar contours with a hole, a surface mesh, and landmarks.",
            clause="§8.5, §8.6, §8.7",
            build=_shape_base,
            warnings=("W912",),
        )
    )
    _CASES.append(
        Case(
            name="cls-staging-and-change",
            description="Per-visit staging with an ordinal scheme, plus a change "
            "label naming the timepoints compared.",
            clause="§9",
            build=_cls_base,
            warnings=("W911", "W912"),
        )
    )
    _invalid(
        "E406-box-lo-gt-hi-boxes",
        "An axis-aligned box with lo greater than hi.",
        "§8.1",
        ["E406"],
        lambda f: None,
        base=lambda p: _det_base(p, bad_box=True),
        warnings=["W912"],
    )
    _invalid(
        "E407-improper-rotation",
        "An `obb` rotation matrix that is not a proper rotation.",
        "§8.3",
        ["E407"],
        lambda f: None,
        base=lambda p: _det_base(p, bad_rotation=True),
        warnings=["W912"],
    )
    _invalid(
        "E412-missing-space",
        "A geometric annotation with no `space`.",
        "§8.1",
        ["E412"],
        lambda f: f["annotations/lesions"].attrs.__delitem__("space"),
        base=_det_base,
        warnings=["W912"],
    )
    _invalid(
        "E413-unknown-skeleton",
        "A `keypoints` annotation naming a skeleton the label set lacks.",
        "§8.4",
        ["E413"],
        lambda f: f["annotations/landmarks"].attrs.__setitem__(
            "skeleton", encode_attr("nope")
        ),
        base=lambda p: _det_base(p, with_keypoints=True, skeleton="pair"),
        warnings=["W912"],
    )
    _invalid(
        "E414-world-space-on-px-grid",
        "World coordinates on an uncalibrated (units='px') grid.",
        "§3.5",
        ["E414"],
        lambda f: f["grids/ct"].attrs.__setitem__("units", encode_attr("px")),
        base=lambda p: _det_base(p, space="world"),
        warnings=["W912"],
    )
    _invalid(
        "E404-single-label-two-positives",
        "multilabel=false with two positive classes in one scope unit.",
        "§9",
        ["E404"],
        _crowd_one_scope_unit,
        base=lambda p: _cls_base(p),
        warnings=["W911", "W912"],
    )
    _invalid(
        "E408-contour-offsets",
        "Contour offsets that do not end at the vertex count.",
        "§8.6",
        ["E408"],
        lambda f: f["annotations/rtstruct/contour_offsets"].__setitem__(
            slice(None), np.array([0, 4, 5], dtype=np.int64)
        ),
        base=_shape_base,
        warnings=["W912"],
    )


_register_geometric_cases()


def _reg_base(
    path: Path,
    *,
    with_displacement: bool = False,
    with_bspline: bool = False,
    with_composite: bool = False,
    with_inverse: bool = False,
    landmarks: bool = True,
) -> None:
    """Two timepoints related by a transform, with landmark ground truth (§10)."""
    rng = np.random.default_rng(SEED)
    shape = (12, 16, 16)
    fixed = np.array([[2.0, 3.0, 4.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    shift = np.array([2.0, -1.0, 0.5])
    matrix = np.eye(4)
    matrix[:3, 3] = shift
    with medh5.create(path, sample_id=path.stem, codec="portable") as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up", days_from_baseline=92)
        w.label_set(_LS)
        for tp, frame in (("tp0", "pseudo:frame-100"), ("tp1", "pseudo:frame-101")):
            w.add_grid(
                f"ct_{tp}",
                shape=shape,
                spacing=(1.5, 0.8, 0.8),
                origin=(0.0, 0.0, 0.0),
                timepoint=tp,
                frame_uid=frame,
            )
            w.add_image(
                f"CT_{tp}",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid=f"ct_{tp}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            from_frame="pseudo:frame-100",
            to_frame="pseudo:frame-101",
            matrix=matrix,
            from_grid="ct_tp0",
            to_grid="ct_tp1",
            invertible=True,
            inverse_id="tp1_to_tp0" if with_inverse else None,
            metrics={"status": "approved", "confidence": 0.88},
        )
        if with_inverse:
            inverse = np.eye(4)
            inverse[:3, 3] = -shift
            w.add_transform(
                "tp1_to_tp0",
                kind="affine",
                from_frame="pseudo:frame-101",
                to_frame="pseudo:frame-100",
                matrix=inverse,
                invertible=True,
                inverse_id="tp0_to_tp1",
            )
        if with_displacement:
            field = np.zeros((3, *shape), dtype=np.float32)
            field[0] = 0.75
            w.add_transform(
                "refine",
                kind="displacement",
                from_frame="pseudo:frame-101",
                to_frame="pseudo:frame-102",
                field=field,
                field_grid="ct_tp1",
                vector_space="world",
            )
        if with_bspline:
            control = np.zeros((3, 6, 6, 6), dtype=np.float64)
            control[1] = 0.5
            w.add_grid(
                "cp",
                shape=(6, 6, 6),
                spacing=(3.0, 3.2, 3.2),
                origin=(0.0, 0.0, 0.0),
                timepoint="tp1",
                frame_uid="pseudo:frame-101",
            )
            w.add_transform(
                "ffd",
                kind="bspline",
                from_frame="pseudo:frame-101",
                to_frame="pseudo:frame-103",
                control_points=control,
                cp_grid="cp",
                order=3,
            )
        if with_composite:
            w.add_transform(
                "tp0_to_refined",
                kind="composite",
                from_frame="pseudo:frame-100",
                to_frame="pseudo:frame-102",
                components=["tp0_to_tp1", "refine"],
            )
        if landmarks:
            w.add_points(
                "landmarks_tp0",
                fixed,
                grid="ct_tp0",
                space="world",
                names=["apex", "carina"],
                weights=[1.0, 1.0],
                correspondence="landmarks_tp1",
                task="registration",
            )
            w.add_points(
                "landmarks_tp1",
                fixed + shift.astype(np.float32),
                grid="ct_tp1",
                space="world",
                names=["apex", "carina"],
                correspondence="landmarks_tp0",
                task="registration",
            )
        w.deidentification(method="dicom-psi-profile")


def _register_transform_cases() -> None:
    _CASES.append(
        Case(
            name="reg-affine-landmarks",
            description="Baseline-to-follow-up affine with paired landmark ground "
            "truth and a metrics record.",
            clause="§10.3, §10.6",
            build=_reg_base,
        )
    )
    _CASES.append(
        Case(
            name="reg-inverse-pair",
            description="Two affines declaring each other as inverses.",
            clause="§10.1",
            build=lambda p: _reg_base(p, with_inverse=True),
        )
    )
    _CASES.append(
        Case(
            name="reg-displacement-composite",
            description="A dense field refining an affine, and the composite of both.",
            clause="§10.4, §10.5",
            build=lambda p: _reg_base(p, with_displacement=True, with_composite=True),
        )
    )
    _CASES.append(
        Case(
            name="reg-bspline",
            description="A cubic free-form deformation on a control-point lattice.",
            clause="§10.5",
            build=lambda p: _reg_base(p, with_bspline=True),
        )
    )
    _invalid(
        "E501-broken-composite-chain",
        "A composite whose components do not chain.",
        "§10.5",
        ["E501"],
        lambda f: f["transforms/tp0_to_refined"].attrs.__setitem__(
            "to_frame", encode_attr("pseudo:frame-999")
        ),
        base=lambda p: _reg_base(p, with_displacement=True, with_composite=True),
    )
    _invalid(
        "E502-unknown-transform-kind",
        "A transform of an unregistered kind.",
        "§10.1",
        ["E502"],
        lambda f: f["transforms/tp0_to_tp1"].attrs.__setitem__(
            "kind", encode_attr("wormhole")
        ),
        base=_reg_base,
    )
    _invalid(
        "E503-field-grid-wrong-frame",
        "A displacement field sampled outside the source frame.",
        "§10.4",
        ["E503"],
        lambda f: f["transforms/refine"].attrs.__setitem__(
            "from_frame", encode_attr("pseudo:frame-100")
        ),
        base=lambda p: _reg_base(p, with_displacement=True),
    )
    _invalid(
        "E504-affine-last-row",
        "An affine whose last row is not [0 … 0 1].",
        "§10.3",
        ["E504"],
        _break_affine_last_row,
        base=_reg_base,
    )
    _invalid(
        "E505-inverse-not-mutual",
        "An `inverse_id` naming a transform that is not the inverse.",
        "§10.1",
        ["E505"],
        lambda f: f["transforms/tp1_to_tp0"].attrs.__setitem__(
            "inverse_id", encode_attr("tp1_to_tp0")
        ),
        base=lambda p: _reg_base(p, with_inverse=True),
    )


def _break_affine_last_row(handle: h5py.File) -> None:
    matrix = np.asarray(handle["transforms/tp0_to_tp1/matrix"][...])
    matrix[-1, 0] = 0.5
    handle["transforms/tp0_to_tp1/matrix"][...] = matrix


_register_transform_cases()


# --------------------------------------------------------------------------
# §2.2 collections and §7.4 tracking
# --------------------------------------------------------------------------


def _tracking_sample(
    path: Path,
    *,
    reclassify: bool = False,
    partial_coverage: bool = False,
) -> None:
    """Two visits of one subject, with the same lesion in both (§7.4).

    ``reclassify`` makes instance 7 a different class at follow-up, which is the
    cross-annotation tracking error W909 exists to catch; ``partial_coverage``
    withdraws the follow-up commitment so absence becomes *unexamined*.
    """
    rng = np.random.default_rng(SEED)
    shape = (12, 16, 16)

    def lesion(z: int, y: int, x: int, r: int) -> npt.NDArray[np.bool_]:
        mask = np.zeros(shape, dtype=bool)
        mask[z - r : z + r, y - r : y + r, x - r : x + r] = True
        return mask

    with medh5.create(
        path, sample_id=path.stem, subject_id="subj-A", codec="portable"
    ) as w:
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="follow_up", days_from_baseline=92)
        w.label_set(_LS)
        rad = w.person("pseudonym:RAD-07", role="annotator")
        act = w.activity("annotate", agent=rad, ended="2026-02-05T14:47:00Z")
        for gid, tp, frame in (
            ("ct_tp0", "tp0", "pseudo:frame-100"),
            ("ct_tp1", "tp1", "pseudo:frame-101"),
        ):
            w.add_grid(
                gid, shape=shape, spacing=(1.5, 0.8, 0.8), timepoint=tp, frame_uid=frame
            )
            w.add_image(
                f"CT_{tp}",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid=gid,
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
        w.add_segmentation(
            "lesions_tp0",
            grid="ct_tp0",
            instances=[
                InstanceInput(class_id=3, instance_id=7, mask=lesion(6, 6, 6, 2)),
                InstanceInput(class_id=3, instance_id=8, mask=lesion(6, 11, 11, 1)),
            ],
            annotated_classes=[3],
            prov=act,
            quality={"status": "approved", "confidence": 0.9},
        )
        w.add_segmentation(
            "lesions_tp1",
            grid="ct_tp1",
            instances=[
                InstanceInput(
                    class_id=1 if reclassify else 3,
                    instance_id=7,
                    mask=lesion(6, 6, 6, 3),
                ),
                InstanceInput(class_id=3, instance_id=9, mask=lesion(7, 3, 12, 1)),
            ],
            # A case isolates one defect: the reclassified follow-up commits to
            # both classes so W904 does not fire alongside the W909 it is for.
            annotated_classes=(
                [] if partial_coverage else [1, 3] if reclassify else [3]
            ),
            prov=act,
            quality={"status": "approved"},
        )
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            matrix=np.eye(4),
            from_frame="pseudo:frame-100",
            to_frame="pseudo:frame-101",
        )
        w.deidentification(method="dicom-psi-profile")
        w.split(set_id="cv5-2026-02", partition="train", fold=1)


@case(
    "longitudinal-instance-tracking",
    "One lesion followed across two visits, joined on `instance_id`.",
    "§7.4, §11.3",
    warnings=["W912"],
)
def _tracking(path: Path) -> None:
    _tracking_sample(path)


@case(
    "W909-instance-reclassified-across-timepoints",
    "One `instance_id` carrying a different class at follow-up.",
    "§7.4",
    warnings=["W909", "W912"],
)
def _tracking_reclassified(path: Path) -> None:
    _tracking_sample(path, reclassify=True)


@case(
    "W904-follow-up-coverage-withdrawn",
    "A follow-up that commits to no class, so absence measures nothing.",
    "§11.3",
    warnings=["W904", "W912"],
)
def _tracking_unexamined(path: Path) -> None:
    _tracking_sample(path, partial_coverage=True)


def _collection(
    path: Path,
    *,
    samples: int = 2,
    drop_content_id: bool = False,
    bad_key: bool = False,
    drop_samples_group: bool = False,
) -> None:
    """A shard built the way a curator builds one: pack standalone samples."""
    import shutil

    from medh5.collection import SAMPLES_GROUP, pack

    # The members are scaffolding, not corpus files: a shipped corpus directory
    # must contain exactly the cases its manifest lists.
    directory = path.parent / f".{path.stem}-members"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    try:
        sources = []
        for i in range(samples):
            member = directory / f"case_{i}.medh5"
            _base(member, timepoints=(("tp0", {"label": "baseline"}),))
            sources.append(member)
        keys = ["case.0", "not a key" if bad_key else "case_1"][:samples]
        pack(sources, path, keys=keys)
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    if drop_content_id:
        _mutate(
            path,
            lambda f: f[f"{SAMPLES_GROUP}/case.0"].attrs.__delitem__("content_id"),
        )
    if drop_samples_group:
        _mutate(path, lambda f: f.__delitem__(SAMPLES_GROUP))


@case(
    "collection-two-samples",
    "Two sample roots in one shard, each independently identifiable.",
    "§2.2",
    suffix=".medh5c",
)
def _collection_valid(path: Path) -> None:
    _collection(path)


@case(
    "E010-collection-member-without-content-id",
    "A packed sample root that lost its own `content_id`.",
    "§2.2",
    errors=["E010"],
    suffix=".medh5c",
    mutated=True,
)
def _collection_no_content_id(path: Path) -> None:
    _collection(path, drop_content_id=True)


@case(
    "E003-collection-bad-sample-key",
    "A sample key outside [A-Za-z0-9_.-]{1,255}.",
    "§2.2",
    errors=["E003"],
    suffix=".medh5c",
    mutated=True,
)
def _collection_bad_key(path: Path) -> None:
    from medh5._hdf5 import open_h5

    _collection(path, samples=1)
    with open_h5(path, "r+") as handle:
        handle["samples"].move("case.0", "not a key")


@case(
    "E008-collection-without-samples-group",
    "A file declaring `collection` with nothing in it.",
    "§2.2",
    errors=["E008"],
    suffix=".medh5c",
    mutated=True,
)
def _collection_empty(path: Path) -> None:
    _collection(path, samples=1, drop_samples_group=True)


def _widen_labelmap(handle: h5py.File) -> None:
    """Store a labelmap that fits in uint8 as uint16 --- §7.1's refusal."""
    group = handle["annotations/organs"]
    values = np.asarray(group["data"][...]).astype(np.uint16)
    attrs = dict(group["data"].attrs)
    del group["data"]
    node = group.create_dataset("data", data=values)
    for key, value in attrs.items():
        node.attrs[key] = value


def _set_split_timestamp(doc: dict[str, Any]) -> None:
    doc["splits"] = [
        {"set_id": "cv5", "partition": "train", "assigned_at": "yesterday"}
    ]


def _set_deidentification_date(doc: dict[str, Any]) -> None:
    doc.setdefault("deidentification", {"method": "dicom-psi-profile"})
    doc["deidentification"]["date"] = "yesterday"


def _register_fourth_batch() -> None:
    """1.3.0: every cross-reference clause gets a case, not only every code."""

    @case(
        "seg-probmap-threshold",
        "Soft labels with a declared decision threshold.",
        "§7.5, §7.6",
        warnings=["W912"],
    )
    def _seg_probmap_threshold(path: Path) -> None:
        rng = np.random.default_rng(SEED)
        shape = (16, 24, 24)
        with medh5.create(path, sample_id=path.stem, codec="portable") as w:
            w.add_timepoint("tp0")
            w.label_set(_LS)
            w.add_grid("ct", shape=shape, spacing=(1.5, 0.8, 0.8), timepoint="tp0")
            w.add_image(
                "CT",
                rng.integers(-1000, 1500, shape).astype(np.int16),
                grid="ct",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
            )
            w.add_segmentation(
                "soft",
                grid="ct",
                probabilities={1: rng.random(shape), 3: rng.random(shape)},
                threshold=0.3,
            )
            w.deidentification(method="dicom-psi-profile")

    _invalid(
        "E703-unknown-digest-algo",
        "A `digest_algo` outside the §2.1 vocabulary.",
        "§2.1",
        ["E703"],
        lambda f: f.attrs.__setitem__("digest_algo", encode_attr("blake3")),
        level="integrity",
    )
    _invalid(
        "E411-labelmap-wide-dtype",
        "A `labelmap` stored uint16 whose ids fit uint8 and carries no ignore voxel.",
        "§7.1",
        ["E411"],
        _widen_labelmap,
        base=lambda p: _seg_base(
            p,
            encoding="labelmap",
            classes={1: (2, 2, 2), 2: (2, 12, 2), 4: (9, 2, 12)},
        ),
        warnings=["W912"],
    )
    _invalid(
        "E109-time-axis-without-time-values",
        "A grid with a `time` axis and no `time_values`.",
        "§3.2",
        ["E109"],
        lambda f: f["grids/dce"].attrs.__delitem__("time_values"),
        base=_core_4d,
    )
    _seg_invalid(
        "E413-dangling-ignore-mask",
        "An `ignore_mask` naming an annotation that does not exist.",
        "§7.7",
        ["E413"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "ignore_mask", encode_attr("nope")
        ),
    )
    _seg_invalid(
        "E413-ignore-mask-not-a-mask",
        "An `ignore_mask` naming an annotation that is not a `mask`.",
        "§7.7",
        ["E413"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "ignore_mask", encode_attr("organs")
        ),
    )
    _seg_invalid(
        "E413-dangling-derived-from",
        "A `derived_from` entry naming an annotation that does not exist.",
        "§6.2",
        ["E413"],
        lambda f: f["annotations/organs"].attrs.__setitem__(
            "derived_from", encode_attr(["ghost"])
        ),
    )
    _invalid(
        "E413-dangling-valid-mask",
        "An image `valid_mask` naming an annotation that does not exist.",
        "§4.4",
        ["E413"],
        lambda f: f["images/CT"].attrs.__setitem__("valid_mask", encode_attr("nope")),
    )
    _invalid(
        "E601-transform-dangling-prov",
        "A transform naming an activity that does not exist.",
        "§10.1, §11.1",
        ["E601"],
        lambda f: f["transforms/tp0_to_tp1"].attrs.__setitem__(
            "prov", encode_attr("act_nope")
        ),
        base=_reg_base,
    )
    _invalid(
        "E602-transform-unknown-metrics",
        "A transform whose `metrics` names no quality record.",
        "§10.1, §11.2",
        ["E602"],
        lambda f: f["transforms/tp0_to_tp1"].attrs.__setitem__(
            "metrics", encode_attr("nope")
        ),
        base=_reg_base,
    )
    _invalid(
        "E604-split-assigned-at",
        "A split claim whose `assigned_at` is not RFC 3339.",
        "§12.3",
        ["E604"],
        lambda f: _set_meta(f, _set_split_timestamp),
    )
    _invalid(
        "E604-deidentification-date",
        "A de-identification record whose `date` is not RFC 3339.",
        "§11.4",
        ["E604"],
        lambda f: _set_meta(f, _set_deidentification_date),
    )


_register_fourth_batch()


CASES: tuple[Case, ...] = tuple(_CASES)


def case_by_name(name: str) -> Case:
    for entry in CASES:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown conformance case {name!r}")


def build_corpus(outdir: str | Path, *, names: Sequence[str] | None = None) -> Path:
    """Write every case and an ``expected.json`` manifest beside them."""
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CASES if names is None or c.name in set(names)]
    manifest = {
        "format": medh5.FORMAT_VERSION,
        "generator": f"medh5 {medh5.__version__}",
        "cases": [],
    }
    for entry in selected:
        path = root / f"{entry.name}{entry.suffix}"
        if path.exists():
            path.unlink()
        entry.build(path)
        record = entry.to_json()
        record["file"] = path.name
        manifest["cases"].append(record)  # type: ignore[attr-defined]
    (root / "expected.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return root / "expected.json"


def run_corpus(
    outdir: str | Path, *, names: Sequence[str] | None = None
) -> list[CaseResult]:
    """Build every case, validate it, and compare against its expected codes."""
    root = Path(outdir)
    build_corpus(root, names=names)
    results: list[CaseResult] = []
    for entry in CASES:
        if names is not None and entry.name not in set(names):
            continue
        path = root / f"{entry.name}{entry.suffix}"
        result = CaseResult(case=entry, path=str(path))
        try:
            report = validate_file(path, level=entry.level)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            results.append(result)
            continue
        got_errors = {d.code for d in report.errors}
        got_warnings = {d.code for d in report.warnings}
        result.got_errors = tuple(sorted(got_errors))
        result.got_warnings = tuple(sorted(got_warnings))
        expected = set(entry.errors) | set(entry.warnings)
        got = got_errors | got_warnings
        result.missing = tuple(sorted(expected - got))
        result.unexpected = tuple(sorted(got - expected))
        result.details = [str(d) for d in report.diagnostics]
        results.append(result)
    return results


__all__ = [
    "CASES",
    "Case",
    "CaseResult",
    "build_corpus",
    "case_by_name",
    "run_corpus",
]
