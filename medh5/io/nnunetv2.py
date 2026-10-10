"""nnU-Net v2 dataset import and export (spec §5, §7).

An nnU-Net v2 dataset is a directory of NIfTI files plus a ``dataset.json``:

.. code-block:: text

    imagesTr/CASE_0000.nii.gz   channel 0        labelsTr/CASE.nii.gz
    imagesTr/CASE_0001.nii.gz   channel 1        dataset.json

Two properties of that layout are worth preserving carefully.

**Label ids are the file's own.**  nnU-Net's ``labels`` maps a name to the
integer written in the label volume, and those integers are meaningful --- a
model trained against them predicts them.  They become MEDH5 class ids
unchanged, so a prediction can be written back without a translation table.
The one id that cannot survive is ``0``: it is nnU-Net's background and MEDH5
reserves it (§5.3), so a class explicitly named for 0 is dropped and reported.

**Regions overlap on purpose.**  A ``labels`` entry whose value is a *list*
(nnU-Net's region-based training) names a union of ids, which is exactly the
overlapping case §7 exists for: the regions are stored as their own classes
alongside the components, and the encoding is chosen by measurement.

**Ignore is not background.**  nnU-Net declares an ``ignore`` label --- the
highest value in ``labels`` --- for voxels nobody examined, which training
and evaluation leave out.  It is read as the annotation's §7.7 ignore region
and written back from it: written as 0 instead, every unexamined voxel became
a verified negative for every class.

``dataset.json`` is stashed verbatim in ``/meta → extra.nnunetv2`` so an export
reproduces the dataset that was imported rather than a reconstruction of it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5ValidationError
from medh5.io._common import sanitize_key
from medh5.io.report import ConversionReport

REQUIRED_KEYS = ("channel_names", "labels", "numTraining", "file_ending")
BACKGROUND = 0
IGNORE = "ignore"
"""nnU-Net v2's name for the ignore label, whose value is the highest."""


def read_dataset_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and check an nnU-Net v2 ``dataset.json``."""
    target = Path(os.fspath(path))
    if target.is_dir():
        target = target / "dataset.json"
    if not target.exists():
        raise MEDH5ValidationError(f"dataset.json not found at {target}")
    raw = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise MEDH5ValidationError("dataset.json must contain an object")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise MEDH5ValidationError(f"dataset.json is missing {missing}")
    channels = raw["channel_names"]
    if not isinstance(channels, dict) or not channels:
        raise MEDH5ValidationError(
            "dataset.json `channel_names` must be a non-empty object"
        )
    indices = sorted(int(k) for k in channels)
    if indices != list(range(len(indices))):
        raise MEDH5ValidationError(
            f"channel_names must cover 0..{len(indices) - 1}, got {indices}"
        )
    if not isinstance(raw["labels"], dict) or not raw["labels"]:
        raise MEDH5ValidationError("dataset.json `labels` must be a non-empty object")
    return raw


def _channels(document: Mapping[str, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in document["channel_names"].items()}


def _labels(document: Mapping[str, Any]) -> dict[str, Any]:
    return dict(document["labels"])


def cases(root: str | os.PathLike[str], document: Mapping[str, Any]) -> list[str]:
    """Case identifiers present in ``imagesTr``."""
    ending = str(document["file_ending"])
    directory = Path(os.fspath(root)) / "imagesTr"
    if not directory.is_dir():
        raise MEDH5ValidationError(f"{directory} does not exist")
    found = set()
    for path in sorted(directory.glob(f"*{ending}")):
        stem = path.name[: -len(ending)]
        if "_" in stem and stem.rsplit("_", 1)[1].isdigit():
            found.add(stem.rsplit("_", 1)[0])
    return sorted(found)


def from_nnunetv2(
    root: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    case_ids: Sequence[str] | None = None,
    coord_system: str = "LPS",
    codec: str = "balanced",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Convert an nnU-Net v2 dataset into one ``.medh5`` per case."""
    import medh5
    from medh5.io.nifti import _same_grid, read_nifti

    log = report or ConversionReport(converter="from-nnunet")
    log.source = os.fspath(root)
    source = Path(os.fspath(root))
    document = read_dataset_json(source)
    channels = _channels(document)
    restore = _recorded_ids(document)
    label_set, regions, ignore_value = _label_set(document, log)
    ending = str(document["file_ending"])
    wanted = list(case_ids) if case_ids is not None else cases(source, document)
    directory = Path(os.fspath(out))
    directory.mkdir(parents=True, exist_ok=True)

    for case in wanted:
        images: dict[str, npt.NDArray[Any]] = {}
        rescales: dict[str, tuple[float, float] | None] = {}
        geometry: dict[str, Any] | None = None
        for index, name in sorted(channels.items()):
            path = source / "imagesTr" / f"{case}_{index:04d}{ending}"
            if not path.exists():
                raise MEDH5ValidationError(f"case {case!r} has no channel {index}")
            data, geo = read_nifti(path, coord_system=coord_system)
            # Every channel is written onto one grid, so a channel that does not
            # share it must be refused rather than filed under it.  nnU-Net
            # requires co-registered channels and most datasets are, but "the
            # inputs were already correct" is not a check: an unchecked channel
            # at a different spacing lands on the first channel's grid with its
            # voxels intact and its position silently wrong.
            geometry = _same_grid(geometry, geo, name, log) if geometry else geo
            images[name] = data
            rescales[name] = geo["rescale"]
        assert geometry is not None
        label_path = source / "labelsTr" / f"{case}{ending}"
        masks: dict[int, npt.NDArray[np.bool_]] | None = None
        ignored: npt.NDArray[np.bool_] | None = None
        if label_path.exists():
            volume, label_geo = read_nifti(label_path, coord_system=coord_system)
            # The label volume above all: a label resampled onto a different grid
            # by some other tool is the ordinary way this goes wrong, and the
            # result annotates voxels nobody drew on.
            _same_grid(geometry, label_geo, f"{case} labels", log)
            ids = _label_ids(volume, label_geo["rescale"], case)
            if ignore_value is not None and np.any(ids == ignore_value):
                ignored = np.asarray(ids == ignore_value)
            ids = _restored(ids, restore)
            masks = _masks_from(ids, label_set, regions)

        target = directory / f"{case}.medh5"
        with medh5.create(
            target, sample_id=case, subject_id=case, codec=codec
        ) as writer:
            tool = writer.software("medh5", medh5.__version__)
            activity = writer.activity(
                "import",
                agent=tool,
                tool="medh5 convert from-nnunet",
                inputs=[f"nnunetv2:{source.name}/{case}"],
            )
            writer.label_set(label_set)
            writer.extra("nnunetv2", dict(document))
            writer.add_grid(
                "ref",
                shape=geometry["shape"],
                spacing=geometry["spacing"],
                origin=geometry["origin"],
                direction=geometry["direction"],
                coord_system=geometry["coord_system"],
                units=geometry["units"],
                timepoint="tp0",
            )
            for name, array in images.items():
                # The header's scl_slope/scl_inter, kept as the image's §4.2
                # rescale: dropping them left a CT with intercept -1024 at the
                # stored counts, 1024 HU off what nnU-Net itself reads.
                rescale = rescales[name]
                if rescale is not None:
                    log.decision(
                        "value_scale",
                        f"case {case}: {name}'s scl_slope/scl_inter were stored as "
                        "its rescale; read(physical=True) applies them (§4.2)",
                        {"case": case, "slope": rescale[0], "intercept": rescale[1]},
                    )
                writer.add_image(
                    name,
                    array,
                    grid="ref",
                    modality=_modality(name),
                    rescale_slope=None if rescale is None else rescale[0],
                    rescale_intercept=None if rescale is None else rescale[1],
                    prov=activity,
                )
            if masks:
                kind, _ = writer.add_segmentation(
                    "seg",
                    grid="ref",
                    masks=masks,
                    annotated_classes="all",
                    ignore=ignored,
                    prov=activity,
                )
                if ignored is not None:
                    log.decision(
                        "ignore",
                        f"case {case}: {int(ignored.sum())} voxel(s) of nnU-Net's "
                        f"ignore label {ignore_value} became the annotation's "
                        "ignore region (§7.7), unexamined rather than background",
                        {"case": case, "voxels": int(ignored.sum())},
                    )
                log.decision(
                    "encoding",
                    f"case {case}: labels were stored as {kind!r}",
                    {"case": case, "kind": kind},
                )
        log.outputs.append(str(target))
    log.decision(
        "coverage",
        "annotated_class_ids covers the whole label set: an nnU-Net label volume "
        "is exhaustive by construction, so a class absent from it is verified "
        "absent rather than unexamined (§11.3)",
        {"classes": len(label_set)},
    )
    return log


def _recorded_ids(document: Mapping[str, Any]) -> dict[int, int]:
    """The class id each renumbered label value stands for, as an export
    records it (``medh5_class_ids``; F23 of the round-4 audit)."""
    recorded = document.get("medh5_class_ids") or {}
    if not isinstance(recorded, Mapping):
        raise MEDH5ValidationError(
            "dataset.json `medh5_class_ids` must map label values to class ids"
        )
    out = {int(value): int(class_id) for value, class_id in recorded.items()}
    if len(set(out.values())) != len(out):
        raise MEDH5ValidationError(
            "dataset.json `medh5_class_ids` gives two label values one class id"
        )
    return out


def _restored(ids: npt.NDArray[Any], restore: Mapping[int, int]) -> npt.NDArray[Any]:
    """*ids* with each renumbered value back at its class id."""
    if not restore:
        return ids
    out = np.array(ids, dtype=np.int64, copy=True)
    for value, class_id in restore.items():
        out[ids == value] = class_id
    return out


def _modality(name: str) -> str:
    """nnU-Net channel names are free text; map the common ones, else ``OT``."""
    known = {
        "ct": "CT",
        "t1": "MR",
        "t1ce": "MR",
        "t2": "MR",
        "flair": "MR",
        "pet": "PT",
    }
    return known.get(name.strip().lower(), "OT")


def _label_set(
    document: Mapping[str, Any], log: ConversionReport
) -> tuple[Any, dict[int, list[int]], int | None]:
    """nnU-Net ``labels`` as a MEDH5 label set, keeping its integer ids, and
    the value of its ignore label, which is no class."""
    from medh5.labels import LabelClass, LabelSet

    scalars: dict[str, int] = {}
    region_values: dict[str, list[int]] = {}
    dropped: list[str] = []
    ignore_value: int | None = None
    restore = _recorded_ids(document)
    for name, value in _labels(document).items():
        if name == IGNORE and not isinstance(value, list):
            ignore_value = int(value)
        elif isinstance(value, list):
            region_values[name] = [restore.get(int(v), int(v)) for v in value]
        elif int(value) == BACKGROUND:
            dropped.append(name)
        else:
            scalars[name] = restore.get(int(value), int(value))
    if restore:
        log.decision(
            "label_ids",
            "dataset.json records the class id each renumbered label value stands "
            "for (medh5_class_ids); the class ids were restored",
            {"restored": {str(v): c for v, c in sorted(restore.items())}},
        )

    next_id = max([*scalars.values(), 0]) + 1
    regions: dict[int, list[int]] = {}
    region_ids: dict[str, int] = {}
    for name, components in region_values.items():
        region_ids[name] = next_id
        regions[next_id] = components
        next_id += 1

    # A region is a union of its components, which is exactly a parent/child
    # relation in the §5.1 DAG --- so it is stored as one, rather than as an
    # opaque list only this converter understands.
    parents: dict[int, list[int]] = {}
    for region_id, components in regions.items():
        for component in components:
            parents.setdefault(component, []).append(region_id)

    classes: list[LabelClass] = [
        LabelClass(
            class_id,
            _key(name),
            name,
            parents=tuple(sorted(parents.get(class_id, ()))),
        )
        for name, class_id in sorted(scalars.items(), key=lambda kv: kv[1])
    ]
    classes.extend(
        LabelClass(
            region_ids[name],
            _key(name),
            name,
            properties={"nnunet_region": components},
        )
        for name, components in sorted(region_values.items())
    )
    if dropped:
        log.decision(
            "background",
            f"label(s) {dropped} map to nnU-Net's background 0, which MEDH5 "
            "reserves (§5.3); they were dropped rather than renumbered, so every "
            "other id still matches the label volume",
            {"dropped": dropped},
        )
    if regions:
        log.decision(
            "regions",
            f"{len(regions)} region label(s) name a union of ids; each became its "
            "own class, with its components recorded as children in the label-set "
            "DAG, which is the overlap §7 handles",
            {"regions": {str(k): v for k, v in regions.items()}},
        )
    log.decision(
        "label_ids",
        "nnU-Net's own integer ids were kept, so a model's predictions map back "
        "without a translation table",
        {"ids": {c.key: c.id for c in classes}},
    )
    return LabelSet("nnunetv2", version="1.0.0", classes=classes), regions, ignore_value


def _label_ids(
    volume: npt.NDArray[Any], rescale: tuple[float, float] | None, case: str
) -> npt.NDArray[Any]:
    """The label volume's class ids as every reader of the file sees them.

    nnU-Net loads labels through nibabel or SimpleITK, which apply the header's
    ``scl_slope``/``scl_inter``, so a file storing 1 with slope 32 is class 32
    to the model; matching the stored integers imported it as class 1, or as
    nothing.  A scaling that leaves a voxel between two ids is no label volume,
    and is refused.
    """
    if rescale is None:
        return np.asarray(volume)
    slope, intercept = rescale
    physical = np.asarray(volume, dtype=np.float64) * slope + intercept
    ids = np.round(physical)
    if not np.allclose(physical, ids, atol=1e-6):
        raise MEDH5ValidationError(
            f"case {case!r}: the label volume's scl_slope/scl_inter ({slope}, "
            f"{intercept}) make labels that are not integers; a label volume "
            "holds class ids"
        )
    return ids.astype(np.int64)


def _masks_from(
    volume: npt.NDArray[Any], label_set: Any, regions: Mapping[int, Sequence[int]]
) -> dict[int, npt.NDArray[np.bool_]]:
    """One boolean mask per class, regions unioned from their components."""
    values = np.asarray(volume)
    masks: dict[int, npt.NDArray[np.bool_]] = {}
    for entry in label_set:
        if entry.id in regions:
            union = np.zeros(values.shape, dtype=bool)
            for component in regions[entry.id]:
                union |= values == component
            masks[entry.id] = union
            continue
        masks[entry.id] = values == entry.id
    return masks


def _key(name: str) -> str:
    return sanitize_key(name)


def to_nnunetv2(
    paths: Sequence[str | os.PathLike[str]],
    out: str | os.PathLike[str],
    *,
    dataset_name: str = "Dataset001_medh5",
    file_ending: str = ".nii.gz",
    annotation: str = "seg",
    classes: Sequence[int | str] | None = None,
    unlabeled: str = "refuse",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Export samples as an nnU-Net v2 dataset.

    When the samples were imported from nnU-Net the stashed ``dataset.json`` is
    reused verbatim, so the export reproduces the original dataset rather than a
    reconstruction of it.

    Everything is checked before anything is written:

    * every case's name is a **file name under the dataset**: its
      ``sample_id`` is a sample key (``[A-Za-z0-9_.-]``), not ``.`` or ``..``,
      and no two differ only in case --- an id was a path, and
      ``../../victim`` or an absolute one wrote outside the export;
    * every case is **labeled**: a case without the annotation is refused, or
      with ``unlabeled="test"`` written to ``imagesTs`` and not counted in
      ``numTraining``;
    * every channel and the labels are **one lattice in one physical space** ---
      shape, spacing, origin and direction, coordinate system, units, and no two
      known frames of reference --- because nnU-Net pairs their voxels by index;
    * the labels are the **dataset's**, every case's classes, and every case was
      **searched for every one** (``annotated_class_ids``): a label volume is
      exhaustive, so 0 says "examined, absent" for every class, which a case
      not searched for one cannot say.  *classes* exports the subset every case
      examined.  A reused ``dataset.json`` that does not name a class some case
      carries is extended with it, or --- region-based --- refused;
    * label values are **consecutive** (``0..K``), as nnU-Net requires: class
      ids with a gap are written as ``1..K`` in ascending order, and
      ``dataset.json`` records the mapping (``medh5_class_ids``), which
      :func:`from_nnunetv2` reads back;
    * every label volume **gives back every class it exports** --- a region
      painted in ``regions_class_order``, overlapping classes refused rather than
      written over each other.
    """
    import medh5
    from medh5.io.nifti import require_nibabel

    nib = require_nibabel()
    if unlabeled not in UNLABELED:
        raise MEDH5ValidationError(
            f"unlabeled= is one of {list(UNLABELED)}, not {unlabeled!r}"
        )
    log = report or ConversionReport(converter="to-nnunet")
    root = _dataset_root(out, dataset_name)
    plan = _plan(paths, annotation, classes, log, unlabeled)
    _require_under(root, plan, file_ending)
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)
    if any(not case.labeled for case in plan.cases):
        (root / "imagesTs").mkdir(parents=True, exist_ok=True)

    labels = dict(plan.labels)
    ignore_value: int | None = None
    for case in plan.cases:
        with medh5.open(case.path) as sample:
            ann = sample.annotations.get(annotation) if case.labeled else None
            if ann is not None:
                # Checked before any of the case's files is written.
                volume = _labelmap_for(ann, labels, plan.order, plan.by_label)
                ignored = np.asarray(sample.ignore_region(annotation), dtype=bool)
                lost = _not_given_back(
                    ann, volume, labels, ignored, plan.by_label, plan.exported
                )
                if lost:
                    raise MEDH5ValidationError(
                        f"{case.path}: the label volume does not give back {lost}: "
                        "nnU-Net holds one value per voxel, so classes that "
                        "overlap must be stated as regions over disjoint "
                        "components; nothing was written for this case"
                    )
            images = "imagesTr" if case.labeled else "imagesTs"
            for index, image_id in enumerate(plan.channels):
                target = root / images / f"{case.name}_{index:04d}{file_ending}"
                _save(
                    nib,
                    sample.images[image_id].grid,
                    sample.images[image_id].read(),
                    target,
                    rescale=sample.images[image_id].rescale,
                )
                log.outputs.append(str(target))
            if ann is not None:
                if ignored.any():
                    # nnU-Net's ignore label is the highest value (§7.7): ignored
                    # voxels written as 0 were verified negatives for every class.
                    if ignore_value is None:
                        ignore_value = _ignore_value(labels)
                        labels[IGNORE] = ignore_value
                    covered = int(np.count_nonzero(ignored & (volume != BACKGROUND)))
                    volume[ignored] = ignore_value
                    log.decision(
                        "ignore",
                        f"{case.path}: {int(ignored.sum())} ignored voxel(s) were "
                        f"written as nnU-Net's ignore label {ignore_value}"
                        + (
                            f", {covered} of them over a class, which nnU-Net "
                            "neither trains nor scores there"
                            if covered
                            else ""
                        ),
                        {
                            "case": case.name,
                            "voxels": int(ignored.sum()),
                            "over": covered,
                        },
                    )
                target = root / "labelsTr" / f"{case.name}{file_ending}"
                _save(nib, ann.grid, volume, target)
                # Listed only when written.  A sample without the annotation
                # produced no label file and the report named one anyway, so a
                # caller checking `outputs` for what to ship was told about a
                # path that did not exist.
                log.outputs.append(str(target))

    stashed = plan.stashed
    document = dict(stashed) if stashed else {}
    document.update(
        {
            "channel_names": {str(i): n for i, n in enumerate(plan.channels)},
            "labels": labels or document.get("labels") or {},
            # The training cases only: a case written to imagesTs is not one.
            "numTraining": sum(1 for case in plan.cases if case.labeled),
            "file_ending": file_ending,
        }
    )
    if plan.order is None:
        document.pop("regions_class_order", None)
    if plan.class_ids:
        document["medh5_class_ids"] = {
            str(value): class_id for value, class_id in sorted(plan.class_ids.items())
        }
    else:
        document.pop("medh5_class_ids", None)
    (root / "dataset.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    log.outputs.append(str(root / "dataset.json"))
    log.decision(
        "dataset_json",
        "the stashed dataset.json was reused verbatim"
        if plan.reused
        else "a dataset.json was generated from the label set",
        {"reused": plan.reused},
    )
    return log


UNLABELED = ("refuse", "test")
"""What :func:`to_nnunetv2` does with a case that lacks the annotation."""


def _dataset_root(out: str | os.PathLike[str], dataset_name: str) -> Path:
    """The dataset's directory: a name, never a path (F05 of the round-4
    audit)."""
    _require_name(dataset_name, "dataset_name")
    return Path(os.fspath(out)) / dataset_name


def _require_name(name: str, what: str) -> None:
    """*name* as one file name: a sample key, and not ``.`` or ``..``.

    Uncoded: ``sample_id`` is free text in a valid sample, so a refusal here
    is the export's, not a defect of the file (§15.2).
    """
    from medh5 import _core

    try:
        _core.validate_sample_key(name)
    except MEDH5ValidationError as e:
        raise MEDH5ValidationError(
            f"{what} {name!r} cannot name a file of the export: {e.message}"
        ) from None
    if name in (".", ".."):
        raise MEDH5ValidationError(f"{what} {name!r} cannot name a file of the export")


def _require_under(root: Path, plan: _Plan, file_ending: str) -> None:
    """Every file the plan writes resolves under *root* --- what the case
    names already ensure, checked where the files are named."""
    top = root.resolve()
    for case in plan.cases:
        folder = "imagesTr" if case.labeled else "imagesTs"
        for target in (
            root / folder / f"{case.name}_0000{file_ending}",
            root / "labelsTr" / f"{case.name}{file_ending}",
        ):
            if not target.resolve().is_relative_to(top):
                raise MEDH5ValidationError(
                    f"{case.path}: case {case.name!r} would write {target}, "
                    f"outside the dataset {root}"
                )


class _Case:
    """One sample of the export: its file, the name its files take, and
    whether it carries the annotation."""

    __slots__ = ("labeled", "name", "path")

    def __init__(self, path: str, name: str, labeled: bool) -> None:
        self.path = path
        self.name = name
        self.labeled = labeled


class _Plan:
    """What an export writes, settled before it writes anything."""

    __slots__ = (
        "by_label",
        "cases",
        "channels",
        "class_ids",
        "exported",
        "labels",
        "order",
        "reused",
        "stashed",
    )

    def __init__(
        self,
        cases: list[_Case],
        channels: list[str],
        labels: dict[str, Any],
        order: list[int] | None,
        stashed: dict[str, Any] | None,
        reused: bool,
        by_label: dict[str, int],
        class_ids: dict[int, int],
        exported: frozenset[int],
    ) -> None:
        self.cases = cases
        self.channels = channels
        self.labels = labels
        self.order = order
        self.stashed = stashed
        self.reused = reused
        # A scalar label's class, where the plan knows it: what its value
        # cannot say once the values are renumbered (F23).
        self.by_label = by_label
        # Written value -> class id, when the values are not the ids.
        self.class_ids = class_ids
        # The classes every label volume must give back (F04).
        self.exported = exported


def _plan(
    paths: Sequence[str | os.PathLike[str]],
    annotation: str,
    classes: Sequence[int | str] | None,
    log: ConversionReport,
    unlabeled: str = "refuse",
) -> _Plan:
    """Every case read once, and every refusal made, before a file is written.

    The labels were the first annotation's and every later case was written
    against them: a class only a later case carried became background there,
    and a case never searched for a class was written as its verified absence
    (N14 of the round-3 audit).  The vocabulary is every case's; a case is held
    to it, or to *classes*; and every channel and the labels are held to the
    first channel's physical space (N17).  Case names are file names (F05 of
    the round-4 audit), a case without labels is no training case (F22), a
    reused ``dataset.json`` names every class (F04), and the label values are
    consecutive (F23).
    """
    import medh5

    stashed: dict[str, Any] | None = None
    channels: list[str] = []
    names: dict[int, str] = {}
    searched: list[tuple[str, frozenset[int]]] = []
    cases: list[_Case] = []
    taken: dict[str, str] = {}
    for path in paths:
        with medh5.open(path) as sample:
            name = sample.identity.sample_id
            _require_name(name, f"{os.fspath(path)}: sample_id")
            # Case-insensitive file systems hold one of two names that differ
            # only in case, and the second case overwrote the first.
            folded = name.casefold()
            if folded in taken:
                raise MEDH5ValidationError(
                    f"{os.fspath(path)}: sample_id {name!r} names the same files "
                    f"as {taken[folded]!r} (on a case-insensitive file system); "
                    "each case of a dataset needs a name of its own"
                )
            taken[folded] = name
            stashed = stashed or sample.document.extra.get("nnunetv2")
            if not channels:
                channels = (
                    [str(v) for _, v in sorted(_stashed_channels(stashed).items())]
                    if stashed
                    else sorted(sample.images)
                )
            for image_id in channels:
                if image_id not in sample.images:
                    raise MEDH5ValidationError(
                        f"{path}: no image {image_id!r}; the export needs the same "
                        f"channels in every case ({channels})"
                    )
            reference = sample.images[channels[0]].grid
            for image_id in channels[1:]:
                _require_one_space(
                    reference,
                    sample.images[image_id].grid,
                    f"{path}: channel {image_id!r}",
                    channels[0],
                )
            labeled = annotation in sample.annotations
            cases.append(_Case(os.fspath(path), name, labeled))
            if not labeled:
                continue
            ann = sample.annotations[annotation]
            # On the annotation's own grid, which nnU-Net needs to be the
            # channels': the reference grid's affine was written whatever grid
            # the labels were on.
            _require_one_space(
                reference, ann.grid, f"{path}: annotation {annotation!r}", channels[0]
            )
            for class_id in ann.class_ids:
                key = ann.class_key(int(class_id))
                if names.setdefault(int(class_id), key) != key:
                    raise MEDH5ValidationError(
                        f"{path}: class {int(class_id)} is {key!r} here and "
                        f"{names[int(class_id)]!r} in an earlier case; one label "
                        "volume value means one class across the dataset"
                    )
            searched.append(
                (os.fspath(path), frozenset(int(c) for c in ann.annotated_class_ids))
            )

    bare = [case.path for case in cases if not case.labeled]
    if bare and unlabeled == "refuse":
        raise MEDH5ValidationError(
            f"{bare} carry no annotation {annotation!r}: an nnU-Net training case "
            "is an image with its label volume, so these cannot be written as "
            "ones; pass unlabeled='test' (--unlabeled test) to write them to "
            "imagesTs, outside numTraining"
        )
    if bare:
        log.decision(
            "unlabeled",
            f"{len(bare)} case(s) without annotation {annotation!r} were written "
            "to imagesTs, and are not counted in numTraining",
            {"cases": bare},
        )

    wanted = set(names)
    if classes is not None:
        wanted = {_class_id(c, names) for c in classes}
        dropped = sorted(set(names) - wanted)
        if dropped:
            log.decision(
                "classes",
                f"classes= exports {sorted(wanted)}; {[names[c] for c in dropped]} "
                "were left out on request",
                {"exported": sorted(wanted), "left_out": dropped},
            )
    for path, looked in searched:
        unsearched = sorted(wanted - looked)
        if unsearched:
            raise MEDH5ValidationError(
                f"{path}: annotation {annotation!r} was not searched for "
                f"{[names[c] for c in unsearched]}, which the dataset labels. An "
                "nnU-Net label volume is exhaustive --- 0 is a verified negative "
                "for every class --- so this case cannot be written; pass "
                "classes= (--class) naming the classes every case examined"
            )
    if searched:
        log.decision(
            "coverage",
            "every case was searched for every exported class, so 0 in each label "
            "volume is the verified absence it means to nnU-Net (§11.3)",
            {"classes": sorted(wanted), "cases": len(searched)},
        )

    reused = bool(stashed and stashed.get("labels")) and classes is None
    by_label: dict[str, int] = {}
    if reused:
        assert stashed is not None
        labels = dict(stashed["labels"])
        order = stashed.get("regions_class_order")
        order = [int(v) for v in order] if isinstance(order, list) else None
        by_label = _stashed_classes(labels, stashed, names)
        _extend_stash(labels, by_label, order, names, wanted, log)
    else:
        labels = {"background": 0}
        labels.update({names[c]: c for c in sorted(wanted)})
        by_label = {names[c]: c for c in sorted(wanted)}
        order = None
        if stashed and classes is not None:
            log.decision(
                "labels",
                "classes= restricts the labels, so the stashed dataset.json's were "
                "not reused; its other fields were",
                {"classes": sorted(wanted)},
            )
    if order is None:
        labels = _consecutive(labels, log)
    # Each written value whose class is another id, which the import reads
    # back: what a renumbered export, or a reused one, needs recorded.
    class_ids = {
        int(labels[name]): class_id
        for name, class_id in by_label.items()
        if not isinstance(labels.get(name, []), list) and int(labels[name]) != class_id
    }
    return _Plan(
        cases,
        channels,
        labels,
        order,
        stashed,
        reused,
        by_label,
        class_ids,
        frozenset(wanted),
    )


def _stashed_classes(
    labels: Mapping[str, Any], stashed: Mapping[str, Any], names: Mapping[int, str]
) -> dict[str, int]:
    """The class each scalar label of a reused ``dataset.json`` writes.

    An export that renumbered recorded the ids (``medh5_class_ids``); an
    imported dataset's values are its class ids; a label naming neither is
    matched by its key.  A label matched by nothing is left to the case,
    which refuses it (E402).
    """
    recorded = {
        int(v): int(c) for v, c in (stashed.get("medh5_class_ids") or {}).items()
    }
    keys = {key: class_id for class_id, key in names.items()}
    out: dict[str, int] = {}
    for name, value in labels.items():
        if name == IGNORE or isinstance(value, list) or int(value) == BACKGROUND:
            continue
        if int(value) in recorded:
            out[name] = recorded[int(value)]
        elif int(value) in names:
            out[name] = int(value)
        elif _key(name) in keys:
            out[name] = keys[_key(name)]
    return out


def _extend_stash(
    labels: dict[str, Any],
    by_label: dict[str, int],
    order: list[int] | None,
    names: Mapping[int, str],
    wanted: set[int],
    log: ConversionReport,
) -> None:
    """Name in a reused ``dataset.json`` every class some case carries.

    A class the stash did not name was written as background: a dataset
    imported with one class and given another later exported the second as 0
    in every case, and ``dataset.json`` without it (F04 of the round-4 audit).
    A scalar stash is extended; a region-based one cannot be without changing
    what ``regions_class_order`` paints, so it is refused.
    """
    keys = {_key(str(name)) for name in labels}
    named = set(by_label.values()) | {c for c in wanted if names[c] in keys}
    missing = sorted(wanted - named)
    if not missing:
        return
    if order is not None or any(isinstance(v, list) for v in labels.values()):
        raise MEDH5ValidationError(
            f"the stashed dataset.json does not name {[names[c] for c in missing]}, "
            "which the cases carry, and it is region-based: a class cannot be "
            "added to its regions without changing what regions_class_order "
            "paints. Pass classes= (--class) to export a new dataset.json"
        )
    used: set[int] = set()
    for value in labels.values():
        used.update(int(v) for v in (value if isinstance(value, list) else [value]))
    for class_id in missing:
        if class_id in used:
            raise MEDH5ValidationError(
                f"the stashed dataset.json writes {class_id} for another label, so "
                f"class {names[class_id]!r} cannot take it; pass classes= "
                "(--class) to export a new dataset.json"
            )
        labels[names[class_id]] = class_id
        by_label[names[class_id]] = class_id
    log.decision(
        "labels",
        f"the stashed dataset.json did not name {[names[c] for c in missing]}, "
        "which the cases carry; they were added to its labels, so no class is "
        "written as background",
        {"added": {names[c]: c for c in missing}},
    )


def _consecutive(labels: dict[str, Any], log: ConversionReport) -> dict[str, Any]:
    """*labels* with values ``0..K``, as nnU-Net requires.

    nnU-Net refuses label values with a gap, and class ids have them --- a
    subset of a dataset's classes, or ids an import kept: ``{1, 3}`` was
    written as given, and nnU-Net's dataset check failed (F23 of the round-4
    audit).  The values are renumbered ``1..K`` in ascending order; labels
    already consecutive are left as they are.
    """
    scalar = sorted(
        int(v)
        for name, v in labels.items()
        if name != IGNORE and not isinstance(v, list) and int(v) != BACKGROUND
    )
    if scalar == list(range(1, len(scalar) + 1)):
        return labels
    value = {old: new for new, old in enumerate(scalar, start=1)}
    out: dict[str, Any] = {}
    for name, v in labels.items():
        if name == IGNORE:
            continue
        if isinstance(v, list):
            out[name] = [value.get(int(x), int(x)) for x in v]
        else:
            out[name] = value.get(int(v), int(v))
    log.decision(
        "label_values",
        f"nnU-Net needs consecutive label values; the values {scalar} were written "
        f"as 1..{len(scalar)}, and dataset.json records each one's class id "
        "(medh5_class_ids), which the import reads back",
        {"values": {str(new): old for old, new in value.items()}},
    )
    return out


def _class_id(key: int | str, names: Mapping[int, str]) -> int:
    """A class named by id or key, among the dataset's."""
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        if int(key) in names:
            return int(key)
    else:
        for class_id, name in names.items():
            if name == key:
                return class_id
    raise MEDH5ValidationError(
        f"classes= names {key!r}, which no case's annotation carries; the "
        f"dataset's are {dict(sorted(names.items()))}",
        code="E402",
    )


def _require_one_space(reference: Any, grid: Any, what: str, channel: str) -> None:
    """Refuse a grid that is not *reference*'s physical support (N17 of the
    round-3 audit).

    nnU-Net pairs channel and label voxels by index, and each NIfTI states its
    own affine: a congruent lattice in LPS against one in RAS, or in metres
    against millimetres, wrote a pair 48 mm --- or a thousandfold --- apart,
    and equal numbers in two known frames of reference are no correspondence.
    Converting between them is resampling, which is not a converter's to do.
    """
    if grid.grid_id == reference.grid_id:
        return
    problems = []
    if not grid.is_congruent(reference):
        problems.append("another lattice")
    if grid.coord_system != reference.coord_system:
        problems.append(f"{grid.coord_system} against {reference.coord_system}")
    if grid.units != reference.units:
        problems.append(f"{grid.units!r} against {reference.units!r}")
    if grid.frame_uid and reference.frame_uid and grid.frame_uid != reference.frame_uid:
        problems.append(
            f"frame of reference {grid.frame_uid!r} against {reference.frame_uid!r}"
        )
    if problems:
        raise MEDH5ValidationError(
            f"{what} is on grid {grid.grid_id!r} and channel {channel!r} on "
            f"{reference.grid_id!r} ({'; '.join(problems)}); nnU-Net needs every "
            "channel and the labels on the channels' grid, and resampling is not "
            "a converter's to do"
        )


def _ignore_value(labels: Mapping[str, Any]) -> int:
    """The value nnU-Net's ignore label takes: its own, or one above every
    other, as nnU-Net requires."""
    if IGNORE in labels and not isinstance(labels[IGNORE], list):
        return int(labels[IGNORE])
    values = [BACKGROUND]
    for value in labels.values():
        values.extend(int(v) for v in (value if isinstance(value, list) else [value]))
    if max(values) + 1 > np.iinfo(np.uint16).max:
        raise MEDH5ValidationError(
            f"labels use {max(values)}, so nnU-Net's ignore label, which must be "
            "the highest, does not fit the uint16 label volume"
        )
    return max(values) + 1


def _stashed_channels(stashed: Mapping[str, Any] | None) -> dict[int, str]:
    if not stashed:
        return {}
    return {int(k): str(v) for k, v in stashed.get("channel_names", {}).items()}


def _labelmap_for(
    ann: Any,
    labels: Mapping[str, Any],
    order: Sequence[int] | None = None,
    by_label: Mapping[str, int] | None = None,
) -> npt.NDArray[np.uint16]:
    """A single-value label volume, which is what nnU-Net reads.

    Without ``regions_class_order`` a region is *not* written as its own value:
    nnU-Net derives it from its components, and writing both would
    double-count every voxel.  With it (*order*), the dataset is region-based:
    each label, in ``labels`` order, is painted with its value in *order*, a
    later one over an earlier --- nnU-Net's own conversion back --- so the
    components a region-based dataset names only inside its regions are written
    too.  They were left as background (N13 of the round-3 audit).

    Classes are matched by **id, not by name**.  The import keeps nnU-Net's own
    integers as class ids precisely so no translation table is needed, and the
    name in ``dataset.json`` is free text that ``_key`` sanitises on the way in
    --- so a dataset naming a class ``"Tumour Core"`` stores the key
    ``tumour_core``, and looking the original name back up finds nothing.  That
    lookup used to fail into a bare ``continue``, so every class of any dataset
    whose labels are not already lowercase identifiers was dropped and the
    export wrote an all-background volume with no indication anything was lost.
    A label whose class the plan knows (*by_label*) writes that class, whatever
    its value: renumbered values are no ids (F23).
    """
    known = set(ann.class_ids)
    out = np.zeros(ann.spatial_shape, dtype=np.uint16)
    missing: list[str] = []
    if order is not None:
        entries = _foreground(labels)
        if len(order) != len(entries):
            raise MEDH5ValidationError(
                f"regions_class_order has {len(order)} value(s) for {len(entries)} "
                "label(s); it gives each label, in order, the value it is written as"
            )
        for (name, value), paint in zip(entries, order, strict=True):
            class_id = _class_of(ann, name, value, known)
            if class_id is None:
                missing.append(name)
                continue
            out[ann.dense([class_id])[0]] = int(paint)
    else:
        scalar = {
            name: int(value)
            for name, value in labels.items()
            if not isinstance(value, list) and name != IGNORE
        }
        for name, value in sorted(scalar.items(), key=lambda kv: kv[1]):
            if value == BACKGROUND:
                continue
            class_id = (by_label or {}).get(name)
            if class_id is None:
                class_id = value if value in known else _resolve_or_none(ann, name)
            if class_id is None:
                missing.append(name)
                continue
            out[ann.dense([class_id])[0]] = value
    if missing:
        raise MEDH5ValidationError(
            f"annotation {ann.ann_id!r} carries no class for {missing}, which "
            f"dataset.json names; exporting would write a label volume missing "
            f"those structures without saying so",
            code="E402",
        )
    return out


def _foreground(labels: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """The labels a volume value or region states, in ``labels`` order."""
    return [
        (name, value)
        for name, value in labels.items()
        if name != IGNORE and (isinstance(value, list) or int(value) != BACKGROUND)
    ]


def _class_of(ann: Any, name: str, value: Any, known: set[int]) -> int | None:
    """The annotation's class a label names: a value by id, a region by name."""
    if not isinstance(value, list) and int(value) in known:
        return int(value)
    return _resolve_or_none(ann, name)


def _not_given_back(
    ann: Any,
    volume: npt.NDArray[Any],
    labels: Mapping[str, Any],
    ignored: npt.NDArray[np.bool_],
    by_label: Mapping[str, int] | None = None,
    exported: frozenset[int] | None = None,
) -> list[str]:
    """The labels whose stored voxels *volume* does not give back, outside the
    ignore region --- what an export would lose without saying so.

    Each label is read back as nnU-Net reads it: a value as the voxels of that
    value, a region as the voxels of any of its components.  And every
    *exported* class the annotation stores voxels of must be some label's: a
    class the vocabulary did not name was written as background, and checking
    the vocabulary alone could not see it (F04 of the round-4 audit).
    """
    known = set(ann.class_ids)
    out: list[str] = []
    read: set[int] = set()
    for name, value in _foreground(labels):
        class_id = None if isinstance(value, list) else (by_label or {}).get(name)
        if class_id is None:
            class_id = _class_of(ann, name, value, known)
        if class_id is None:
            continue
        read.add(class_id)
        components = (
            [int(v) for v in value] if isinstance(value, list) else [int(value)]
        )
        stored = np.asarray(ann.dense([class_id])[0], dtype=bool) & ~ignored
        written = np.isin(volume, components) & ~ignored
        if not np.array_equal(stored, written):
            lost = int(np.count_nonzero(stored & ~written))
            gained = int(np.count_nonzero(written & ~stored))
            out.append(f"{name!r} ({lost} voxel(s) lost, {gained} gained)")
    for class_id in sorted((exported or frozenset()) & known - read):
        stored = np.asarray(ann.dense([class_id])[0], dtype=bool) & ~ignored
        if stored.any():
            out.append(
                f"class {class_id} ({int(stored.sum())} voxel(s) no label names)"
            )
    return out


def _resolve_or_none(ann: Any, name: str) -> int | None:
    """Fall back to name resolution for a label set that renumbered."""
    for candidate in (name, _key(name)):
        try:
            return int(ann.resolve_class(candidate))
        except Exception:
            continue
    return None


def _save(
    nib: Any,
    grid: Any,
    array: npt.NDArray[Any],
    path: Path,
    *,
    rescale: tuple[float, float] | None = None,
) -> None:
    """Write one channel or label volume, through the one NIfTI writer.

    *rescale* is the image's §4.2 scale.  ``Image.read()`` returns **stored**
    values, so writing them with no ``scl_slope``/``scl_inter`` handed nnU-Net
    a CT 1024 HU off its own units, in the exporter's default path, with
    nothing in ``dataset.json`` or the report to say so.
    """
    del nib  # every write goes through `write_nifti`, which requires nibabel
    from medh5.io.nifti import for_export, write_nifti

    data, affine = for_export(grid, np.asarray(array))
    write_nifti(
        data,
        affine,
        path,
        rescale=rescale,
        units=grid.units,
        time_values=grid.time_values,
        time_units=grid.time_units,
    )


__all__ = [
    "BACKGROUND",
    "REQUIRED_KEYS",
    "cases",
    "from_nnunetv2",
    "read_dataset_json",
    "to_nnunetv2",
]
