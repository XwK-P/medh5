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
            masks = _masks_from(ids, label_set, regions)
            if ignore_value is not None and np.any(ids == ignore_value):
                ignored = np.asarray(ids == ignore_value)

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
    for name, value in _labels(document).items():
        if name == IGNORE and not isinstance(value, list):
            ignore_value = int(value)
        elif isinstance(value, list):
            region_values[name] = [int(v) for v in value]
        elif int(value) == BACKGROUND:
            dropped.append(name)
        else:
            scalars[name] = int(value)

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
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Export samples as an nnU-Net v2 dataset.

    When the samples were imported from nnU-Net the stashed ``dataset.json`` is
    reused verbatim, so the export reproduces the original dataset rather than a
    reconstruction of it.
    """
    import medh5
    from medh5.io.nifti import require_nibabel

    nib = require_nibabel()
    log = report or ConversionReport(converter="to-nnunet")
    root = Path(os.fspath(out)) / dataset_name
    (root / "imagesTr").mkdir(parents=True, exist_ok=True)
    (root / "labelsTr").mkdir(parents=True, exist_ok=True)

    stashed: dict[str, Any] | None = None
    channel_order: list[str] = []
    labels: dict[str, Any] = {}
    ignore_value: int | None = None
    for path in paths:
        with medh5.open(path) as sample:
            case = sample.identity.sample_id
            stashed = stashed or sample.document.extra.get("nnunetv2")
            if not channel_order:
                channel_order = (
                    [str(v) for _, v in sorted(_stashed_channels(stashed).items())]
                    if stashed
                    else sorted(sample.images)
                )
            for index, image_id in enumerate(channel_order):
                if image_id not in sample.images:
                    raise MEDH5ValidationError(
                        f"{path}: no image {image_id!r}; the export needs the same "
                        f"channels in every case ({channel_order})"
                    )
                _save(
                    nib,
                    sample.images[image_id].grid,
                    sample.images[image_id].read(),
                    root / "imagesTr" / f"{case}_{index:04d}{file_ending}",
                    rescale=sample.images[image_id].rescale,
                )
                log.outputs.append(
                    str(root / "imagesTr" / f"{case}_{index:04d}{file_ending}")
                )
            if annotation in sample.annotations:
                ann = sample.annotations[annotation]
                labels = labels or _labels_for(ann, stashed)
                # On the annotation's own grid, which nnU-Net needs to be the
                # channels': the reference grid's affine was written whatever
                # grid the labels were on.
                channels = sample.images[channel_order[0]].grid
                if not ann.grid.is_congruent(channels):
                    raise MEDH5ValidationError(
                        f"{path}: annotation {annotation!r} is on grid "
                        f"{ann.grid.grid_id!r} and channel {channel_order[0]!r} on "
                        f"{channels.grid_id!r}; nnU-Net needs labels on the "
                        "channels' grid, and resampling is not a converter's to do"
                    )
                volume = _labelmap_for(ann, labels)
                ignored = np.asarray(sample.ignore_region(annotation), dtype=bool)
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
                        f"{path}: {int(ignored.sum())} ignored voxel(s) were written "
                        f"as nnU-Net's ignore label {ignore_value}"
                        + (
                            f", {covered} of them over a class, which nnU-Net "
                            "neither trains nor scores there"
                            if covered
                            else ""
                        ),
                        {"case": case, "voxels": int(ignored.sum()), "over": covered},
                    )
                _save(nib, ann.grid, volume, root / "labelsTr" / f"{case}{file_ending}")
                # Listed only when written.  A sample without the annotation
                # produced no label file and the report named one anyway, so a
                # caller checking `outputs` for what to ship was told about a
                # path that did not exist.
                log.outputs.append(str(root / "labelsTr" / f"{case}{file_ending}"))

    document = dict(stashed) if stashed else {}
    document.update(
        {
            "channel_names": {str(i): n for i, n in enumerate(channel_order)},
            "labels": labels or document.get("labels") or {},
            "numTraining": len(list(paths)),
            "file_ending": file_ending,
        }
    )
    (root / "dataset.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    log.outputs.append(str(root / "dataset.json"))
    log.decision(
        "dataset_json",
        "the stashed dataset.json was reused verbatim"
        if stashed
        else "a dataset.json was generated from the label set",
        {"reused": bool(stashed)},
    )
    return log


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


def _labels_for(ann: Any, stashed: Mapping[str, Any] | None) -> dict[str, Any]:
    if stashed and stashed.get("labels"):
        return dict(stashed["labels"])
    out: dict[str, Any] = {"background": 0}
    for class_id in ann.class_ids:
        out[ann.class_key(int(class_id))] = int(class_id)
    return out


def _labelmap_for(ann: Any, labels: Mapping[str, Any]) -> npt.NDArray[np.uint16]:
    """A single-value label volume, which is what nnU-Net reads.

    Region labels are *not* written as their own value: nnU-Net derives them
    from their components, and writing both would double-count every voxel.

    Classes are matched by **id, not by name**.  The import keeps nnU-Net's own
    integers as class ids precisely so no translation table is needed, and the
    name in ``dataset.json`` is free text that ``_key`` sanitises on the way in
    --- so a dataset naming a class ``"Tumour Core"`` stores the key
    ``tumour_core``, and looking the original name back up finds nothing.  That
    lookup used to fail into a bare ``continue``, so every class of any dataset
    whose labels are not already lowercase identifiers was dropped and the
    export wrote an all-background volume with no indication anything was lost.
    """
    scalar = {
        name: int(value)
        for name, value in labels.items()
        if not isinstance(value, list) and name != IGNORE
    }
    known = set(ann.class_ids)
    out = np.zeros(ann.spatial_shape, dtype=np.uint16)
    missing: list[str] = []
    for name, value in sorted(scalar.items(), key=lambda kv: kv[1]):
        if value == BACKGROUND:
            continue
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
