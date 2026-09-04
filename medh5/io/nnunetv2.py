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


def read_dataset_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and check an nnU-Net v2 ``dataset.json``."""
    target = Path(os.fspath(path))
    if target.is_dir():
        target = target / "dataset.json"
    if not target.exists():
        raise MEDH5ValidationError(f"dataset.json not found at {target}")
    raw = json.loads(target.read_text())
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
    label_set, regions = _label_set(document, log)
    ending = str(document["file_ending"])
    wanted = list(case_ids) if case_ids is not None else cases(source, document)
    directory = Path(os.fspath(out))
    directory.mkdir(parents=True, exist_ok=True)

    for case in wanted:
        images: dict[str, npt.NDArray[Any]] = {}
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
        assert geometry is not None
        label_path = source / "labelsTr" / f"{case}{ending}"
        masks: dict[int, npt.NDArray[np.bool_]] | None = None
        if label_path.exists():
            volume, label_geo = read_nifti(label_path, coord_system=coord_system)
            # The label volume above all: a label resampled onto a different grid
            # by some other tool is the ordinary way this goes wrong, and the
            # result annotates voxels nobody drew on.
            _same_grid(geometry, label_geo, f"{case} labels", log)
            masks = _masks_from(volume, label_set, regions)

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
                writer.add_image(
                    name, array, grid="ref", modality=_modality(name), prov=activity
                )
            if masks:
                kind, _ = writer.add_segmentation(
                    "seg",
                    grid="ref",
                    masks=masks,
                    annotated_classes="all",
                    prov=activity,
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
) -> tuple[Any, dict[int, list[int]]]:
    """nnU-Net ``labels`` as a MEDH5 label set, keeping its integer ids."""
    from medh5.labels.labelset import LabelClass, LabelSet

    scalars: dict[str, int] = {}
    region_values: dict[str, list[int]] = {}
    dropped: list[str] = []
    for name, value in _labels(document).items():
        if isinstance(value, list):
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
    return LabelSet("nnunetv2", version="1.0.0", classes=classes), regions


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
                    sample,
                    sample.images[image_id].read(),
                    root / "imagesTr" / f"{case}_{index:04d}{file_ending}",
                    image_id,
                )
            if annotation in sample.annotations:
                ann = sample.annotations[annotation]
                labels = labels or _labels_for(ann, stashed)
                _save(
                    nib,
                    sample,
                    _labelmap_for(ann, labels),
                    root / "labelsTr" / f"{case}{file_ending}",
                    None,
                )
        log.outputs.append(str(root / "labelsTr" / f"{case}{file_ending}"))

    document = dict(stashed) if stashed else {}
    document.update(
        {
            "channel_names": {str(i): n for i, n in enumerate(channel_order)},
            "labels": document.get("labels") or labels,
            "numTraining": len(list(paths)),
            "file_ending": file_ending,
        }
    )
    (root / "dataset.json").write_text(json.dumps(document, indent=2) + "\n")
    log.outputs.append(str(root / "dataset.json"))
    log.decision(
        "dataset_json",
        "the stashed dataset.json was reused verbatim"
        if stashed
        else "a dataset.json was generated from the label set",
        {"reused": bool(stashed)},
    )
    return log


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
        if not isinstance(value, list)
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
    nib: Any, sample: Any, array: npt.NDArray[Any], path: Path, image: str | None
) -> None:
    from medh5.geometry.affine import build_affine, decompose_affine
    from medh5.io.nifti import convert_world

    grid = sample.images[image].grid if image else sample.reference_grid
    affine = convert_world(grid.affine, source=grid.coord_system, target="RAS")
    spacing, origin, direction = decompose_affine(affine)
    data = np.asarray(array)
    if data.ndim >= 3:
        flip = tuple(reversed(range(data.ndim - 3, data.ndim)))
        data = np.transpose(data, tuple(range(data.ndim - 3)) + flip)
        index = [f - (data.ndim - 3) for f in flip]
        spacing = spacing[index]
        direction = direction[:, index]
    nib.save(
        nib.Nifti1Image(
            np.ascontiguousarray(data), build_affine(spacing, origin, direction)
        ),
        str(path),
    )


__all__ = [
    "BACKGROUND",
    "REQUIRED_KEYS",
    "cases",
    "from_nnunetv2",
    "read_dataset_json",
    "to_nnunetv2",
]
