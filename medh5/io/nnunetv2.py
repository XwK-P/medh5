"""nnU-Net v2 ⇄ medh5 dataset converters.

Converts a raw nnU-Net v2 dataset folder (``imagesTr/``/``labelsTr/``/
``imagesTs/`` + ``dataset.json``) into a directory of per-case ``.medh5``
files and back. Each case becomes exactly one ``.medh5`` file that bundles
every channel plus one boolean segmentation mask per foreground class from
``dataset.json``; the original ``dataset.json`` payload is stashed in
``extra["nnunetv2"]`` so export can reconstruct the exact source layout.

Requires ``nibabel`` (install with ``pip install medh5[nifti]``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5ValidationError
from medh5.io.nifti import (
    _compose_affine,
    _decompose_affine,
    _load_nifti,
    _require_nibabel,
)

try:
    import nibabel as nib

    _NIBABEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NIBABEL_AVAILABLE = False


_REQUIRED_DATASET_KEYS = ("channel_names", "labels", "numTraining", "file_ending")


# ---------------------------------------------------------------------------
# dataset.json parsing
# ---------------------------------------------------------------------------


def _parse_dataset_json(path: Path) -> dict[str, Any]:
    """Load and validate an nnU-Net v2 ``dataset.json`` file."""
    if not path.is_file():
        raise MEDH5ValidationError(f"dataset.json not found at {path}")
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MEDH5ValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise MEDH5ValidationError(
            f"dataset.json root must be an object, got {type(raw).__name__}"
        )
    missing = [k for k in _REQUIRED_DATASET_KEYS if k not in raw]
    if missing:
        raise MEDH5ValidationError(f"dataset.json is missing required keys: {missing}")

    channel_names_raw = raw["channel_names"]
    if not isinstance(channel_names_raw, dict) or not channel_names_raw:
        raise MEDH5ValidationError(
            "'channel_names' must be a non-empty object mapping channel "
            "index to modality name"
        )
    channel_names: dict[int, str] = {}
    for key, value in channel_names_raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as exc:
            raise MEDH5ValidationError(
                f"channel_names keys must be integer strings, got '{key}'"
            ) from exc
        if not isinstance(value, str) or not value:
            raise MEDH5ValidationError(
                f"channel_names['{key}'] must be a non-empty string"
            )
        channel_names[idx] = value
    expected_indices = set(range(len(channel_names)))
    if set(channel_names.keys()) != expected_indices:
        last = len(channel_names) - 1
        raise MEDH5ValidationError(
            f"channel_names must cover consecutive indices 0..{last}, "
            f"got {sorted(channel_names.keys())}"
        )

    labels_raw = raw["labels"]
    if not isinstance(labels_raw, dict) or not labels_raw:
        raise MEDH5ValidationError(
            "'labels' must be a non-empty object mapping class name to integer"
        )
    labels: dict[str, int] = {}
    for name, value in labels_raw.items():
        if isinstance(value, list):
            raise MEDH5ValidationError(
                f"Region-based labels are not supported (label '{name}' is a list). "
                "Convert to integer labels first."
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise MEDH5ValidationError(
                f"labels['{name}'] must be an integer, got {type(value).__name__}"
            )
        if not isinstance(name, str) or not name:
            raise MEDH5ValidationError("labels keys must be non-empty strings")
        labels[name] = value
    if "background" not in labels or labels["background"] != 0:
        raise MEDH5ValidationError("'labels' must contain 'background' mapped to 0")
    seen_values = sorted(labels.values())
    if seen_values != list(range(len(seen_values))):
        raise MEDH5ValidationError(
            "label integer values must be consecutive starting from 0, "
            f"got {seen_values}"
        )

    num_training = raw["numTraining"]
    if not isinstance(num_training, int) or isinstance(num_training, bool):
        raise MEDH5ValidationError(
            f"'numTraining' must be an integer, got {type(num_training).__name__}"
        )
    file_ending = raw["file_ending"]
    if not isinstance(file_ending, str) or not file_ending.startswith("."):
        raise MEDH5ValidationError(
            f"'file_ending' must be a string starting with '.', got {file_ending!r}"
        )

    parsed: dict[str, Any] = {
        "channel_names": channel_names,
        "labels": labels,
        "numTraining": num_training,
        "file_ending": file_ending,
    }
    if "overwrite_image_reader_writer" in raw:
        parsed["overwrite_image_reader_writer"] = raw["overwrite_image_reader_writer"]
    if "regions_class_order" in raw:
        parsed["regions_class_order"] = raw["regions_class_order"]
    if "name" in raw:
        parsed["name"] = raw["name"]
    if "description" in raw:
        parsed["description"] = raw["description"]
    return parsed


_NNUNETV2_SCHEMA_VERSION = 1


def _nnunet_meta_for_extra(parsed: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON-friendly copy suitable for storage in ``extra``.

    ``channel_names`` is serialized with string keys (matching dataset.json)
    so it survives HDF5 attribute round-tripping cleanly.
    """
    meta: dict[str, Any] = dict(parsed)
    meta["channel_names"] = {
        str(idx): name for idx, name in parsed["channel_names"].items()
    }
    meta["labels"] = dict(parsed["labels"])
    meta["schema_version"] = _NNUNETV2_SCHEMA_VERSION
    return meta


# ---------------------------------------------------------------------------
# Case discovery
# ---------------------------------------------------------------------------


def _discover_cases(
    split_dir: Path,
    file_ending: str,
    n_channels: int,
) -> dict[str, list[Path]]:
    """Group ``{case_id}_{XXXX}{ending}`` files by case id.

    Returns a dict mapping case id → list of channel paths indexed by
    channel number (list[0] is channel 0000, list[1] is channel 0001, …).
    Raises if a case is missing any channel or has extra files.
    """
    if not split_dir.is_dir():
        return {}
    pattern = re.compile(
        r"^(?P<case>.+)_(?P<channel>\d{4})" + re.escape(file_ending) + r"$"
    )
    # Collect (case_id, channel_idx, path) tuples.
    by_case: dict[str, dict[int, Path]] = {}
    for entry in sorted(split_dir.iterdir()):
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if not match:
            continue
        case_id = match.group("case")
        channel = int(match.group("channel"))
        slot = by_case.setdefault(case_id, {})
        if channel in slot:
            raise MEDH5ValidationError(
                f"Duplicate channel {channel:04d} for case '{case_id}' in {split_dir}"
            )
        slot[channel] = entry

    ordered: dict[str, list[Path]] = {}
    for case_id in sorted(by_case.keys()):
        channels = by_case[case_id]
        expected = set(range(n_channels))
        if set(channels.keys()) != expected:
            missing = sorted(expected - set(channels.keys()))
            extra = sorted(set(channels.keys()) - expected)
            raise MEDH5ValidationError(
                f"Case '{case_id}' in {split_dir}: expected channels "
                f"{sorted(expected)}, missing={missing}, extra={extra}"
            )
        ordered[case_id] = [channels[i] for i in range(n_channels)]
    return ordered


def _find_label_file(labels_dir: Path, case_id: str, file_ending: str) -> Path:
    candidate = labels_dir / f"{case_id}{file_ending}"
    if not candidate.is_file():
        raise MEDH5ValidationError(
            f"Label file not found for case '{case_id}': expected {candidate}"
        )
    return candidate


# ---------------------------------------------------------------------------
# Per-case conversion
# ---------------------------------------------------------------------------


def _split_label_volume(
    label_arr: np.ndarray,
    labels: dict[str, int],
    *,
    case_id: str,
) -> dict[str, np.ndarray]:
    """Split an integer label volume into per-class boolean masks.

    Drops background (value 0) and returns one mask per foreground class.
    Raises ``MEDH5ValidationError`` if the volume contains non-integer
    voxel values or any value not declared in ``labels`` — otherwise
    the round-trip would silently drop those voxels.
    """
    raw = np.asarray(label_arr)
    # Reject non-integer voxels: a float label like 1.9 would pass the
    # undeclared-value check (int(1.9) == 1) yet fail the equality mask
    # (1.9 != 1), silently dropping those voxels.
    if not np.issubdtype(raw.dtype, np.integer):
        rounded = np.round(raw)
        if not np.array_equal(raw, rounded):
            raise MEDH5ValidationError(
                f"Case '{case_id}': label volume has dtype {raw.dtype} and "
                f"contains non-integer values; expected an integer mask"
            )
        raw = rounded.astype(np.intp)
    declared = set(labels.values())
    actual = {int(v) for v in np.unique(raw)}
    unexpected = sorted(actual - declared)
    if unexpected:
        raise MEDH5ValidationError(
            f"Case '{case_id}': label volume contains undeclared values "
            f"{unexpected}; dataset.json declares {sorted(declared)}"
        )
    seg: dict[str, np.ndarray] = {}
    for class_name, class_value in labels.items():
        if class_value == 0:
            continue
        seg[class_name] = raw == class_value
    return seg


def _convert_case(
    case_id: str,
    channel_paths: list[Path],
    label_path: Path | None,
    *,
    channel_names: dict[int, str],
    labels: dict[str, int],
    out_path: Path,
    nnunet_meta: dict[str, Any],
    compression: str | None,
    checksum: bool,
) -> None:
    """Read NIfTI files for one case and write a ``.medh5`` file."""
    image_arrays: dict[str, np.ndarray] = {}
    ref_shape: tuple[int, ...] | None = None
    ref_affine: np.ndarray | None = None
    ref_name: str | None = None
    for idx, path in enumerate(channel_paths):
        name = channel_names[idx]
        data, affine = _load_nifti(path)
        if ref_shape is None:
            ref_shape = tuple(data.shape)
            ref_affine = affine
            ref_name = name
        else:
            if tuple(data.shape) != ref_shape:
                raise MEDH5ValidationError(
                    f"Case '{case_id}': channel '{name}' shape {tuple(data.shape)} "
                    f"does not match '{ref_name}' shape {ref_shape}"
                )
            if ref_affine is not None and not np.allclose(
                affine, ref_affine, atol=1e-5
            ):
                raise MEDH5ValidationError(
                    f"Case '{case_id}': channel '{name}' affine does not match "
                    f"'{ref_name}'"
                )
        image_arrays[name] = np.asarray(data)

    assert ref_shape is not None
    assert ref_affine is not None

    seg_arrays: dict[str, np.ndarray] | None = None
    if label_path is not None:
        label_data, label_affine = _load_nifti(label_path)
        if tuple(label_data.shape) != ref_shape:
            raise MEDH5ValidationError(
                f"Case '{case_id}': label shape {tuple(label_data.shape)} does not "
                f"match image shape {ref_shape}"
            )
        if not np.allclose(label_affine, ref_affine, atol=1e-5):
            raise MEDH5ValidationError(
                f"Case '{case_id}': label affine does not match images"
            )
        seg_arrays = _split_label_volume(label_data, labels, case_id=case_id)

    spacing, origin, direction = _decompose_affine(ref_affine)
    ndim = len(ref_shape)
    spacing = spacing[:ndim]
    origin = origin[:ndim]
    direction = [row[:ndim] for row in direction[:ndim]]

    MEDH5File.write(
        out_path,
        images=image_arrays,
        seg=seg_arrays,
        spacing=spacing,
        origin=origin,
        direction=direction,
        coord_system="RAS",
        extra={"nnunetv2": nnunet_meta},
        compression=compression,
        checksum=checksum,
    )


# ---------------------------------------------------------------------------
# Public API — nnU-Net v2 → medh5
# ---------------------------------------------------------------------------


def from_nnunetv2(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    include_test: bool = True,
    compression: str | None = "balanced",
    checksum: bool = False,
) -> dict[str, list[Path]]:
    """Convert a raw nnU-Net v2 dataset folder into per-case ``.medh5`` files.

    Parameters
    ----------
    src_dir : str or Path
        Path to a ``DatasetXXX_NAME/`` folder containing ``dataset.json``,
        ``imagesTr/``, ``labelsTr/``, and optionally ``imagesTs/``.
    out_dir : str or Path
        Destination directory. Will be created if missing. Training cases
        are written to ``<out_dir>/imagesTr/{case}.medh5``; test cases (if
        present and ``include_test=True``) to ``<out_dir>/imagesTs/{case}.medh5``.
    include_test : bool
        If True (default), also convert ``imagesTs/`` cases (with ``seg=None``
        since nnU-Net v2 has no ``labelsTs/``).
    compression : str
        Compression preset forwarded to :meth:`MEDH5File.write`.
    checksum : bool
        If True, compute SHA-256 checksums for each written file.

    Returns
    -------
    dict[str, list[Path]]
        ``{"train": [...], "test": [...]}`` — the paths of the written files.

    Raises
    ------
    MEDH5ValidationError
        On a malformed ``dataset.json``, missing channels/labels, or grid
        mismatches between channels of a single case.
    """
    _require_nibabel()

    src = Path(src_dir)
    if not src.is_dir():
        raise MEDH5ValidationError(f"nnU-Net dataset directory not found: {src}")

    parsed = _parse_dataset_json(src / "dataset.json")
    channel_names: dict[int, str] = parsed["channel_names"]
    labels: dict[str, int] = parsed["labels"]
    file_ending: str = parsed["file_ending"]
    n_channels = len(channel_names)
    nnunet_meta = _nnunet_meta_for_extra(parsed)

    images_tr = src / "imagesTr"
    labels_tr = src / "labelsTr"
    if not images_tr.is_dir():
        raise MEDH5ValidationError(f"Missing required directory: {images_tr}")
    if not labels_tr.is_dir():
        raise MEDH5ValidationError(f"Missing required directory: {labels_tr}")

    out = Path(out_dir)
    train_out = out / "imagesTr"
    train_out.mkdir(parents=True, exist_ok=True)

    written: dict[str, list[Path]] = {"train": [], "test": []}

    train_cases = _discover_cases(images_tr, file_ending, n_channels)
    if not train_cases:
        raise MEDH5ValidationError(
            f"No training cases found in {images_tr} with file ending '{file_ending}'"
        )
    for case_id, channel_paths in train_cases.items():
        label_path = _find_label_file(labels_tr, case_id, file_ending)
        out_path = train_out / f"{case_id}.medh5"
        _convert_case(
            case_id,
            channel_paths,
            label_path,
            channel_names=channel_names,
            labels=labels,
            out_path=out_path,
            nnunet_meta=nnunet_meta,
            compression=compression,
            checksum=checksum,
        )
        written["train"].append(out_path)

    if include_test:
        images_ts = src / "imagesTs"
        if images_ts.is_dir():
            test_cases = _discover_cases(images_ts, file_ending, n_channels)
            if test_cases:
                test_out = out / "imagesTs"
                test_out.mkdir(parents=True, exist_ok=True)
                for case_id, channel_paths in test_cases.items():
                    out_path = test_out / f"{case_id}.medh5"
                    _convert_case(
                        case_id,
                        channel_paths,
                        None,
                        channel_names=channel_names,
                        labels=labels,
                        out_path=out_path,
                        nnunet_meta=nnunet_meta,
                        compression=compression,
                        checksum=checksum,
                    )
                    written["test"].append(out_path)

    return written


# ---------------------------------------------------------------------------
# Public API — medh5 → nnU-Net v2
# ---------------------------------------------------------------------------


def _collect_medh5_cases(src: Path) -> dict[str, list[Path]]:
    """Locate training/test ``.medh5`` files under ``src``.

    Recognizes two layouts:

    1. Mirrored: ``src/imagesTr/*.medh5`` (required) plus ``src/imagesTs/*.medh5``.
    2. Flat: ``src/*.medh5`` — all cases treated as training.
    """
    result: dict[str, list[Path]] = {"train": [], "test": []}
    images_tr = src / "imagesTr"
    if images_tr.is_dir():
        result["train"] = sorted(images_tr.glob("*.medh5"))
        images_ts = src / "imagesTs"
        if images_ts.is_dir():
            result["test"] = sorted(images_ts.glob("*.medh5"))
    else:
        result["train"] = sorted(src.glob("*.medh5"))
    return result


def _resolve_channel_order(
    nnunet_meta: dict[str, Any] | None,
    image_names: list[str],
) -> dict[int, str]:
    """Decide the channel index → modality name mapping for export.

    Prefers the original order stored in ``extra["nnunetv2"]["channel_names"]``.
    Falls back to alphabetical order (mirroring how ``MEDH5File.write`` stores
    image names) if no nnU-Net metadata is available.
    """
    if nnunet_meta is not None and "channel_names" in nnunet_meta:
        raw = nnunet_meta["channel_names"]
        parsed: dict[int, str] = {}
        for key, value in raw.items():
            parsed[int(key)] = str(value)
        stored_names = set(parsed.values())
        missing = stored_names - set(image_names)
        if missing:
            raise MEDH5ValidationError(
                f"Stored nnU-Net channel names {sorted(missing)} not present in "
                f"medh5 file (has: {image_names})"
            )
        return parsed
    return {i: name for i, name in enumerate(image_names)}


def _resolve_label_mapping(
    nnunet_meta: dict[str, Any] | None,
    seg_names: list[str] | None,
) -> dict[str, int]:
    """Decide the class name → integer mapping for export."""
    if nnunet_meta is not None and "labels" in nnunet_meta:
        return {str(k): int(v) for k, v in nnunet_meta["labels"].items()}
    mapping: dict[str, int] = {"background": 0}
    for i, name in enumerate(sorted(seg_names or []), start=1):
        mapping[name] = i
    return mapping


def _merge_seg_to_label(
    seg: dict[str, np.ndarray] | None,
    labels: dict[str, int],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Combine per-class boolean masks into one integer label volume."""
    max_value = max(labels.values()) if labels else 0
    dtype = np.uint8 if max_value <= 255 else np.uint16
    out: np.ndarray = np.zeros(shape, dtype=dtype)
    if seg is None:
        return out
    # Paint in declared nnU-Net order so later classes overwrite earlier ones,
    # matching how nnU-Net itself resolves overlaps.
    ordered = sorted(
        ((name, value) for name, value in labels.items() if value != 0),
        key=lambda item: item[1],
    )
    for name, value in ordered:
        if name in seg:
            out[np.asarray(seg[name], dtype=bool)] = value
    return out


def _write_case_nifti(
    case_id: str,
    sample: Any,
    channel_order: dict[int, str],
    labels: dict[str, int],
    images_out: Path,
    labels_out: Path | None,
    file_ending: str,
) -> None:
    expected_channels = set(channel_order.values())
    actual_channels = set(sample.images.keys())
    missing_channels = sorted(expected_channels - actual_channels)
    extra_channels = sorted(actual_channels - expected_channels)
    if missing_channels or extra_channels:
        raise MEDH5ValidationError(
            f"Case '{case_id}': image channels do not match the dataset "
            f"channel set {sorted(expected_channels)} — "
            f"missing={missing_channels}, extra={extra_channels}"
        )

    shape = tuple(next(iter(sample.images.values())).shape)
    spatial = sample.meta.spatial
    affine = _compose_affine(
        spatial.spacing, spatial.origin, spatial.direction, len(shape)
    )
    for idx in sorted(channel_order.keys()):
        name = channel_order[idx]
        arr = np.asarray(sample.images[name])
        out_path = images_out / f"{case_id}_{idx:04d}{file_ending}"
        nib.save(nib.Nifti1Image(arr, affine), str(out_path))

    if labels_out is not None:
        if sample.seg is not None:
            declared_seg_names = {name for name, value in labels.items() if value != 0}
            unexpected_seg = sorted(set(sample.seg.keys()) - declared_seg_names)
            if unexpected_seg:
                raise MEDH5ValidationError(
                    f"Case '{case_id}': seg masks {unexpected_seg} are not "
                    f"declared in the nnU-Net label map "
                    f"{sorted(declared_seg_names)}. Update "
                    f"extra['nnunetv2']['labels'] or drop the masks before export."
                )
        label_arr = _merge_seg_to_label(sample.seg, labels, shape)
        out_path = labels_out / f"{case_id}{file_ending}"
        nib.save(nib.Nifti1Image(label_arr, affine), str(out_path))


def to_nnunetv2(
    src_dir: str | Path,
    out_dir: str | Path,
    *,
    dataset_name: str | None = None,
    file_ending: str = ".nii.gz",
) -> Path:
    """Export a directory of ``.medh5`` files as a raw nnU-Net v2 dataset.

    Parameters
    ----------
    src_dir : str or Path
        Directory containing ``.medh5`` files. Supports two layouts: a
        mirrored layout with ``imagesTr/`` (and optional ``imagesTs/``)
        subdirectories, or a flat layout (all files treated as training).
    out_dir : str or Path
        Destination ``DatasetXXX_NAME/`` folder. Created if missing. Will
        contain ``imagesTr/``, ``labelsTr/``, optional ``imagesTs/``, and
        ``dataset.json``.
    dataset_name : str, optional
        Overrides the ``name`` field in ``dataset.json``. Defaults to the
        source folder name or the value stashed in ``extra["nnunetv2"]``.
    file_ending : str
        File extension for emitted NIfTI files. Defaults to ``".nii.gz"``.

    Returns
    -------
    Path
        Path to the written ``dataset.json``.

    Raises
    ------
    MEDH5ValidationError
        If no ``.medh5`` files are found, or if the files disagree on their
        nnU-Net metadata (channel set, label set).
    """
    _require_nibabel()

    src = Path(src_dir)
    if not src.is_dir():
        raise MEDH5ValidationError(f"Source directory not found: {src}")

    cases = _collect_medh5_cases(src)
    if not cases["train"]:
        raise MEDH5ValidationError(
            f"No training .medh5 files found under {src} "
            "(expected <src>/imagesTr/*.medh5 or <src>/*.medh5)"
        )

    # Use the first training file as the reference for channel/label metadata.
    ref_sample = MEDH5File.read(cases["train"][0])
    ref_extra = ref_sample.meta.extra or {}
    ref_nnunet_meta = ref_extra.get("nnunetv2") if isinstance(ref_extra, dict) else None
    channel_order = _resolve_channel_order(
        ref_nnunet_meta, list(ref_sample.images.keys())
    )
    labels = _resolve_label_mapping(ref_nnunet_meta, ref_sample.meta.seg_names)

    out = Path(out_dir)
    images_tr = out / "imagesTr"
    labels_tr = out / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for path in cases["train"]:
        sample = MEDH5File.read(path)
        case_id = path.stem
        _write_case_nifti(
            case_id,
            sample,
            channel_order,
            labels,
            images_tr,
            labels_tr,
            file_ending,
        )

    if cases["test"]:
        images_ts = out / "imagesTs"
        images_ts.mkdir(parents=True, exist_ok=True)
        for path in cases["test"]:
            sample = MEDH5File.read(path)
            case_id = path.stem
            _write_case_nifti(
                case_id,
                sample,
                channel_order,
                labels,
                images_ts,
                None,
                file_ending,
            )

    dataset_json: dict[str, Any] = {
        "channel_names": {
            str(idx): channel_order[idx] for idx in sorted(channel_order.keys())
        },
        "labels": dict(labels),
        "numTraining": len(cases["train"]),
        "file_ending": file_ending,
    }
    if ref_nnunet_meta is not None:
        for key in ("overwrite_image_reader_writer", "regions_class_order"):
            if key in ref_nnunet_meta:
                dataset_json[key] = ref_nnunet_meta[key]
    resolved_name = dataset_name
    if resolved_name is None and ref_nnunet_meta is not None:
        resolved_name = ref_nnunet_meta.get("name")
    if resolved_name is None:
        resolved_name = out.name
    dataset_json["name"] = resolved_name

    dataset_json_path = out / "dataset.json"
    dataset_json_path.write_text(json.dumps(dataset_json, indent=2))
    return dataset_json_path
