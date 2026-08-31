"""The validation rules, one section of the spec at a time (spec §15).

Rules are plain generators over a :class:`Context`.  Each yields
:class:`~medh5.validate.report.Diagnostic` objects and never raises: a validator
that stops at the first problem is useless for curation, where the point is to
see everything wrong with a file in one pass.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import h5py
import numpy as np

from medh5._hdf5 import ID_PATTERN, SAMPLE_KEY_PATTERN, as_str, as_str_tuple
from medh5.annotations.base import (
    ANNOTATION_KINDS,
    GEOMETRIC_KINDS,
    RESERVED_KINDS,
    TASKS,
    VOXEL_KINDS,
)
from medh5.annotations.classification import SCOPES
from medh5.annotations.geometric import ROTATION_TOL, SPACES
from medh5.collection import SAMPLES_GROUP
from medh5.curation.provenance import ACTIVITY_TYPES, RFC3339
from medh5.document import SampleDocument, schema_available, validate_against_schema
from medh5.errors import CODES
from medh5.geometry.affine import is_orthonormal, is_proper_rotation
from medh5.geometry.grid import AXIS_KINDS
from medh5.image import VALUE_TYPES
from medh5.labels.labelset import BACKGROUND_ID, CLOSURES, IGNORE_ID
from medh5.sample import PROFILES
from medh5.storage.codecs import is_bulk
from medh5.transforms.base import TRANSFORM_KINDS, VECTOR_SPACES
from medh5.validate.report import Diagnostic, Level, Report

SUPPORTED_MAJOR = "1"

REQUIRED_DATASETS: dict[str, tuple[str, ...]] = {
    "labelmap": ("data",),
    "layers": ("data", "layer_class_ids"),
    "bitmask": ("data", "bit_class_ids"),
    "probmap": ("data",),
    "mask": ("data",),
    "instances": ("boxes", "class_ids", "instance_ids"),
    "boxes": ("boxes", "class_ids"),
    "obb": ("centers", "sizes", "rotations", "class_ids"),
    "keypoints": ("points", "keypoint_class_ids", "class_ids"),
    "points": ("points",),
    "contours": ("vertices", "contour_offsets", "contour_class_ids"),
    "mesh": ("vertices", "faces"),
    "classification": ("class_ids", "values"),
}

ALLOWED_DTYPES: dict[str, tuple[str, ...]] = {
    "labelmap": ("uint8", "uint16"),
    "layers": ("uint8", "uint16"),
    "bitmask": ("uint64",),
    "probmap": ("float16", "float32"),
    "mask": ("bool", "uint8"),
}

GEOMETRIC_DTYPES: dict[str, dict[str, tuple[str, ...]]] = {
    "boxes": {"boxes": ("float32", "float64"), "class_ids": ("uint16",)},
    "obb": {
        "centers": ("float32", "float64"),
        "sizes": ("float32", "float64"),
        "rotations": ("float32", "float64"),
    },
    "keypoints": {"points": ("float32", "float64"), "visibility": ("uint8",)},
    "points": {"points": ("float32", "float64")},
    "contours": {"vertices": ("float32", "float64"), "contour_offsets": ("int64",)},
    "mesh": {"vertices": ("float32", "float64"), "faces": ("int32", "int64")},
    "classification": {"class_ids": ("uint16",), "values": ("float32", "float64")},
}

W908_TOLERANCE = 2
"""Extra layers beyond the greedy optimum before W908 fires."""


@dataclass
class Context:
    """What every rule gets: the file, the parsed document, and the level."""

    root: h5py.Group
    path: str
    level: Level
    profiles: tuple[str, ...] = ()
    document: SampleDocument | None = None
    grids: dict[str, Any] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def err(self, code: str, location: str, message: str) -> Diagnostic:
        return Diagnostic(
            code=code,
            location=location,
            message=message,
            severity=CODES[code].severity if code in CODES else "error",
            level=self.level,
        )

    @property
    def annotation_groups(self) -> dict[str, h5py.Group]:
        node = self.root.get("annotations")
        return {name: node[name] for name in sorted(node)} if node is not None else {}

    @property
    def image_nodes(self) -> dict[str, Any]:
        node = self.root.get("images")
        return {name: node[name] for name in sorted(node)} if node is not None else {}


# --------------------------------------------------------------------------
# §2 container
# --------------------------------------------------------------------------


def check_container(ctx: Context) -> Iterator[Diagnostic]:
    attrs = ctx.root.attrs
    if "medh5_version" not in attrs:
        yield ctx.err("E001", "/", "root has no `medh5_version` attribute")
    else:
        version = as_str(attrs["medh5_version"])
        major = version.split(".", 1)[0]
        if major != SUPPORTED_MAJOR:
            yield ctx.err(
                "E002",
                "/",
                f"declares MEDH5 {version}; this validator implements "
                f"{SUPPORTED_MAJOR}.x",
            )
    if "medh5_kind" not in attrs:
        yield ctx.err("E006", "/", "root has no `medh5_kind` attribute")
    elif as_str(attrs["medh5_kind"]) not in ("sample", "collection"):
        yield ctx.err(
            "E006",
            "/",
            f"unknown `medh5_kind` {as_str(attrs['medh5_kind'])!r}",
        )
    if "medh5_profiles" not in attrs:
        yield ctx.err("E007", "/", "root has no `medh5_profiles` attribute")
    else:
        unknown = set(as_str_tuple(attrs["medh5_profiles"])) - set(PROFILES)
        if unknown:
            yield ctx.err("E007", "/", f"unknown profile(s) {sorted(unknown)}")
    for required in ("grids", "images"):
        if required not in ctx.root:
            yield ctx.err(
                "E008", f"/{required}", f"required group `{required}` is absent"
            )
    if "meta" not in ctx.root:
        yield ctx.err("E004", "/meta", "required dataset `meta` is absent")

    for group_name in ("grids", "images", "annotations", "transforms"):
        node = ctx.root.get(group_name)
        if node is None:
            continue
        for name in node:
            if not ID_PATTERN.match(name) or name == "meta":
                yield ctx.err(
                    "E003",
                    f"/{group_name}/{name}",
                    f"identifier {name!r} does not match [A-Za-z0-9_.-]{{1,128}}",
                )


def check_collection(ctx: Context) -> Iterator[Diagnostic]:
    """Rules that apply to a ``collection`` root itself (spec §2.2).

    The members are validated as ordinary samples --- that is the point of the
    containment --- so this checks only what is true of the shard: that it says
    what it is, that its keys are legal, and that every member still carries the
    two attributes that make it independently identifiable once extracted.
    """
    attrs = ctx.root.attrs
    if "medh5_version" not in attrs:
        yield ctx.err("E001", "/", "collection root has no `medh5_version` attribute")
    elif as_str(attrs["medh5_version"]).split(".", 1)[0] != SUPPORTED_MAJOR:
        yield ctx.err(
            "E002",
            "/",
            f"declares MEDH5 {as_str(attrs['medh5_version'])}; this validator "
            f"implements {SUPPORTED_MAJOR}.x",
        )
    node = ctx.root.get(SAMPLES_GROUP)
    if node is None:
        yield ctx.err(
            "E008",
            f"/{SAMPLES_GROUP}",
            f"a `collection` requires a `{SAMPLES_GROUP}` group",
        )
        return
    if len(node) == 0:
        yield ctx.err(
            "E008", f"/{SAMPLES_GROUP}", "collection contains no sample roots"
        )
    for key in sorted(node):
        location = f"/{SAMPLES_GROUP}/{key}"
        if not SAMPLE_KEY_PATTERN.match(key):
            yield ctx.err(
                "E003",
                location,
                f"sample key {key!r} does not match [A-Za-z0-9_.-]{{1,255}}",
            )
        member = node[key]
        if "medh5_profiles" not in member.attrs:
            yield ctx.err(
                "E007",
                location,
                "a sample root in a collection carries its own `medh5_profiles`",
            )
        if "content_id" not in member.attrs:
            yield ctx.err(
                "E010",
                location,
                "a sample root in a collection carries its own `content_id`, so "
                "extracting it yields an identifiable sample",
            )


def check_document(ctx: Context) -> Iterator[Diagnostic]:
    """Parse and check ``/meta`` in three separable steps.

    Keeping them separate is what makes the codes mean what they say: E004 is
    "not JSON", E005 is "JSON that the schema rejects", and a document that is
    schema-valid but semantically impossible (a non-dense timepoint index, say)
    keeps its own specific code rather than being flattened into E005.
    """
    if "meta" not in ctx.root:
        return
    raw = ctx.root["meta"][()]
    text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        yield ctx.err("E004", "/meta", f"`meta` is not valid JSON: {exc}")
        return
    if not isinstance(parsed, dict):
        yield ctx.err("E004", "/meta", "`meta` must hold a JSON object")
        return

    ctx.notes["schema_checked"] = schema_available()
    schema_failed = False
    if schema_available():
        for message in validate_against_schema(parsed):
            schema_failed = True
            location, _, detail = message.partition(": ")
            yield ctx.err("E005", f"/meta#{location}", detail or message)

    try:
        ctx.document = SampleDocument.from_json(parsed)
    except Exception as exc:  # noqa: BLE001 - reported, never raised
        code = getattr(exc, "code", None) or "E005"
        if not (schema_failed and code == "E005"):
            yield ctx.err(code, "/meta", str(exc))


def check_bulk_storage(ctx: Context) -> Iterator[Diagnostic]:
    def visit(name: str, obj: h5py.HLObject) -> None:
        return None

    del visit
    for name, node in _iter_datasets(ctx.root):
        if name == "meta" or name.startswith("index/"):
            continue
        if is_bulk(node) and (node.chunks is None or not _has_filter(node)):
            yield ctx.err(
                "W902",
                f"/{name}",
                f"{node.nbytes / 1024 / 1024:.1f} MiB dataset is "
                f"{'unchunked' if node.chunks is None else 'uncompressed'}",
            )


def _has_filter(node: h5py.Dataset) -> bool:
    if node.compression:
        return True
    plist = node.id.get_create_plist()
    return int(plist.get_nfilters()) > 0


def _iter_datasets(root: h5py.Group) -> Iterator[tuple[str, h5py.Dataset]]:
    found: list[tuple[str, h5py.Dataset]] = []

    def visit(name: str, obj: h5py.HLObject) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append((name, obj))

    root.visititems(visit)
    return iter(found)


# --------------------------------------------------------------------------
# §3 geometry and timepoints
# --------------------------------------------------------------------------


def check_geometry(ctx: Context) -> Iterator[Diagnostic]:
    node = ctx.root.get("grids")
    if node is None:
        return
    if len(node) == 0:
        yield ctx.err("E111", "/grids", "a sample must declare at least one grid")
        return
    for name in sorted(node):
        group = node[name]
        location = f"/grids/{name}"
        attrs = group.attrs
        missing = [
            key
            for key in (
                "shape",
                "axis_names",
                "axis_kinds",
                "spacing",
                "origin",
                "direction",
                "coord_system",
                "units",
            )
            if key not in attrs
        ]
        if missing:
            yield ctx.err("E109", location, f"missing required attribute(s) {missing}")
            continue
        shape = np.atleast_1d(attrs["shape"])
        kinds = as_str_tuple(attrs["axis_kinds"])
        names = as_str_tuple(attrs["axis_names"])
        if len(kinds) != shape.size or len(names) != shape.size:
            yield ctx.err(
                "E109",
                location,
                f"axis_names/axis_kinds have {len(names)}/{len(kinds)} entries "
                f"for a {shape.size}-D shape",
            )
            continue
        unknown = set(kinds) - set(AXIS_KINDS)
        if unknown:
            yield ctx.err("E110", location, f"unknown axis kinds {sorted(unknown)}")
            continue
        n_spatial = kinds.count("spatial")
        if not 2 <= n_spatial <= 3:  # noqa: PLR2004
            yield ctx.err(
                "E110", location, f"{n_spatial} spatial axes; the spec allows 2 or 3"
            )
        if kinds.count("time") > 1 or kinds.count("channel") > 1:
            yield ctx.err(
                "E110",
                location,
                "at most one `time` and one `channel` axis are allowed",
            )
        spatial_positions = tuple(i for i, k in enumerate(kinds) if k == "spatial")
        expected = tuple(range(len(kinds) - n_spatial, len(kinds)))
        if spatial_positions != expected:
            yield ctx.err(
                "E103",
                location,
                f"spatial axes at {spatial_positions} must be contiguous and trailing "
                f"(expected {expected})",
            )
        spacing = np.atleast_1d(np.asarray(attrs["spacing"], dtype=float))
        if np.any(spacing <= 0):
            yield ctx.err(
                "E104",
                location,
                f"spacing {spacing.tolist()} must be strictly positive",
            )
        direction = np.asarray(attrs["direction"], dtype=float)
        if direction.ndim != 2:  # noqa: PLR2004
            yield ctx.err(
                "E109",
                location,
                f"`direction` must be stored 2-D, got shape {direction.shape}",
            )
        elif not is_orthonormal(direction):
            residual = float(
                np.max(np.abs(direction.T @ direction - np.eye(direction.shape[0])))
            )
            yield ctx.err(
                "E102",
                location,
                f"`direction` is not orthonormal (max residual {residual:.3g})",
            )


def check_timepoints(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    if document is None:
        return
    declared = set(document.timepoints.ids)
    multi = len(declared) > 1
    node = ctx.root.get("grids")
    if node is None:
        return
    frames: dict[str, set[str]] = {}
    for name in sorted(node):
        attrs = node[name].attrs
        location = f"/grids/{name}"
        timepoint = as_str(attrs["timepoint"]) if "timepoint" in attrs else None
        if timepoint is None:
            if multi:
                yield ctx.err(
                    "E106",
                    location,
                    f"grid has no `timepoint`, but the sample declares "
                    f"{len(declared)} timepoints",
                )
            continue
        if timepoint not in declared:
            yield ctx.err(
                "E107",
                location,
                f"`timepoint` {timepoint!r} is not declared "
                f"(declared: {sorted(declared)})",
            )
            continue
        frame = as_str(attrs["frame_uid"]) if "frame_uid" in attrs else None
        if frame:
            frames.setdefault(frame, set()).add(timepoint)
    for frame, timepoints in sorted(frames.items()):
        if len(timepoints) > 1:
            yield ctx.err(
                "W910",
                "/grids",
                f"frame_uid {frame!r} is shared by timepoints {sorted(timepoints)}; "
                "follow-up imaging is a new frame unless the subject was never "
                "repositioned",
            )
    if multi and not _has_relating_transform(ctx, frames):
        yield ctx.err(
            "W911",
            "/transforms",
            f"the sample declares {len(declared)} timepoints but no transform "
            "relates any two of them",
        )


def _has_relating_transform(ctx: Context, frames: dict[str, set[str]]) -> bool:
    node = ctx.root.get("transforms")
    if node is None or len(node) == 0:
        return False
    for name in node:
        attrs = node[name].attrs
        src = as_str(attrs["from_frame"]) if "from_frame" in attrs else None
        dst = as_str(attrs["to_frame"]) if "to_frame" in attrs else None
        if src and dst and frames.get(src, set()) != frames.get(dst, set()):
            return True
    return False


# --------------------------------------------------------------------------
# §4 images
# --------------------------------------------------------------------------


def check_images(ctx: Context) -> Iterator[Diagnostic]:
    node = ctx.root.get("images")
    if node is None:
        return
    if len(node) == 0:
        yield ctx.err("E201", "/images", "a sample must contain at least one image")
        return
    grids_node = ctx.root.get("grids")
    for name in sorted(node):
        image = node[name]
        location = f"/images/{name}"
        attrs = image.attrs
        for key, code in (
            ("grid", "E205"),
            ("modality", "E205"),
            ("value_type", "E205"),
        ):
            if key not in attrs:
                yield ctx.err(code, location, f"missing required attribute {key!r}")
        if "value_type" in attrs and as_str(attrs["value_type"]) not in VALUE_TYPES:
            yield ctx.err(
                "E203",
                location,
                f"unknown value_type {as_str(attrs['value_type'])!r}",
            )
        if "grid" not in attrs:
            continue
        grid_id = as_str(attrs["grid"])
        if grids_node is None or grid_id not in grids_node:
            yield ctx.err(
                "E101", location, f"names grid {grid_id!r}, which does not exist"
            )
            continue
        grid_shape = tuple(
            int(v) for v in np.atleast_1d(grids_node[grid_id].attrs["shape"])
        )
        dataset = image["0"] if isinstance(image, h5py.Group) else image
        if tuple(dataset.shape) != grid_shape:
            yield ctx.err(
                "E202",
                location,
                f"shape {tuple(dataset.shape)} != grid {grid_id!r} shape {grid_shape}",
            )
        if "channel_names" in attrs:
            kinds = as_str_tuple(grids_node[grid_id].attrs["axis_kinds"])
            if "channel" not in kinds:
                yield ctx.err(
                    "E204", location, "`channel_names` on a grid with no channel axis"
                )
            else:
                extent = grid_shape[kinds.index("channel")]
                if len(as_str_tuple(attrs["channel_names"])) != extent:
                    yield ctx.err(
                        "E204",
                        location,
                        f"`channel_names` has "
                        f"{len(as_str_tuple(attrs['channel_names']))} "
                        f"entries for a channel axis of extent {extent}",
                    )
        if dataset.dtype.kind == "f" and _int16_lossless(dataset):
            yield ctx.err(
                "W907",
                location,
                f"stored as {dataset.dtype.name} but every value is an integer within "
                "int16 range; int16 + rescale is lossless and ~3x smaller",
            )


def _int16_lossless(dataset: h5py.Dataset, sample_limit: int = 4_000_000) -> bool:
    if dataset.size == 0 or dataset.size > sample_limit:
        return False
    values = np.asarray(dataset[...])
    finite = values[np.isfinite(values)]
    if finite.size == 0 or not np.array_equal(finite, np.rint(finite)):
        return False
    return bool(
        finite.min() >= np.iinfo(np.int16).min
        and finite.max() <= np.iinfo(np.int16).max
    )


def check_multiscale(ctx: Context) -> Iterator[Diagnostic]:
    from medh5.geometry.grid import read_grid
    from medh5.geometry.multiscale import check_pyramid

    node = ctx.root.get("images")
    grids_node = ctx.root.get("grids")
    if node is None or grids_node is None:
        return
    for name in sorted(node):
        image = node[name]
        if not isinstance(image, h5py.Group):
            continue
        attrs = image.attrs
        location = f"/images/{name}"
        if "grid_levels" not in attrs or "downsample_factors" not in attrs:
            yield ctx.err(
                "E105",
                location,
                "multiscale image needs `grid_levels` and `downsample_factors`",
            )
            continue
        level_ids = as_str_tuple(attrs["grid_levels"])
        if any(gid not in grids_node for gid in level_ids):
            yield ctx.err(
                "E101",
                location,
                f"grid_levels reference missing grids {list(level_ids)}",
            )
            continue
        grids = [read_grid(grids_node[gid], gid) for gid in level_ids]
        problems = check_pyramid(
            grids[0], grids, np.asarray(attrs["downsample_factors"], dtype=float)
        )
        for problem in problems:
            yield ctx.err("E105", location, problem)


# --------------------------------------------------------------------------
# §5 label set
# --------------------------------------------------------------------------


def check_label_set(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    if document is None:
        return
    needs_labels = {"seg", "det", "cls"} & set(ctx.profiles)
    label_set = document.label_set
    if label_set is None:
        if needs_labels:
            yield ctx.err(
                "E301",
                "/meta#label_set",
                f"profile(s) {sorted(needs_labels)} require a label set",
            )
        return
    if label_set.form == "ref" and not label_set.uri:
        yield ctx.err("E305", "/meta#label_set", "`form: ref` requires a `uri`")
    seen_ids: set[int] = set()
    seen_keys: set[str] = set()
    for entry in label_set.classes:
        if entry.id in (BACKGROUND_ID, IGNORE_ID):
            yield ctx.err(
                "E303",
                f"/meta#label_set/classes/{entry.key}",
                f"id {entry.id} is reserved",
            )
        if entry.id in seen_ids:
            yield ctx.err(
                "E302",
                f"/meta#label_set/classes/{entry.key}",
                f"duplicate class id {entry.id}",
            )
        if entry.key in seen_keys:
            yield ctx.err(
                "E302",
                f"/meta#label_set/classes/{entry.key}",
                f"duplicate class key {entry.key!r}",
            )
        seen_ids.add(entry.id)
        seen_keys.add(entry.key)
    try:
        label_set.check()
    except Exception as exc:  # noqa: BLE001 - surfaced as a diagnostic
        code = getattr(exc, "code", None) or "E306"
        yield ctx.err(code, "/meta#label_set", str(exc))


def check_ontology_bindings(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    if document is None or document.label_set is None:
        return
    used: set[int] = set()
    for group in ctx.annotation_groups.values():
        if "class_ids" in group.attrs:
            used.update(int(c) for c in np.atleast_1d(group.attrs["class_ids"]))
    unbound = sorted(
        c
        for c in used
        if (entry := document.label_set.get(c)) is not None and not entry.codes
    )
    if unbound:
        yield ctx.err(
            "W912",
            "/meta#label_set",
            f"{len(unbound)} class(es) used by annotations have no ontology binding: "
            f"{unbound[:8]}{'...' if len(unbound) > 8 else ''}",
        )


# --------------------------------------------------------------------------
# §6-§7 annotations
# --------------------------------------------------------------------------


def check_annotations(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    grids_node = ctx.root.get("grids")
    declared_timepoints = set(document.timepoints.ids) if document else set()
    label_set = document.label_set if document else None

    for name, group in ctx.annotation_groups.items():
        location = f"/annotations/{name}"
        attrs = group.attrs
        if "kind" not in attrs:
            yield ctx.err("E412", location, "missing required attribute `kind`")
            continue
        kind = as_str(attrs["kind"])
        if kind in RESERVED_KINDS:
            yield ctx.err(
                "E401",
                location,
                f"kind {kind!r} is reserved by spec §16 and must not appear in a 1.0 "
                f"file",
            )
            continue
        if kind not in ANNOTATION_KINDS:
            yield ctx.err("E401", location, f"unknown annotation kind {kind!r}")
            continue
        if "task" in attrs and as_str(attrs["task"]) not in TASKS:
            yield ctx.err("E412", location, f"unknown task {as_str(attrs['task'])!r}")
        if "closure" in attrs and as_str(attrs["closure"]) not in CLOSURES:
            yield ctx.err(
                "E412", location, f"unknown closure {as_str(attrs['closure'])!r}"
            )
        if "annotated_class_ids" not in attrs and kind != "mask":
            yield ctx.err(
                "E412",
                location,
                "missing `annotated_class_ids`; the coverage contract is required",
            )
        class_ids = (
            {int(c) for c in np.atleast_1d(attrs["class_ids"])}
            if "class_ids" in attrs
            else set()
        )
        annotated = (
            {int(c) for c in np.atleast_1d(attrs["annotated_class_ids"])}
            if "annotated_class_ids" in attrs
            else set()
        )
        if not annotated <= class_ids and kind != "mask":
            yield ctx.err(
                "E403",
                location,
                f"annotated_class_ids {sorted(annotated - class_ids)} are not in "
                f"class_ids",
            )
        if label_set is not None and label_set.form == "inline":
            missing = label_set.missing(class_ids)
            if missing:
                yield ctx.err(
                    "E402",
                    location,
                    f"class ids {list(missing)} are not in label set {label_set.id!r}",
                )
        reserved = class_ids & {BACKGROUND_ID, IGNORE_ID}
        if reserved:
            yield ctx.err(
                "E303", location, f"class_ids uses reserved id(s) {sorted(reserved)}"
            )
        if "timepoints" in attrs:
            for timepoint in as_str_tuple(attrs["timepoints"]):
                if timepoint not in declared_timepoints:
                    yield ctx.err(
                        "E409", location, f"undeclared timepoint {timepoint!r}"
                    )
        grid_id = as_str(attrs["grid"]) if "grid" in attrs else None
        if kind in VOXEL_KINDS:
            if grid_id is None:
                yield ctx.err("E412", location, f"kind {kind!r} requires a `grid`")
            elif grids_node is None or grid_id not in grids_node:
                yield ctx.err(
                    "E101", location, f"names grid {grid_id!r}, which does not exist"
                )
        for required in REQUIRED_DATASETS.get(kind, ()):
            if required not in group:
                yield ctx.err(
                    "E410", location, f"kind {kind!r} requires dataset {required!r}"
                )
        if "data" in group and kind in ALLOWED_DTYPES:
            dtype_name = group["data"].dtype.name
            if dtype_name not in ALLOWED_DTYPES[kind]:
                yield ctx.err(
                    "E411",
                    f"{location}/data",
                    f"dtype {dtype_name} is not permitted for kind {kind!r} "
                    f"(expected one of {list(ALLOWED_DTYPES[kind])})",
                )
        yield from _check_voxel_shape(ctx, name, group, kind, grid_id, grids_node)
        yield from _check_encoding_invariants(ctx, name, group, kind)
        yield from _check_dataset_dtypes(ctx, name, group, kind)
        yield from _check_geometric(ctx, name, group, kind, grid_id, grids_node)
        yield from _check_classification(ctx, name, group, kind, declared_timepoints)
        if kind != "mask" and annotated < class_ids and not _has_ignore(group, kind):
            yield ctx.err(
                "W904",
                location,
                f"{len(class_ids - annotated)} class(es) are encodable but not "
                "annotated, and there is no ignore region; `0` cannot be read as "
                "a verified negative for them",
            )


def _has_ignore(group: h5py.Group, kind: str) -> bool:
    if "ignore_mask" in group.attrs:
        return True
    if kind in ("labelmap", "layers") and "data" in group:
        ignore_id = int(group.attrs.get("ignore_id", IGNORE_ID))
        data = group["data"]
        if data.size <= 64_000_000:  # noqa: PLR2004 - bounded scan
            return bool(np.any(np.asarray(data[...]) == ignore_id))
    return False


def _check_voxel_shape(
    ctx: Context,
    name: str,
    group: h5py.Group,
    kind: str,
    grid_id: str | None,
    grids_node: h5py.Group | None,
) -> Iterator[Diagnostic]:
    if kind not in VOXEL_KINDS or grid_id is None or grids_node is None:
        return
    if grid_id not in grids_node or "data" not in group:
        return
    attrs = grids_node[grid_id].attrs
    shape = tuple(int(v) for v in np.atleast_1d(attrs["shape"]))
    kinds = as_str_tuple(attrs["axis_kinds"])
    spatial = tuple(s for s, k in zip(shape, kinds, strict=True) if k == "spatial")
    data_shape = tuple(int(v) for v in group["data"].shape)
    expected_stacked = kind in ("layers", "bitmask", "probmap")
    tail = data_shape[1:] if expected_stacked else data_shape
    if tail != spatial:
        yield ctx.err(
            "E405",
            f"/annotations/{name}/data",
            f"spatial shape {tail} != grid {grid_id!r} spatial shape {spatial}",
        )
    if (
        expected_stacked
        and group["data"].chunks is not None
        and group["data"].chunks[0] != 1
    ):
        yield ctx.err(
            "W902",
            f"/annotations/{name}/data",
            f"chunk shape {group['data'].chunks} spans the stacked axis; the spec "
            "requires (1, *spatial_chunk) so one plane reads without the others",
        )


def _check_encoding_invariants(
    ctx: Context, name: str, group: h5py.Group, kind: str
) -> Iterator[Diagnostic]:
    location = f"/annotations/{name}"
    declared = (
        {int(c) for c in np.atleast_1d(group.attrs["class_ids"])}
        if "class_ids" in group.attrs
        else set()
    )
    if kind == "layers" and "layer_class_ids" in group:
        table = np.asarray(group["layer_class_ids"][...])
        seen: dict[int, int] = {}
        for layer in range(table.shape[0]):
            for value in table[layer]:
                class_id = int(value)
                if class_id == 0:
                    continue
                if class_id in seen:
                    yield ctx.err(
                        "E404",
                        location,
                        f"class {class_id} appears in layers {seen[class_id]} and "
                        f"{layer}; "
                        "every class must be in exactly one layer",
                    )
                seen[class_id] = layer
        missing = declared - set(seen)
        if missing:
            yield ctx.err(
                "E404",
                location,
                f"class_ids {sorted(missing)} are not assigned to any layer",
            )
        yield from _check_layer_optimality(
            ctx, name, group, table.shape[0], len(declared)
        )
    if kind == "bitmask" and "bit_class_ids" in group and "data" in group:
        n_classes = int(np.asarray(group["bit_class_ids"]).size)
        expected = max(1, math.ceil(n_classes / 64))
        planes = int(group["data"].shape[0])
        if planes != expected:
            yield ctx.err(
                "E404",
                location,
                f"{planes} bitplanes for {n_classes} classes; expected {expected}",
            )
    if kind == "instances":
        yield from _check_instances(ctx, name, group)


def _check_layer_optimality(
    ctx: Context, name: str, group: h5py.Group, n_layers: int, n_classes: int
) -> Iterator[Diagnostic]:
    if ctx.level not in ("semantic", "strict") or n_classes == 0:
        return
    from medh5.annotations.payload import AnnotationPayload
    from medh5.annotations.voxel.select import analyse
    from medh5.annotations.voxel.transcode import payload_to_masks

    data = group["data"]
    if data.size > 64_000_000:  # noqa: PLR2004 - bounded: colouring needs the masks
        return
    payload = AnnotationPayload(
        kind="layers",
        datasets={
            "data": np.asarray(data[...]),
            "layer_class_ids": np.asarray(group["layer_class_ids"][...]),
        },
    )
    masks = payload_to_masks(payload)
    optimal = analyse(masks).n_layers
    if n_layers > optimal + W908_TOLERANCE:
        yield ctx.err(
            "W908",
            f"/annotations/{name}",
            f"{n_layers} layers where a greedy colouring of the overlap graph needs "
            f"{optimal}; transcoding would cut the label volume by "
            f"{100 * (1 - optimal / n_layers):.0f}%",
        )


def _check_dataset_dtypes(
    ctx: Context, name: str, group: h5py.Group, kind: str
) -> Iterator[Diagnostic]:
    """Per-kind dtype checks for the §8/§9 datasets (E411)."""
    for dataset, allowed in GEOMETRIC_DTYPES.get(kind, {}).items():
        if dataset not in group:
            continue
        found = group[dataset].dtype.name
        if found not in allowed:
            yield ctx.err(
                "E411",
                f"/annotations/{name}/{dataset}",
                f"dtype {found} is not permitted for {kind!r}.{dataset} "
                f"(expected one of {list(allowed)})",
            )


def _check_geometric(
    ctx: Context,
    name: str,
    group: h5py.Group,
    kind: str,
    grid_id: str | None,
    grids_node: h5py.Group | None,
) -> Iterator[Diagnostic]:
    """Coordinate-space and per-kind invariants for §8 annotations."""
    if kind not in GEOMETRIC_KINDS:
        return
    location = f"/annotations/{name}"
    attrs = group.attrs
    space = as_str(attrs["space"]) if "space" in attrs else None
    if space is None:
        yield ctx.err("E412", location, f"kind {kind!r} requires a `space` attribute")
    elif space not in SPACES:
        yield ctx.err("E412", location, f"unknown space {space!r}")
    elif space == "index" and grid_id is None:
        yield ctx.err(
            "E412", location, "space='index' names a grid's coordinates, but no `grid`"
        )
    elif space == "world" and "frame_uid" not in attrs:
        yield ctx.err(
            "E412", location, "space='world' names a physical frame, but no `frame_uid`"
        )
    if (
        space is not None
        and space != "index"
        and grid_id is not None
        and grids_node is not None
        and grid_id in grids_node
        and as_str(grids_node[grid_id].attrs.get("units", "mm")) == "px"
    ):
        yield ctx.err(
            "E414",
            location,
            f"grid {grid_id!r} is uncalibrated (units='px'), so a geometric "
            "annotation on it must use space='index'",
        )

    if kind == "boxes" and "boxes" in group:
        boxes = np.asarray(group["boxes"][...])
        if boxes.ndim == 3 and boxes.size and np.any(boxes[..., 0] > boxes[..., 1]):  # noqa: PLR2004
            bad = int(np.sum(np.any(boxes[..., 0] > boxes[..., 1], axis=1)))
            yield ctx.err("E406", location, f"{bad} box(es) have lo > hi")
        # `slice_index` names the plane each 2-D box sits on (§8.2). The rule
        # lives in `check_slice_index` and the writer and `as_slices` call the
        # same function, so this cannot drift from what they enforce.
        if "slice_index" in group and boxes.ndim == 3:  # noqa: PLR2004
            from medh5.annotations.geometric import check_slice_index

            planes = np.asarray(group["slice_index"][...])
            # The range half needs index-space boxes and the grid they index.
            # A world-space box's plane is not known until the affine has been
            # applied, so those are range-checked on the way out instead; the
            # shape half applies either way.
            spatial: tuple[int, ...] | None = None
            if (
                space == "index"
                and grid_id is not None
                and grids_node is not None
                and grid_id in grids_node
                and boxes.ndim == 3  # noqa: PLR2004
            ):
                full = [
                    int(v) for v in np.atleast_1d(grids_node[grid_id].attrs["shape"])
                ]
                dims = int(boxes.shape[1])
                if len(full) >= dims:
                    spatial = tuple(full[-dims:])
            problem = check_slice_index(
                planes,
                boxes.shape[0],
                boxes=boxes.astype(np.float64) if spatial else None,
                shape=spatial,
            )
            if problem:
                yield ctx.err("E405", location, problem)
    if kind == "obb" and "rotations" in group:
        rotations = np.asarray(group["rotations"][...], dtype=np.float64)
        offenders = [
            i
            for i in range(rotations.shape[0])
            if not is_proper_rotation(rotations[i], ROTATION_TOL)
        ]
        if offenders:
            yield ctx.err(
                "E407",
                location,
                f"{len(offenders)} rotation(s) are not proper rotations "
                f"(orthonormal with det = +1); first at index {offenders[0]}",
            )
        if "sizes" in group and np.any(np.asarray(group["sizes"][...]) < 0):
            yield ctx.err("E406", location, "`sizes` must be non-negative edge lengths")
    if kind == "keypoints":
        yield from _check_keypoints(ctx, name, group)
    if kind == "points":
        target = as_str(attrs["correspondence"]) if "correspondence" in attrs else None
        if target is not None and target not in ctx.annotation_groups:
            yield ctx.err(
                "E413",
                location,
                f"`correspondence` names annotation {target!r}, which does not exist",
            )
    if kind == "contours":
        yield from _check_offsets(ctx, name, group, "contour_offsets", "vertices")
    if kind == "mesh":
        yield from _check_mesh(ctx, name, group)


def _check_keypoints(
    ctx: Context, name: str, group: h5py.Group
) -> Iterator[Diagnostic]:
    location = f"/annotations/{name}"
    if "points" not in group:
        return
    points = group["points"]
    if points.ndim != 3:  # noqa: PLR2004
        yield ctx.err(
            "E405", f"{location}/points", f"expected (N, K, S), got {points.shape}"
        )
        return
    n, k = int(points.shape[0]), int(points.shape[1])
    if "keypoint_class_ids" in group and int(group["keypoint_class_ids"].shape[0]) != k:
        yield ctx.err(
            "E405",
            location,
            f"`keypoint_class_ids` has {group['keypoint_class_ids'].shape[0]} entries "
            f"for {k} keypoint slots",
        )
    if "visibility" in group:
        visibility = np.asarray(group["visibility"][...])
        if visibility.shape != (n, k):
            yield ctx.err(
                "E405", location, f"`visibility` {visibility.shape} must be ({n}, {k})"
            )
        elif visibility.size and int(visibility.max()) > 2:  # noqa: PLR2004
            yield ctx.err(
                "E411",
                f"{location}/visibility",
                "values must be 0 (unlabelled), 1 (occluded) or 2 (visible)",
            )
    skeleton = as_str(group.attrs["skeleton"]) if "skeleton" in group.attrs else None
    if skeleton is not None:
        label_set = ctx.document.label_set if ctx.document else None
        known = {sk.id for sk in label_set.skeletons} if label_set else set()
        if skeleton not in known:
            yield ctx.err(
                "E413",
                location,
                f"`skeleton` names {skeleton!r}, which the label set does not declare",
            )


def _check_offsets(
    ctx: Context, name: str, group: h5py.Group, offsets_name: str, target: str
) -> Iterator[Diagnostic]:
    if offsets_name not in group:
        return
    offsets = np.asarray(group[offsets_name][...]).astype(np.int64)
    location = f"/annotations/{name}/{offsets_name}"
    if offsets.size and np.any(np.diff(offsets) < 0):
        yield ctx.err("E408", location, "offsets are not monotonically increasing")
    if (
        offsets.size
        and target in group
        and int(offsets[-1]) != int(group[target].shape[0])
    ):
        yield ctx.err(
            "E408",
            location,
            f"last offset {int(offsets[-1])} != {target} length "
            f"{int(group[target].shape[0])}",
        )


def _check_mesh(ctx: Context, name: str, group: h5py.Group) -> Iterator[Diagnostic]:
    location = f"/annotations/{name}"
    yield from _check_offsets(ctx, name, group, "mesh_offsets", "faces")
    if "vertices" not in group or "faces" not in group:
        return
    n_vertices = int(group["vertices"].shape[0])
    faces = np.asarray(group["faces"][...])
    if faces.size and (int(faces.min()) < 0 or int(faces.max()) >= n_vertices):
        yield ctx.err(
            "E405",
            f"{location}/faces",
            f"face indices reach outside the {n_vertices} vertices",
        )
    if "normals" in group and tuple(group["normals"].shape) != tuple(
        group["vertices"].shape
    ):
        yield ctx.err(
            "E405",
            f"{location}/normals",
            f"{tuple(group['normals'].shape)} must match vertices "
            f"{tuple(group['vertices'].shape)}",
        )


def _check_classification(
    ctx: Context,
    name: str,
    group: h5py.Group,
    kind: str,
    declared_timepoints: set[str],
) -> Iterator[Diagnostic]:
    """§9 invariants: scope, value range, and single-label exclusivity."""
    if kind != "classification":
        return
    location = f"/annotations/{name}"
    attrs = group.attrs
    if "scope" not in attrs:
        yield ctx.err("E412", location, "`classification` requires a `scope` attribute")
        scope = None
    else:
        scope = as_str(attrs["scope"])
        if scope not in SCOPES:
            yield ctx.err("E412", location, f"unknown classification scope {scope!r}")
    if "class_ids" not in group or "values" not in group:
        return
    class_ids = np.asarray(group["class_ids"][...])
    values = np.asarray(group["values"][...], dtype=np.float64)
    if values.shape != class_ids.shape:
        yield ctx.err(
            "E405",
            location,
            f"`values` {values.shape} must match `class_ids` {class_ids.shape}",
        )
        return
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        yield ctx.err(
            "E404",
            location,
            "values must lie in [0, 1]; 1.0 is a hard positive, 0.0 an explicit "
            "negative",
        )
    scope_ids = np.asarray(group["scope_ids"][...]) if "scope_ids" in group else None
    if scope_ids is not None and scope_ids.shape != class_ids.shape:
        yield ctx.err(
            "E405",
            location,
            f"`scope_ids` {scope_ids.shape} must match `class_ids` {class_ids.shape}",
        )
        scope_ids = None
    multilabel = bool(attrs.get("multilabel", True))
    if not multilabel and values.size:
        units = scope_ids if scope_ids is not None else np.zeros_like(values, dtype=int)
        crowded = sorted(
            {
                int(u)
                for u in np.unique(units)
                if int((values > 0.0)[units == u].sum()) > 1
            }
        )
        if crowded:
            yield ctx.err(
                "E404",
                location,
                f"multilabel=false allows one positive class per scope unit, but "
                f"unit(s) {crowded} carry several",
            )
    if scope == "timepoint" and scope_ids is not None and ctx.document is not None:
        declared = len(ctx.document.timepoints)
        unknown = sorted({int(v) for v in scope_ids if not 0 <= int(v) < declared})
        if unknown:
            yield ctx.err(
                "E409",
                location,
                f"scope='timepoint' scope_ids {unknown} are not timepoint indices "
                f"(0..{declared - 1})",
            )
    for column in ("schemes", "scheme_values"):
        if column in group and tuple(group[column].shape) != tuple(class_ids.shape):
            yield ctx.err(
                "E405",
                f"{location}/{column}",
                f"{tuple(group[column].shape)} must match `class_ids` "
                f"{tuple(class_ids.shape)}",
            )
    del declared_timepoints


def _check_instances(
    ctx: Context, name: str, group: h5py.Group
) -> Iterator[Diagnostic]:
    location = f"/annotations/{name}"
    if "boxes" in group:
        boxes = np.asarray(group["boxes"][...])
        if boxes.ndim == 3 and np.any(boxes[..., 0] > boxes[..., 1]):  # noqa: PLR2004
            bad = int(np.sum(np.any(boxes[..., 0] > boxes[..., 1], axis=1)))
            yield ctx.err("E406", location, f"{bad} box(es) have lo > hi")
    if "mask_offsets" in group:
        offsets = np.asarray(group["mask_offsets"][...])
        if np.any(np.diff(offsets.astype(np.int64)) < 0):
            yield ctx.err("E408", location, "`mask_offsets` are not monotonic")
    if "instance_ids" in group and "class_ids" in group:
        ids = np.asarray(group["instance_ids"][...])
        classes = np.asarray(group["class_ids"][...])
        table: dict[int, int] = {}
        conflicting: list[int] = []
        for instance_id, class_id in zip(ids.tolist(), classes.tolist(), strict=True):
            if table.setdefault(instance_id, class_id) != class_id:
                conflicting.append(instance_id)
        if conflicting:
            yield ctx.err(
                "W909",
                location,
                f"instance id(s) {sorted(set(conflicting))} carry more than one class "
                f"id; "
                "this is almost always a tracking error rather than a reclassification",
            )


# --------------------------------------------------------------------------
# §10 transforms
# --------------------------------------------------------------------------


def check_instance_identity(ctx: Context) -> Iterator[Diagnostic]:
    """``instance_id`` is sample-scoped, so the check has to be too (§7.4).

    :func:`_check_instances` already catches a conflict inside one annotation.
    The conflict that actually costs a study is the one *between* annotations:
    lesion 7 is a metastasis at baseline and a cyst at follow-up, each file
    internally consistent, the tracking join silently wrong.  Nothing but a
    sample-wide pass can see it.
    """
    table: dict[int, dict[int, list[str]]] = {}
    for name, group in ctx.annotation_groups.items():
        if "instance_ids" not in group or "class_ids" not in group:
            continue
        ids = np.asarray(group["instance_ids"][...]).reshape(-1)
        classes = np.asarray(group["class_ids"][...]).reshape(-1)
        if ids.shape != classes.shape:
            continue  # E405 reports the mismatch; do not compound it
        for instance_id, class_id in zip(ids.tolist(), classes.tolist(), strict=True):
            table.setdefault(int(instance_id), {}).setdefault(int(class_id), []).append(
                name
            )
    for instance_id, by_class in sorted(table.items()):
        if len(by_class) < 2:  # noqa: PLR2004 - one class is the healthy case
            continue
        where = sorted({n for names in by_class.values() for n in names})
        if len(where) < 2:  # noqa: PLR2004 - single-annotation case: _check_instances
            continue
        detail = ", ".join(
            f"class {c} in {sorted(set(names))}"
            for c, names in sorted(by_class.items())
        )
        yield ctx.err(
            "W909",
            "/annotations",
            f"instance id {instance_id} carries several class ids across "
            f"annotations ({detail}); the longitudinal join treats these as one "
            "object",
        )


def check_transforms(ctx: Context) -> Iterator[Diagnostic]:
    """Transform structure, frame chaining and inverse consistency (§10)."""
    node = ctx.root.get("transforms")
    if node is None:
        return
    grids_node = ctx.root.get("grids")
    declared: dict[str, tuple[str, str]] = {}
    for name in sorted(node):
        group = node[name]
        location = f"/transforms/{name}"
        attrs = group.attrs
        if "kind" not in attrs:
            yield ctx.err("E502", location, "transform has no `kind` attribute")
            continue
        kind = as_str(attrs["kind"])
        if kind not in TRANSFORM_KINDS:
            yield ctx.err(
                "E502",
                location,
                f"unknown transform kind {kind!r}; expected one of "
                f"{list(TRANSFORM_KINDS)}",
            )
            continue
        missing = [k for k in ("from_frame", "to_frame") if k not in attrs]
        if missing:
            yield ctx.err("E502", location, f"missing {missing}")
            continue
        source, target = as_str(attrs["from_frame"]), as_str(attrs["to_frame"])
        declared[name] = (source, target)
        if source == target:
            yield ctx.err(
                "E502",
                location,
                f"maps frame {source!r} to itself; grids sharing a frame need no "
                "transform (§3.4)",
            )
        if kind == "affine":
            yield from _check_affine(ctx, name, group)
        if kind in ("displacement", "bspline"):
            yield from _check_field_transform(
                ctx, name, group, kind, source, grids_node
            )
        if kind == "composite":
            yield from _check_composite(ctx, name, group, declared, source, target)
    yield from _check_inverses(ctx, node, declared)


def _check_affine(ctx: Context, name: str, group: h5py.Group) -> Iterator[Diagnostic]:
    location = f"/transforms/{name}"
    if "matrix" not in group:
        yield ctx.err("E502", location, "kind 'affine' requires a `matrix` dataset")
        return
    matrix = np.asarray(group["matrix"][...], dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:  # noqa: PLR2004
        yield ctx.err(
            "E504", location, f"`matrix` must be square (S+1, S+1), got {matrix.shape}"
        )
        return
    expected = np.zeros(matrix.shape[0])
    expected[-1] = 1.0
    if not np.allclose(matrix[-1], expected, atol=1e-9):
        yield ctx.err(
            "E504",
            location,
            f"last row must be [0 … 0 1], got {matrix[-1].tolist()}",
        )


def _check_field_transform(
    ctx: Context,
    name: str,
    group: h5py.Group,
    kind: str,
    from_frame: str,
    grids_node: h5py.Group | None,
) -> Iterator[Diagnostic]:
    location = f"/transforms/{name}"
    dataset, grid_attr = (
        ("field", "field_grid")
        if kind == "displacement"
        else ("control_points", "cp_grid")
    )
    if dataset not in group:
        yield ctx.err("E502", location, f"kind {kind!r} requires a {dataset!r} dataset")
    if grid_attr not in group.attrs:
        yield ctx.err("E503", location, f"kind {kind!r} requires {grid_attr!r}")
        return
    grid_id = as_str(group.attrs[grid_attr])
    if grids_node is None or grid_id not in grids_node:
        yield ctx.err("E101", location, f"{grid_attr} {grid_id!r} does not exist")
        return
    grid_attrs = grids_node[grid_id].attrs
    frame = as_str(grid_attrs["frame_uid"]) if "frame_uid" in grid_attrs else None
    if frame is not None and frame != from_frame:
        yield ctx.err(
            "E503",
            location,
            f"{grid_attr} {grid_id!r} is in frame {frame!r} but the transform starts "
            f"in {from_frame!r}; the field is sampled in the source frame",
        )
    space = as_str(group.attrs.get("vector_space", "world"))
    if space not in VECTOR_SPACES:
        yield ctx.err("E502", location, f"unknown vector_space {space!r}")
    if dataset in group:
        data = group[dataset]
        kinds = as_str_tuple(grid_attrs["axis_kinds"])
        n_spatial = kinds.count("spatial")
        if int(data.shape[0]) != n_spatial:
            yield ctx.err(
                "E503",
                f"{location}/{dataset}",
                f"{data.shape[0]} components on a {n_spatial}-D lattice; they must "
                "match",
            )
        if kind == "displacement":
            shape = tuple(int(v) for v in np.atleast_1d(grid_attrs["shape"]))
            spatial = tuple(
                s for s, k in zip(shape, kinds, strict=True) if k == "spatial"
            )
            if tuple(int(v) for v in data.shape[1:]) != spatial:
                yield ctx.err(
                    "E503",
                    f"{location}/{dataset}",
                    f"field lattice {tuple(data.shape[1:])} != grid {grid_id!r} "
                    f"spatial shape {spatial}",
                )


def _check_composite(
    ctx: Context,
    name: str,
    group: h5py.Group,
    declared: dict[str, tuple[str, str]],
    source: str,
    target: str,
) -> Iterator[Diagnostic]:
    location = f"/transforms/{name}"
    if "components" not in group.attrs:
        yield ctx.err("E501", location, "kind 'composite' requires `components`")
        return
    components = as_str_tuple(group.attrs["components"])
    node = ctx.root["transforms"]
    unknown = [c for c in components if c not in node]
    if unknown:
        yield ctx.err("E501", location, f"names components {unknown} that do not exist")
        return
    frames = []
    for component in components:
        attrs = node[component].attrs
        if "from_frame" not in attrs or "to_frame" not in attrs:
            yield ctx.err(
                "E501", location, f"component {component!r} declares no frames"
            )
            return
        frames.append((as_str(attrs["from_frame"]), as_str(attrs["to_frame"])))
    if frames[0][0] != source:
        yield ctx.err(
            "E501",
            location,
            f"first component starts in {frames[0][0]!r} but the composite declares "
            f"{source!r}",
        )
    if frames[-1][1] != target:
        yield ctx.err(
            "E501",
            location,
            f"last component ends in {frames[-1][1]!r} but the composite declares "
            f"{target!r}",
        )
    for (left, right), (a, b) in zip(
        zip(components, components[1:], strict=False),
        zip(frames, frames[1:], strict=False),
        strict=False,
    ):
        if a[1] != b[0]:
            yield ctx.err(
                "E501",
                location,
                f"{left!r} ends in {a[1]!r} but {right!r} starts in {b[0]!r}",
            )
    # Units, alongside the frames.  §10.1 makes `units` a MUST, and this rule is
    # what `medh5 validate` reports -- so while `CompositeTransform.check_chain`
    # rejected an `mm` leg chained to a `um` leg, the advertised conformance
    # check still passed the same file. The two implementations have to agree
    # about what a sound chain is.
    declared_units = as_str(group.attrs["units"]) if "units" in group.attrs else None
    if declared_units is not None:
        mixed = [
            (component, as_str(node[component].attrs["units"]))
            for component in components
            if "units" in node[component].attrs
            and as_str(node[component].attrs["units"]) != declared_units
        ]
        if mixed:
            listed = ", ".join(f"{c!r} in {u!r}" for c, u in mixed)
            yield ctx.err(
                "E501",
                location,
                f"declares units {declared_units!r} but {listed} --- a chain "
                "whose legs are in different units does not compose",
            )
    del declared


def _check_inverses(
    ctx: Context, node: h5py.Group, declared: dict[str, tuple[str, str]]
) -> Iterator[Diagnostic]:
    """``inverse_id`` must name a transform that really is the inverse (E505)."""
    for name, (source, target) in sorted(declared.items()):
        attrs = node[name].attrs
        if "inverse_id" not in attrs:
            continue
        other = as_str(attrs["inverse_id"])
        location = f"/transforms/{name}"
        if other not in declared:
            yield ctx.err(
                "E505", location, f"`inverse_id` names {other!r}, which does not exist"
            )
            continue
        other_source, other_target = declared[other]
        if (other_source, other_target) != (target, source):
            yield ctx.err(
                "E505",
                location,
                f"`inverse_id` names {other!r}, which maps {other_source!r} -> "
                f"{other_target!r}; an inverse must map {target!r} -> {source!r}",
            )
            continue
        back = node[other].attrs.get("inverse_id")
        if back is not None and as_str(back) != name:
            yield ctx.err(
                "E505",
                location,
                f"{other!r} names {as_str(back)!r} as its inverse, not {name!r}; the "
                "relation must be mutual",
            )


# --------------------------------------------------------------------------
# §11-§12 curation
# --------------------------------------------------------------------------


def check_curation(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    if document is None:
        return
    provenance = document.provenance
    for activity in provenance.activities:
        if activity.type not in ACTIVITY_TYPES:
            yield ctx.err(
                "E603",
                f"/meta#provenance/activities/{activity.id}",
                f"unknown activity type {activity.type!r}",
            )
        for field_name in ("started", "ended"):
            value = getattr(activity, field_name)
            if value is not None and not RFC3339.match(value):
                yield ctx.err(
                    "E604",
                    f"/meta#provenance/activities/{activity.id}",
                    f"{field_name} {value!r} is not RFC 3339",
                )
    for activity_id, agent_id in provenance.dangling_agent_refs():
        yield ctx.err(
            "E605",
            f"/meta#provenance/activities/{activity_id}",
            f"names agent {agent_id!r}, which is not declared",
        )
    for name, group in ctx.annotation_groups.items():
        yield from _check_links(ctx, f"/annotations/{name}", group.attrs, document)
    for name, node in ctx.image_nodes.items():
        yield from _check_links(ctx, f"/images/{name}", node.attrs, document)
    if document.deidentification is None:
        yield ctx.err(
            "W903",
            "/meta#deidentification",
            "no de-identification record; tooling must treat this file as "
            "potentially identifying",
        )
    if "curation" in ctx.profiles:
        for name, group in ctx.annotation_groups.items():
            if "quality" not in group.attrs:
                yield ctx.err(
                    "E009",
                    f"/annotations/{name}",
                    "the `curation` profile requires `quality` on every annotation",
                )


def _check_links(
    ctx: Context, location: str, attrs: Any, document: SampleDocument
) -> Iterator[Diagnostic]:
    if "prov" in attrs:
        activity_id = as_str(attrs["prov"])
        if not document.provenance.has_activity(activity_id):
            yield ctx.err(
                "E601", location, f"`prov` names unknown activity {activity_id!r}"
            )
    if "quality" in attrs:
        key = as_str(attrs["quality"])
        if key not in document.quality:
            yield ctx.err("E602", location, f"`quality` names unknown record {key!r}")


def check_splits(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    if document is None:
        return
    by_set: dict[str, set[str]] = {}
    for claim in document.splits:
        if claim.manifest_sha256:
            by_set.setdefault(claim.set_id, set()).add(claim.manifest_sha256)
    for set_id, hashes in sorted(by_set.items()):
        if len(hashes) > 1:
            yield ctx.err(
                "W906",
                "/meta#splits",
                f"split set {set_id!r} is claimed against {len(hashes)} different "
                "manifests in one file",
            )


# --------------------------------------------------------------------------
# §13 integrity
# --------------------------------------------------------------------------


def check_integrity(ctx: Context) -> Iterator[Diagnostic]:
    from medh5.integrity.digest import parse_digest
    from medh5.integrity.verify import stale_index_entries, verify_root

    attr_names = ctx.notes.get("attr_names")
    result = verify_root(ctx.root, attr_names)
    if not result.checked and not result.undigested:
        return
    if result.undigested and not result.checked:
        yield ctx.err("W901", "/", "no dataset carries a `digest` attribute")
    for path in result.malformed:
        yield ctx.err("E703", f"/{path}", "malformed digest string")
    for path in result.mismatched:
        yield ctx.err("E701", f"/{path}", "digest does not match the stored data")
    if result.content_id_ok is False:
        yield ctx.err(
            "E702",
            "/",
            f"`content_id` {result.content_id_declared} does not match the computed "
            f"{result.content_id_computed}",
        )
    for name in stale_index_entries(ctx.root):
        yield ctx.err(
            "W905",
            f"/index/{name}",
            "index `source_digest` does not match its annotation; readers must "
            "ignore this entry and rebuild it",
        )
    if "digest_algo" in ctx.root.attrs:
        try:
            parse_digest(f"{as_str(ctx.root.attrs['digest_algo'])}:00")
        except Exception:  # noqa: BLE001 - reported as a diagnostic
            yield ctx.err(
                "E703",
                "/",
                f"unsupported digest_algo {as_str(ctx.root.attrs['digest_algo'])!r}",
            )


# --------------------------------------------------------------------------
# §1.3 profiles
# --------------------------------------------------------------------------


def check_profiles(ctx: Context) -> Iterator[Diagnostic]:
    document = ctx.document
    kinds = {
        as_str(g.attrs["kind"])
        for g in ctx.annotation_groups.values()
        if "kind" in g.attrs
    }
    declared = set(ctx.profiles)
    if "seg" in declared and not (kinds & set(VOXEL_KINDS) - {"mask"}):
        yield ctx.err(
            "E009", "/", "profile `seg` is declared but no voxel annotation is present"
        )
    tasks = {
        as_str(g.attrs["task"])
        for g in ctx.annotation_groups.values()
        if "task" in g.attrs
    }
    if "det" in declared and "detection" not in tasks:
        yield ctx.err(
            "E009",
            "/",
            "profile `det` is declared but no annotation declares task='detection'",
        )
    if "cls" in declared and "classification" not in kinds:
        yield ctx.err(
            "E009",
            "/",
            "profile `cls` is declared but no classification annotation is present",
        )
    if "reg" in declared and (
        "transforms" not in ctx.root or len(ctx.root["transforms"]) == 0
    ):
        yield ctx.err(
            "E009", "/", "profile `reg` is declared but no transform is present"
        )
    if "training" in declared and (
        "index" not in ctx.root or len(ctx.root["index"]) == 0
    ):
        yield ctx.err(
            "E009",
            "/",
            "profile `training` is declared but no sampling index is present",
        )
    if (
        "longitudinal" in declared
        and document is not None
        and len(document.timepoints) < 2  # noqa: PLR2004
    ):
        yield ctx.err(
            "E009",
            "/meta#timepoints",
            "profile `longitudinal` requires at least two declared timepoints",
        )


COLLECTION_RULES = (check_collection,)
"""Rules for a ``collection`` root; its members are validated as samples."""

STRUCTURAL_RULES = (
    check_container,
    check_document,
    check_geometry,
    check_images,
    check_annotations,
    check_bulk_storage,
)

SEMANTIC_RULES = (
    check_timepoints,
    check_instance_identity,
    check_transforms,
    check_label_set,
    check_multiscale,
    check_curation,
    check_splits,
    check_profiles,
    check_ontology_bindings,
)

INTEGRITY_RULES = (check_integrity,)


def rules_for(level: Level) -> tuple[Any, ...]:
    """The rules a level runs.  Each level includes every earlier level's rules."""
    if level == "structural":
        return STRUCTURAL_RULES
    if level == "semantic":
        return STRUCTURAL_RULES + SEMANTIC_RULES
    return STRUCTURAL_RULES + SEMANTIC_RULES + INTEGRITY_RULES


__all__ = [
    "COLLECTION_RULES",
    "INTEGRITY_RULES",
    "SEMANTIC_RULES",
    "STRUCTURAL_RULES",
    "Context",
    "Diagnostic",
    "Report",
    "rules_for",
]
