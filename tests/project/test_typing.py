"""The engine's Python face is typed, and the types are the 1.x API's.

``medh5/_core.pyi`` types the compiled extension; ``mypy --strict`` checks the
package against it in CI.  This module holds the stub to the module actually
built --- every name, every parameter --- with mypy's own ``stubtest``, and
holds the value classes to the dataclass protocol 1.x users relied on.
"""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from medh5.errors import MEDH5ValidationError

# Every class 1.x shipped as a dataclass, by its public home, with its fields.
DATACLASSES = {
    "medh5.curation.Activity": (
        "id",
        "type",
        "agent",
        "started",
        "ended",
        "tool",
        "inputs",
        "outputs",
        "params",
    ),
    "medh5.curation.Agent": (
        "id",
        "type",
        "name",
        "role",
        "version",
        "qualification",
        "organization",
    ),
    "medh5.curation.Agreement": ("metric", "value", "against", "per_class"),
    "medh5.annotations.base.AnnotationHeader": (
        "kind",
        "task",
        "grid",
        "timepoints",
        "space",
        "frame_uid",
        "class_ids",
        "annotated_class_ids",
        "closure",
        "ignore_id",
        "ignore_mask",
        "prov",
        "quality",
        "derived_from",
        "extra",
    ),
    "medh5.annotations.base.AnnotationPayload": (
        "kind",
        "datasets",
        "attrs",
        "stacked_axes",
        "class_ids",
    ),
    "medh5.curation.Cohort": (
        "dataset_id",
        "site_id",
        "scanner_id",
        "group_id",
        "acquisition_protocol",
        "extra",
    ),
    "medh5.annotations.voxel.CostModel": (
        "labelmap",
        "layers",
        "bitmask",
        "instances",
        "probmap",
        "detail",
    ),
    "medh5.curation.Deidentification": (
        "method",
        "profile",
        "date_shift_days",
        "id_mapping",
        "performed_by",
        "date",
        "burned_in_annotation_checked",
        "extra",
    ),
    "medh5.geometry.Grid": (
        "grid_id",
        "shape",
        "axis_names",
        "axis_kinds",
        "spacing",
        "origin",
        "direction",
        "coord_system",
        "units",
        "timepoint",
        "frame_uid",
        "time_values",
        "time_units",
        "chunk_hint",
        "patch_hint",
        "extra",
    ),
    "medh5.curation.Identity": (
        "sample_id",
        "subject_id",
        "sex",
        "laterality",
        "bodypart",
        "extra",
    ),
    "medh5.storage.IndexPayload": (
        "ann_id",
        "class_ids",
        "voxel_counts",
        "class_bboxes",
        "fg_coords",
        "occupancy",
        "source_digest",
        "max_coords",
        "seed",
        "stats",
    ),
    "medh5.curation.Issue": ("code", "severity", "class_ids", "note"),
    "medh5.labels.LabelClass": (
        "id",
        "key",
        "name",
        "parents",
        "category",
        "color",
        "codes",
        "laterality",
        "properties",
    ),
    "medh5.curation.Observation": (
        "timepoint",
        "annotation",
        "index",
        "instance_id",
        "class_id",
        "box",
        "voxel_count",
        "volume",
        "units",
        "score",
        "grid",
    ),
    "medh5.labels.OntologyCode": ("system", "code", "name"),
    "medh5.annotations.voxel.OverlapStats": (
        "class_ids",
        "spatial_shape",
        "counts",
        "edges",
        "colouring",
        "localized",
        "n_labelled_voxels",
    ),
    "medh5.geometry.Pyramid": (
        "levels",
        "downsample_factors",
        "downsample_method",
        "grid_levels",
    ),
    "medh5.curation.QualityRecord": (
        "status",
        "confidence",
        "reviewed_by",
        "agreement",
        "issues",
        "edit_effort_s",
    ),
    "medh5.labels.Relation": ("subject", "predicate", "object"),
    "medh5.document.SampleDocument": (
        "identity",
        "timepoints",
        "cohort",
        "label_set",
        "provenance",
        "quality",
        "splits",
        "acquisition",
        "deidentification",
        "extra",
    ),
    "medh5.labels.Skeleton": ("id", "keypoints", "edges"),
    "medh5.curation.SplitClaim": (
        "set_id",
        "partition",
        "fold",
        "assigned_by",
        "assigned_at",
        "manifest_sha256",
    ),
    "medh5.curation.Timepoint": (
        "id",
        "index",
        "label",
        "date",
        "days_from_baseline",
        "study_uid",
        "series_uids",
        "subject_age_years",
        "description",
    ),
    "medh5.curation.Track": (
        "instance_id",
        "class_ids",
        "observations",
        "class_key",
    ),
    "medh5.transforms.TransformHeader": (
        "kind",
        "from_frame",
        "to_frame",
        "units",
        "from_grid",
        "to_grid",
        "invertible",
        "inverse_id",
        "prov",
        "metrics",
        "extra",
    ),
}


def _public(dotted: str) -> type:
    module, _, name = dotted.rpartition(".")
    found: type = getattr(importlib.import_module(module), name)
    return found


def test_the_stub_matches_the_extension(tmp_path: Path) -> None:
    pytest.importorskip("mypy.stubtest")  # the dev extra installs it
    config = tmp_path / "stubtest.ini"
    config.write_text("[mypy]\nignore_missing_imports = True\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy.stubtest",
            "medh5._core",
            "--mypy-config-file",
            str(config),
            "--allowlist",
            str(Path(__file__).with_name("stubtest_allowlist.txt")),
            "--ignore-unused-allowlist",
        ],
        capture_output=True,
        text=True,
        # Not the checkout: `-m` imports from the working directory first, and
        # the module to check is the one installed --- an editable build or a
        # wheel --- not the source tree beside it.
        cwd=tmp_path,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-2000:]


@pytest.mark.parametrize("dotted", sorted(DATACLASSES))
def test_a_1x_dataclass_is_still_a_dataclass(dotted: str) -> None:
    cls = _public(dotted)
    fields = DATACLASSES[dotted]
    assert dataclasses.is_dataclass(cls)
    assert tuple(f.name for f in dataclasses.fields(cls)) == fields
    assert cls.__match_args__ == fields


def test_every_engine_class_is_found_where_its_module_says() -> None:
    """``pickle`` finds a class by ``__module__`` and ``__qualname__``, so the
    module each binding names (``#[pyclass(module = ...)]``) must export it."""
    import inspect

    from medh5 import _core

    for name, cls in vars(_core).items():
        if inspect.isclass(cls) and cls.__module__ != "medh5._core":
            module = importlib.import_module(cls.__module__)
            assert getattr(module, cls.__qualname__, None) is cls, name


def test_replace_builds_a_changed_copy() -> None:
    from medh5.annotations.base import AnnotationHeader
    from medh5.curation import Identity
    from medh5.geometry import Grid
    from medh5.labels import LabelClass
    from medh5.transforms import TransformHeader

    liver = LabelClass(1, "liver", "Liver")
    assert dataclasses.replace(liver, name="Hepar").name == "Hepar"
    assert liver.__replace__(color=(1, 2, 3, 255)).color == (1, 2, 3, 255)
    assert dataclasses.asdict(liver)["key"] == "liver"
    ident = Identity("s1", "p1")
    assert ident.__replace__(subject_id="p2") == Identity("s1", "p2")
    header = TransformHeader("affine", "a", "b")
    assert dataclasses.replace(header, units="m").units == "m"
    ann = AnnotationHeader("labelmap", "segmentation")
    assert dataclasses.replace(ann, grid="g").grid == "g"
    grid = Grid(
        "g",
        (2, 2, 2),
        ("i", "j", "k"),
        ("spatial",) * 3,
        (1.0,) * 3,
        (0.0,) * 3,
        np.eye(3),
    )
    moved = dataclasses.replace(grid, origin=(1.0, 2.0, 3.0))
    assert moved.origin == (1.0, 2.0, 3.0) and moved.grid_id == "g"
    # A replacement is validated like any construction.
    with pytest.raises(MEDH5ValidationError):
        dataclasses.replace(liver, color=(1, 2, 3))


def test_match_reads_the_fields_in_order() -> None:
    from medh5.labels import LabelClass

    match LabelClass(1, "liver", "Liver"):
        case LabelClass(cid, key, name):
            assert (cid, key, name) == (1, "liver", "Liver")
        case _:  # pragma: no cover - the case above must match
            pytest.fail("LabelClass did not match its fields")


def test_a_missing_required_field_is_the_dataclass_TypeError() -> None:
    from medh5.curation import Agent, Identity

    with pytest.raises(
        TypeError, match=r"missing 1 required positional argument: 'subject_id'$"
    ):
        Identity("s1")
    with pytest.raises(
        TypeError, match=r"missing 2 required positional arguments: 'type' and 'name'$"
    ):
        Agent("a")
    with pytest.raises(
        TypeError,
        match=r"missing 3 required positional arguments: 'id', 'type', and 'name'$",
    ):
        Agent()


def test_too_many_arguments_is_the_dataclass_TypeError() -> None:
    """Counted as a dataclass counts them: ``self`` and the required fields."""
    from medh5.curation import Agent, Identity

    with pytest.raises(
        TypeError,
        match=r"^Identity.__init__\(\) takes from 3 to 7 positional arguments "
        r"but 8 were given$",
    ):
        Identity(*map(str, range(7)))
    with pytest.raises(TypeError, match=r"takes from 4 to 8 positional arguments"):
        Agent(*map(str, range(8)))


def test_a_record_signature_names_its_fields() -> None:
    """``help()`` and ``inspect`` see the fields, not ``(*args, **kwargs)``."""
    import inspect

    from medh5.curation import Identity, Issue

    records = (
        "Activity",
        "Agent",
        "Agreement",
        "Cohort",
        "Deidentification",
        "Identity",
        "Issue",
        "QualityRecord",
        "SplitClaim",
        "Timepoint",
    )
    for name in records:
        dotted = f"medh5.curation.{name}"
        parameters = inspect.signature(_public(dotted)).parameters
        assert tuple(parameters) == DATACLASSES[dotted], dotted
    identity = inspect.signature(Identity).parameters
    empty = inspect.Parameter.empty
    assert [n for n, p in identity.items() if p.default is empty] == [
        "sample_id",
        "subject_id",
    ]
    assert identity["sex"].default is None
    assert inspect.signature(Issue).parameters["severity"].default == "info"


def _former_dataclasses(tmp_path: Path, label_set: Any, masks: Any) -> dict[str, Any]:
    """One of each class 1.x shipped as a dataclass that ``__reduce__``
    rebuilds from its fields: what the engine builds, and the headers."""
    import medh5
    from medh5.annotations.base import AnnotationHeader
    from medh5.annotations.voxel import analyse, cost_model, encode_labelmap
    from medh5.geometry import Pyramid
    from medh5.storage import build_index
    from medh5.transforms import TransformHeader
    from tests.helpers import write_sample, write_series

    payload = encode_labelmap({1: masks[1], 2: masks[2]})
    payload.attrs["custom"] = 1  # an edit, which a copy keeps
    stats = analyse(masks)
    indexed = write_sample(tmp_path / "case.medh5", label_set=label_set, masks=masks)
    with medh5.open(indexed) as sample:
        index = build_index(sample.annotations["organs_tp0"])
    with medh5.open(write_series(tmp_path / "series.medh5", label_set)) as sample:
        tracking = sample.tracks()
    return {
        "AnnotationPayload": payload,
        "AnnotationHeader": AnnotationHeader("labelmap", "segmentation", grid="g"),
        "OverlapStats": stats,
        "CostModel": cost_model(stats),
        "TransformHeader": TransformHeader("affine", "a", "b", units="m"),
        "IndexPayload": index,
        "Pyramid": Pyramid(
            levels=2,
            downsample_factors=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            downsample_method="mean",
            grid_levels=("l0", "l1"),
        ),
        "Observation": tracking[7].observations[0],
        "Track": tracking[7],
        "Tracking": tracking,
    }


def _same(a: Any, b: Any) -> bool:
    """Field by field, arrays by value (``==`` on a dataclass of arrays is
    ambiguous, and ``Tracking`` is a mapping without one)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    if dataclasses.is_dataclass(a):
        names = [f.name for f in dataclasses.fields(a)]
        return all(_same(getattr(a, n), getattr(b, n)) for n in names)
    if hasattr(a, "to_json"):
        return bool(a.to_json() == b.to_json())
    return bool(a == b)


def test_every_former_dataclass_pickles_and_copies(
    tmp_path: Path, label_set: Any, masks: Any
) -> None:
    """1.x's dataclasses pickled and copied, so worker processes and caches
    could hold them; each rebuilds from its fields."""
    import copy
    import pickle

    values = _former_dataclasses(tmp_path, label_set, masks)
    assert len(values) == 10
    for name, value in values.items():
        for copied in (
            pickle.loads(pickle.dumps(value)),
            copy.copy(value),
            copy.deepcopy(value),
        ):
            assert _same(copied, value), name
            assert repr(copied) == repr(value), name
    payload = pickle.loads(pickle.dumps(values["AnnotationPayload"]))
    assert payload.attrs["custom"] == 1
    assert pickle.loads(pickle.dumps(values["Observation"])).kind == "instances"


def test_no_public_class_is_named_after_the_extension() -> None:
    """``pickle`` and ``help()`` name a class by ``__module__``: a public
    class says where it is exported, never ``medh5._core``."""
    import inspect
    import pkgutil

    import medh5
    from medh5 import _core

    engine = {id(c) for c in vars(_core).values() if inspect.isclass(c)}
    for info in pkgutil.walk_packages(medh5.__path__, "medh5."):
        if "._" in info.name or info.name.startswith(("medh5.torch", "medh5.monai")):
            continue
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # pragma: no cover - an optional dependency
            continue
        for name in getattr(module, "__all__", ()):
            found = getattr(module, name, None)
            if inspect.isclass(found) and id(found) in engine:
                assert found.__module__ != "medh5._core", f"{info.name}.{name}"
    for cls in (_core.Attrs, _core.Dataset, _core.Group):
        assert cls.__module__ == "medh5.nodes"
    assert _core.SamplingIndex.__module__ == "medh5.storage"
