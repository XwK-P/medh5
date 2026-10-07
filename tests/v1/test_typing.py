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

import numpy as np
import pytest

from medh5.errors import MEDH5ValidationError

ROOT = Path(__file__).resolve().parents[2]

# Every class 1.x shipped as a dataclass, by its public home, with its fields.
DATACLASSES = {
    "medh5.curation.provenance.Activity": (
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
    "medh5.curation.provenance.Agent": (
        "id",
        "type",
        "name",
        "role",
        "version",
        "qualification",
        "organization",
    ),
    "medh5.curation.quality.Agreement": ("metric", "value", "against", "per_class"),
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
    "medh5.annotations.payload.AnnotationPayload": (
        "kind",
        "datasets",
        "attrs",
        "stacked_axes",
        "class_ids",
    ),
    "medh5.curation.identity.Cohort": (
        "dataset_id",
        "site_id",
        "scanner_id",
        "group_id",
        "acquisition_protocol",
        "extra",
    ),
    "medh5.annotations.voxel.select.CostModel": (
        "labelmap",
        "layers",
        "bitmask",
        "instances",
        "probmap",
        "detail",
    ),
    "medh5.curation.identity.Deidentification": (
        "method",
        "profile",
        "date_shift_days",
        "id_mapping",
        "performed_by",
        "date",
        "burned_in_annotation_checked",
        "extra",
    ),
    "medh5.geometry.grid.Grid": (
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
    "medh5.curation.identity.Identity": (
        "sample_id",
        "subject_id",
        "sex",
        "laterality",
        "bodypart",
        "extra",
    ),
    "medh5.storage.index.IndexPayload": (
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
    "medh5.curation.quality.Issue": ("code", "severity", "class_ids", "note"),
    "medh5.labels.labelset.LabelClass": (
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
    "medh5.curation.tracking.Observation": (
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
    "medh5.labels.labelset.OntologyCode": ("system", "code", "name"),
    "medh5.annotations.voxel.select.OverlapStats": (
        "class_ids",
        "spatial_shape",
        "counts",
        "edges",
        "colouring",
        "localized",
        "n_labelled_voxels",
    ),
    "medh5.geometry.multiscale.Pyramid": (
        "levels",
        "downsample_factors",
        "downsample_method",
        "grid_levels",
    ),
    "medh5.curation.quality.QualityRecord": (
        "status",
        "confidence",
        "reviewed_by",
        "agreement",
        "issues",
        "edit_effort_s",
    ),
    "medh5.labels.labelset.Relation": ("subject", "predicate", "object"),
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
    "medh5.labels.labelset.Skeleton": ("id", "keypoints", "edges"),
    "medh5.curation.identity.SplitClaim": (
        "set_id",
        "partition",
        "fold",
        "assigned_by",
        "assigned_at",
        "manifest_sha256",
    ),
    "medh5.curation.timeline.Timepoint": (
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
    "medh5.curation.tracking.Track": (
        "instance_id",
        "class_ids",
        "observations",
        "class_key",
    ),
    "medh5.transforms.base.TransformHeader": (
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
            str(ROOT / "tests" / "v1" / "stubtest_allowlist.txt"),
            "--ignore-unused-allowlist",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
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


def test_replace_builds_a_changed_copy() -> None:
    from medh5.annotations.base import AnnotationHeader
    from medh5.curation.identity import Identity
    from medh5.geometry.grid import Grid
    from medh5.labels.labelset import LabelClass
    from medh5.transforms.base import TransformHeader

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
    from medh5.labels.labelset import LabelClass

    match LabelClass(1, "liver", "Liver"):
        case LabelClass(cid, key, name):
            assert (cid, key, name) == (1, "liver", "Liver")
        case _:  # pragma: no cover - the case above must match
            pytest.fail("LabelClass did not match its fields")


def test_a_missing_required_field_is_the_dataclass_TypeError() -> None:
    from medh5.curation.identity import Identity
    from medh5.curation.provenance import Agent

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
