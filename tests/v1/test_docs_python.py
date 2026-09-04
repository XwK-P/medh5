"""The documentation's Python examples are executed, not proofread.

``test_docs_examples`` holds the *prose* to the code: it checks that every
documented CLI flag exists and that corrected claims stay corrected.  It never
ran a line of Python, and the second audit found a page recommending
``add_segmentation(..., ignore=uncertain_mask, encoding="auto")`` as the way to
write an ignore region while the writer dropped that region under three of the
five encodings.  A one-line round-trip assertion in the example itself would
have caught it years earlier.

So every fenced ``python`` block on every page runs here, against a real sample
built by the suite's own fixture writer, with a namespace holding the names the
pages use by convention (``s``, ``w``, ``ann``, ``liver``, ``paths`` ...).  A
block whose only failure is a name it never binds is a **fragment**: the page
shows the call and leaves the setup to the prose around it, and that is the one
failure tolerated here.  Everything else --- a method that no longer exists, a
signature that changed, a key that is wrong --- is a claim the code has stopped
honouring, and fails.  A block that genuinely cannot be executed says so with an
``<!-- illustrative -->`` comment above its fence, on the page, where a reader
sees it too.
"""

from __future__ import annotations

import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import medh5
from medh5.annotations.voxel import InstanceInput
from medh5.collection import pack
from medh5.dataset.manifest import scan
from medh5.labels.labelset import LabelClass, LabelSet
from tests.v1.conftest import SHAPE, block, write_sample

DOCS = Path(__file__).resolve().parents[2] / "docs"

MARKER = "<!-- illustrative -->"
"""What a block that cannot run must carry, on the line before its fence."""

# The specification is excluded for the same reason `test_docs_examples` excludes
# it: its examples illustrate the *format*, not this package's API.
PAGES = sorted(p for p in DOCS.rglob("*.md") if "spec/medh5-1.0.md" not in str(p))

_FENCE = re.compile(r"(?P<marker>[^\n]*)\n```python\n(?P<code>.*?)```", re.S)

OPTIONAL = frozenset(
    {
        "highdicom",
        "hdf5plugin",
        "jsonschema",
        "monai",
        "nibabel",
        "pydicom",
        "scipy",
        "SimpleITK",
        "torch",
    }
)
"""Packages a page may import that a given CI job does not install."""


def _blocks() -> list[tuple[Path, int, str, bool]]:
    """``(page, line, code, marked)`` for every fenced Python block."""
    out: list[tuple[Path, int, str, bool]] = []
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        for match in _FENCE.finditer(text):
            line = text.count("\n", 0, match.start("code")) + 1
            out.append(
                (
                    page,
                    line,
                    match.group("code"),
                    MARKER in match.group("marker"),
                )
            )
    return out


BLOCKS = _blocks()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The world the documentation writes as though the reader already has.

    One directory holding the file names the pages use --- ``case.medh5``,
    ``cohort.json``, ``shard.medh5c``, ``ct.nii.gz`` --- and one sample whose
    ids are the ones they name: grid ``ct``, image ``CT``, annotations
    ``organs``, ``lesions`` and ``response``, two visits and the registration
    between them.  Blocks run with this directory as the working directory, so
    ``medh5.open("case.medh5")`` on the page is the call the test makes.
    """
    root = tmp_path_factory.mktemp("docs")
    label_set = LabelSet(
        "docs",
        version="1.0.0",
        classes=[
            LabelClass(1, "liver", "Liver", category="organ"),
            LabelClass(2, "spleen", "Spleen", category="organ"),
            LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
            LabelClass(4, "vessel", "Vessel", category="vessel"),
            LabelClass(5, "progressive", "Progressive disease", category="response"),
            LabelClass(6, "stable", "Stable disease", category="response"),
            LabelClass(7, "progressive_disease", "Progressive", category="response"),
            LabelClass(8, "non_diagnostic", "Non-diagnostic", category="quality"),
            LabelClass(9, "birads_4", "BI-RADS 4", category="birads"),
        ],
    )
    masks = {
        1: block(SHAPE, (2, 2, 2), 8),
        2: block(SHAPE, (2, 14, 2), 6),
        3: block(SHAPE, (4, 4, 4), 3),
    }
    rng = np.random.default_rng(11)
    image = rng.integers(-1000, 1500, SHAPE).astype(np.int16)
    matrix = np.eye(4)
    matrix[:3, 3] = [1.0, 0.5, -0.5]

    case = root / "case.medh5"
    with medh5.create(
        case, sample_id="case", subject_id="subj-A", codec="portable"
    ) as w:
        w.identity(sex="F", bodypart="abdomen")
        w.cohort(dataset_id="docs", site_id="site-A")
        w.add_timepoint("tp0", label="baseline", days_from_baseline=0)
        w.add_timepoint("tp1", label="fu1", days_from_baseline=92)
        w.label_set(label_set)
        w.software("medh5", medh5.__version__)
        rad = w.person("RAD-07", role="annotator")
        act = w.activity("annotate", agent=rad, tool="docs")
        for suffix, tp, frame in (
            ("", "tp0", "a"),
            ("_tp0", "tp0", "a"),
            ("_tp1", "tp1", "b"),
        ):
            w.add_grid(
                f"ct{suffix}",
                shape=SHAPE,
                spacing=(1.5, 0.8, 0.8),
                origin=(-12.0, -9.6, -9.6),
                timepoint=tp,
                frame_uid=f"pseudo:frame-{frame}",
                patch_hint=(8, 8, 8),
            )
            w.add_image(
                f"CT{suffix}",
                image,
                grid=f"ct{suffix}",
                modality="CT",
                value_type="quantitative",
                value_units="HU",
                rescale_slope=1.0,
                rescale_intercept=-1024.0,
                prov=act,
            )
            w.add_segmentation(
                f"organs{suffix}",
                grid=f"ct{suffix}",
                masks=masks,
                annotated_classes="all",
                prov=act,
                quality={"status": "approved", "reviewed_by": [rad.id]},
            )
        w.add_segmentation(
            "objects",
            grid="ct",
            instances=[
                InstanceInput(class_id=3, instance_id=7, mask=masks[3], score=0.91)
            ],
            prov=act,
        )
        w.add_segmentation(
            "vessels", grid="ct", masks={4: block(SHAPE, (6, 6, 6), 3)}, prov=act
        )
        # The pages that compare two readings of the same thing name them like
        # this: a prediction against a reference, and two raters.
        for name in ("organs_pred", "organs_a", "organs_b", "pred", "truth"):
            w.add_segmentation(name, grid="ct", masks=masks, prov=act)
        # `lesions` is what the detection pages name their *boxes* annotation.
        w.add_boxes(
            "lesions",
            np.array([[[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]], dtype=np.float32),
            class_ids=["lesion"],
            grid="ct",
            space="index",
            scores=[0.91],
            instance_ids=[7],
            prov=act,
        )
        w.add_classification(
            "response",
            {"progressive": 1.0},
            scope="sample",
            timepoints=["tp0", "tp1"],
        )
        w.add_grid(
            "pet",
            shape=SHAPE,
            spacing=(1.5, 0.8, 0.8),
            timepoint="tp0",
            frame_uid="pseudo:frame-a",
        )
        w.add_image("PET", image, grid="pet", modality="PT", prov=act)
        w.add_transform(
            "tp0_to_tp1",
            kind="affine",
            from_frame="pseudo:frame-a",
            to_frame="pseudo:frame-b",
            matrix=matrix,
            invertible=True,
            prov=act,
        )
        w.split(set_id="cv5", partition="train", fold=1)
        w.deidentification(method="dicom-psi-profile", date_shift_days=-117)
        w.build_index(max_coords=64)

    other = write_sample(
        root / "case2.medh5", label_set=label_set, masks=masks, sample_id="case2"
    )
    pack([case, other], root / "shard.medh5c", keys=["case_0001", "case_0002"])

    # The tutorial's own file name, so its opening example runs as written.
    shutil.copyfile(case, root / "case_0001.medh5")
    manifest, _ = scan(root)
    manifest.save(root / "cohort.json")

    nifti = None
    try:
        import nibabel as nib

        affine = np.diag([0.8, 0.9, 1.5, 1.0])
        nib.save(nib.Nifti1Image(image, affine), str(root / "ct.nii.gz"))
        nib.save(
            nib.Nifti1Image(masks[1].astype(np.uint8), affine),
            str(root / "liver.nii.gz"),
        )
        nib.save(
            nib.Nifti1Image(masks[3].astype(np.uint8), affine),
            str(root / "lesion.nii.gz"),
        )
        nifti = str(root / "ct.nii.gz")
    except ImportError:  # pragma: no cover - nibabel is a test dependency
        pass

    return {
        "root": root,
        "case": case,
        "label_set": label_set,
        "masks": masks,
        "manifest": manifest,
        "paths": [str(case), str(other)],
        "nifti": nifti,
    }


def _torch_names(workspace: dict[str, Any]) -> dict[str, Any]:
    """The loader names, or nothing where torch is not installed."""
    try:
        import torch
        from torch.utils.data import DataLoader

        from medh5.torch import PairedPatchDataset, PatchDataset, VolumeDataset
    except ImportError:
        return {}
    from medh5.sampling import PatchSampler

    case = str(workspace["case"])
    volumes = VolumeDataset([case], images=["CT"])
    patches = PatchDataset(
        [case],
        PatchSampler((8, 8, 8), strategy="foreground"),
        images=["CT"],
        annotations={"organs": ["liver", "lesion"]},
        annotation="organs",
    )
    return {
        "torch": torch,
        "DataLoader": DataLoader,
        "PatchDataset": PatchDataset,
        "PairedPatchDataset": PairedPatchDataset,
        "VolumeDataset": VolumeDataset,
        "dataset": volumes,
        "ds": volumes,
        # `annotations=` is what puts `item["label"]` there; a tutorial reads it.
        "item": patches[0],
        # `patches` on the performance page is a list of patch *meta* dicts, and
        # `p["used_index"]` is the line that says so.
        "patches": [patches[i]["meta"]["patch"] for i in range(len(patches))],
        "batch": next(iter(DataLoader(patches, batch_size=2))),
    }


def _namespace(workspace: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """The names a documentation example may assume are already bound.

    Deliberately generous.  The alternative is a page that spells out five
    lines of setup before the one line it is about, which is worse
    documentation --- so the runner supplies the setup instead, under the names
    the pages already use.
    """

    from medh5.dataset import Manifest, compute_stats
    from medh5.sampling import PatchSampler, TimepointPairSampler

    sample = medh5.open(workspace["case"])
    # A live writer, because half the pages show one call of one.  It carries
    # the two visits and the label set the examples assume and *no* grids: the
    # blocks declare their own, and pre-declaring them made every such block
    # fail with "already declared".
    writer = medh5.create(tmp_path / "scratch.medh5", sample_id="scratch")
    writer.add_timepoint("tp0", label="baseline", days_from_baseline=0)
    writer.add_timepoint("tp1", label="fu1", days_from_baseline=92)
    writer.label_set(workspace["label_set"])
    for name, tp in (("ct", "tp0"), ("ct_tp0", "tp0"), ("ct_tp1", "tp1")):
        writer.add_grid(
            name,
            shape=SHAPE,
            spacing=(1.5, 0.8, 0.8),
            timepoint=tp,
            frame_uid=f"pseudo:frame-{tp}",
        )

    annotation = sample.annotations["organs"]
    grid = sample.grids["ct"]
    image = sample.images["CT"]
    uncertain = np.zeros(SHAPE, dtype=bool)
    uncertain[14:, :, :] = True
    return {
        "medh5": medh5,
        "np": np,
        "numpy": np,
        "Path": Path,
        "Manifest": Manifest,
        "compute_stats": compute_stats,
        "PatchSampler": PatchSampler,
        "TimepointPairSampler": TimepointPairSampler,
        # The loader names, where the job installs torch.  Without them a block
        # that needs one raises NameError (a fragment) or ImportError (skipped),
        # which is what the minimum-dependency and Windows jobs should see ---
        # importing torch to *build* the namespace failed every block instead.
        **_torch_names(workspace),
        # The sample, under every name the pages give it.
        "s": sample,
        "sample": sample,
        "opened": sample,
        "w": writer,
        "path": "case.medh5",
        "paths": workspace["paths"],
        "out": str(tmp_path / "out.medh5"),
        "root": str(workspace["root"]),
        # Its objects.
        "ann": annotation,
        "annotation": annotation,
        "organs": annotation,
        "lesions": sample.annotations["lesions"],
        "b": sample.annotations["lesions"],
        "boxes": np.array([[[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]], dtype=np.float32),
        "c": sample.annotations["response"],
        "objects": sample.annotations["objects"],
        "g": grid,
        "grid": grid,
        "img": image,
        "image": image,
        "t": sample.transforms["tp0_to_tp1"],
        # `index` is a voxel index on these pages, not the sampling index ---
        # `ann.contains(class_id, index)` is the line that says so.
        "index": (4, 4, 4),
        "sampling_index": sample.index["organs"],
        "tracking": sample.tracks(),
        "manifest": workspace["manifest"],
        "e": next(iter(workspace["manifest"])),
        "label_set": workspace["label_set"],
        "labels": workspace["label_set"],
        "masks": workspace["masks"],
        "liver": workspace["masks"][1],
        "spleen": workspace["masks"][2],
        "lesion": workspace["masks"][3],
        "uncertain_mask": uncertain,
        "array": image.read(),
        "roi": (slice(0, 8),) * 3,
        "class_id": 1,
        "instance_id": 7,
        "class_key": "liver",
        "ANN": "organs",
        "SALT": "docs-salt",
        "frame_a": "pseudo:frame-a",
        "frame_b": "pseudo:frame-b",
        "z": 4,
        "y": 4,
        "x": 4,
        "tmp_path": tmp_path,
        "_writer": writer,
    }


def _runnable() -> list[tuple[Path, int, str]]:
    return [(p, ln, code) for p, ln, code, marked in BLOCKS if not marked]


@pytest.mark.parametrize(
    ("page", "line", "code"),
    _runnable(),
    ids=lambda v: f"{v.name}" if isinstance(v, Path) else str(v)[:20],
)
def test_documented_python_runs(
    page: Path, line: int, code: str, workspace: dict[str, Any], tmp_path: Path
) -> None:
    """Every unmarked Python block executes against a real sample.

    The failure message names the page and line, because that is what a writer
    needs: the block is the unit of documentation, and one that no longer runs
    is a claim the code has stopped honouring.
    """
    # A private copy per block, because the examples write: several pages
    # create `case.medh5`, and one block clobbering the fixture for the next is
    # a test order dependency rather than a documentation problem.
    scratch = tmp_path / "docs"
    shutil.copytree(workspace["root"], scratch)
    namespace = _namespace(
        {**workspace, "root": scratch, "case": scratch / "case.medh5"}, tmp_path
    )
    here = Path.cwd()
    os.chdir(scratch)
    try:
        with warnings.catch_warnings():
            # `labelmap()` warns when it flattens overlap, which several pages
            # demonstrate on purpose.
            warnings.simplefilter("ignore")
            exec(compile(code, f"{page}:{line}", "exec"), namespace)
    except ImportError as exc:
        # An optional dependency this job does not install.  `require()` names
        # the extra, and `pip install 'medh5[...]'` is in the message either
        # way; the rest of the suite reaches the same conclusion with
        # `importorskip`.  A name removed from *this* package raises
        # `ImportError` too, so only the optional ones are skipped.
        named = re.search(r"No module named '([^'.]+)", str(exc))
        root = str(getattr(exc, "name", "") or "").split(".")[0]
        missing = root or (named.group(1) if named else "")
        if "pip install 'medh5[" in str(exc) or missing in OPTIONAL:
            pytest.skip(f"{page.relative_to(DOCS)}:{line}: {exc}")
        raise AssertionError(
            f"{page.relative_to(DOCS)}:{line} does not run: {exc}\n---\n{code}"
        ) from exc
    except SystemExit:
        # A page that ends in `raise SystemExit(...)` is showing a pipeline
        # gate; running it is the point, exiting the test run is not.
        pass
    except NameError as exc:
        # A fragment: the page shows the call and leaves its setup to the prose
        # around it, so a name it never binds is missing.  That is the one
        # failure tolerated here, and it is bounded --- a name the *package*
        # used to export raises ImportError or AttributeError instead, so a
        # renamed or deleted API still fails.
        missing = re.search(r"name '([^']+)'", str(exc))
        if missing and hasattr(medh5, missing.group(1)):
            pytest.fail(f"{page.relative_to(DOCS)}:{line}: {exc}\n---\n{code}")
    except Exception as exc:  # pragma: no cover - the message is the point
        raise AssertionError(
            f"{page.relative_to(DOCS)}:{line} does not run: "
            f"{type(exc).__name__}: {exc}\n"
            f"Fix the example, or mark it `{MARKER}` on the line above its fence "
            "if it cannot be executed here.\n---\n" + code
        ) from exc
    finally:
        os.chdir(here)
        namespace["s"].close()
        namespace["_writer"].abort()


def test_the_ignore_example_round_trips(workspace: dict[str, Any], tmp_path: Path):
    """The example that started this file, asserted rather than shown.

    `add_segmentation(..., ignore=..., encoding="auto")` is what the API page
    recommends; under the encoding measurement happens to pick, the region has
    to come back.
    """
    uncertain = np.zeros(SHAPE, dtype=bool)
    uncertain[14:, :, :] = True
    path = tmp_path / "ignore.medh5"
    with medh5.create(path, sample_id="x", codec="portable") as w:
        w.label_set(workspace["label_set"])
        w.add_grid("ct", shape=SHAPE, spacing=(1.5, 0.8, 0.8))
        w.add_image("CT", np.zeros(SHAPE, np.int16), grid="ct", modality="CT")
        w.add_segmentation(
            "organs",
            grid="ct",
            masks=workspace["masks"],
            encoding="auto",
            annotated_classes=["liver", "spleen", "lesion"],
            ignore=uncertain,
        )
    with medh5.open(path) as sample:
        annotation = sample.annotations["organs"]
        assert annotation.has_ignore_region
        referenced = annotation.header.ignore_mask
        region = (
            sample.annotations[referenced].read()
            if referenced
            else annotation.ignore_mask()
        )
        assert np.array_equal(region, uncertain)


def test_every_illustrative_block_says_why_it_cannot_run() -> None:
    """The marker is a claim too, so it is bounded.

    A page allowed to mark every block would pass this file while documenting
    nothing that works.  The cap is generous and the point is the direction of
    travel: a new example runs unless there is a reason it cannot.
    """
    marked = [(p, ln) for p, ln, _, is_marked in BLOCKS if is_marked]
    assert len(marked) < len(BLOCKS) / 2, (
        f"{len(marked)} of {len(BLOCKS)} Python blocks are marked "
        f"`{MARKER}`; the documentation should mostly be executable"
    )
