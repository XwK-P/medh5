"""``convert`` and ``migrate`` --- the importers and exporters (plan §7).

Every command writes a conversion report, because the interesting part of an
import is not that it succeeded but what it had to decide: which encoding, which
class ids, whether a half-voxel convention changed, whether a timepoint order
was read or guessed.  ``--report FILE`` keeps it; without it the guesses and
warnings still print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from medh5.cli._common import (
    EXIT_ERROR,
    EXIT_OK,
    add_json_flag,
    emit,
    fail,
)
from medh5.errors import MEDH5Error
from medh5.io.report import ConversionReport

GROUPING = ("subject", "study")


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    convert = sub.add_parser("convert", help="import from and export to other formats")
    group = convert.add_subparsers(dest="convert_command", metavar="COMMAND")

    nifti = group.add_parser("from-nifti", help="NIfTI volumes -> one sample")
    nifti.add_argument("out", help="the .medh5 file to write")
    nifti.add_argument(
        "--image",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="an image channel; repeatable",
    )
    nifti.add_argument(
        "--mask", action="append", metavar="NAME=PATH", help="a mask; repeatable"
    )
    nifti.add_argument(
        "--modality",
        action="append",
        metavar="NAME=CODE",
        help="NAME=CODE, the modality for an image; repeatable",
    )
    nifti.add_argument(
        "--coord-system",
        choices=("LPS", "RAS"),
        default="LPS",
        help="world coordinate system to store: LPS (default) or RAS",
    )
    nifti.add_argument(
        "--fourth-axis",
        choices=("auto", "time", "channel"),
        default="auto",
        help=(
            "what a 4-D series' extra axis is: time (cine, DCE, 4-D CT) or "
            "channel (multi-b-value DWI, multi-echo). auto reads the file and "
            "reports a guess where it cannot tell"
        ),
    )
    nifti.add_argument(
        "--assume-geometry",
        action="store_true",
        help=(
            "import a NIfTI that declares no spatial mapping (sform_code = "
            "qform_code = 0) by taking the pixdim fallback. Off by default: "
            "that grid is assumed, not measured. Recorded as a guess."
        ),
    )
    nifti.add_argument(
        "--sample-id", help="sample id to write; defaults to the output filename"
    )
    nifti.add_argument("--subject-id", help="subject id to write")
    _common(nifti)

    to_nifti = group.add_parser("to-nifti", help="one image or class -> NIfTI")
    to_nifti.add_argument("path", help="the sample to read")
    to_nifti.add_argument("image", help="the image id to export")
    to_nifti.add_argument("out", help="the .nii.gz file to write")
    to_nifti.add_argument(
        "--annotation", help="export this annotation instead of the image"
    )
    to_nifti.add_argument(
        "--class",
        dest="class_key",
        help="with --annotation, the single class to export",
    )
    to_nifti.add_argument("--stored", action="store_true", help="skip the rescale")

    dicom = group.add_parser("from-dicom", help="a DICOM tree -> samples")
    dicom.add_argument("root", help="directory tree of DICOM files")
    dicom.add_argument(
        "out", help="output file, or a directory when several samples result"
    )
    dicom.add_argument(
        "--group-by",
        choices=GROUPING,
        default="subject",
        help="one sample per subject (default) or per study",
    )
    dicom.add_argument(
        "--modality",
        action="append",
        dest="modalities",
        help="only import these modalities; repeatable",
    )
    dicom.add_argument(
        "--series",
        action="append",
        dest="series_uids",
        help="only import these SeriesInstanceUIDs; repeatable",
    )
    _common(dicom)

    seg_in = group.add_parser("from-dicom-seg", help="a DICOM SEG -> an annotation")
    seg_in.add_argument("seg", help="the DICOM SEG file to import")
    seg_in.add_argument("sample", help="the sample to add the annotation to")
    seg_in.add_argument(
        "--id", dest="ann_id", default="seg", help="id for the new annotation"
    )
    seg_in.add_argument(
        "--grid", help="grid to place the frames on; inferred when omitted"
    )
    _common(seg_in)

    seg_out = group.add_parser("to-dicom-seg", help="an annotation -> a DICOM SEG")
    seg_out.add_argument("path", help="the sample to read")
    seg_out.add_argument("annotation", help="the annotation to export")
    seg_out.add_argument("out", help="the DICOM SEG file to write")
    seg_out.add_argument(
        "--source", action="append", required=True, help="a source DICOM file"
    )
    _common(seg_out)

    rt_in = group.add_parser("from-rtstruct", help="an RTSTRUCT -> contours")
    rt_in.add_argument("rtstruct", help="the RTSTRUCT file to import")
    rt_in.add_argument("sample", help="the sample to add the contours to")
    rt_in.add_argument(
        "--id", dest="ann_id", default="contours", help="id for the new annotation"
    )
    rt_in.add_argument("--grid", help="grid the contours are measured against")
    rt_in.add_argument(
        "--rasterize",
        action="store_true",
        help="also derive a voxel annotation; the rule is recorded in provenance",
    )
    _common(rt_in)

    rt_out = group.add_parser("to-rtstruct", help="contours -> an RTSTRUCT")
    rt_out.add_argument("path", help="the sample to read")
    rt_out.add_argument("annotation", help="the contour annotation to export")
    rt_out.add_argument("out", help="the RTSTRUCT file to write")
    rt_out.add_argument(
        "--source",
        action="append",
        required=True,
        help="a source DICOM file; repeat once per slice",
    )
    _common(rt_out)

    nn_in = group.add_parser("from-nnunet", help="an nnU-Net v2 dataset -> samples")
    nn_in.add_argument("root", help="the nnU-Net v2 dataset directory")
    nn_in.add_argument("out", help="directory to write the samples into")
    nn_in.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="only import these case ids; repeatable",
    )
    _common(nn_in)

    nn_out = group.add_parser("to-nnunet", help="samples -> an nnU-Net v2 dataset")
    nn_out.add_argument("out", help="directory to write the dataset into")
    nn_out.add_argument("paths", nargs="+", help="the samples to export")
    nn_out.add_argument(
        "--dataset-name",
        default="Dataset001_medh5",
        help="nnU-Net dataset name, e.g. Dataset001_Liver",
    )
    nn_out.add_argument(
        "--annotation", default="seg", help="the annotation to export as labels"
    )
    _common(nn_out)

    migrate = sub.add_parser("migrate", help="0.x files -> 1.0 samples (Appendix B)")
    migrate.add_argument("paths", nargs="+", help="the 0.x files to convert")
    migrate.add_argument("-o", "--out", required=True, help="output directory")
    migrate.add_argument(
        "--group-by",
        choices=GROUPING,
        default="study",
        help="merge files sharing --subject-key into one sample per subject",
    )
    migrate.add_argument(
        "--subject-key",
        help="dotted path to a subject key in 0.x extra, e.g. extra.patient_id",
    )
    migrate.add_argument(
        "--label-set", help="a reviewed label-set sidecar to reuse (see --write-labels)"
    )
    migrate.add_argument(
        "--write-labels",
        metavar="FILE",
        help="mint the cohort's label set, write it for review, and stop",
    )
    _common(migrate)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", metavar="FILE", help="write the report as JSON")
    add_json_flag(parser)


def dispatch(command: str, args: argparse.Namespace) -> int | None:
    if command == "migrate":
        return _migrate(args)
    if command != "convert":
        return None
    handlers = {
        "from-nifti": _from_nifti,
        "to-nifti": _to_nifti,
        "from-dicom": _from_dicom,
        "from-dicom-seg": _from_dicom_seg,
        "to-dicom-seg": _to_dicom_seg,
        "from-rtstruct": _from_rtstruct,
        "to-rtstruct": _to_rtstruct,
        "from-nnunet": _from_nnunet,
        "to-nnunet": _to_nnunet,
    }
    handler = handlers.get(getattr(args, "convert_command", None) or "")
    if handler is None:
        return fail("usage: medh5 convert COMMAND ... (see --help)")
    try:
        return handler(args)
    except (MEDH5Error, ImportError, FileNotFoundError) as exc:
        return fail(str(exc))


def _pairs(values: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in values or []:
        if "=" not in entry:
            raise MEDH5Error(f"--{what} expects NAME=VALUE, got {entry!r}")
        name, _, value = entry.partition("=")
        out[name] = value
    return out


def _finish(report: ConversionReport, args: argparse.Namespace) -> int:
    if getattr(args, "report", None):
        Path(args.report).write_text(json.dumps(report.to_json(), indent=2) + "\n")
    if getattr(args, "json", False):
        emit(report.to_json(), as_json=True)
    else:
        print(report.format(verbose=True))
    return EXIT_OK if report.ok else EXIT_ERROR


# -- NIfTI -----------------------------------------------------------------


def _from_nifti(args: argparse.Namespace) -> int:
    from medh5.io.nifti import from_nifti

    report = from_nifti(
        _pairs(args.image, "image"),
        args.out,
        masks=_pairs(args.mask, "mask") or None,
        modalities=_pairs(args.modality, "modality") or None,
        coord_system=args.coord_system,
        fourth_axis=args.fourth_axis,
        assume_geometry=args.assume_geometry,
        sample_id=args.sample_id,
        subject_id=args.subject_id,
    )
    return _finish(report, args)


def _to_nifti(args: argparse.Namespace) -> int:
    from medh5.io.nifti import to_nifti

    written = to_nifti(
        args.path,
        args.image,
        args.out,
        physical=not args.stored,
        annotation=args.annotation,
        class_key=args.class_key,
    )
    print(written)
    return EXIT_OK


# -- DICOM -----------------------------------------------------------------


def _from_dicom(args: argparse.Namespace) -> int:
    from medh5.io.dicom import from_dicom

    report = from_dicom(
        args.root,
        args.out,
        group_by=args.group_by,
        modalities=args.modalities,
        series_uids=args.series_uids,
    )
    return _finish(report, args)


def _from_dicom_seg(args: argparse.Namespace) -> int:
    from medh5.io.dicom_seg import from_dicom_seg

    report = from_dicom_seg(args.seg, args.sample, ann_id=args.ann_id, grid=args.grid)
    return _finish(report, args)


def _to_dicom_seg(args: argparse.Namespace) -> int:
    from medh5.io.dicom_seg import to_dicom_seg

    report = ConversionReport(converter="to-dicom-seg")
    to_dicom_seg(args.path, args.annotation, args.source, args.out, report=report)
    return _finish(report, args)


# -- RTSTRUCT --------------------------------------------------------------


def _from_rtstruct(args: argparse.Namespace) -> int:
    from medh5.io.rtstruct import from_rtstruct

    report = from_rtstruct(
        args.rtstruct,
        args.sample,
        ann_id=args.ann_id,
        grid=args.grid,
        rasterize=args.rasterize,
    )
    return _finish(report, args)


def _to_rtstruct(args: argparse.Namespace) -> int:
    from medh5.io.rtstruct import to_rtstruct

    report = ConversionReport(converter="to-rtstruct")
    to_rtstruct(args.path, args.annotation, args.source, args.out, report=report)
    return _finish(report, args)


# -- nnU-Net ---------------------------------------------------------------


def _from_nnunet(args: argparse.Namespace) -> int:
    from medh5.io.nnunetv2 import from_nnunetv2

    report = from_nnunetv2(args.root, args.out, case_ids=args.case_ids)
    return _finish(report, args)


def _to_nnunet(args: argparse.Namespace) -> int:
    from medh5.io.nnunetv2 import to_nnunetv2

    report = to_nnunetv2(
        args.paths,
        args.out,
        dataset_name=args.dataset_name,
        annotation=args.annotation,
    )
    return _finish(report, args)


# -- migrate ---------------------------------------------------------------


def _migrate(args: argparse.Namespace) -> int:
    from medh5.io.legacy import (
        build_label_set,
        load_sidecar,
        migrate_paths,
        write_sidecar,
    )

    try:
        if args.write_labels:
            report = ConversionReport(converter="migrate")
            label_set = build_label_set(args.paths, report=report)
            written = write_sidecar(label_set, args.write_labels)
            print(f"{written}: {len(label_set)} classes --- review before migrating")
            return _finish(report, args)
        reviewed: Any = load_sidecar(args.label_set) if args.label_set else None
        report = migrate_paths(
            args.paths,
            args.out,
            group_by=args.group_by,
            subject_key=args.subject_key,
            label_set=reviewed,
        )
    except (MEDH5Error, FileNotFoundError) as exc:
        return fail(str(exc))
    return _finish(report, args)


__all__ = ["dispatch", "register"]
