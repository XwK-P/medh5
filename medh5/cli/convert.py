"""``import`` / ``export`` subgroups (NIfTI, DICOM, nnU-Net v2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medh5.cli._common import Handler


def _cmd_import_nifti(args: argparse.Namespace) -> int:
    from medh5.io import from_nifti

    if not args.image:
        print("--image NAME PATH is required (one or more)", file=sys.stderr)
        return 2
    images: dict[str, str | Path] = {name: Path(path) for name, path in args.image}
    seg_d: dict[str, str | Path] = {name: Path(path) for name, path in (args.seg or [])}
    from_nifti(
        images=images,
        seg=seg_d or None,
        out_path=Path(args.output),
        label=args.label,
        label_name=args.label_name,
        compression=args.compression,
        checksum=args.checksum,
        resample_to=args.resample_to,
        interpolator=args.interpolator,
    )
    print(f"Wrote {args.output}")
    return 0


def _cmd_import_dicom(args: argparse.Namespace) -> int:
    from medh5.io import from_dicom

    from_dicom(
        Path(args.dicom_dir),
        Path(args.output),
        modality_name=args.modality,
        series_uid=args.series_uid,
        apply_modality_lut=not args.no_modality_lut,
        compression=args.compression,
        checksum=args.checksum,
    )
    print(f"Wrote {args.output}")
    return 0


def _cmd_import_nnunetv2(args: argparse.Namespace) -> int:
    from medh5.io import from_nnunetv2

    written = from_nnunetv2(
        Path(args.src),
        Path(args.output),
        include_test=not args.no_test,
        compression=args.compression,
        checksum=args.checksum,
    )
    print(
        f"Wrote {len(written['train'])} train + {len(written['test'])} test "
        f"files to {args.output}"
    )
    return 0


def _cmd_export_nifti(args: argparse.Namespace) -> int:
    from medh5.io import to_nifti

    written = to_nifti(
        Path(args.file),
        Path(args.output),
        modalities=args.modalities,
        seg=args.seg,
    )
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


def _cmd_export_nnunetv2(args: argparse.Namespace) -> int:
    from medh5.io import to_nnunetv2

    dataset_json = to_nnunetv2(
        Path(args.src),
        Path(args.output),
        dataset_name=args.dataset_name,
        file_ending=args.file_ending,
    )
    print(f"Wrote nnU-Net v2 dataset → {dataset_json.parent}")
    return 0


IMPORT_HANDLERS: dict[str, Handler] = {
    "nifti": _cmd_import_nifti,
    "dicom": _cmd_import_dicom,
    "nnunetv2": _cmd_import_nnunetv2,
}

EXPORT_HANDLERS: dict[str, Handler] = {
    "nifti": _cmd_export_nifti,
    "nnunetv2": _cmd_export_nnunetv2,
}


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("import", help="Import external formats into .medh5")
    isub = p.add_subparsers(dest="import_command")

    pn = isub.add_parser("nifti", help="Import NIfTI volumes")
    pn.add_argument(
        "--image",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Modality name and NIfTI path (repeatable)",
    )
    pn.add_argument(
        "--seg",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Segmentation name and NIfTI path (repeatable)",
    )
    pn.add_argument("-o", "--output", required=True)
    pn.add_argument("--label", default=None)
    pn.add_argument("--label-name", default=None)
    pn.add_argument(
        "--resample-to",
        default=None,
        help="Reference modality name or NIfTI path for SimpleITK resampling",
    )
    pn.add_argument(
        "--interpolator",
        default="linear",
        choices=["linear", "nearest", "bspline"],
    )
    pn.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    pn.add_argument("--checksum", action="store_true")

    pd = isub.add_parser("dicom", help="Import a DICOM series directory")
    pd.add_argument("dicom_dir")
    pd.add_argument("-o", "--output", required=True)
    pd.add_argument("--modality", default="CT")
    pd.add_argument("--series-uid", default=None)
    pd.add_argument("--no-modality-lut", action="store_true")
    pd.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    pd.add_argument("--checksum", action="store_true")

    pnu = isub.add_parser("nnunetv2", help="Import a raw nnU-Net v2 dataset folder")
    pnu.add_argument(
        "src", help="Path to Dataset###_NAME/ folder containing dataset.json"
    )
    pnu.add_argument(
        "-o", "--output", required=True, help="Output directory for .medh5 files"
    )
    pnu.add_argument(
        "--no-test",
        action="store_true",
        help="Skip imagesTs/ (test cases)",
    )
    pnu.add_argument(
        "--compression", default="balanced", choices=["fast", "balanced", "max"]
    )
    pnu.add_argument("--checksum", action="store_true")

    p = sub.add_parser("export", help="Export .medh5 to external formats")
    esub = p.add_subparsers(dest="export_command")
    pe = esub.add_parser("nifti", help="Export images and masks as NIfTI")
    pe.add_argument("file", help="Source .medh5 file")
    pe.add_argument("-o", "--output", required=True, help="Output directory")
    pe.add_argument("--modalities", nargs="*", default=None)
    pe.add_argument("--seg", nargs="*", default=None)

    pen = esub.add_parser(
        "nnunetv2", help="Export .medh5 files as a raw nnU-Net v2 dataset"
    )
    pen.add_argument(
        "src",
        help="Directory of .medh5 files (with optional imagesTr/ and imagesTs/)",
    )
    pen.add_argument(
        "-o", "--output", required=True, help="Output DatasetXXX_NAME/ folder"
    )
    pen.add_argument(
        "--dataset-name",
        default=None,
        help="Overrides 'name' field in dataset.json",
    )
    pen.add_argument("--file-ending", default=".nii.gz")


def dispatch(cmd: str, args: argparse.Namespace) -> int | None:
    if cmd == "import":
        sub = getattr(args, "import_command", None)
        if sub in IMPORT_HANDLERS:
            return IMPORT_HANDLERS[sub](args)
        print(
            f"medh5 import: missing subcommand "
            f"(choose from {', '.join(sorted(IMPORT_HANDLERS))})",
            file=sys.stderr,
        )
        return 2
    if cmd == "export":
        sub = getattr(args, "export_command", None)
        if sub in EXPORT_HANDLERS:
            return EXPORT_HANDLERS[sub](args)
        print(
            f"medh5 export: missing subcommand "
            f"(choose from {', '.join(sorted(EXPORT_HANDLERS))})",
            file=sys.stderr,
        )
        return 2
    return None
