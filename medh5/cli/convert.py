"""``medh5 convert`` --- the format converters, run for the native CLI.

The command line is native (``medh5-cli``): it parses ``convert``, and hands
each converter subcommand here with its arguments, keyed as the 1.x parser
named them.  The converters are Python integrations (nibabel, pydicom,
highdicom); ``migrate`` --- 0.x files --- is the engine's and never reaches this
module.

Every converter writes a conversion report, because the interesting part of an
import is not that it succeeded but what it had to decide: which encoding, which
class ids, whether a half-voxel convention changed, whether a timepoint order
was read or guessed.  ``--report FILE`` keeps it; without it the guesses and
warnings still print.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from medh5.cli._common import EXIT_ERROR, EXIT_OK, emit, fail
from medh5.errors import MEDH5Error
from medh5.io.report import ConversionReport


def run(command: str, args: Mapping[str, Any]) -> int:
    """Run converter *command* (``from-nifti``, ``to-dicom-seg``, ...) on its
    parsed arguments; the exit code.

    A handled failure --- a format error, a missing optional dependency, a
    missing file, a name the file does not have --- prints ``medh5: <why>`` and
    exits 1, as every native command does.
    """
    from medh5.cli import _what

    handler = HANDLERS.get(command)
    if handler is None:
        return fail("usage: medh5 convert COMMAND ... (see --help)")
    try:
        return handler(SimpleNamespace(**args))
    except (MEDH5Error, ImportError, FileNotFoundError) as exc:
        return fail(str(exc))
    except LookupError as exc:
        return fail(_what(exc))
    except BrokenPipeError:  # pragma: no cover - `medh5 convert ... | head`
        return EXIT_OK


def _pairs(values: list[str] | None, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in values or []:
        if "=" not in entry:
            raise MEDH5Error(f"--{what} expects NAME=VALUE, got {entry!r}")
        name, _, value = entry.partition("=")
        out[name] = value
    return out


def _finish(report: ConversionReport, args: Any) -> int:
    if getattr(args, "report", None):
        Path(args.report).write_text(
            json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8"
        )
    if getattr(args, "json", False):
        emit(report.to_json(), as_json=True)
    else:
        print(report.format(verbose=True))
    return EXIT_OK if report.ok else EXIT_ERROR


# -- NIfTI -----------------------------------------------------------------


def _from_nifti(args: Any) -> int:
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


def _to_nifti(args: Any) -> int:
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


def _from_dicom(args: Any) -> int:
    from medh5.io.dicom import from_dicom

    report = from_dicom(
        args.root,
        args.out,
        group_by=args.group_by,
        modalities=args.modalities,
        series_uids=args.series_uids,
    )
    return _finish(report, args)


def _from_dicom_seg(args: Any) -> int:
    from medh5.io.dicom_seg import from_dicom_seg

    report = from_dicom_seg(args.seg, args.sample, ann_id=args.ann_id, grid=args.grid)
    return _finish(report, args)


def _to_dicom_seg(args: Any) -> int:
    from medh5.io.dicom_seg import to_dicom_seg

    report = ConversionReport(converter="to-dicom-seg")
    to_dicom_seg(args.path, args.annotation, args.source, args.out, report=report)
    return _finish(report, args)


# -- RTSTRUCT --------------------------------------------------------------


def _from_rtstruct(args: Any) -> int:
    from medh5.io.rtstruct import from_rtstruct

    report = from_rtstruct(
        args.rtstruct,
        args.sample,
        ann_id=args.ann_id,
        grid=args.grid,
        rasterize=args.rasterize,
    )
    return _finish(report, args)


def _to_rtstruct(args: Any) -> int:
    from medh5.io.rtstruct import to_rtstruct

    report = ConversionReport(converter="to-rtstruct")
    to_rtstruct(args.path, args.annotation, args.source, args.out, report=report)
    return _finish(report, args)


# -- nnU-Net ---------------------------------------------------------------


def _from_nnunet(args: Any) -> int:
    from medh5.io.nnunetv2 import from_nnunetv2

    report = from_nnunetv2(args.root, args.out, case_ids=args.case_ids)
    return _finish(report, args)


def _to_nnunet(args: Any) -> int:
    from medh5.io.nnunetv2 import to_nnunetv2

    report = to_nnunetv2(
        args.paths,
        args.out,
        dataset_name=args.dataset_name,
        annotation=args.annotation,
    )
    return _finish(report, args)


HANDLERS: dict[str, Callable[[Any], int]] = {
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
"""Converter subcommand -> its handler."""


__all__ = ["HANDLERS", "run"]
