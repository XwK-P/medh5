"""The ``medh5`` command line.

One native application over the format engine (the ``medh5-cli`` crate): the
same grammar, output and exit codes --- 0 success, 1 a handled error, 2 a usage
error --- whether it runs as the standalone ``medh5`` binary or as this
package's console script.  The commands only Python can run are handed back to
the package: the format converters (NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net),
which wrap nibabel, pydicom and highdicom, and the PyTorch dataloader
benchmark.  The standalone binary reaches them through ``python -m medh5.cli``.

Every converter writes a conversion report, because the interesting part of an
import is not that it succeeded but what it had to decide: which encoding, which
class ids, whether a half-voxel convention changed, whether a timepoint order
was read or guessed.  ``--report FILE`` keeps it; without it the guesses and
warnings still print.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from medh5 import _core
from medh5.errors import MEDH5Error

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.io.report import ConversionReport

EXIT_OK: int = _core.EXIT_OK
EXIT_ERROR: int = _core.EXIT_ERROR
EXIT_USAGE: int = _core.EXIT_USAGE


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``medh5`` on *argv* (``sys.argv[1:]`` by default); the exit code.

    ``--help`` and ``--version`` return 0 and a usage error 2, like every other
    outcome: nothing raises ``SystemExit`` but the console script itself.
    """
    args = [str(a) for a in (sys.argv[1:] if argv is None else argv)]
    return int(_core.cli_main(args, _Host()))


def command_tree() -> dict[str, Any]:
    """The grammar as data: ``{"options", "positionals", "commands"}``,
    recursively --- what documentation is checked against."""
    found: dict[str, Any] = _core.cli_command_tree()
    return found


class _Host:
    """What only the Python package can run, on behalf of the native CLI."""

    def convert(self, argv: list[str], command: str, args: dict[str, Any]) -> int:
        return _convert(command, args)

    def throughput(
        self, path: str, patch: int, workers: int, annotation: str | None
    ) -> dict[str, Any]:
        """Sustained patches/s through the real dataloader, as a measurement
        record.  Raises ``ImportError`` without PyTorch; the command line
        reports that as a skipped measurement rather than a failure."""
        from medh5.bench import throughput

        found: dict[str, Any] = throughput(
            [path], patch=patch, workers=workers, annotation=annotation
        ).to_json()
        return found


def _what(exc: LookupError) -> str:
    """The message a lookup failed with, or what was looked up.

    ``str(KeyError('x'))`` is ``"'x'"``: the key alone, in quotes.  Most of
    this package's lookups raise with a sentence that names what is available,
    which is printed as it is; a bare key is named as one.  The rule is the
    native CLI's, so a converter's error reads as an engine error does.
    """
    detail = exc.args[0] if len(exc.args) == 1 else exc
    return str(_core.cli_lookup_message(str(detail)))


def _fail(message: str) -> int:
    """Report a handled error on stderr: ``medh5: <message>``, exit code 1."""
    print(f"medh5: {message}", file=sys.stderr)
    return EXIT_ERROR


# -- the converters ``medh5 convert`` hands back to the package -------------------


def _convert(command: str, args: Mapping[str, Any]) -> int:
    """Run converter *command* (``from-nifti``, ``to-dicom-seg``, ...) on its
    parsed arguments, keyed as the 1.x parser named them; the exit code.

    A handled failure --- a format error, a missing optional dependency, a
    missing file, a name the file does not have --- prints ``medh5: <why>`` and
    exits 1, as every native command does.
    """
    handler = _HANDLERS.get(command)
    if handler is None:
        return _fail("usage: medh5 convert COMMAND ... (see --help)")
    try:
        return handler(SimpleNamespace(**args))
    except (MEDH5Error, ImportError, FileNotFoundError) as exc:
        return _fail(str(exc))
    except LookupError as exc:
        return _fail(_what(exc))
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
        print(json.dumps(report.to_json(), indent=2, default=str))
    else:
        print(report.format(verbose=True))
    return EXIT_OK if report.ok else EXIT_ERROR


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

    report = from_dicom_seg(
        args.seg,
        args.sample,
        ann_id=args.ann_id,
        grid=args.grid,
        frame_salt=args.frame_salt,
    )
    return _finish(report, args)


def _to_dicom_seg(args: Any) -> int:
    from medh5.io.dicom_seg import to_dicom_seg
    from medh5.io.report import ConversionReport

    report = ConversionReport(converter="to-dicom-seg")
    to_dicom_seg(args.path, args.annotation, args.source, args.out, report=report)
    return _finish(report, args)


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
    from medh5.io.report import ConversionReport
    from medh5.io.rtstruct import to_rtstruct

    report = ConversionReport(converter="to-rtstruct")
    to_rtstruct(
        args.path,
        args.annotation,
        args.source,
        args.out,
        frame_salt=args.frame_salt,
        report=report,
    )
    return _finish(report, args)


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
        classes=args.classes,
        unlabeled=args.unlabeled,
    )
    return _finish(report, args)


_HANDLERS: dict[str, Callable[[Any], int]] = {
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


__all__ = ["EXIT_ERROR", "EXIT_OK", "EXIT_USAGE", "command_tree", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
