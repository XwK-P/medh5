"""``review`` subgroup: set / get / list / import-seg."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medh5.cli._common import Handler, iter_medh5
from medh5.core import MEDH5File
from medh5.exceptions import MEDH5Error


def _cmd_review_set(args: argparse.Namespace) -> int:
    MEDH5File.set_review_status(
        Path(args.file),
        status=args.status,
        annotator=args.annotator,
        notes=args.notes,
    )
    print(f"OK: {args.file} → {args.status}")
    return 0


def _cmd_review_get(args: argparse.Namespace) -> int:
    st = MEDH5File.get_review_status(Path(args.file))
    if args.json:
        print(
            json.dumps(
                {
                    "status": st.status,
                    "annotator": st.annotator,
                    "timestamp": st.timestamp,
                    "notes": st.notes,
                    "history": st.history,
                },
                indent=2,
            )
        )
        return 0
    print(f"status:    {st.status}")
    print(f"annotator: {st.annotator}")
    print(f"timestamp: {st.timestamp}")
    print(f"notes:     {st.notes}")
    if st.history:
        n = len(st.history)
        word = "entry" if n == 1 else "entries"
        print(f"history:   {n} prior {word}")
    return 0


def _cmd_review_list(args: argparse.Namespace) -> int:
    root = Path(args.dir)
    matches: list[Path] = []
    for path in iter_medh5(root):
        try:
            st = MEDH5File.get_review_status(path)
        except MEDH5Error:
            continue
        if args.status is None or st.status == args.status:
            matches.append(path)
    for p in matches:
        print(p)
    return 0


def _cmd_review_import_seg(args: argparse.Namespace) -> int:
    from medh5.io import import_seg_nifti

    import_seg_nifti(
        Path(args.file),
        Path(args.from_nifti),
        name=args.name,
        resample=args.resample,
        replace=args.replace,
    )
    print(f"OK: added mask '{args.name}' to {args.file}")
    return 0


HANDLERS: dict[str, Handler] = {
    "set": _cmd_review_set,
    "get": _cmd_review_get,
    "list": _cmd_review_list,
    "import-seg": _cmd_review_import_seg,
}


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    p = sub.add_parser("review", help="Review/QA workflow helpers")
    rsub = p.add_subparsers(dest="review_command")

    rs = rsub.add_parser("set", help="Record a review status entry")
    rs.add_argument("file")
    rs.add_argument(
        "--status",
        required=True,
        choices=["pending", "reviewed", "flagged", "rejected"],
    )
    rs.add_argument("--annotator", default=None)
    rs.add_argument("--notes", default=None)

    rg = rsub.add_parser("get", help="Print the current review status")
    rg.add_argument("file")
    rg.add_argument("--json", action="store_true")

    rl = rsub.add_parser("list", help="List files matching a status filter")
    rl.add_argument("dir")
    rl.add_argument("--status", default=None)

    ri = rsub.add_parser("import-seg", help="Import a (NIfTI) mask back into a file")
    ri.add_argument("file")
    ri.add_argument("--name", required=True)
    ri.add_argument("--from", dest="from_nifti", required=True)
    ri.add_argument("--resample", action="store_true")
    ri.add_argument("--replace", action="store_true")


def dispatch(cmd: str, args: argparse.Namespace) -> int | None:
    if cmd != "review":
        return None
    sub = getattr(args, "review_command", None)
    if sub in HANDLERS:
        return HANDLERS[sub](args)
    print(
        f"medh5 review: missing subcommand (choose from {', '.join(sorted(HANDLERS))})",
        file=sys.stderr,
    )
    return 2
