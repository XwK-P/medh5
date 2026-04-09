"""Command-line interface for medh5.

Usage::

    medh5 info sample.medh5
    medh5 validate sample.medh5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medh5.core import MEDH5File
from medh5.exceptions import MEDH5Error


def _cmd_info(args: argparse.Namespace) -> int:
    """Print metadata summary for a .medh5 file."""
    path = Path(args.file)
    try:
        with MEDH5File(path) as f:
            meta = f.meta
            img_grp = f.images
            first_key = next(iter(img_grp))
            first_ds = img_grp[first_key]
            shape = first_ds.shape
            dtype = first_ds.dtype

        print(f"File:         {path}")
        print(f"Schema:       v{meta.schema_version}")
        print(f"Images:       {meta.image_names}")
        print(f"Shape:        {shape}")
        print(f"Dtype:        {dtype}")
        if meta.label is not None:
            label_str = repr(meta.label)
            if meta.label_name:
                label_str += f" ({meta.label_name})"
            print(f"Label:        {label_str}")
        if meta.has_seg:
            print(f"Seg masks:    {meta.seg_names}")
        if meta.has_bbox:
            print("Bboxes:       yes")
        if meta.spatial.spacing is not None:
            print(f"Spacing:      {meta.spatial.spacing}")
        if meta.spatial.origin is not None:
            print(f"Origin:       {meta.spatial.origin}")
        if meta.spatial.coord_system is not None:
            print(f"Coord system: {meta.spatial.coord_system}")
        if meta.patch_size is not None:
            print(f"Patch size:   {meta.patch_size}")
        if meta.extra is not None:
            print(f"Extra:        {meta.extra}")
    except MEDH5Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a .medh5 file structure and schema."""
    path = Path(args.file)
    errors: list[str] = []

    try:
        with MEDH5File(path) as f:
            h5 = f.h5

            if "images" not in h5:
                errors.append("Missing required 'images' group")
            else:
                img_grp = h5["images"]
                if len(img_grp) == 0:
                    errors.append("'images' group is empty")
                else:
                    shapes = {name: img_grp[name].shape for name in img_grp}
                    unique_shapes = set(shapes.values())
                    if len(unique_shapes) > 1:
                        errors.append(f"Image shape mismatch: {shapes}")

            if "schema_version" not in h5.attrs:
                errors.append("Missing 'schema_version' attribute")

            meta = f.meta

            if meta.has_seg and "seg" not in h5:
                errors.append("has_seg=True but no 'seg' group found")
            if meta.has_bbox and "bboxes" not in h5:
                errors.append("has_bbox=True but no 'bboxes' dataset found")

    except MEDH5Error as exc:
        errors.append(str(exc))

    if errors:
        print(f"INVALID: {path}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"VALID: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``medh5`` CLI."""
    parser = argparse.ArgumentParser(
        prog="medh5",
        description="medh5 file utility",
    )
    sub = parser.add_subparsers(dest="command")

    info_p = sub.add_parser("info", help="Print metadata summary")
    info_p.add_argument("file", help="Path to .medh5 file")

    validate_p = sub.add_parser("validate", help="Validate file structure")
    validate_p.add_argument("file", help="Path to .medh5 file")

    args = parser.parse_args(argv)

    if args.command == "info":
        return _cmd_info(args)
    elif args.command == "validate":
        return _cmd_validate(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
