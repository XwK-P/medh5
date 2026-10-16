"""Record what medh5 2.1.0's NIfTI converter did with every case.

Run under the released 2.1.0, in a virtual environment of its own, from the
repository root::

    python -m venv /tmp/medh5-2.1.0
    /tmp/medh5-2.1.0/bin/pip install "medh5[nifti]==2.1.0" h5py
    /tmp/medh5-2.1.0/bin/python tests/data/nifti-parity/record.py

It refuses to run under any other version: the records are the contract the
native converter is held to (2.2.0 decision E3), so they must be 2.1.0's.

Each case in ``cases.json`` runs in a directory of its own, and its record,
``records/<id>.json``, holds what came out:

- ``outcome``: ``ok``, or ``refused`` with the exception's type, ``code`` and
  text;
- for an import, the sample: every object's attributes (with its digest),
  every dataset's type and shape, ``/meta`` as JSON, and the root's
  attributes;
- the conversion report's JSON;
- for an export, each file written: a NIfTI file's header fields, as stored,
  and a digest of its voxel bytes; a sidecar's text;
- for ``read_nifti``, the geometry it returned and a digest of the array;
- the files the case's directory holds afterwards.

Paths are written relative to the corpus (``{corpus}/inputs/...``) and to the
case's directory (``{out}/...``), so a record does not depend on where it was
made.  Floats that JSON has no spelling for are written as strings.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import nibabel as nib
import numpy as np

import medh5
from medh5.io import nifti
from medh5.io.report import ConversionReport

HERE = Path(__file__).resolve().parent
RECORDS = HERE / "records"
VERSION = "2.1.0"


def plain(value: Any) -> Any:
    """*value* as JSON: arrays as lists, bytes as text, non-finite floats as
    strings."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", "replace")
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    return value


def relative(value: Any, out: Path) -> Any:
    """Every path in *value* relative to the corpus and the case's directory."""
    if isinstance(value, dict):
        return {k: relative(v, out) for k, v in value.items()}
    if isinstance(value, list):
        return [relative(v, out) for v in value]
    if isinstance(value, str):
        return value.replace(str(out), "{out}").replace(str(HERE), "{corpus}")
    return value


def sample_record(path: Path) -> dict[str, Any]:
    """What an import wrote: every object, ``/meta``, the root."""
    objects: dict[str, Any] = {}
    with h5py.File(path, "r") as f:

        def visit(name: str, obj: Any) -> None:
            entry: dict[str, Any] = {
                "type": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                "attrs": {k: plain(obj.attrs[k]) for k in sorted(obj.attrs)},
            }
            if isinstance(obj, h5py.Dataset):
                entry["dtype"] = obj.dtype.str
                entry["shape"] = list(obj.shape)
            objects[name] = entry

        f.visititems(visit)
        meta = json.loads(f["meta"][()])
        root = {k: plain(f.attrs[k]) for k in sorted(f.attrs)}
    root.pop("created", None)
    return {"root": root, "meta": meta, "objects": objects}


def nifti_record(path: Path) -> dict[str, Any]:
    """A NIfTI file as stored: its header fields and a digest of its voxels.

    The header is decoded from the bytes, not through ``nibabel.load``, which
    consumes ``scl_slope``/``scl_inter`` and resets them.
    """
    raw = path.read_bytes()
    compressed = raw[:2] == b"\x1f\x8b"
    if compressed:
        raw = gzip.decompress(raw)
    klass = nib.Nifti2Header if raw[:4] in (b"\x1c\x02\0\0", b"\0\0\x02\x1c") else (
        nib.Nifti1Header
    )  # fmt: skip
    header = klass.from_fileobj(io.BytesIO(raw), check=False)
    fields = {k: plain(header[k]) for k in header.keys()}  # noqa: SIM118
    offset = int(header["vox_offset"])
    dtype = header.get_data_dtype()
    shape = header.get_data_shape()
    size = int(np.prod(shape)) * dtype.itemsize
    voxels = raw[offset : offset + size]
    return {
        "gzip": compressed,
        "endianness": header.endianness,
        "header": fields,
        "data": {
            "dtype": dtype.str,
            "shape": [int(v) for v in shape],
            "bytes": len(voxels),
            "trailing": len(raw) - offset - size,
            "sha256": hashlib.sha256(voxels).hexdigest(),
        },
    }


def array_record(data: np.ndarray) -> dict[str, Any]:
    """An array by type, shape and a digest of its little-endian C-order bytes."""
    little = np.ascontiguousarray(data, dtype=data.dtype.newbyteorder("<"))
    return {
        "dtype": data.dtype.str,
        "shape": [int(v) for v in data.shape],
        "sha256": hashlib.sha256(little.tobytes()).hexdigest(),
    }


def corpus(path: str) -> Path:
    return HERE / path


def options_of(case: dict[str, Any]) -> dict[str, Any]:
    options = dict(case["options"])
    if "label_set" in options:
        options["label_set"] = corpus(options["label_set"])
    return options


def run(case: dict[str, Any], out: Path) -> dict[str, Any]:
    """One case; what it produced."""
    record: dict[str, Any] = {}
    call = case["call"]
    if call == "from_nifti":
        target = out / case["out"]
        images = {n: corpus(p) for n, p in case["images"]}
        masks = {n: corpus(p) for n, p in case["masks"]} or None
        report = nifti.from_nifti(images, target, masks=masks, **options_of(case))
        record["report"] = report.to_json()
        record["sample"] = sample_record(target)
    elif call == "import_seg_nifti":
        target = out / Path(case["sample"]).name
        shutil.copyfile(corpus(case["sample"]), target)
        masks = {n: corpus(p) for n, p in case["masks"]}
        report = nifti.import_seg_nifti(target, masks, **options_of(case))
        record["report"] = report.to_json()
        record["sample"] = sample_record(target)
    elif call == "to_nifti":
        source = out / Path(case["sample"]).name
        shutil.copyfile(corpus(case["sample"]), source)
        for name, beside in case["beside"].items():
            if beside is None:
                (out / name).mkdir()
            else:
                shutil.copyfile(corpus(beside), out / name)
        report = ConversionReport(converter="to-nifti")
        written = nifti.to_nifti(
            source, case["image"], out / case["out"], report=report, **case["options"]
        )
        record["returned"] = str(written)
        record["report"] = report.to_json()
        files: dict[str, Any] = {}
        for path in sorted(out.iterdir()):
            if path.name == source.name or path.is_dir():
                continue
            if path.name.endswith((".nii", ".nii.gz")):
                files[path.name] = nifti_record(path)
            else:
                files[path.name] = {"text": path.read_text(encoding="utf-8")}
        record["export"] = files
    elif call == "read_nifti":
        data, geometry = nifti.read_nifti(corpus(case["path"]), **case["options"])
        record["read"] = {"geometry": plain(geometry), "array": array_record(data)}
    else:  # pragma: no cover - a table error
        raise ValueError(f"unknown call {call!r}")
    return record


def outcome(case: dict[str, Any], out: Path) -> dict[str, Any]:
    """Run *case* in the empty directory *out*; its record, but for the
    versions it was made with."""
    record: dict[str, Any] = {}
    try:
        record.update(run(case, out))
        record["outcome"] = "ok"
    except Exception as exc:  # every refusal is recorded
        record["outcome"] = "refused"
        record["error"] = {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", None),
            "message": str(exc),
        }
    record["files"] = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    result: dict[str, Any] = relative(plain(record), out)
    return result


def main() -> None:
    if medh5.__version__ != VERSION:
        sys.exit(f"record.py runs under medh5 {VERSION}, not {medh5.__version__}")
    table = json.loads((HERE / "cases.json").read_text(encoding="utf-8"))
    if RECORDS.exists():
        shutil.rmtree(RECORDS)
    RECORDS.mkdir()
    refused = 0
    for case in table["cases"]:
        with tempfile.TemporaryDirectory() as scratch:
            out = Path(scratch).resolve() / "out"
            out.mkdir()
            record = {
                "id": case["id"],
                "generator": f"medh5 {medh5.__version__}",
                "nibabel": nib.__version__,
                "numpy": np.__version__,
                **outcome(case, out),
            }
        refused += record["outcome"] == "refused"
        (RECORDS / f"{case['id']}.json").write_text(
            json.dumps(record, indent=1, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"{len(table['cases'])} cases recorded, {refused} refused")


if __name__ == "__main__":
    main()
