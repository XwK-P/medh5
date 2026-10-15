"""ITK transform files: affines, their composites, and displacement fields.

ITK's convention is MEDH5's (§10.2): a transform from a *fixed* image to a
*moving* one maps fixed points to moving points, ``x_M = T(x_F)``, so a
registration's output is stored as it is, from the fixed grid's frame to the
moving grid's.  What differs is the world each is written in: ITK's points
are LPS millimetres, a grid's are its own coordinate system and units.  The
matrix is re-expressed on both sides (``H_M⁻¹ · T · H_F``), never resampled.

Two things an ITK file does not say are taken from the caller:

- **The frames.** A transform file names no image, so the caller names the
  fixed and the moving grid, in ITK's words.
- **A field's vector convention.** A displacement field's vectors are written
  in LPS by some versions of ITK and in RAS by others, and the file does not
  say which, so ``vectors`` is required whenever a field is read or written.

A composite applies its *last* listed transform first, as ITK does; §10.5
applies its first component first, so components are listed in reverse.  An
ANTs pair (an affine and a warp, the warp applied first) is stored as the warp
from the fixed frame to an intermediate one, the affine from there to the
moving frame, and a ``composite`` between the two grids' frames, which earns
``reg`` (1.0 §1.3) and is the one shortest route between them.

Parsed with the standard library (``.tfm``/``.txt``, and ANTs' MATLAB v4
``.mat`` with :mod:`struct`); fields need nibabel (the ``nifti`` extra).
ITK's HDF5 transform files and B-splines are not read before medh5 2.2.
"""

from __future__ import annotations

import math
import os
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from medh5 import _core
from medh5._files import write_text
from medh5.errors import MEDH5ValidationError
from medh5.io.report import ConversionReport

Vectors = Literal["lps", "ras"]

_NAME = re.compile(
    r"^(?P<kind>[A-Za-z][A-Za-z0-9]*)_(?P<precision>double|float)_"
    r"(?P<input>\d+)_(?P<output>\d+)$"
)

_LINEAR = (
    "AffineTransform",
    "MatrixOffsetTransformBase",
    "TranslationTransform",
    "IdentityTransform",
    "Euler3DTransform",
    "Euler2DTransform",
    "VersorRigid3DTransform",
    "Similarity3DTransform",
    "Similarity2DTransform",
    "ScaleTransform",
)
"""The ITK transform types this module reads: every one a matrix."""

_LATER = ("BSpline", "DisplacementField", "GaussianSmoothingOnUpdate", "TimeVarying")
"""Prefixes of ITK's non-linear types: none is stored in a text or `.mat` file
this module reads, and B-splines wait for medh5 2.2."""

_FIXED_DIMENSION = {
    "Euler3DTransform": 3,
    "VersorRigid3DTransform": 3,
    "Similarity3DTransform": 3,
    "Euler2DTransform": 2,
    "Similarity2DTransform": 2,
}

_MAGIC = "#Insight Transform File V1.0"

_INTENT_DISPVECT = 1006
"""NIfTI's intent for a displacement vector field."""


@dataclass(frozen=True, eq=False)
class ItkTransform:
    """One ITK transform as a homogeneous matrix in LPS millimetres, mapping
    fixed points to moving ones (``x_moving = T(x_fixed)``).

    *name* is ITK's type name (``AffineTransform_double_3_3``).  A composite
    has *components*, in the order the file lists them --- the last is applied
    first --- and *matrix* is their product, the whole mapping.
    """

    name: str
    matrix: npt.NDArray[np.float64]
    components: tuple[ItkTransform, ...] = ()

    @property
    def dimension(self) -> int:
        return int(self.matrix.shape[0]) - 1

    @property
    def is_composite(self) -> bool:
        return bool(self.components)


# -- reading ---------------------------------------------------------------------


def read_itk_transform(path: str | os.PathLike[str]) -> ItkTransform:
    """Read an ITK transform file: text (``.tfm``, ``.txt``) or ANTs' MATLAB
    v4 ``.mat``, holding one of the linear types or a composite of them.

    Refused: ITK's HDF5 files, B-splines and other non-linear types (medh5
    2.2), a transform whose input and output dimensions differ, dimensions
    other than 2 and 3, and parameter counts the type does not have.
    """
    source = Path(path)
    where = os.fspath(path)
    if source.suffix.lower() in (".h5", ".hdf5", ".hdf"):
        raise MEDH5ValidationError(
            f"{where}: ITK's HDF5 transform files are not read before medh5 "
            "2.2; write the transform as .tfm or .mat"
        )
    data = source.read_bytes()
    if data.lstrip().startswith(b"#Insight Transform File"):
        return _read_text(data.decode("utf-8", errors="replace"), where)
    if source.suffix.lower() in (".tfm", ".txt"):
        raise MEDH5ValidationError(
            f"{where} is not an ITK transform file: it does not start with {_MAGIC!r}"
        )
    return _read_mat(data, where)


def _read_text(text: str, where: str) -> ItkTransform:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, colon, value = line.partition(":")
        key = key.strip()
        if not colon:
            raise MEDH5ValidationError(
                f"{where}, line {number}: {line!r} is not `Key: value`"
            )
        if key == "Transform":
            current = {"name": value.strip(), "Parameters": [], "FixedParameters": []}
            entries.append(current)
        elif key in ("Parameters", "FixedParameters"):
            if current is None:
                raise MEDH5ValidationError(
                    f"{where}, line {number}: {key} before any `Transform:`"
                )
            try:
                current[key] = [float(v) for v in value.split()]
            except ValueError:
                raise MEDH5ValidationError(
                    f"{where}, line {number}: {key} holds a value that is not a number"
                ) from None
        else:
            raise MEDH5ValidationError(f"{where}, line {number}: unknown key {key!r}")
    if not entries:
        raise MEDH5ValidationError(f"{where} holds no transform")
    head = _parse_name(entries[0]["name"], where)
    if head[0] == "CompositeTransform":
        if len(entries) == 1:
            raise MEDH5ValidationError(f"{where}: the composite has no components")
        parts = tuple(
            _linear(e["name"], e["Parameters"], e["FixedParameters"], where)
            for e in entries[1:]
        )
        return _composite(entries[0]["name"], parts, where)
    if len(entries) > 1:
        raise MEDH5ValidationError(
            f"{where} lists {len(entries)} transforms without a "
            "CompositeTransform saying how they compose"
        )
    first = entries[0]
    return _linear(first["name"], first["Parameters"], first["FixedParameters"], where)


def _composite(name: str, parts: tuple[ItkTransform, ...], where: str) -> ItkTransform:
    kind, dimension = _parse_name(name, where)
    nested = [p.name for p in parts if p.is_composite]
    if nested:
        raise MEDH5ValidationError(f"{where}: a composite nested in a composite")
    other = [p.name for p in parts if p.dimension != dimension]
    if other:
        raise MEDH5ValidationError(
            f"{where}: the {dimension}-D composite holds {other}, of another dimension"
        )
    matrix = np.eye(dimension + 1)
    for part in parts:
        matrix = matrix @ part.matrix
    return ItkTransform(name, matrix, parts)


def _parse_name(name: str, where: str) -> tuple[str, int]:
    """``(type, dimension)`` of an ITK type name, refusing what is not read."""
    found = _NAME.match(name)
    if found is None:
        raise MEDH5ValidationError(f"{where}: {name!r} is not an ITK transform type")
    kind = found["kind"]
    if kind.startswith(_LATER):
        raise MEDH5ValidationError(
            f"{where}: {name} is not read before medh5 2.2 (B-splines and other "
            "non-linear ITK transforms); a displacement field is read from its "
            "NIfTI file with warp="
        )
    if kind not in (*_LINEAR, "CompositeTransform"):
        raise MEDH5ValidationError(f"{where}: ITK type {name} is not one this reads")
    source, target = int(found["input"]), int(found["output"])
    if source != target:
        raise MEDH5ValidationError(
            f"{where}: {name} maps {source}-D points to {target}-D ones; a "
            "registration maps a space onto one of its own dimension"
        )
    if source not in (2, 3):
        raise MEDH5ValidationError(
            f"{where}: {name} is {source}-D; grids are 2-D or 3-D"
        )
    expected = _FIXED_DIMENSION.get(kind)
    if expected is not None and expected != source:
        raise MEDH5ValidationError(f"{where}: {kind} is {expected}-D, not {source}-D")
    return kind, source


def _linear(
    name: str, parameters: Sequence[float], fixed: Sequence[float], where: str
) -> ItkTransform:
    kind, n = _parse_name(name, where)
    if kind == "CompositeTransform":
        raise MEDH5ValidationError(f"{where}: a composite nested in a composite")
    p = [float(v) for v in parameters]
    f = [float(v) for v in fixed]

    def counts(params: int, fixed_counts: Sequence[int]) -> None:
        if len(p) != params or len(f) not in fixed_counts:
            raise MEDH5ValidationError(
                f"{where}: {name} takes {params} parameters and "
                f"{' or '.join(str(c) for c in fixed_counts)} fixed parameters; "
                f"the file gives {len(p)} and {len(f)}"
            )

    centre = np.zeros(n)
    translation = np.zeros(n)
    if kind in ("AffineTransform", "MatrixOffsetTransformBase"):
        counts(n * n + n, (0, n))
        linear = np.array(p[: n * n]).reshape(n, n)
        translation = np.array(p[n * n :])
    elif kind == "TranslationTransform":
        counts(n, (0,))
        linear, translation = np.eye(n), np.array(p)
    elif kind == "IdentityTransform":
        counts(0, (0,))
        linear = np.eye(n)
    elif kind == "ScaleTransform":
        counts(n, (0, n))
        linear = np.diag(p)
    elif kind == "Euler3DTransform":
        counts(6, (0, 3, 4))
        zyx = len(f) == 4 and f[3] != 0.0
        linear = _euler3d(p[0], p[1], p[2], zyx=zyx)
        translation = np.array(p[3:6])
        f = f[:3]
    elif kind == "Euler2DTransform":
        counts(3, (0, 2))
        linear, translation = _rotation2d(p[0]), np.array(p[1:3])
    elif kind == "Similarity2DTransform":
        counts(4, (0, 2))
        linear, translation = p[0] * _rotation2d(p[1]), np.array(p[2:4])
    else:  # VersorRigid3DTransform, Similarity3DTransform
        counts(7 if kind == "Similarity3DTransform" else 6, (0, 3))
        linear = _versor(p[0], p[1], p[2], where)
        if kind == "Similarity3DTransform":
            linear = p[6] * linear
        translation = np.array(p[3:6])
    if f:
        centre = np.array(f[:n])
    matrix = np.eye(n + 1)
    matrix[:n, :n] = linear
    # ITK's MatrixOffsetTransformBase: x' = A (x - c) + t + c.
    matrix[:n, n] = translation + centre - linear @ centre
    return ItkTransform(name, matrix)


def _rotation2d(angle: float) -> npt.NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def _euler3d(x: float, y: float, z: float, *, zyx: bool) -> npt.NDArray[np.float64]:
    """ITK's ``Euler3DTransform::ComputeMatrix``: ``Rz·Rx·Ry``, or ``Rz·Ry·Rx``
    when the file's fourth fixed parameter sets ``ComputeZYX``."""
    cx, sx, cy, sy, cz, sz = (
        math.cos(x),
        math.sin(x),
        math.cos(y),
        math.sin(y),
        math.cos(z),
        math.sin(z),
    )
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rotation = rz @ ry @ rx if zyx else rz @ rx @ ry
    return np.asarray(rotation, dtype=np.float64)


def _versor(x: float, y: float, z: float, where: str) -> npt.NDArray[np.float64]:
    """ITK's ``Versor::GetMatrix`` for the versor's vector part."""
    norm = x * x + y * y + z * z
    if norm > 1.0 + 1e-12:
        raise MEDH5ValidationError(
            f"{where}: the versor ({x}, {y}, {z}) has a norm above 1, so it is "
            "no rotation"
        )
    w = math.sqrt(max(0.0, 1.0 - norm))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _read_mat(data: bytes, where: str) -> ItkTransform:
    """ANTs' MATLAB v4 file: a variable named for the type, and ``fixed``."""
    variables: dict[str, list[float]] = {}
    offset = 0
    while offset < len(data):
        if len(data) - offset < 20:
            raise MEDH5ValidationError(f"{where}: a truncated MATLAB v4 record")
        header = None
        for order, digit in (("<", 0), (">", 1)):
            mopt, rows, cols, imagf, namlen = struct.unpack_from(
                order + "5i", data, offset
            )
            if 0 <= mopt < 10000 and mopt // 1000 == digit:
                header = (order, mopt, rows, cols, imagf, namlen)
                break
        if header is None:
            raise MEDH5ValidationError(
                f"{where} is neither an ITK text transform nor a MATLAB v4 file"
            )
        order, mopt, rows, cols, imagf, namlen = header
        precision, other, matrix_type = (mopt // 10) % 10, (mopt // 100) % 10, mopt % 10
        if other != 0 or matrix_type != 0 or imagf != 0 or precision not in (0, 1):
            raise MEDH5ValidationError(
                f"{where}: a MATLAB v4 record of type {mopt:04d}; a transform "
                "is a real double or single matrix"
            )
        if rows < 0 or cols < 0 or namlen <= 0:
            raise MEDH5ValidationError(f"{where}: a malformed MATLAB v4 record")
        offset += 20
        name = (
            data[offset : offset + namlen].split(b"\0", 1)[0].decode("ascii", "replace")
        )
        offset += namlen
        dtype = np.dtype(order + ("f8" if precision == 0 else "f4"))
        size = rows * cols * dtype.itemsize
        if len(data) - offset < size:
            raise MEDH5ValidationError(f"{where}: variable {name!r} is truncated")
        values = np.frombuffer(data, dtype=dtype, count=rows * cols, offset=offset)
        offset += size
        variables[name] = [float(v) for v in values]
    named = [n for n in variables if _NAME.match(n)]
    if len(named) != 1:
        raise MEDH5ValidationError(
            f"{where}: a MATLAB transform file holds one variable named for its "
            f"ITK type, and this one holds {sorted(variables)}"
        )
    return _linear(named[0], variables[named[0]], variables.get("fixed", []), where)


# -- writing ---------------------------------------------------------------------


def write_itk_transform(path: str | os.PathLike[str], transform: ItkTransform) -> Path:
    """Write *transform* as ITK reads it, picking the format by suffix:
    ``.tfm`` or ``.txt`` text, or ``.mat`` as ANTs writes it.

    Every matrix is written as ``AffineTransform_double_S_S`` with its centre
    at the origin, which maps every point as the type it was read as did.  A
    ``.mat`` holds one transform, so a composite is refused there.
    """
    target = Path(path)
    if _transform_suffix(target) in (".tfm", ".txt"):
        write_text(target, _text(transform))
    else:
        if transform.is_composite:
            raise MEDH5ValidationError(
                f"{os.fspath(path)}: a .mat file holds one transform, and this "
                f"is a composite of {len(transform.components)}; write it as .tfm"
            )
        _write_bytes(target, _mat(transform.matrix))
    return target


def _transform_suffix(path: Path) -> str:
    """*path*'s suffix, which names the format: ``.tfm``, ``.txt`` or
    ``.mat``; anything else is refused."""
    suffix = path.suffix.lower()
    if suffix not in (".tfm", ".txt", ".mat"):
        raise MEDH5ValidationError(
            f"{os.fspath(path)}: name a .tfm, .txt or .mat file (ITK's HDF5 "
            "transform files are not written before medh5 2.2)"
        )
    return suffix


def _affine_name(dimension: int) -> str:
    return f"AffineTransform_double_{dimension}_{dimension}"


def _number(value: float) -> str:
    return repr(float(value))


def _affine_lines(matrix: npt.NDArray[np.float64], index: int) -> list[str]:
    n = matrix.shape[0] - 1
    parameters = [*matrix[:n, :n].ravel(), *matrix[:n, n]]
    return [
        f"#Transform {index}",
        f"Transform: {_affine_name(n)}",
        "Parameters: " + " ".join(_number(v) for v in parameters),
        "FixedParameters: " + " ".join("0" for _ in range(n)),
    ]


def _text(transform: ItkTransform) -> str:
    lines = [_MAGIC]
    if transform.is_composite:
        n = transform.dimension
        lines += ["#Transform 0", f"Transform: CompositeTransform_double_{n}_{n}"]
        for index, part in enumerate(transform.components, 1):
            lines += _affine_lines(part.matrix, index)
    else:
        lines += _affine_lines(transform.matrix, 0)
    return "\n".join(lines) + "\n"


def _mat_record(name: str, values: Sequence[float]) -> bytes:
    encoded = name.encode("ascii") + b"\0"
    head = struct.pack("<5i", 0, len(values), 1, 0, len(encoded))
    return head + encoded + struct.pack(f"<{len(values)}d", *values)


def _mat(matrix: npt.NDArray[np.float64]) -> bytes:
    n = matrix.shape[0] - 1
    parameters = [*matrix[:n, :n].ravel(), *matrix[:n, n]]
    return _mat_record(_affine_name(n), [float(v) for v in parameters]) + _mat_record(
        "fixed", [0.0] * n
    )


def _write_bytes(target: Path, data: bytes) -> None:
    """*data* as *target*'s content, through the engine's replace (§14.4)."""
    _core.write_atomic(target, data)


# -- coordinates -----------------------------------------------------------------


def _world_to_itk(grid: Any, what: str) -> npt.NDArray[np.float64]:
    """``H``: a point of *grid*'s world (its coordinate system and units) as
    ITK's LPS millimetres, homogeneous.  A grid in ``px`` or another
    convention has no LPS millimetres to give."""
    from medh5.geometry import COORD_SYSTEMS, MM_PER_UNIT

    n = len(grid.spacing)
    if grid.coord_system not in COORD_SYSTEMS:
        raise MEDH5ValidationError(
            f"{what} grid {grid.grid_id!r} is in {grid.coord_system!r}, which "
            "ITK's LPS cannot be converted to"
        )
    scale = MM_PER_UNIT.get(grid.units)
    if scale is None:
        raise MEDH5ValidationError(
            f"{what} grid {grid.grid_id!r} is in {grid.units!r}, which has no "
            "length in millimetres, so an ITK transform cannot be expressed on it"
        )
    out = np.eye(n + 1)
    flip = np.ones(n)
    if grid.coord_system == "RAS":
        flip[:2] = -1.0
    out[:n, :n] = np.diag(flip * scale)
    return out


def _vector_flip(vectors: str, n: int) -> npt.NDArray[np.float64]:
    flip = np.ones(n)
    if vectors == "ras":
        flip[:2] = -1.0
    return flip


def _check_vectors(vectors: str | None, what: str) -> str:
    if vectors is None:
        raise MEDH5ValidationError(
            f"{what}: vectors='lps' or 'ras' is required --- a displacement "
            "field's vectors are written in LPS by some versions of ITK and in "
            "RAS by others, and the file does not say which"
        )
    if vectors not in ("lps", "ras"):
        raise ValueError(f"vectors must be 'lps' or 'ras', not {vectors!r}")
    return vectors


# -- import ----------------------------------------------------------------------


def from_itk_transform(
    path: str | os.PathLike[str],
    sample: str | os.PathLike[str],
    *,
    transform_id: str,
    fixed: str,
    moving: str,
    warp: str | os.PathLike[str] | None = None,
    vectors: Vectors | None = None,
    inverse_warp: str | os.PathLike[str] | None = None,
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Add the ITK transform at *path* to *sample*, from the frame of grid
    *fixed* to that of grid *moving* --- ITK's fixed and moving images.

    One linear transform is stored as an ``affine`` (or ``identity``)
    *transform_id*.  A composite is stored as its components, in §10.5's
    order (ITK applies the last listed first), between intermediate frames
    ``pseudo:<id>.mid…``, and a ``composite`` *transform_id* between the two
    grids' frames.  *warp*, a displacement field as ANTs writes one beside
    its affine (applied first), is stored as ``<id>.warp`` from the fixed
    frame to ``pseudo:<id>.mid``, the linear part after it, and the composite
    *transform_id*; *inverse_warp*, its inverse, as ``<id>.inverse_warp``,
    linked to it by ``inverse_id``. *vectors* is required with a field.

    The field is stored on grid *fixed* when its geometry is that grid's, and
    otherwise on a new grid ``<id>_field`` in the fixed frame; its vectors as
    ``world`` vectors in the fixed grid's convention and units.
    """
    import medh5

    if inverse_warp is not None and warp is None:
        raise MEDH5ValidationError(
            "inverse_warp is the inverse of a warp; give the warp too"
        )
    if warp is not None:
        vectors = _check_vectors(vectors, "a warp")  # type: ignore[assignment]
    log = report or ConversionReport(converter="from-itk-transform")
    log.source = ", ".join(
        os.fspath(p) for p in (path, warp, inverse_warp) if p is not None
    )
    transform = read_itk_transform(path)
    with medh5.open(sample) as s:
        grids = {
            role: s.grids[gid] for role, gid in (("fixed", fixed), ("moving", moving))
        }
        taken = set(s.transforms) | set(s.grids)
        frames_in_use = (
            {g.frame_uid for g in s.grids.values() if g.frame_uid}
            | {
                frame
                for t in s.transforms.values()
                for frame in (t.from_frame, t.to_frame)
                if frame
            }
            | {a.frame_uid for a in s.annotations.values() if a.frame_uid}
        )
        visit = grids["fixed"].timepoint
        visit_frames = {
            g.frame_uid
            for g in s.grids.values()
            if g.timepoint == visit and g.frame_uid
        }
    if inverse_warp is not None and visit is not None and len(visit_frames) == 1:
        # §10.4 puts an inverse warp's field on a grid of the intermediate
        # frame, which joins the fixed grid's visit (E106 asks every grid for
        # one): a visit on one frame would gain a second, and transform_between
        # by timepoint, which answered for it, would refuse.
        raise MEDH5ValidationError(
            "inverse_warp needs a grid in the intermediate frame, which would "
            f"join visit {visit!r} and give it a second frame, so that "
            "transform_between could no longer answer for the visit; import "
            "the forward warp alone"
        )
    for role, grid in grids.items():
        if not grid.frame_uid:
            raise MEDH5ValidationError(
                f"{role} grid {grid.grid_id!r} has no frame_uid, and a transform "
                "relates frames (§10); give the grid its frame first"
            )
        if len(grid.spacing) != transform.dimension:
            raise MEDH5ValidationError(
                f"the transform is {transform.dimension}-D, and {role} grid "
                f"{grid.grid_id!r} has {len(grid.spacing)} spatial axes"
            )
    fixed_grid, moving_grid = grids["fixed"], grids["moving"]
    if fixed_grid.frame_uid == moving_grid.frame_uid:
        raise MEDH5ValidationError(
            f"grids {fixed!r} and {moving!r} are on one frame "
            f"({fixed_grid.frame_uid!r}), which no transform relates"
        )
    h_fixed = _world_to_itk(fixed_grid, "fixed")
    h_moving = _world_to_itk(moving_grid, "moving")
    log.decision(
        "frames",
        f"the transform maps grid {fixed!r}'s frame {fixed_grid.frame_uid!r} "
        f"(ITK's fixed image) to grid {moving!r}'s {moving_grid.frame_uid!r} "
        "(ITK's moving image), x_moving = T(x_fixed) (§10.2)",
        {"from_frame": fixed_grid.frame_uid, "to_frame": moving_grid.frame_uid},
    )
    _coordinates_note(log, fixed_grid, moving_grid)

    linear = list(transform.components) if transform.is_composite else [transform]
    if transform.is_composite:
        log.decision(
            "order",
            f"the composite's {len(linear)} components are stored in reverse: "
            "ITK applies the last listed first, §10.5 the first",
            {"itk_order": [c.name for c in linear]},
        )
        linear.reverse()
    fields: list[tuple[str, Any]] = []
    if warp is not None:
        fields.append(("warp", _read_field(warp, fixed_grid, vectors, log, "warp")))  # type: ignore[arg-type]
    if inverse_warp is not None:
        fields.append(
            (
                "inverse_warp",
                _read_field(inverse_warp, fixed_grid, vectors, log, "inverse warp"),  # type: ignore[arg-type]
            )
        )

    steps = len(linear) + (1 if warp is not None else 0)
    interior = (
        [f"pseudo:{transform_id}.mid"]
        if steps == 2
        else [f"pseudo:{transform_id}.mid{j}" for j in range(1, steps)]
    )
    frames = [str(fixed_grid.frame_uid), *interior, str(moving_grid.frame_uid)]
    names: list[str] = []
    if steps == 1:
        names = [transform_id]
    else:
        if warp is not None:
            names.append(f"{transform_id}.warp")
        if len(linear) == 1:
            names.append(f"{transform_id}.affine")
        else:
            names += [f"{transform_id}.affine{j}" for j in range(1, len(linear) + 1)]
    if inverse_warp is not None:
        names_all = [*names, f"{transform_id}.inverse_warp"]
    else:
        names_all = list(names)
    if steps > 1:
        names_all.append(transform_id)
    clash = sorted(n for n in names_all if n in taken)
    if clash:
        raise MEDH5ValidationError(
            f"the sample already holds {clash}; choose another transform_id"
        )
    # The steps would join a frame already in use, and with it routes that
    # are not this registration's.
    joined = sorted(f for f in interior if f in frames_in_use)
    if joined:
        raise MEDH5ValidationError(
            f"the sample already has frame {joined[0]!r}, where this import's "
            "steps would meet; choose another transform_id"
        )
    if interior:
        log.decision(
            "interior_frames",
            f"the steps meet in {interior}, frames of no grid, whose points are "
            f"expressed as grid {fixed!r}'s are ({fixed_grid.coord_system}, "
            f"{fixed_grid.units})",
            {"frames": interior},
        )

    with medh5.amend(sample) as w:
        tool = w.software("medh5", medh5.__version__)
        activity = w.activity(
            "import", agent=tool, tool="medh5.io.itk.from_itk_transform"
        )
        index = 0
        if warp is not None:
            field_grid, field = _place_field(
                w, fields[0][1], fixed_grid, f"{transform_id}_field", log
            )
            w.add_transform(
                names[0],
                kind="displacement",
                from_frame=frames[0],
                to_frame=frames[1],
                field=field,
                field_grid=field_grid,
                vector_space="world",
                units=fixed_grid.units,
                from_grid=fixed,
                inverse_id=(
                    f"{transform_id}.inverse_warp" if inverse_warp is not None else None
                ),
                prov=activity,
            )
            if inverse_warp is not None:
                back_grid, back = _place_field(
                    w,
                    fields[1][1],
                    fixed_grid,
                    f"{transform_id}_inverse_field",
                    log,
                    frame=frames[1],
                )
                w.add_transform(
                    f"{transform_id}.inverse_warp",
                    kind="displacement",
                    from_frame=frames[1],
                    to_frame=frames[0],
                    field=back,
                    field_grid=back_grid,
                    vector_space="world",
                    units=fixed_grid.units,
                    to_grid=fixed,
                    inverse_id=names[0],
                    prov=activity,
                )
            index = 1
        for offset, part in enumerate(linear):
            j = index + offset
            last = j == steps - 1
            # Interior frames are expressed as the fixed grid's are.
            before = h_fixed
            after = h_moving if last else h_fixed
            matrix = np.linalg.inv(after) @ part.matrix @ before
            identity = part.name.startswith("IdentityTransform") and np.allclose(
                matrix, np.eye(matrix.shape[0])
            )
            w.add_transform(
                names[j],
                kind="identity" if identity else "affine",
                from_frame=frames[j],
                to_frame=frames[j + 1],
                matrix=None if identity else matrix,
                units=fixed_grid.units,
                from_grid=fixed if j == 0 else None,
                to_grid=moving if last else None,
                prov=activity,
            )
        if steps > 1:
            w.add_transform(
                transform_id,
                kind="composite",
                from_frame=frames[0],
                to_frame=frames[-1],
                components=names,
                units=fixed_grid.units,
                from_grid=fixed,
                to_grid=moving,
                prov=activity,
            )
            log.decision(
                "composite",
                f"{transform_id!r} composes {names} between the two grids' "
                "frames, so it is the one shortest route between them",
                {"components": names},
            )
    log.outputs.append(os.fspath(sample))
    return log


def _coordinates_note(log: ConversionReport, *grids: Any) -> None:
    for grid in grids:
        if grid.coord_system != "LPS" or grid.units != "mm":
            log.decision(
                "coordinates",
                f"grid {grid.grid_id!r} is in {grid.coord_system} {grid.units}, "
                "and ITK's points are LPS millimetres: the matrix is "
                "re-expressed on that side",
                {
                    "grid": grid.grid_id,
                    "coord_system": grid.coord_system,
                    "units": grid.units,
                },
            )


def _read_field(
    path: str | os.PathLike[str],
    fixed_grid: Any,
    vectors: str,
    log: ConversionReport,
    what: str,
) -> dict[str, Any]:
    """A NIfTI displacement field as ``(S, *spatial)`` world vectors in
    *fixed_grid*'s convention and units, with its geometry in the same."""
    from medh5.geometry import MM_PER_UNIT
    from medh5.io.nifti import require_nibabel

    nib = require_nibabel()
    from medh5.geometry import decompose_affine
    from medh5.io.nifti import _units, convert_world

    where = os.fspath(path)
    try:
        image = nib.load(where)
    except nib.filebasedimages.ImageFileError as exc:
        raise MEDH5ValidationError(f"{where} is not a NIfTI file: {exc}") from None
    if not isinstance(image, nib.Nifti1Pair):
        raise MEDH5ValidationError(
            f"{where} is not a NIfTI file: nibabel reads it as {type(image).__name__}"
        )
    n = len(fixed_grid.spacing)
    data = np.asarray(image.dataobj, dtype=np.float64)
    if data.ndim == 5 and data.shape[3] == 1:
        data = data[:, :, :, 0, :]
    if data.ndim != 4 or data.shape[3] != n or n != 3:
        raise MEDH5ValidationError(
            f"{what} {where} has shape {image.shape}; a 3-D displacement field "
            "is (X, Y, Z, 1, 3)"
        )
    if (
        int(image.header.get_sform(coded=True)[1] or 0) == 0
        and int(image.header.get_qform(coded=True)[1] or 0) == 0
    ):
        raise MEDH5ValidationError(
            f"{what} {where} declares no spatial mapping (sform_code = "
            "qform_code = 0), so where its vectors sit is unknown"
        )
    intent = int(image.header["intent_code"])
    log.decision(
        "vectors",
        f"the {what}'s vectors were read as {vectors.upper()}, as the caller "
        f"said (the file's intent code is {intent})",
        {"file": where, "vectors": vectors, "intent_code": intent},
    )
    # The file's lengths, its affine's and its vectors', in its spatial unit;
    # millimetres where it states none, as read_nifti reads them.
    unit, stated = _units(image)
    if stated is not None:
        log.guess(
            "units",
            f"the {what} states its spatial unit as {stated!r}, so its lengths "
            "and vectors were read as millimetres, as ITK writes them",
            {"file": where, "xyz_units": stated},
        )
    to_mm = MM_PER_UNIT[unit]
    scale = MM_PER_UNIT[fixed_grid.units]
    in_mm = np.array(image.affine, dtype=np.float64)
    in_mm[:3, :] *= to_mm
    affine = convert_world(in_mm, source="RAS", target=fixed_grid.coord_system)
    spacing, origin, direction = decompose_affine(affine)
    # (X, Y, Z, 3) -> (3, Z, Y, X), the grid's axes reversed as read_nifti does
    field = np.transpose(data, (3, 2, 1, 0))
    spatial = [2, 1, 0]
    spacing = np.asarray(spacing)[spatial] / scale
    direction = np.asarray(direction)[:, spatial]
    origin = np.asarray(origin) / scale
    # LPS (or RAS) millimetres -> the grid's convention and units
    flip = _vector_flip(vectors, n)
    if fixed_grid.coord_system == "RAS":
        flip = flip * np.array([-1.0, -1.0, 1.0])
    field = field * (flip * to_mm / scale)[:, None, None, None]
    return {
        "field": np.ascontiguousarray(field, dtype=np.float32)
        if image.get_data_dtype() == np.float32
        else np.ascontiguousarray(field),
        "shape": tuple(int(v) for v in field.shape[1:]),
        "spacing": [float(v) for v in spacing],
        "origin": [float(v) for v in origin],
        "direction": np.asarray(direction, dtype=np.float64),
        "file": where,
    }


def _place_field(
    w: Any,
    found: dict[str, Any],
    fixed_grid: Any,
    grid_id: str,
    log: ConversionReport,
    *,
    frame: str | None = None,
) -> tuple[str, npt.NDArray[Any]]:
    """The grid a field is stored on: *fixed_grid* when the field's geometry
    is its own and *frame* is its frame, else a new grid *grid_id*."""
    same = (
        frame is None
        and tuple(found["shape"]) == tuple(fixed_grid.spatial_shape)
        and np.allclose(found["spacing"], fixed_grid.spacing, rtol=1e-6, atol=1e-6)
        and np.allclose(found["origin"], fixed_grid.origin, rtol=1e-6, atol=1e-4)
        and np.allclose(found["direction"], np.asarray(fixed_grid.direction), atol=1e-6)
    )
    if same:
        log.decision(
            "field_grid",
            f"the field {found['file']} is on grid {fixed_grid.grid_id!r}'s "
            "geometry, and is stored on that grid",
            {"grid": fixed_grid.grid_id},
        )
        return str(fixed_grid.grid_id), found["field"]
    w.add_grid(
        grid_id,
        shape=found["shape"],
        spacing=found["spacing"],
        origin=found["origin"],
        direction=found["direction"],
        coord_system=fixed_grid.coord_system,
        units=fixed_grid.units,
        frame_uid=frame or fixed_grid.frame_uid,
        # the fixed grid's visit; an intermediate frame's grid joins it only
        # where the visit already holds several frames (see from_itk_transform)
        timepoint=fixed_grid.timepoint,
    )
    log.decision(
        "field_grid",
        f"the field {found['file']} is sampled on its own lattice, stored as "
        f"grid {grid_id!r} in frame {frame or fixed_grid.frame_uid!r}",
        {"grid": grid_id},
    )
    return grid_id, found["field"]


# -- export ----------------------------------------------------------------------


def to_itk_transform(
    sample: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    transform_id: str,
    warp_out: str | os.PathLike[str] | None = None,
    vectors: Vectors | None = None,
) -> ConversionReport:
    """Write transform *transform_id* of *sample* as ITK reads it: its linear
    part to *out* (``.tfm``, ``.txt`` or ``.mat``), and a displacement field
    applied first, as ANTs pairs them, to *warp_out* as a DISPVECT NIfTI
    (intent 1006) whose vectors are in the *vectors* convention, required.

    A composite is flattened; it may hold one displacement, as its first
    step, and linear steps after it.  Several linear steps are written as an
    ITK composite in a ``.tfm`` (last applied listed first) or multiplied into
    one affine in a ``.mat``. A displacement alone writes the identity to
    *out*. B-splines are refused.
    """
    import medh5

    log = ConversionReport(converter="to-itk-transform")
    log.source = os.fspath(sample)
    # Both files are checked before either is written, so a refusal leaves
    # an existing pair as it was.
    _transform_suffix(Path(out))
    with medh5.open(sample) as s:
        transform = s.transforms[transform_id]
        steps = _flatten(transform)
        start = transform.grid_in(transform.from_frame)
        if start is None:
            raise MEDH5ValidationError(
                f"transform {transform_id!r} starts in frame "
                f"{transform.from_frame!r}, which no grid of the sample is in, "
                "so the points' convention is unknown"
            )

        def convention(frame: str) -> Any:
            grid = transform.grid_in(frame)
            if grid is None:
                grid = next((g for g in s.grids.values() if g.frame_uid == frame), None)
            return grid if grid is not None else start

        fields = [t for t in steps if t.kind == "displacement"]
        if any(t.kind == "displacement" for t in steps[1:]):
            raise MEDH5ValidationError(
                f"transform {transform_id!r} applies a displacement after "
                "another step; ITK's affine-and-warp pair applies the warp first"
            )
        if fields and warp_out is None:
            raise MEDH5ValidationError(
                f"transform {transform_id!r} holds a displacement field; name "
                "the NIfTI file to write it to with warp_out="
            )
        if warp_out is not None and not fields:
            raise MEDH5ValidationError(
                f"transform {transform_id!r} holds no displacement field to "
                "write to warp_out"
            )
        matrices: list[npt.NDArray[np.float64]] = []
        for step in steps:
            if step.kind == "displacement":
                continue
            n = len(start.spacing)
            matrix = (
                np.eye(n + 1) if step.kind == "identity" else np.asarray(step.matrix)
            )
            before = _world_to_itk(convention(step.from_frame), "the step's")
            after = _world_to_itk(convention(step.to_frame), "the step's")
            matrices.append(after @ matrix @ np.linalg.inv(before))
        if fields:
            written = _write_field(fields[0], warp_out, vectors, log)  # type: ignore[arg-type]
            log.outputs.append(os.fspath(written))
    n = len(start.spacing)
    if not matrices:
        itk = ItkTransform(_affine_name(n), np.eye(n + 1))
        log.decision(
            "identity",
            f"transform {transform_id!r} is a displacement alone; the identity "
            f"was written to {os.fspath(out)} to pair with it",
        )
    elif len(matrices) == 1:
        itk = ItkTransform(_affine_name(n), matrices[0])
    elif Path(out).suffix.lower() == ".mat":
        product = np.eye(n + 1)
        for matrix in matrices:
            product = matrix @ product
        itk = ItkTransform(_affine_name(n), product)
        log.decision(
            "composite",
            f"the {len(matrices)} linear steps were multiplied into one affine, "
            "which is all a .mat file holds",
        )
    else:
        parts = tuple(ItkTransform(_affine_name(n), m) for m in reversed(matrices))
        product = np.eye(n + 1)
        for matrix in matrices:
            product = matrix @ product
        itk = ItkTransform(f"CompositeTransform_double_{n}_{n}", product, parts)
        log.decision(
            "order",
            f"the {len(matrices)} linear steps are listed in reverse: ITK applies "
            "the last listed first",
        )
    written = write_itk_transform(out, itk)
    log.outputs.insert(0, os.fspath(written))
    return log


def _flatten(transform: Any) -> list[Any]:
    """A stored transform's steps, composites opened, in application order."""
    if transform.kind == "composite":
        out: list[Any] = []
        for component in transform.components():
            out += _flatten(component)
        return out
    if transform.kind in ("affine", "identity", "displacement"):
        return [transform]
    raise MEDH5ValidationError(
        f"transform {transform.transform_id!r} is a {transform.kind}, which is "
        "not written as an ITK file before medh5 2.2"
    )


def _write_field(
    transform: Any,
    target: str | os.PathLike[str],
    vectors: str | None,
    log: ConversionReport,
) -> Path:
    from medh5.geometry import MM_PER_UNIT
    from medh5.io.nifti import for_export, require_nibabel
    from medh5.transforms import to_world_vectors

    vectors = _check_vectors(vectors, "a warp")
    nib = require_nibabel()
    grid = transform.field_grid
    path = Path(target)
    if not path.name.endswith((".nii", ".nii.gz")):
        raise MEDH5ValidationError(f"{os.fspath(target)}: name a .nii or .nii.gz file")
    n = len(grid.spacing)
    if n != 3:
        raise MEDH5ValidationError("a displacement field is written for 3-D grids")
    _world_to_itk(grid, "field")
    stored = np.asarray(transform.read_field(), dtype=np.float64)
    flat = np.moveaxis(stored, 0, -1).reshape(-1, n)
    world = to_world_vectors(flat, grid, transform.vector_space)
    scale = MM_PER_UNIT[grid.units]
    flip = np.ones(n)
    if grid.coord_system == "RAS":
        flip[:2] = -1.0
    lps = world * (flip * scale)
    out = lps * _vector_flip(vectors, n)
    field = np.moveaxis(out.reshape(*stored.shape[1:], n), -1, 0)
    data, affine = for_export(grid, field)
    affine = np.array(affine, dtype=np.float64)
    affine[:3, :] *= scale
    dtype = np.float32 if transform.field.dtype == np.float32 else np.float64
    data = np.ascontiguousarray(data[:, :, :, None, :], dtype=dtype)
    image = nib.Nifti1Image(data, affine)
    image.header.set_intent(_INTENT_DISPVECT)
    image.header.set_xyzt_units(xyz="mm")
    with _core.stage_file(path) as staged:
        nib.save(image, str(staged.path))
    log.decision(
        "vectors",
        f"the field's vectors were written as {vectors.upper()} millimetres, "
        "as the caller said, in a DISPVECT NIfTI (intent 1006)",
        {"file": os.fspath(path), "vectors": vectors},
    )
    return path


__all__ = [
    "ItkTransform",
    "from_itk_transform",
    "read_itk_transform",
    "to_itk_transform",
    "write_itk_transform",
]
