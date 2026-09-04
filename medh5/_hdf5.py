"""HDF5 plumbing: attribute codecs, identifier rules, atomic create, copy-on-write.

Everything in this module is about *how* values reach HDF5, never about what
they mean.  Spec §2.5 fixes the attribute encoding; spec §14.4 fixes the write
model (atomic create, copy-on-write amend, unknown-object preservation).
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import numpy.typing as npt

from medh5.errors import MEDH5FileError, MEDH5ValidationError

try:  # noqa: SIM105 - the failure mode matters, see below
    # Importing hdf5plugin registers the Blosc2 filter with the HDF5 library.
    # Without it, *reading* a Blosc2-compressed dataset fails deep inside h5py
    # with an unrelated-looking "can't open directory .../plugin" error.  It is
    # imported here because this is the module every HDF5 path goes through, and
    # it is guarded because a `portable` file (spec §14.2) must stay readable on
    # an installation where the plugin is missing or broken.
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover - only on a broken plugin install
    pass

ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
SAMPLE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,255}$")

RESERVED_IDS = frozenset({"meta"})
"""Object names an identifier must not take (spec §2.3)."""


def str_dtype() -> Any:
    """The variable-length UTF-8 string dtype used for every string in a file."""
    return h5py.string_dtype(encoding="utf-8")


def validate_id(name: str, *, what: str = "identifier") -> str:
    """Check an object identifier against spec §2.3 and return it unchanged."""
    if not ID_PATTERN.match(name):
        raise MEDH5ValidationError(
            f"{what} {name!r} must match [A-Za-z0-9_.-]{{1,128}}", code="E003"
        )
    if name in RESERVED_IDS:
        raise MEDH5ValidationError(f"{what} {name!r} is reserved", code="E003")
    return name


def validate_sample_key(name: str) -> str:
    """Check a collection sample key against spec §2.2 and return it unchanged."""
    if not SAMPLE_KEY_PATTERN.match(name):
        raise MEDH5ValidationError(
            f"sample key {name!r} must match [A-Za-z0-9_.-]{{1,255}}", code="E003"
        )
    return name


# --------------------------------------------------------------------------
# Attribute decoding.  h5py returns bytes or str depending on version and on
# how the file was written; spec §2.5 requires readers to accept both.
# --------------------------------------------------------------------------


def as_str(value: Any) -> str:
    """Normalise an HDF5 string attribute to :class:`str`."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return as_str(value[()])
    return str(value)


def as_str_tuple(value: Any) -> tuple[str, ...]:
    """Normalise an HDF5 string-list attribute to a tuple of :class:`str`."""
    if isinstance(value, (str, bytes)):
        return (as_str(value),)
    return tuple(as_str(v) for v in value)


def as_int(value: Any) -> int:
    return int(value)


def as_int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(v) for v in np.atleast_1d(np.asarray(value)))


def as_float(value: Any) -> float:
    return float(value)


def as_float_tuple(value: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in np.atleast_1d(np.asarray(value)))


def as_bool(value: Any) -> bool:
    return bool(np.asarray(value).reshape(()).item())


def as_matrix(value: Any) -> npt.NDArray[np.float64]:
    """Normalise a matrix attribute to a 2-D float64 array (spec §2.5)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise MEDH5ValidationError(
            f"matrix attribute must be stored 2-D, got shape {arr.shape}", code="E109"
        )
    return arr


# --------------------------------------------------------------------------
# Attribute encoding
# --------------------------------------------------------------------------


def encode_attr(value: Any) -> Any:
    """Encode a Python value for ``obj.attrs[...]`` following spec §2.5.

    Strings become variable-length UTF-8, string sequences become 1-D arrays of
    them (never a JSON blob), matrices stay 2-D, and scalars keep an explicit
    width so a reader never has to guess.
    """
    if isinstance(value, str):
        return np.array(value, dtype=str_dtype())
    if isinstance(value, (bool, np.bool_)):
        return np.bool_(value)
    if isinstance(value, (int, np.integer)):
        return np.int64(value)
    if isinstance(value, (float, np.floating)):
        return np.float64(value)
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, Sequence):
        seq = list(value)
        if not seq:
            return np.empty((0,), dtype=np.int64)
        if all(isinstance(v, str) for v in seq):
            return np.array(seq, dtype=str_dtype())
        if all(isinstance(v, (bool, np.bool_)) for v in seq):
            return np.array(seq, dtype=np.bool_)
        if all(isinstance(v, (int, np.integer)) for v in seq):
            return np.array(seq, dtype=np.int64)
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in seq):
            return np.array(seq, dtype=np.float64)
        return np.asarray(seq)
    raise MEDH5ValidationError(f"cannot encode attribute value of type {type(value)!r}")


def set_attrs(obj: Any, attrs: Mapping[str, Any]) -> None:
    """Write a mapping of attributes, skipping ``None`` values."""
    for key, value in attrs.items():
        if value is None:
            continue
        obj.attrs[key] = encode_attr(value)


def has_attr(obj: Any, name: str) -> bool:
    return name in obj.attrs


def require_attr(obj: Any, name: str, *, code: str = "E109") -> Any:
    """Fetch an attribute, raising a coded validation error when it is absent."""
    try:
        return obj.attrs[name]
    except KeyError:
        raise MEDH5ValidationError(
            f"{obj.name}: required attribute {name!r} is missing", code=code
        ) from None


# --------------------------------------------------------------------------
# File open / atomic create / copy-on-write
# --------------------------------------------------------------------------


def open_h5(path: str | os.PathLike[str], mode: str = "r") -> h5py.File:
    """Open an HDF5 file, mapping OS-level failures onto :class:`MEDH5FileError`."""
    try:
        return h5py.File(str(path), mode)
    except OSError as exc:
        raise MEDH5FileError(f"failed to open {os.fspath(path)!r}: {exc}") from exc


def _fsync_path(path: Path) -> None:
    """Flush a closed file to stable storage before it is renamed into place.

    Windows commits a file only through a handle opened for writing ---
    ``os.fsync`` on a read-only descriptor fails with EBADF there --- so the
    descriptor is read-write on that platform and read-only everywhere else,
    where write access would need the file's mode to allow it.
    """
    flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
    fd = os.open(str(path), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory fds
        return
    try:
        with contextlib.suppress(OSError):  # not every filesystem supports it
            os.fsync(fd)
    finally:
        os.close(fd)


def _temporary_name(name: str) -> str:
    """A sibling name no other writer in any process is using.

    The pid alone told two threads of one process apart from nothing: both
    built ``.x.medh5.tmp-1234``, and the second ``os.replace`` moved a file the
    first was still writing.
    """
    return f".{name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _existing_mode(target: Path) -> int | None:
    """The permission bits of *target*, or ``None`` when it does not exist."""
    try:
        return stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        return None


@contextmanager
def atomic_h5(
    path: str | os.PathLike[str], *, libver: str | tuple[str, str] = "latest"
) -> Iterator[h5py.File]:
    """Create an HDF5 file atomically (spec §14.4).

    Writes to a sibling temporary file, fsyncs it, then ``os.replace``s it onto
    *path* and fsyncs the directory.  A reader therefore never observes a
    partially written file, and a crash leaves the previous file intact.

    When *path* already exists its permission bits are carried onto the
    replacement.  Every copy-on-write command --- ``amend``, ``scrub --apply``,
    ``fix``, ``recompress`` --- goes through here, and these are the commands
    most likely to be pointed at data whose mode *is* its access control: a
    ``0o600`` sample came back ``0o644`` and became world-readable, which is a
    quiet way to widen access to exactly the files a site restricted on purpose.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(_temporary_name(target.name))
    mode = _existing_mode(target)
    handle = None
    try:
        handle = h5py.File(str(tmp), "w", libver=libver)
        yield handle
        handle.close()
        handle = None
        if mode is not None:
            with contextlib.suppress(OSError):  # best effort; not every FS obeys
                os.chmod(tmp, mode)
        _fsync_path(tmp)
        os.replace(str(tmp), str(target))
        _fsync_dir(target.parent)
    except BaseException:
        if handle is not None:
            handle.close()
        if tmp.exists():
            with contextlib.suppress(OSError):  # best-effort cleanup
                tmp.unlink()
        raise


def copy_object(src: h5py.Group, name: str, dst: h5py.Group) -> None:
    """Copy one object (group or dataset), attributes included, into *dst*."""
    src.copy(name, dst, name=name, expand_soft=True, expand_external=True)


@contextmanager
def atomic_rewrite(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str] | None = None,
    *,
    libver: str | tuple[str, str] = "latest",
) -> Iterator[tuple[h5py.File, h5py.File]]:
    """Rebuild a file from an existing one, atomically.  Yields ``(src, dst)``.

    The source handle is closed **before** the replace, which
    ``with open_h5(...) as src, atomic_h5(...) as dst:`` cannot do --- context
    managers exit right-to-left, so the replace ran while the source was still
    open.  POSIX allows that; Windows does not, so every rewrite of a file onto
    itself (`repack`, and `recompress` without ``out=``) failed there.

    *target* defaults to *source*, which is the rewrite-in-place case.
    """
    src_path = Path(os.fspath(source))
    dst_path = Path(os.fspath(target)) if target is not None else src_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(_temporary_name(dst_path.name))
    mode = _existing_mode(dst_path)
    src: h5py.File | None = None
    dst: h5py.File | None = None
    try:
        src = open_h5(src_path, "r")
        dst = h5py.File(str(tmp), "w", libver=libver)
        yield src, dst
        dst.close()
        dst = None
        src.close()
        src = None
        if mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(tmp, mode)
        _fsync_path(tmp)
        os.replace(str(tmp), str(dst_path))
        _fsync_dir(dst_path.parent)
    except BaseException:
        for handle in (dst, src):
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
        raise


def repack(path: str | os.PathLike[str]) -> None:
    """Rewrite *path* so freed space is not carried forward (spec §14.4).

    HDF5 does not reclaim storage.  An amend that copies an object and *then*
    rewrites one of its attributes leaves the superseded value physically in the
    new file, where ``strings`` still finds it even though every API read
    returns the new one.  For most edits that is only wasted bytes; for
    de-identification it is the difference between a pseudonymised file and one
    that still carries the original DICOM UID.

    This copies each top-level object into a fresh file, so only current values
    are written.  Filters, chunking and attributes come across untouched --- it
    is a compaction, not a re-encode, so every digest and the ``content_id``
    survive it.
    """
    with atomic_rewrite(path) as (src, dst):
        for name in src:
            copy_object(src, name, dst)
        for key, value in src.attrs.items():
            dst.attrs[key] = value


def copy_unknown(
    src: h5py.Group, dst: h5py.Group, known: Sequence[str]
) -> tuple[str, ...]:
    """Copy every child of *src* not in *known* into *dst* (spec §14.4).

    Amending a file written by a future minor version must not silently drop the
    objects that version added, so an amend copies everything it does not
    recognise straight through.  Returns the names it copied.
    """
    kept = tuple(name for name in src if name not in set(known))
    for name in kept:
        copy_object(src, name, dst)
    return kept


def copy_root_attrs(src: h5py.Group, dst: h5py.Group, skip: Sequence[str] = ()) -> None:
    """Copy attributes from *src* to *dst*, skipping the named ones."""
    blocked = set(skip)
    for key, value in src.attrs.items():
        if key not in blocked:
            dst.attrs[key] = value


__all__ = [
    "ID_PATTERN",
    "RESERVED_IDS",
    "SAMPLE_KEY_PATTERN",
    "as_bool",
    "as_float",
    "as_float_tuple",
    "as_int",
    "as_int_tuple",
    "as_matrix",
    "as_str",
    "as_str_tuple",
    "atomic_h5",
    "atomic_rewrite",
    "copy_object",
    "copy_root_attrs",
    "copy_unknown",
    "encode_attr",
    "has_attr",
    "open_h5",
    "repack",
    "require_attr",
    "set_attrs",
    "str_dtype",
    "validate_id",
    "validate_sample_key",
]
