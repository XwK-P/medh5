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
    if isinstance(value, (bytes, np.bytes_)):
        # A fixed-length string as h5py reads one back.  `bytes` is a
        # Sequence, and the branch below would store its code points.
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
    """Open an HDF5 file, mapping OS-level failures onto :class:`MEDH5FileError`.

    Every file is also checked to be self-contained before it is handed back
    (see :func:`check_self_contained`), because this is the one door that
    readers, validators and every copy-on-write path open files through.
    """
    try:
        handle = h5py.File(str(path), mode)
    except OSError as exc:
        raise MEDH5FileError(f"failed to open {os.fspath(path)!r}: {exc}") from exc
    try:
        check_self_contained(handle, path)
    except BaseException:
        handle.close()
        raise
    return handle


_SELF_CONTAINED: dict[tuple[int, ...], None] = {}
"""Files already checked, keyed by the identity of the file a handle had open:
``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)``.

Every write is an atomic replace (§14.4), so a file that changes gets a new
inode; an in-place edit changes the ctime even where the mtime is put back.  A
training loop re-opens the same few thousand files for the length of a run;
this keeps the check off that path after the first open of each.
"""

_SELF_CONTAINED_LIMIT = 65_536


def _opened_key(
    handle: h5py.File, path: str | os.PathLike[str] | None
) -> tuple[int, ...] | None:
    """The identity of the file *handle* holds open --- not whatever *path* names now.

    Between an open and a stat by name, another process can atomically replace
    the path; caching the replacement's identity as checked would let it skip
    the check on its next open, and the next ``recompress`` would copy whatever
    it points at.  So on POSIX the identity comes from the descriptor HDF5 read
    through.  Windows refuses to replace a file another handle holds open, so
    there the path names the opened file for as long as the handle lives --- and
    HDF5's descriptor belongs to its own C runtime, which Python's cannot
    ``fstat`` --- so a stat by name gives the same answer.
    """
    try:
        if os.name == "nt":
            if path is None:
                return None
            st = os.stat(os.fspath(path))
        else:
            fd = handle.id.get_vfd_handle()
            if not isinstance(fd, int):
                return None
            st = os.fstat(fd)
    except Exception:
        # A driver without a descriptor, or a vanished path: no key, so the
        # file is checked on every open rather than trusted on a guess.
        return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def outside_references(handle: h5py.Group) -> list[tuple[str, str]]:
    """Objects that read bytes from outside the file: ``(path, what)`` pairs.

    Three HDF5 features do: a dataset whose raw data lives in *external
    storage* (any file on the reader's disk), a *virtual dataset* mapping other
    files, and an *external link*.  A MEDH5 file is a self-contained sample
    (§2), and a tool that follows one of these copies bytes it was never given
    --- ``recompress`` of a crafted file used to write the contents of a local
    private key into its output.
    """
    from h5py import h5d, h5l, h5o

    fid = handle.id
    found: list[tuple[str, str]] = []

    def visit(name: bytes, info: Any) -> None:
        text = name.decode("utf-8", "replace")
        if info.type == h5l.TYPE_EXTERNAL:
            found.append((text, "an external link"))
            return None
        if info.type != h5l.TYPE_HARD:
            return None
        try:
            if h5o.get_info(fid, name).type != h5o.TYPE_DATASET:
                return None
            plist = h5d.open(fid, name).get_create_plist()
            layout, external = plist.get_layout(), plist.get_external_count()
        except Exception:
            # An object whose header cannot be read cannot have its data read
            # either, so it cannot leak anything; the rules that touch it report
            # the damage.  Raising here, inside HDF5's visit, surfaced as an
            # unrelated-looking SystemError.
            return None
        if layout == h5d.VIRTUAL:
            found.append((text, "a virtual dataset"))
        elif external:
            found.append((text, "external raw-data storage"))
        return None

    try:
        fid.links.visit(visit, info=True)
    except Exception as exc:
        raise MEDH5FileError(
            f"{handle.filename!r} could not be walked to check that it is "
            f"self-contained: {type(exc).__name__}: {exc}"
        ) from exc
    return found


def check_self_contained(
    handle: h5py.File, path: str | os.PathLike[str] | None = None
) -> None:
    """Refuse a file that reads bytes from outside itself (``MEDH5FileError``).

    See :func:`outside_references`.  The result is memoised per identity of the
    file *handle* has open (:func:`_opened_key`), so re-opening an unchanged
    file costs one ``stat``.
    """
    key = _opened_key(handle, path)
    if key is not None and key in _SELF_CONTAINED:
        return
    found = outside_references(handle)
    if found:
        named = "; ".join(f"/{name} is {what}" for name, what in found[:5])
        more = f" (and {len(found) - 5} more)" if len(found) > 5 else ""
        where = os.fspath(path) if path is not None else handle.filename
        raise MEDH5FileError(
            f"{where!r} is not self-contained: {named}{more}. A MEDH5 file holds "
            "its own bytes (§2); following these would read files on this "
            "machine that the file's author chose, so the file is refused"
        )
    if key is not None:
        if len(_SELF_CONTAINED) >= _SELF_CONTAINED_LIMIT:
            _SELF_CONTAINED.clear()
        _SELF_CONTAINED[key] = None


def _fsync_path(path: Path) -> None:
    """Flush a closed file to stable storage before it is renamed into place.

    Windows commits a file only through a handle opened for writing ---
    ``os.fsync`` on a read-only descriptor fails with EBADF there --- so the
    descriptor is read-write on that platform and read-only everywhere else,
    where write access would need the file's mode to allow it.  It is also
    opened in binary mode there: the C runtime's default text mode treats a
    trailing 0x1A as an end-of-file mark and strips it from a writable file
    on open, and one HDF5 file in 256 ends with that byte.
    """
    flags = os.O_RDONLY
    if os.name == "nt":
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
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


def _precreate(tmp: Path, mode: int | None) -> None:
    """Create the temporary file before any data, readable by its owner only.

    Restoring the mode after the write left a window: for as long as a large
    amend ran, a ``0o600`` sample's new contents sat in a sibling created with
    the umask's default --- usually world-readable.  HDF5 truncates an existing
    file without touching its mode, so creating it first, restricted, closes the
    window, and the ``chmod`` after the write sets the target's exact bits.

    Restricted means owner read-write, not the target's own mode: HDF5 reopens
    the file for writing, which a read-only mode such as ``0o444`` refuses to
    its owner, so a read-only sample could no longer be amended at all.  A new
    file, with no target mode to protect, is created as a writer would create
    it, subject to the umask.
    """
    fd = os.open(
        str(tmp),
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
        0o666 if mode is None else 0o600,
    )
    os.close(fd)


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
        _precreate(tmp, mode)
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
    """Copy one object (group or dataset), attributes included, into *dst*.

    External links are *not* expanded.  Expanding one copies another file's
    contents into this one; :func:`open_h5` refuses a file that carries one, so
    this is the second line of defence rather than the first.
    """
    src.copy(name, dst, name=name, expand_soft=True, expand_external=False)


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
        _precreate(tmp, mode)
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
    from medh5.sample import require_major

    with atomic_rewrite(path) as (src, dst):
        require_major(src, path)
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
    standard = set(known)
    kept = tuple(name for name in src if name not in standard)
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
    "check_self_contained",
    "copy_object",
    "copy_root_attrs",
    "copy_unknown",
    "encode_attr",
    "has_attr",
    "open_h5",
    "outside_references",
    "repack",
    "require_attr",
    "set_attrs",
    "str_dtype",
    "validate_id",
    "validate_sample_key",
]
