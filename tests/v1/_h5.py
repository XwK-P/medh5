"""h5py helpers for tests that craft files the writer would refuse.

The package no longer reads or writes through h5py --- the format engine does
both --- but a test that has to break a file on purpose (drop an attribute,
corrupt a dataset, write a 0.x layout) still needs a second, independent
writer.  These are the attribute codecs 1.x used, kept here for that purpose
only.  Close every h5py handle before the package reads the file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np


def str_dtype() -> Any:
    """The variable-length UTF-8 string dtype used for every string in a file."""
    return h5py.string_dtype(encoding="utf-8")


def as_str(value: Any) -> str:
    """Normalise an HDF5 string attribute to :class:`str`."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return as_str(value[()])
    return str(value)


def as_str_tuple(value: Any) -> tuple[str, ...]:
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


def encode_attr(value: Any) -> Any:
    """Encode a Python value for ``obj.attrs[...]`` following spec §2.5."""
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
    raise TypeError(f"cannot encode attribute value of type {type(value)!r}")


__all__ = [
    "as_bool",
    "as_float",
    "as_float_tuple",
    "as_int",
    "as_int_tuple",
    "as_str",
    "as_str_tuple",
    "encode_attr",
    "str_dtype",
]
