"""File integrity helpers -- checksum storage and verification.

Checksums are stored as HDF5 root attributes using SHA-256 over the
datasets and critical metadata that define a sample's contents.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import h5py
import numpy as np

from medh5.meta import _IMAGE_META_ATTRS, _ROOT_META_ATTRS

_CHECKSUM_ATTR = "checksum_sha256"


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _normalize_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, np.generic):
        return _normalize_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U", "O"}:
            payload = _json_safe(value.tolist())
            return json.dumps(payload, ensure_ascii=True, sort_keys=True).encode(
                "utf-8"
            )
        arr = np.ascontiguousarray(value)
        return arr.tobytes()
    if isinstance(value, (list, tuple, dict, bool, int, float)) or value is None:
        return json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return repr(value).encode("utf-8")


def _hash_attrs(h: Any, attrs: h5py.AttributeManager, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in attrs:
            continue
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(_normalize_value(attrs[key]))
        h.update(b"\0")


def _hash_dataset(h: Any, name: str, ds: h5py.Dataset) -> None:
    h.update(name.encode("utf-8"))
    h.update(b"\0")
    h.update(str(ds.shape).encode("ascii"))
    h.update(b"\0")
    h.update(str(ds.dtype).encode("ascii"))
    h.update(b"\0")
    if ds.chunks is not None:
        for slices in ds.iter_chunks():
            data = ds[slices]
            h.update(_normalize_value(np.asarray(data)))
    else:
        h.update(_normalize_value(np.asarray(ds[...])))
    h.update(b"\0")


def compute_checksum(f: h5py.File) -> str:
    """Compute a SHA-256 hex digest over datasets and critical metadata in *f*."""
    h = hashlib.sha256()
    _hash_attrs(h, f.attrs, _ROOT_META_ATTRS)
    img_grp = f["images"]
    _hash_attrs(h, img_grp.attrs, _IMAGE_META_ATTRS)
    for name in sorted(img_grp.keys()):
        _hash_dataset(h, f"images/{name}", img_grp[name])
    seg_grp = f.get("seg")
    if isinstance(seg_grp, h5py.Group):
        for name in sorted(seg_grp.keys()):
            _hash_dataset(h, f"seg/{name}", seg_grp[name])
    for name in ("bboxes", "bbox_scores", "bbox_labels"):
        if name in f:
            _hash_dataset(h, name, f[name])
    return h.hexdigest()


def write_checksum(f: h5py.File) -> str:
    """Compute and store a checksum on *f*.  Returns the hex digest."""
    digest = compute_checksum(f)
    f.attrs[_CHECKSUM_ATTR] = digest
    return digest


def verify_checksum(f: h5py.File) -> bool:
    """Return *True* if the stored checksum matches the data.

    Returns *True* if no checksum is stored (opt-in verification).
    """
    stored = f.attrs.get(_CHECKSUM_ATTR)
    if stored is None:
        return True
    if isinstance(stored, bytes):
        stored = stored.decode()
    return bool(compute_checksum(f) == stored)
