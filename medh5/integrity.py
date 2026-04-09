"""File integrity helpers -- checksum storage and verification.

Checksums are stored as HDF5 root attributes using SHA-256 over the
concatenated raw bytes of all image datasets (sorted by name).
"""

from __future__ import annotations

import hashlib

import h5py
import numpy as np

_CHECKSUM_ATTR = "checksum_sha256"


def compute_checksum(f: h5py.File) -> str:
    """Compute a SHA-256 hex digest over all image data in *f*."""
    h = hashlib.sha256()
    img_grp = f["images"]
    for name in sorted(img_grp.keys()):
        data = np.ascontiguousarray(img_grp[name][...])
        h.update(data.tobytes())
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
