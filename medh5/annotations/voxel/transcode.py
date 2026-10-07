"""Converting between voxel encodings, losslessly or not at all (spec §7.6).

A transcode refuses rather than dropping what the target cannot express: an
in-band ignore region, object identity, class identity itself.
"""

from __future__ import annotations

from medh5 import _core

TRANSCODABLE: tuple[str, ...] = _core.TRANSCODABLE
IN_BAND_IGNORE_KINDS: tuple[str, ...] = _core.IN_BAND_IGNORE_KINDS

payload_to_masks = _core.payload_to_masks
annotation_to_masks = _core.annotation_to_masks
encode_masks = _core.encode_masks
transcode_payload = _core.transcode_payload
transcode = _core.transcode
masks_equal = _core.masks_equal
check_roundtrip = _core.check_roundtrip

__all__ = [
    "IN_BAND_IGNORE_KINDS",
    "TRANSCODABLE",
    "annotation_to_masks",
    "check_roundtrip",
    "encode_masks",
    "masks_equal",
    "payload_to_masks",
    "transcode",
    "transcode_payload",
]
