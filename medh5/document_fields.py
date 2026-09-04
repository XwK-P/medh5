"""One rule for the closed objects of the sample document (§2.4).

Four of the document's objects are ``additionalProperties: false`` in the
schema --- ``timepoint``, ``qualityRecord``, ``activity`` and ``agent`` --- so a
key outside their vocabulary makes the whole document **E005** and the writer
refuses it at ``commit()``.  Three of the four none the less had an ``extra``
mapping that collected unknown keys and wrote them straight back out: an API
that looked like an extension point, could not reach a file, and failed at the
end of the build rather than at the call that introduced the value.

Rejecting the key where it is parsed is the same answer, given earlier and with
the field named.  It also keeps an amend honest: a mapping that could hold what
the schema forbids would either be dropped silently on the way out or make the
rewrite fail, and neither is something a caller can act on.

The open objects --- ``identity``, ``cohort``, ``extra`` and ``acquisition`` ---
keep their ``extra`` mappings, because for them the schema permits it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

from medh5.errors import MEDH5ValidationError


def check_known(doc: Mapping[str, Any], known: Collection[str], *, what: str) -> None:
    """Refuse a key the schema does not allow on this object (E005)."""
    unknown = sorted(k for k in doc if k not in known)
    if unknown:
        raise MEDH5ValidationError(
            f"{what}: {unknown} is not a {what} field; the schema closes this "
            f"object, so it may hold only {sorted(known)}",
            code="E005",
        )


__all__ = ["check_known"]
