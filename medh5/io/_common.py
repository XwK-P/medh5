"""Helpers every converter needs, each of which used to carry its own copy.

Four importers minted label-set keys with four private ``_key`` functions and
two built filename stems with two more.  Copies drift: the keys were built to
§2.3's identifier rule --- ``[A-Za-z0-9_.-]`` --- when the sample-document
schema binds a class ``key`` to ``^[a-z0-9][a-z0-9_]*$``, so an ROI named
``GTV-1`` minted ``gtv-1`` and the write failed E005 at commit, one copy at a
time.  One function, held to the schema, cannot disagree with itself.
"""

from __future__ import annotations

from medh5 import _core


def sanitize_key(name: str, *, fallback: str = "class") -> str:
    """A label-set ``key`` from free text (§5.2): ``^[a-z0-9][a-z0-9_]*$``.

    Lowercased ASCII letters and digits are kept, everything else becomes an
    underscore, leading and trailing underscores are dropped, and *fallback*
    stands in for a name with nothing left --- or prefixes one that would
    otherwise start with an underscore, which the schema refuses.
    """
    return str(_core.io_sanitize_key(name, fallback=fallback))


def sanitize_stem(text: str, *, limit: int = 200) -> str:
    """A filename stem from free text: identifier characters only, truncated."""
    return str(_core.io_sanitize_stem(text, limit=limit))


__all__ = ["sanitize_key", "sanitize_stem"]
