"""Helpers every converter needs, each of which used to carry its own copy.

Four importers minted label-set keys with four private ``_key`` functions and
two built filename stems with two more.  Copies drift: the keys were built to
§2.3's identifier rule --- ``[A-Za-z0-9_.-]`` --- when the sample-document
schema binds a class ``key`` to ``^[a-z0-9][a-z0-9_]*$``, so an ROI named
``GTV-1`` minted ``gtv-1`` and the write failed E005 at commit, one copy at a
time.  One function, held to the schema, cannot disagree with itself.
"""

from __future__ import annotations

import unicodedata

from medh5 import _core


def sanitize_key(name: str, *, fallback: str = "class") -> str:
    """A label-set ``key`` from free text (§5.2): ``^[a-z0-9][a-z0-9_]*$``.

    The text is lowercased; ASCII letters, digits and underscores are kept, and
    everything else becomes an underscore.  Leading and trailing underscores
    are dropped, so what is left starts with a letter or a digit, as the
    schema requires; *fallback* stands in for a name with nothing left, and a
    key is cut at 128 characters.
    """
    return str(_core.io_sanitize_key(name, fallback=fallback))


def sanitize_stem(text: str, *, limit: int = 200) -> str:
    """A filename stem from free text: identifier characters only, truncated."""
    return str(_core.io_sanitize_stem(text, limit=limit))


def sanitize_ascii_stem(text: str, *, limit: int = 200) -> str:
    """*text* as a sample id or a file name stem: ASCII identifier characters.

    :func:`sanitize_stem` keeps every alphanumeric character, and a sample id
    is ``[A-Za-z0-9_.-]`` (§2.2), so a name like ``Müller Hans`` made an id
    the writer refused (E003).  Accents are dropped --- ``Muller_Hans`` ---
    and any other character outside ASCII becomes ``_``, as punctuation does.
    Text that is already such an id comes back unchanged (up to *limit*).
    """
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return "".join(
        c if c.isascii() else "_" for c in sanitize_stem(folded, limit=limit)
    )


__all__ = ["sanitize_ascii_stem", "sanitize_key", "sanitize_stem"]
