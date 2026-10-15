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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from medh5 import _core

if TYPE_CHECKING:  # pragma: no cover - typing only
    from medh5.io.report import ConversionReport


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
    """A filename stem from free text: ASCII identifier characters, truncated.

    Latin letters lose their accents (``Müller`` is ``Muller``), and any other
    character outside ``[A-Za-z0-9_.-]`` becomes ``_``.
    """
    return str(_core.io_sanitize_stem(text, limit=limit))


def sanitize_ascii_stem(text: str, *, limit: int = 200) -> str:
    """*text* as a sample id or a file name stem: ASCII identifier characters.

    A sample id is ``[A-Za-z0-9_.-]`` (§2.2), so a name like ``Müller Hans``
    must lose its accents --- ``Muller_Hans`` --- or the writer refuses it
    (E003).  :func:`sanitize_stem` folds the Latin-1 and Latin Extended-A
    letters itself; this decomposes *text* first (NFKD), so an accented
    letter beyond them folds to its base letter too.  Any other character
    outside ASCII becomes ``_``, as punctuation does.  Text that is already
    such an id comes back unchanged (up to *limit*).
    """
    folded = "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )
    return "".join(
        c if c.isascii() else "_" for c in sanitize_stem(folded, limit=limit)
    )


def examined_classes(
    annotated_classes: Sequence[int | str] | str,
    label_set: Any,
    *,
    mint: bool,
    log: ConversionReport,
    supplied: str,
) -> tuple[Any, Sequence[int] | str]:
    """*label_set* and *annotated_classes*, a class named that the input
    does not supply recorded as examined and found absent (§11.3).

    ``"all_given"`` and ``"all"`` pass through.  A list names classes by key,
    by the key a name makes (:func:`sanitize_key`), or by id, and comes back
    as their ids.  A name *label_set* does not hold is a class someone looked
    for and did not find.  Where the import mints its label set (*mint*), the
    class is minted after the ids the input supplies, with a ``label_set``
    decision naming it; where the label set was given (the sample's, or the
    caller's), it must hold the class already, and the import is refused
    (E402).  Looked up as a key, such a class was a bare ``KeyError`` from
    the writer.  *supplied* names the input's classes, for the note and the
    refusal: ``"the masks"``, ``"the SEG's segments"``, ``"the ROIs"``.
    """
    from medh5.errors import MEDH5ValidationError
    from medh5.labels import LabelClass, LabelSet

    if isinstance(annotated_classes, str):
        return label_set, annotated_classes

    def found(entry: int | str) -> Any:
        if isinstance(entry, str):
            hit = label_set.get(entry)
            return hit if hit is not None else label_set.get(sanitize_key(entry))
        return label_set.get(int(entry))

    absent = [entry for entry in annotated_classes if found(entry) is None]
    if absent and not mint:
        raise MEDH5ValidationError(
            f"annotated_classes names {[str(e) for e in absent]}, which label "
            f"set {label_set.id!r} does not hold and {supplied} do not supply: "
            "a class examined and found absent must be a class of the label "
            "set --- add it there, or leave it out of annotated_classes "
            "(--annotated-classes)",
            code="E402",
        )
    if absent:
        numbers = [e for e in absent if not isinstance(e, str)]
        if numbers:
            raise MEDH5ValidationError(
                f"annotated_classes names class id(s) {numbers}, which "
                f"{supplied} do not supply; a class to mint is named by its "
                "key, not its id",
                code="E402",
            )
        classes = list(label_set.classes)
        minted: dict[str, int] = {}
        next_id = max(label_set.ids, default=0) + 1
        for entry in absent:
            key = sanitize_key(str(entry))
            if key in minted:
                continue
            name = key.replace("_", " ").title() if key == entry else str(entry)
            classes.append(LabelClass(next_id, key, name))
            minted[key] = next_id
            next_id += 1
        label_set = LabelSet(label_set.id, version=label_set.version, classes=classes)
        log.decision(
            "label_set",
            f"class(es) {sorted(minted)} are named in annotated_classes and "
            f"{supplied} do not supply them: each was examined and found "
            "absent, so it was minted after the supplied ids, with no voxel "
            "or shape of its own",
            {"examined_absent": minted},
        )
    ids: list[int] = []
    for entry in annotated_classes:
        class_id = int(found(entry).id)
        if class_id not in ids:
            ids.append(class_id)
    return label_set, ids


__all__ = ["examined_classes", "sanitize_ascii_stem", "sanitize_key", "sanitize_stem"]
