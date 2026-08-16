"""Label sets: id -> meaning (spec §5).

Annotations reference classes by ``uint16`` id and never by name, so a label set
is the only thing standing between an integer and a diagnosis.  Two properties
matter more than the data model:

* The hierarchy is a **DAG, not a tree**.  ``left_kidney`` is a ``kidney`` and is
  part of the urinary system; forcing that into a tree loses one of the two.
* ``closure`` is declared per annotation, never inferred.  A reader that helpfully
  adds ``liver`` because ``liver_segment_iv`` is present has invented ground
  truth, so the spec forbids it unless ``closure = "implicit"`` says otherwise.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from medh5.errors import MEDH5ValidationError

BACKGROUND_ID = 0
"""Explicitly *not* any class.  Never appears in ``classes``."""

IGNORE_ID = 65535
"""Outside what was annotated: neither foreground nor background."""

MAX_CLASS_ID = 65534

CLOSURES = ("explicit", "implicit")

FORMS = ("inline", "ref")

INLINE_REQUIRED_BELOW = 4096
"""At or below this class count, ``form: inline`` is REQUIRED (spec §5.1)."""


@dataclass(frozen=True, slots=True)
class OntologyCode:
    """A binding to an external vocabulary (SNOMED-CT, RadLex, FMA, UBERON, ...)."""

    system: str
    code: str
    name: str | None = None

    def to_json(self) -> dict[str, Any]:
        out = {"system": self.system, "code": self.code}
        if self.name is not None:
            out["name"] = self.name
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> OntologyCode:
        return cls(
            system=str(doc["system"]), code=str(doc["code"]), name=doc.get("name")
        )


@dataclass(frozen=True, slots=True)
class Relation:
    """A non-``is_a`` edge between classes, e.g. ``part_of`` or ``adjacent_to``."""

    subject: int
    predicate: str
    object: int

    def to_json(self) -> dict[str, Any]:
        return {
            "subject": int(self.subject),
            "predicate": self.predicate,
            "object": int(self.object),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Relation:
        return cls(
            subject=int(doc["subject"]),
            predicate=str(doc["predicate"]),
            object=int(doc["object"]),
        )


@dataclass(frozen=True, slots=True)
class Skeleton:
    """A keypoint topology declared by the vocabulary (spec §5.5)."""

    id: str
    keypoints: tuple[int, ...]
    edges: tuple[tuple[int, int], ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "keypoints": [int(k) for k in self.keypoints],
            "edges": [[int(a), int(b)] for a, b in self.edges],
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Skeleton:
        return cls(
            id=str(doc["id"]),
            keypoints=tuple(int(k) for k in doc["keypoints"]),
            edges=tuple((int(a), int(b)) for a, b in doc.get("edges") or ()),
        )


@dataclass(frozen=True, slots=True)
class LabelClass:
    """One semantic class (spec §5.2)."""

    id: int
    key: str
    name: str
    parents: tuple[int, ...] = ()
    category: str | None = None
    color: tuple[int, int, int, int] | None = None
    codes: tuple[OntologyCode, ...] = ()
    laterality: str | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not BACKGROUND_ID < self.id <= MAX_CLASS_ID:
            raise MEDH5ValidationError(
                f"class {self.key!r}: id {self.id} must be in 1..{MAX_CLASS_ID} "
                f"({BACKGROUND_ID} is background, {IGNORE_ID} is ignore)",
                code="E303",
            )
        if not self.key or not self.name:
            raise MEDH5ValidationError(
                f"class id {self.id}: both key and name are required", code="E306"
            )
        object.__setattr__(self, "parents", tuple(int(p) for p in self.parents))
        object.__setattr__(self, "codes", tuple(self.codes))
        if self.color is not None:
            color = tuple(int(c) for c in self.color)
            if len(color) != 4 or any(not 0 <= c <= 255 for c in color):  # noqa: PLR2004
                raise MEDH5ValidationError(
                    f"class {self.key!r}: color must be four 0-255 RGBA values",
                    code="E306",
                )
            object.__setattr__(self, "color", color)

    @property
    def is_lesion(self) -> bool:
        return bool(self.properties.get("is_lesion", self.category == "lesion"))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": int(self.id), "key": self.key, "name": self.name}
        if self.parents:
            out["parents"] = [int(p) for p in self.parents]
        if self.category is not None:
            out["category"] = self.category
        if self.color is not None:
            out["color"] = list(self.color)
        if self.codes:
            out["codes"] = [c.to_json() for c in self.codes]
        if self.laterality is not None:
            out["laterality"] = self.laterality
        if self.properties:
            out["properties"] = dict(self.properties)
        return out

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> LabelClass:
        color = doc.get("color")
        return cls(
            id=int(doc["id"]),
            key=str(doc["key"]),
            name=str(doc["name"]),
            parents=tuple(doc.get("parents") or ()),
            category=doc.get("category"),
            color=cast(
                "tuple[int, int, int, int] | None", tuple(color) if color else None
            ),
            codes=tuple(OntologyCode.from_json(c) for c in doc.get("codes") or ()),
            laterality=doc.get("laterality"),
            properties=dict(doc.get("properties") or {}),
        )


def canonical_json(doc: Mapping[str, Any]) -> bytes:
    """Deterministic UTF-8 serialization used for label-set digests.

    Sorted keys, no insignificant whitespace, non-ASCII kept as UTF-8.  Defining
    this is what makes ``label_set.sha256`` comparable across implementations
    and languages --- without it, "the digest of the canonical serialization"
    would mean whatever each writer's JSON library happened to emit.
    """
    return json.dumps(
        doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


class LabelSet:
    """A controlled vocabulary, inline in the file or referenced by URI."""

    __slots__ = (
        "_by_id",
        "_by_key",
        "_classes",
        "_declared_sha256",
        "form",
        "id",
        "relations",
        "skeletons",
        "uri",
        "version",
    )

    def __init__(
        self,
        id: str,
        classes: Sequence[LabelClass] = (),
        *,
        version: str = "1.0.0",
        relations: Sequence[Relation] = (),
        skeletons: Sequence[Skeleton] = (),
        form: str = "inline",
        uri: str | None = None,
        sha256: str | None = None,
    ) -> None:
        if form not in FORMS:
            raise MEDH5ValidationError(
                f"label set form {form!r} must be one of {list(FORMS)}", code="E305"
            )
        self.id = id
        self.version = version
        self.form = form
        self.uri = uri
        self._declared_sha256 = sha256
        self.relations = tuple(relations)
        self.skeletons = tuple(skeletons)
        self._classes = tuple(sorted(classes, key=lambda c: c.id))
        self._by_id = {c.id: c for c in self._classes}
        self._by_key = {c.key: c for c in self._classes}
        self.check()

    # -- validation --------------------------------------------------------

    def check(self) -> None:
        """Validate spec §5.1-§5.3 (E302, E303, E304, E305)."""
        if len(self._by_id) != len(self._classes):
            dupes = _duplicates(c.id for c in self._classes)
            raise MEDH5ValidationError(
                f"label set {self.id!r}: duplicate class ids {sorted(dupes)}",
                code="E302",
            )
        if len(self._by_key) != len(self._classes):
            dupes = _duplicates(c.key for c in self._classes)
            raise MEDH5ValidationError(
                f"label set {self.id!r}: duplicate class keys {sorted(dupes)}",
                code="E302",
            )
        for cls_ in self._classes:
            for parent in cls_.parents:
                if parent not in self._by_id:
                    raise MEDH5ValidationError(
                        f"label set {self.id!r}: class {cls_.key!r} names "
                        f"unknown parent id {parent}",
                        code="E306",
                    )
        self._check_acyclic()
        if self.form == "ref" and not self.uri:
            raise MEDH5ValidationError(
                f"label set {self.id!r}: form 'ref' requires a uri", code="E305"
            )
        if self.form == "ref" and not self._declared_sha256:
            raise MEDH5ValidationError(
                f"label set {self.id!r}: form 'ref' requires a sha256, so a reader "
                "that cannot resolve the uri still knows which vocabulary it needs",
                code="E305",
            )
        if self.form == "ref" and self._classes:
            raise MEDH5ValidationError(
                f"label set {self.id!r}: form 'ref' must not carry inline classes",
                code="E305",
            )
        for relation in self.relations:
            for endpoint in (relation.subject, relation.object):
                if endpoint not in self._by_id and self.form == "inline":
                    raise MEDH5ValidationError(
                        f"label set {self.id!r}: relation names unknown class "
                        f"{endpoint}",
                        code="E306",
                    )

    def _check_acyclic(self) -> None:
        colour: dict[int, int] = {}

        def visit(node: int, path: tuple[int, ...]) -> None:
            state = colour.get(node, 0)
            if state == 1:
                cycle = " -> ".join(self._by_id[n].key for n in (*path, node))
                raise MEDH5ValidationError(
                    f"label set {self.id!r}: hierarchy cycle {cycle}", code="E304"
                )
            if state == 2:  # noqa: PLR2004 - fully explored
                return
            colour[node] = 1
            for parent in self._by_id[node].parents:
                visit(parent, (*path, node))
            colour[node] = 2

        for cls_ in self._classes:
            visit(cls_.id, ())

    # -- lookup ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._classes)

    def __iter__(self) -> Iterator[LabelClass]:
        return iter(self._classes)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, int):
            return item in self._by_id
        return item in self._by_key

    def __repr__(self) -> str:
        return (
            f"LabelSet({self.id!r}, version={self.version!r}, "
            f"{len(self._classes)} classes, form={self.form!r})"
        )

    @property
    def classes(self) -> tuple[LabelClass, ...]:
        return self._classes

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(c.id for c in self._classes)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self._classes)

    def __getitem__(self, key: int | str) -> LabelClass:
        """Look up a class by id or by key."""
        table = cast(
            "Mapping[Any, LabelClass]",
            self._by_id if isinstance(key, int) else self._by_key,
        )
        try:
            return table[key]
        except KeyError:
            raise KeyError(f"label set {self.id!r} has no class {key!r}") from None

    def get(self, key: int | str) -> LabelClass | None:
        try:
            return self[key]
        except KeyError:
            return None

    def resolve(self, keys: Iterable[int | str]) -> tuple[LabelClass, ...]:
        """Resolve a mixed sequence of ids and keys to classes, in order."""
        return tuple(self[k] for k in keys)

    def ids_for(self, keys: Iterable[int | str]) -> tuple[int, ...]:
        """Resolve a mixed sequence of ids and keys to class ids, in order."""
        return tuple(self[k].id for k in keys)

    def missing(self, ids: Iterable[int]) -> tuple[int, ...]:
        """Ids not present in this label set --- the E402 check."""
        return tuple(sorted({i for i in ids if i not in self._by_id}))

    # -- hierarchy ---------------------------------------------------------

    def ancestors(self, key: int | str) -> tuple[int, ...]:
        """Transitive ``is_a`` ancestors of a class, nearest first, deduplicated."""
        start = self[key]
        seen: dict[int, None] = {}
        frontier = list(start.parents)
        while frontier:
            node = frontier.pop(0)
            if node in seen:
                continue
            seen[node] = None
            frontier.extend(self._by_id[node].parents)
        return tuple(seen)

    def descendants(self, key: int | str) -> tuple[int, ...]:
        """Every class having this one among its transitive ancestors."""
        target = self[key].id
        return tuple(c.id for c in self._classes if target in self.ancestors(c.id))

    def close(self, ids: Iterable[int], closure: str) -> tuple[int, ...]:
        """Apply a ``closure`` (spec §5.4) to a set of class ids.

        ``explicit`` returns the ids unchanged; ``implicit`` adds every ``is_a``
        ancestor.  Readers must never do this without a declared closure.
        """
        if closure not in CLOSURES:
            raise MEDH5ValidationError(
                f"closure {closure!r} must be one of {list(CLOSURES)}", code="E412"
            )
        ordered = list(dict.fromkeys(int(i) for i in ids))
        if closure == "explicit":
            return tuple(ordered)
        out: dict[int, None] = dict.fromkeys(ordered)
        for i in ordered:
            for ancestor in self.ancestors(i):
                out.setdefault(ancestor, None)
        return tuple(out)

    def relations_of(
        self, key: int | str, predicate: str | None = None
    ) -> tuple[Relation, ...]:
        subject = self[key].id
        return tuple(
            r
            for r in self.relations
            if r.subject == subject and (predicate is None or r.predicate == predicate)
        )

    def skeleton(self, skeleton_id: str) -> Skeleton:
        for sk in self.skeletons:
            if sk.id == skeleton_id:
                return sk
        raise KeyError(f"label set {self.id!r} has no skeleton {skeleton_id!r}")

    def colors(self) -> dict[int, tuple[int, int, int, int]]:
        """Class id -> RGBA, for viewers.  Classes without a colour are omitted."""
        return {c.id: c.color for c in self._classes if c.color is not None}

    # -- serialization -----------------------------------------------------

    def content_doc(self) -> dict[str, Any]:
        """The digested part of the label set: identity plus content, no carriage.

        ``form``, ``uri`` and ``sha256`` are excluded because they describe how
        the vocabulary is *carried*, not what it *says*; an inline copy and a
        referenced copy of the same vocabulary must digest identically.
        """
        doc: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "classes": [c.to_json() for c in self._classes],
        }
        if self.relations:
            doc["relations"] = sorted(
                (r.to_json() for r in self.relations),
                key=lambda r: (r["subject"], r["predicate"], r["object"]),
            )
        if self.skeletons:
            doc["skeletons"] = sorted(
                (s.to_json() for s in self.skeletons), key=lambda s: str(s["id"])
            )
        return doc

    def digest(self, algo: str = "sha256") -> str:
        """Hex digest of :meth:`content_doc` under :func:`canonical_json`.

        A ``form: ref`` set carries the digest rather than computing one: it has
        no inline classes to hash, and the point of the field is to identify the
        vocabulary a reader must fetch.
        """
        if self._declared_sha256 is not None and not self._classes:
            return self._declared_sha256
        return hashlib.new(algo, canonical_json(self.content_doc())).hexdigest()

    def to_json(self, *, form: str | None = None) -> dict[str, Any]:
        resolved_form = form or self.form
        doc: dict[str, Any] = {
            "id": self.id,
            "version": self.version,
            "sha256": self.digest(),
            "form": resolved_form,
        }
        if self.uri is not None:
            doc["uri"] = self.uri
        if resolved_form == "inline":
            doc["classes"] = [c.to_json() for c in self._classes]
            if self.relations:
                doc["relations"] = [r.to_json() for r in self.relations]
            if self.skeletons:
                doc["skeletons"] = [s.to_json() for s in self.skeletons]
        return doc

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | None) -> LabelSet | None:
        if not doc:
            return None
        return cls(
            id=str(doc["id"]),
            version=str(doc.get("version", "1.0.0")),
            form=str(doc.get("form", "inline")),
            uri=doc.get("uri"),
            sha256=doc.get("sha256"),
            classes=[LabelClass.from_json(c) for c in doc.get("classes") or ()],
            relations=[Relation.from_json(r) for r in doc.get("relations") or ()],
            skeletons=[Skeleton.from_json(s) for s in doc.get("skeletons") or ()],
        )

    def as_ref(self, uri: str) -> LabelSet:
        """A ``form: ref`` view of this vocabulary, for collection-level sharing."""
        return LabelSet(
            self.id, version=self.version, form="ref", uri=uri, sha256=self.digest()
        )

    def subset(self, keys: Iterable[int | str], *, id: str | None = None) -> LabelSet:
        """A vocabulary restricted to *keys* plus their ancestors."""
        wanted = {self[k].id for k in keys}
        for cid in list(wanted):
            wanted.update(self.ancestors(cid))
        return LabelSet(
            id=id or f"{self.id}-subset",
            version=self.version,
            classes=[c for c in self._classes if c.id in wanted],
            relations=[
                r for r in self.relations if r.subject in wanted and r.object in wanted
            ],
        )


def _duplicates(values: Iterable[Any]) -> set[Any]:
    seen: set[Any] = set()
    dupes: set[Any] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes


def from_keys(
    keys: Sequence[str], *, id: str, version: str = "1.0.0", start: int = 1
) -> LabelSet:
    """Mint a vocabulary from bare names --- what converters do on ingest."""
    return LabelSet(
        id=id,
        version=version,
        classes=[
            LabelClass(id=start + i, key=key, name=key.replace("_", " ").title())
            for i, key in enumerate(keys)
        ],
    )


__all__ = [
    "BACKGROUND_ID",
    "CLOSURES",
    "FORMS",
    "IGNORE_ID",
    "INLINE_REQUIRED_BELOW",
    "MAX_CLASS_ID",
    "LabelClass",
    "LabelSet",
    "OntologyCode",
    "Relation",
    "Skeleton",
    "canonical_json",
    "from_keys",
]
