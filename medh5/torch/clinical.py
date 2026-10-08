"""Cutoff-aware multimodal longitudinal training over a task (format 1.1).

:class:`ClinicalTaskDataset` turns the rows of a ``medh5.task/1`` manifest
into training items.  Every decision about *what a row may see* is the
engine's, made once at construction by the task's preflight
(:func:`medh5.task.preflight`): which event versions the strict policy admits
at the row's cutoff, which image fills each modality slot, which annotations
may serve as labels, and what the target is.  The dataset only reads what a
row admits, when the item is built --- voxels from the slot's image, text or
cached features from the admitted documents --- so a later report, a revised
measurement or a follow-up scan cannot reach an input.

An item is a ``dict`` whose masks stay distinct, because they mean different
things to a loss:

``present[slot]``
    whether an eligible image filled the slot (*modality availability*);
``valid[slot]``
    where the slot's window holds data: inside the image and its
    ``valid_mask`` (*field of view*); never the padding;
``annotated[slot]``
    per class, whether an eligible annotation *examined* it (*coverage*); a 0
    in an unexamined class is not a negative;
``ignore[slot]``
    voxels a loss must not score (§7.7 ignore regions, and padding);
``target["observed"]``
    whether the row's target is observed; a censored row trains its inputs
    with the target masked out;
``events["mask"]`` / ``documents["mask"]``
    after :func:`collate_clinical`, which entries of a padded sequence are real.

Learned preprocessing --- the concept vocabulary and per-concept value
statistics --- is fitted on the task's **training partition** only, and
records the split it was fitted on (``fitted_on``); a vocabulary fitted on
anything else is refused (T405).

Handles follow §14.4: sources --- files or collection members --- are read
through the PID-keyed cache of :mod:`medh5.torch.handles`, and a feature cache
is opened lazily in the process that reads it and *abandoned*, never closed,
across a fork.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from medh5.cache import FeatureCache, fitted_on, validate_cache
from medh5.clinical import EVENT_KINDS, HOUR, Event
from medh5.errors import MEDH5ValidationError
from medh5.sample import Sample
from medh5.sampling import window_around
from medh5.task import Preflight, RowView, Slot, TaskManifest
from medh5.torch._compat import dataset_base, require_torch, to_tensor
from medh5.torch.handles import CACHE

PathLike = str | os.PathLike[str]

PAD = 0
"""Concept index of padding."""
UNKNOWN = 1
"""Concept index of a concept the vocabulary was not fitted on."""


def concept_of(event: Event) -> str:
    """An event's concept token: ``kind|code_system|code`` (``kind|`` uncoded)."""
    if event.code_system is None or event.code is None:
        return f"{event.kind}|"
    return f"{event.kind}|{event.code_system}|{event.code}"


@dataclass(frozen=True)
class ConceptVocabulary:
    """Concept indices and per-concept value statistics, fitted on one
    partition of one task --- and saying which.

    Index 0 is padding and 1 an unseen concept; fitted concepts start at 2.
    """

    concepts: tuple[str, ...]
    stats: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fitted_on: Mapping[str, Any] | None = None

    @classmethod
    def fit(
        cls,
        task: TaskManifest,
        preflight: Preflight | None = None,
        *,
        partition: str | None = None,
    ) -> ConceptVocabulary:
        """Fit on the eligible rows of ``partition`` (default: the training
        partition; every row when the task declares no split)."""
        report = preflight if preflight is not None else task.preflight()
        chosen = partition if partition is not None else task.training_partition
        seen: set[str] = set()
        values: dict[str, list[float]] = {}
        counted: set[tuple[str, str]] = set()
        for row in report.rows:
            if not row.eligible or (chosen is not None and row.partition != chosen):
                continue
            for event in row.events:
                token = concept_of(event)
                seen.add(token)
                key = (row.subject_id, event.event_id)
                if event.value_num is not None and key not in counted:
                    counted.add(key)
                    values.setdefault(token, []).append(float(event.value_num))
        stats = {}
        for token, found in sorted(values.items()):
            array = np.asarray(found, dtype=np.float64)
            std = float(array.std()) if array.size > 1 else 0.0
            stats[token] = (float(array.mean()), std if std > 1e-12 else 1.0)
        record = None if chosen is None else fitted_on(task, chosen)
        return cls(tuple(sorted(seen)), stats, record)

    def index(self, token: str) -> int:
        try:
            return self._lookup()[token]
        except KeyError:
            return UNKNOWN

    def _lookup(self) -> dict[str, int]:
        cached: dict[str, int] | None = self.__dict__.get("_index")
        if cached is None:
            cached = {c: i + 2 for i, c in enumerate(self.concepts)}
            object.__setattr__(self, "_index", cached)
        return cached

    def normalise(self, token: str, value: float) -> float:
        """``(value - mean) / std`` with the fitted statistics (identity when
        the concept was not fitted)."""
        mean, std = self.stats.get(token, (0.0, 1.0))
        return (float(value) - mean) / std

    def __len__(self) -> int:
        """Indices in use: padding, unknown, and the fitted concepts."""
        return len(self.concepts) + 2

    def to_json(self) -> dict[str, Any]:
        return {
            "concepts": list(self.concepts),
            "stats": {k: list(v) for k, v in sorted(self.stats.items())},
            "fitted_on": None if self.fitted_on is None else dict(self.fitted_on),
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> ConceptVocabulary:
        return cls(
            tuple(doc["concepts"]),
            {
                str(k): (float(v[0]), float(v[1]))
                for k, v in doc.get("stats", {}).items()
            },
            doc.get("fitted_on"),
        )

    @property
    def digest(self) -> str:
        text = json.dumps(self.to_json(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: PathLike) -> ConceptVocabulary:
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def check_fitted_on(self, task: TaskManifest) -> None:
        """Refuse a vocabulary fitted on anything but this task's training
        partition (T405)."""
        if self.fitted_on is None:
            if task.split is not None:
                raise MEDH5ValidationError(
                    "the vocabulary records no split it was fitted on, and this task "
                    "declares one: fit it on the training partition",
                    "T405",
                )
            return
        wanted = task.training_partition
        if wanted is None:
            raise MEDH5ValidationError(
                "the vocabulary was fitted on a split, and this task declares none",
                "T405",
            )
        expected = fitted_on(task, wanted)
        for key in ("task_fingerprint", "partition", "subjects_digest"):
            if self.fitted_on.get(key) != expected[key]:
                raise MEDH5ValidationError(
                    f"the vocabulary was fitted on {key} {self.fitted_on.get(key)!r}; "
                    f"this task's training partition is {expected[key]!r}",
                    "T405",
                )


class _LazyCache:
    """A feature cache opened in the process that reads it (§14.4)."""

    __slots__ = ("_cache", "_pid", "path")

    def __init__(self, path: PathLike) -> None:
        self.path = os.fspath(path)
        self._cache: FeatureCache | None = None
        self._pid = os.getpid()

    def get(self) -> FeatureCache:
        if self._pid != os.getpid():
            if self._cache is not None:
                self._cache.abandon()
            self._cache = None
            self._pid = os.getpid()
        if self._cache is None:
            self._cache = FeatureCache.open(self.path)
        return self._cache

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.path = state["path"]
        self._cache = None
        self._pid = os.getpid()


_DatasetBase = dataset_base()


def _scalar(value: Any, dtype: Any) -> Any:
    """A 0-d tensor (``to_tensor`` would make it 1-d)."""
    import torch

    return torch.tensor(value, dtype=dtype)


class ClinicalTaskDataset(_DatasetBase):  # type: ignore[misc,valid-type]
    """One item per admitted row of a task.

    ``partition`` restricts to one partition of the task's split;
    ``statuses`` to rows of those preflight statuses (``eligible`` only, by
    default: an uncertifiable row is a row strict selection cannot vouch for).
    A preflight with findings --- a stale pin, an unreconciled fragment --- is
    refused unless ``strict=False``, which keeps only the unaffected rows.

    ``documents`` gives each admitted document a feature: a feature-cache path
    (event-level, validated at construction), or an encoder with
    ``encode(text) -> array`` that reads the admitted text when the item is
    built.  ``row_features`` is a patient-level cache built for this task; it
    is validated against the task (T404--T406) before any row reads it.
    """

    def __init__(
        self,
        task: TaskManifest | PathLike,
        *,
        partition: str | None = None,
        statuses: Sequence[str] = ("eligible",),
        concepts: ConceptVocabulary | None = None,
        documents: PathLike | Any | None = None,
        row_features: PathLike | None = None,
        base: PathLike | None = None,
        strict: bool = True,
        preflight: Preflight | None = None,
    ) -> None:
        require_torch()
        self.task = task if isinstance(task, TaskManifest) else TaskManifest.load(task)
        self.base = Path(base) if base is not None else self.task.base
        report = preflight if preflight is not None else self.task.preflight(self.base)
        if report.task_fingerprint != self.task.task_fingerprint:
            raise MEDH5ValidationError(
                "the preflight is of another task definition", "T404"
            )
        if strict and not report.ok:
            listed = "; ".join(str(f) for f in report.findings[:5])
            raise MEDH5ValidationError(
                f"the task's preflight has {len(report.findings)} finding(s): {listed}"
                " --- fix them, or pass strict=False to train on the unaffected rows",
                report.findings[0].code,
            )
        split = self.task.split
        if partition is not None and (split is None or partition not in split[1]):
            raise MEDH5ValidationError(
                f"partition {partition!r} is not one of the task's split", "T202"
            )
        self.partition = partition
        self.preflight = report
        self.rows: tuple[RowView, ...] = tuple(
            r
            for r in report.rows
            if r.status in statuses and (partition is None or r.partition == partition)
        )
        self.slots: tuple[Slot, ...] = self.task.slots
        self.concepts = (
            concepts
            if concepts is not None
            else ConceptVocabulary.fit(self.task, report)
        )
        self.concepts.check_fitted_on(self.task)
        self._documents: _LazyCache | None = None
        self._encoder: Any = None
        if documents is not None:
            if isinstance(documents, (str, os.PathLike)):
                checked = validate_cache(documents, base=self.base)
                if not checked.ok:
                    raise MEDH5ValidationError(
                        f"the document cache does not validate: {checked.findings[0]}",
                        checked.findings[0].code,
                    )
                self._documents = _LazyCache(documents)
            else:
                self._encoder = documents
        self._rows_cache: _LazyCache | None = None
        if row_features is not None:
            checked = validate_cache(row_features, base=self.base, task=self.task)
            if not checked.ok:
                raise MEDH5ValidationError(
                    f"the row-feature cache does not validate: {checked.findings[0]}",
                    checked.findings[0].code,
                )
            self._rows_cache = _LazyCache(row_features)

    def __len__(self) -> int:
        return len(self.rows)

    # -- reading -----------------------------------------------------------

    def _source(self, row: RowView, fragment: int) -> tuple[str, str | None]:
        source = row.sources[fragment]
        return os.fspath(source.resolve(self.base)), source.sample_key

    def _slot(self, sample: Sample | None, slot: Slot, row: RowView) -> dict[str, Any]:
        fill = row.slots[slot.name]
        classes = tuple(slot.classes)
        if sample is None or fill.image_id is None:
            shape = tuple(slot.patch) if slot.patch is not None else ()
            return {
                "image": np.zeros((1, *shape), dtype=np.float32),
                "valid": np.zeros(shape, dtype=bool),
                "present": False,
                "label": np.zeros((len(classes), *shape), dtype=np.float32),
                "annotated": np.zeros(len(classes), dtype=bool),
                "ignore": np.ones(shape, dtype=bool),
                "age_h": 0.0,
            }
        image = sample.images[fill.image_id]
        spatial = image.grid.spatial_shape
        if slot.patch is None:
            roi = tuple(slice(0, n) for n in spatial)
            pad: tuple[tuple[int, int], ...] = ((0, 0),) * len(spatial)
        else:
            if len(slot.patch) != len(spatial):
                raise MEDH5ValidationError(
                    f"slot {slot.name!r} reads a {len(slot.patch)}-D patch from "
                    f"{fill.image_id!r}, whose grid is {len(spatial)}-D",
                    "T102",
                )
            center = (
                fill.center
                if fill.center is not None
                else tuple(n // 2 for n in spatial)
            )
            roi, pad = window_around(center, slot.patch, spatial)
        array = image.read(list(roi), physical=True, dtype=np.float32)
        array = array.reshape((-1, *array.shape[array.ndim - len(spatial) :]))
        array = np.pad(array, ((0, 0), *pad), mode="constant")
        valid = np.pad(
            sample.valid_region(fill.image_id, list(roi)), pad, constant_values=False
        )
        label = np.zeros((len(classes), *valid.shape), dtype=np.float32)
        annotated = np.zeros(len(classes), dtype=bool)
        # Padding is never scored; inside the window, the annotations decide.
        inner = [s.stop - s.start for s in roi]
        ignore = np.pad(np.zeros(inner, dtype=bool), pad, constant_values=True)
        # Labels are supervision, like the target: read from every annotation
        # on the grid, including ones drawn after the cutoff.  They never enter
        # an input --- the ROI above was centred on eligible annotations only.
        for ann_id in fill.label_annotations if classes else ():
            ann = sample.annotations[ann_id]
            present = set(ann.class_ids)
            examined = set(ann.annotated_class_ids)
            wanted = [c for c in classes if c in present]
            if wanted:
                planes = np.pad(
                    np.asarray(ann.dense(wanted, roi=list(roi)), dtype=np.float32),
                    ((0, 0), *pad),
                )
                for plane, c in zip(planes, wanted, strict=True):
                    i = classes.index(c)
                    label[i] = np.maximum(label[i], plane)
            for i, c in enumerate(classes):
                annotated[i] |= c in examined
            ignore |= np.pad(
                sample.ignore_region(ann_id, list(roi)), pad, constant_values=True
            )
        age = 0.0
        if row.selection is not None:
            when = next(
                (e for e in row.selection.events if e.event_id == fill.event_id), None
            )
            if when is not None and when.order_us is not None:
                age = (row.cutoff_us - when.order_us[1]) / HOUR
        return {
            "image": array,
            "valid": valid,
            "present": True,
            "label": label,
            "annotated": annotated,
            "ignore": ignore,
            "age_h": float(age),
        }

    def _events(self, row: RowView) -> dict[str, npt.NDArray[Any]]:
        order = {}
        if row.selection is not None:
            order = {e.event_id: e.order_us for e in row.selection.events}
        items = []
        for event in row.events:
            bounds = order.get(event.event_id)
            known = bounds is not None
            age = (row.cutoff_us - bounds[1]) / HOUR if bounds is not None else 0.0
            token = concept_of(event)
            has_value = event.value_num is not None
            value = (
                0.0
                if event.value_num is None
                else self.concepts.normalise(token, event.value_num)
            )
            items.append(
                (
                    (0 if not known else 1, -age, event.event_id),
                    self.concepts.index(token),
                    EVENT_KINDS.index(event.kind) + 1,
                    value,
                    has_value,
                    age,
                    known,
                )
            )
        items.sort(key=lambda t: t[0])
        return {
            "concept": np.asarray([t[1] for t in items], dtype=np.int64),
            "kind": np.asarray([t[2] for t in items], dtype=np.int64),
            "value": np.asarray([t[3] for t in items], dtype=np.float32),
            "has_value": np.asarray([t[4] for t in items], dtype=bool),
            "age_h": np.asarray([t[5] for t in items], dtype=np.float32),
            "time_known": np.asarray([t[6] for t in items], dtype=bool),
        }

    def _document_features(
        self, row: RowView, samples: dict[int, Sample]
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], list[str]] | None:
        if self._documents is None and self._encoder is None:
            return None
        if row.selection is None:
            return None
        order = {e.event_id: e.order_us for e in row.selection.events}
        found: list[tuple[float, str, npt.NDArray[np.float32]]] = []
        for event, fragment in zip(row.events, row.event_fragments, strict=True):
            if event.kind != "document":
                continue
            sample = samples[fragment]
            clinical = sample.clinical
            if clinical is None:  # pragma: no cover - a selected event has a profile
                continue
            for link in clinical.links:
                if not (
                    link.relation == "describes"
                    and link.source_type == "event"
                    and link.source_id == event.event_id
                    and link.target_type == "document"
                ):
                    continue
                if not row.selection.admits("document", link.target_id, fragment):
                    continue  # pragma: no cover - a structural link is admitted
                if self._documents is not None:
                    content_id = row.sources[fragment].content_id
                    feature = self._documents.get().event_feature(
                        content_id, event.event_id
                    )
                    if feature is None:
                        raise MEDH5ValidationError(
                            f"the document cache has no feature for event "
                            f"{event.event_id!r} of {row.sources[fragment].locator} "
                            f"at {content_id}: rebuild it for these sources",
                            "T403",
                        )
                else:
                    feature = np.asarray(
                        self._encoder.encode(clinical.text(link.target_id)),
                        dtype=np.float32,
                    )
                bounds = order.get(event.event_id)
                age = (row.cutoff_us - bounds[1]) / HOUR if bounds is not None else 0.0
                found.append(
                    (age, event.event_id, np.asarray(feature, dtype=np.float32))
                )
        found.sort(key=lambda t: (-t[0], t[1]))
        if not found:
            dim = self._feature_dim()
            return (
                np.zeros((0, dim), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [],
            )
        return (
            np.stack([t[2] for t in found]),
            np.asarray([t[0] for t in found], dtype=np.float32),
            [t[1] for t in found],
        )

    def _feature_dim(self) -> int:
        if self._documents is not None:
            shape = self._documents.get().header["output"]["shape"]
            return int(shape[0]) if shape else 1
        dim = getattr(self._encoder, "dim", None)
        if dim is None:
            raise MEDH5ValidationError("the document encoder declares no `dim`")
        return int(dim)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        reads_documents = self._documents is not None or self._encoder is not None
        needed = {f.fragment for f in row.slots.values() if f.fragment is not None}
        if reads_documents:
            needed |= set(row.event_fragments)
        with contextlib.ExitStack() as stack:
            samples: dict[int, Sample] = {
                i: stack.enter_context(CACHE.lease(*self._source(row, i)))
                for i in sorted(needed)
            }
            images: dict[str, Any] = {}
            valid: dict[str, Any] = {}
            present: dict[str, Any] = {}
            label: dict[str, Any] = {}
            annotated: dict[str, Any] = {}
            ignore: dict[str, Any] = {}
            age: dict[str, Any] = {}
            visits: dict[str, Any] = {}
            for slot in self.slots:
                fill = row.slots[slot.name]
                sample = None if fill.fragment is None else samples[fill.fragment]
                read = self._slot(sample, slot, row)
                images[slot.name] = to_tensor(read["image"])
                valid[slot.name] = to_tensor(read["valid"])
                present[slot.name] = _scalar(read["present"], torch.bool)
                age[slot.name] = _scalar(read["age_h"], torch.float32)
                ignore[slot.name] = to_tensor(read["ignore"])
                if slot.classes:
                    label[slot.name] = to_tensor(read["label"])
                    annotated[slot.name] = to_tensor(read["annotated"])
                visits[slot.name] = {
                    "image_id": fill.image_id,
                    "event_id": fill.event_id,
                    "timepoint": None
                    if sample is None or fill.image_id is None
                    else sample.images[fill.image_id].timepoint,
                    "source": None
                    if fill.fragment is None
                    else row.sources[fill.fragment].source_id,
                    "roi": fill.roi,
                }
            events = {k: to_tensor(v) for k, v in self._events(row).items()}
            target_value = row.target.value if row.target.value is not None else 0.0
            item: dict[str, Any] = {
                "images": images,
                "valid": valid,
                "present": present,
                "image_age_h": age,
                "ignore": ignore,
                "events": events,
                "target": {
                    "value": _scalar(target_value, torch.float32),
                    "observed": _scalar(row.target.observed, torch.bool),
                },
                "meta": {
                    "row_id": row.row_id,
                    "subject_id": row.subject_id,
                    "partition": row.partition,
                    "cutoff_us": row.cutoff_us,
                    "fingerprint": row.fingerprint,
                    "status": row.status,
                    "target_status": row.target.status,
                    "event_ids": [e.event_id for e in row.events],
                    "visits": visits,
                },
            }
            if label:
                item["label"] = label
                item["annotated"] = annotated
            docs = self._document_features(row, samples)
            if docs is not None:
                features, doc_age, doc_ids = docs
                item["documents"] = {
                    "features": to_tensor(features),
                    "age_h": to_tensor(doc_age),
                }
                item["meta"]["document_events"] = doc_ids
            if self._rows_cache is not None:
                feature = self._rows_cache.get().row_feature(row.row_id)
                if feature is None:
                    raise MEDH5ValidationError(
                        f"the row-feature cache has no entry for row {row.row_id!r}",
                        "T404",
                    )
                item["row_feature"] = to_tensor(np.asarray(feature))
            return item


_SEQUENCES = ("events", "documents")


def collate_clinical(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``DataLoader(collate_fn=...)`` for :class:`ClinicalTaskDataset` items.

    Images, masks and targets stack; event and document sequences pad to the
    longest in the batch, with ``mask`` marking the real entries --- a padded
    position is never an event.  ``meta`` stays a list of dicts.
    """
    require_torch()
    from medh5.torch.collate import _collate_value

    if not batch:
        raise MEDH5ValidationError("cannot collate an empty batch")
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if key in _SEQUENCES:
            out[key] = _pad_sequences(values)
        elif key == "meta":
            out[key] = list(values)
        else:
            out[key] = _collate_value(key, values)
    return out


def _pad_sequences(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    lengths = [int(next(iter(v.values())).shape[0]) for v in values]
    longest = max(lengths) if lengths else 0
    out: dict[str, Any] = {}
    for key in values[0]:
        first = values[0][key]
        shape = (len(values), longest, *first.shape[1:])
        padded = torch.zeros(shape, dtype=first.dtype)
        for i, v in enumerate(values):
            n = lengths[i]
            if n:
                padded[i, :n] = v[key]
        out[key] = padded
    mask = torch.zeros((len(values), longest), dtype=torch.bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    out["mask"] = mask
    out["length"] = torch.as_tensor(lengths, dtype=torch.int64)
    return out


__all__ = [
    "PAD",
    "UNKNOWN",
    "ClinicalTaskDataset",
    "ConceptVocabulary",
    "collate_clinical",
    "concept_of",
]
