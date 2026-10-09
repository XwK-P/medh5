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

Time is never collapsed to one number.  Every time an input carries --- an
event's effective start and end, its availability, an image's acquisition
--- is given as its inclusive bounds, as *ages* before the cutoff in hours:
``[..., 0]`` the least it can be, ``[..., 1]`` the most, equal for an exact
instant, so a day-precision diagnosis keeps its whole day.  A time that is not
known has a ``*_known`` of ``False`` and zeros, which are not a time.  A
``static`` event has no effective time at all, which ``temporal_type`` says,
apart from one whose time is unknown.  Ages are never negative but for what a
version available at the cutoff itself recorded about later: a plan's start
(``plan``) or a course's recorded end.  Events whose ordering times overlap
share a ``tie_group``: their order in the sequence is the storage tie-break,
not evidence.  An absent value is ``has_value = False``; a value known only as
``< 5`` has its ``comparator``; an expected result that is missing for a
reason has ``missing`` --- never a measured zero.  A value is normalised only
in the unit its concept was fitted in (``unit`` 2; another unit is 1, with
``value`` 0), and a text value is a fitted category (``value_index``).

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

from medh5.cache import FeatureCache, fitted_on, fitted_on_mismatches, validate_cache
from medh5.clinical import COMPARATORS, EVENT_KINDS, HOUR, Event
from medh5.errors import MEDH5ValidationError
from medh5.sample import Sample
from medh5.sampling import window_around
from medh5.task import Packed, Preflight, RowView, Slot, SubjectHistory, TaskManifest
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
    """Concept indices, per-concept value statistics and the categories each
    concept's text values took, fitted on one partition of one task --- and
    saying which.

    Index 0 is padding and 1 an unseen concept; fitted concepts start at 2.
    Categorical values (an event's ``value_text``) are indexed the same way,
    as ``(concept, value)`` pairs: 0 for no value, 1 for one the concept never
    took in training, fitted pairs from 2.  A concept's statistics are in the
    one unit its training values were in (``units``): values in two units are
    refused at fit time, and a value in another unit is never normalised with
    them.  Without both, ``female`` and ``male`` --- and 5 mg/dL and 5 mmol/L
    --- were the same input (B07 of the 2.0 audit).
    """

    concepts: tuple[str, ...]
    stats: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fitted_on: Mapping[str, Any] | None = None
    units: Mapping[str, str | None] = field(default_factory=dict)
    categories: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

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
        _require_preflight_of(task, report)
        chosen = partition if partition is not None else task.training_partition
        # Each subject's admitted versions, counted once however many of its
        # rows admit them.
        admitted: dict[int, list[npt.NDArray[np.int32]]] = {}
        for row in report.rows:
            if not row.eligible or (chosen is not None and row.partition != chosen):
                continue
            k = row.subject_index
            if k is not None:
                admitted.setdefault(k, []).append(row.selected)
        seen: set[str] = set()
        values: dict[str, list[float]] = {}
        units: dict[str, set[str | None]] = {}
        texts: dict[str, set[str]] = {}
        for k, parts in admitted.items():
            events = report.subjects[k].events
            tokens, codes = events.concepts()
            index = np.unique(np.concatenate(parts))
            seen.update(tokens[c] for c in np.unique(codes[index]))
            valid = events.column("value_num_valid")
            numbers = events.column("value_num")
            unit, text = events.column("unit"), events.column("value_text")
            for i in index.tolist():
                token = tokens[codes[i]]
                if valid[i]:
                    values.setdefault(token, []).append(float(numbers[i]))
                    units.setdefault(token, set()).add(_text_at(unit, i))
                label = _text_at(text, i)
                if label is not None:
                    texts.setdefault(token, set()).add(label)
        mixed = {token: found for token, found in units.items() if len(found) > 1}
        if mixed:
            listed = "; ".join(
                f"{token}: {sorted(found, key=str)}"
                for token, found in sorted(mixed.items())
            )
            raise MEDH5ValidationError(
                "these concepts have values in more than one unit among the "
                f"training rows ({listed}); one mean and deviation cannot "
                "normalise both --- convert them to one unit, or code them as "
                "different concepts"
            )
        stats = {}
        for token, found in sorted(values.items()):
            array = np.asarray(found, dtype=np.float64)
            std = float(array.std()) if array.size > 1 else 0.0
            stats[token] = (float(array.mean()), std if std > 1e-12 else 1.0)
        record = None if chosen is None else fitted_on(task, chosen)
        return cls(
            tuple(sorted(seen)),
            stats,
            record,
            {token: next(iter(found)) for token, found in sorted(units.items())},
            {token: tuple(sorted(found)) for token, found in sorted(texts.items())},
        )

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
        the concept was not fitted).  The statistics are in
        ``units[token]``: a value in another unit must not be passed."""
        mean, std = self.stats.get(token, (0.0, 1.0))
        return (float(value) - mean) / std

    def fits_unit(self, token: str, unit: str | None) -> bool:
        """Whether a value of *token* in *unit* is one the statistics describe:
        the concept was fitted, and in this unit."""
        return token in self.stats and self.units.get(token) == unit

    def value_index(self, token: str, text: str | None) -> int:
        """A categorical value's index: 0 none, 1 unseen, fitted pairs from 2."""
        if text is None:
            return PAD
        return self._value_lookup().get((token, text), UNKNOWN)

    def _value_lookup(self) -> dict[tuple[str, str], int]:
        cached: dict[tuple[str, str], int] | None = self.__dict__.get("_values")
        if cached is None:
            pairs = [
                (t, v) for t in sorted(self.categories) for v in self.categories[t]
            ]
            cached = {pair: i + 2 for i, pair in enumerate(pairs)}
            object.__setattr__(self, "_values", cached)
        return cached

    @property
    def n_values(self) -> int:
        """Categorical indices in use: none, unseen, and the fitted pairs."""
        return sum(len(v) for v in self.categories.values()) + 2

    def __len__(self) -> int:
        """Indices in use: padding, unknown, and the fitted concepts."""
        return len(self.concepts) + 2

    def to_json(self) -> dict[str, Any]:
        return {
            "concepts": list(self.concepts),
            "stats": {k: list(v) for k, v in sorted(self.stats.items())},
            "units": dict(sorted(self.units.items())),
            "categories": {k: list(v) for k, v in sorted(self.categories.items())},
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
            {
                str(k): None if v is None else str(v)
                for k, v in doc.get("units", {}).items()
            },
            {
                str(k): tuple(str(x) for x in v)
                for k, v in doc.get("categories", {}).items()
            },
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
        if task.training_partition is None:
            raise MEDH5ValidationError(
                "the vocabulary was fitted on a split, and this task declares none",
                "T405",
            )
        # The cache's comparison, set_id included: a vocabulary fitted under
        # another split of the same subjects passed its own (N05 of the 2.0
        # re-audit).
        problems = fitted_on_mismatches(task, self.fitted_on)
        if problems:
            raise MEDH5ValidationError(f"the vocabulary was {problems[0]}", "T405")


def _text_at(column: Any, i: int) -> str | None:
    """Cell *i* of a text column that may be all null; empty reads as null."""
    value = None if column is None else column[i]
    return value or None


def _require_preflight_of(task: TaskManifest, report: Preflight) -> None:
    """Refuse a preflight of anything but this task instance (T404).

    The *definition* fingerprint is shared by every instance of a task: two
    manifests with the rows' partitions swapped had the same one, and a
    preflight of the other put this task's validation subjects in its training
    rows --- and its vocabulary fit on them while recording this task's
    training split.  The manifest fingerprint covers the split, rows and pins.
    """
    if report.task_fingerprint != task.task_fingerprint:
        raise MEDH5ValidationError(
            "the preflight is of another task definition", "T404"
        )
    if report.manifest_fingerprint != task.manifest_fingerprint:
        raise MEDH5ValidationError(
            "the preflight is of another instance of this task --- its split, rows "
            "or pins differ: run this manifest's own preflight",
            "T404",
        )


def _require_level(path: PathLike, level: str, role: str) -> None:
    """A cache serves the role its level answers (T404): an event-level cache
    encodes event versions any row may read; a patient-level one, one row's
    whole history at its cutoff, which no other row may read."""
    with FeatureCache.open(path) as cache:
        found = cache.level
    if found != level:
        raise MEDH5ValidationError(
            f"{role}= takes a cache of level {level!r}; {os.fspath(path)!r} is "
            f"of level {found!r}",
            "T404",
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
    (event-level, validated against the task at construction), or an encoder with
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
        _require_preflight_of(self.task, report)
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
                _require_level(documents, "event", "documents")
                # With the task, so a feature fitted on another split is
                # refused (T405); an event-level cache has no row entries, so
                # the rows are not re-checked.
                checked = validate_cache(
                    documents, base=self.base, task=self.task, check_rows=False
                )
                if not checked.ok:
                    raise MEDH5ValidationError(
                        f"the document cache does not validate: {checked.findings[0]}",
                        checked.findings[0].code,
                    )
                self._documents = _LazyCache(documents)
            else:
                self._encoder = documents
        self._vocabularies: dict[
            int,
            tuple[
                npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]
            ],
        ] = {}
        self._values: dict[
            int, tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]
        ] = {}
        self._rows_cache: _LazyCache | None = None
        if row_features is not None:
            _require_level(row_features, "patient", "row_features")
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
                "time": _no_time(),
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
        time = _no_time()
        subject = row.subject
        if subject is not None and fill.event_id is not None:
            # The acquisition and its availability, as the imaging version
            # that owns the image records them.
            k = np.asarray([subject.events.index(fill.event_id)], dtype=np.int64)
            for name in ("effective_start", "available"):
                ages, known = _ages(subject, k, name, row.cutoff_us)
                key = "start" if name == "effective_start" else "available"
                time[f"{key}_age_h"] = ages[0]
                time[f"{key}_known"] = bool(known[0])
        return {
            "image": array,
            "valid": valid,
            "present": True,
            "label": label,
            "annotated": annotated,
            "ignore": ignore,
            "time": time,
        }

    def _vocabulary_of(
        self, k: int, subject: SubjectHistory
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Per concept of one subject: its vocabulary index, and the fitted
        mean and standard deviation its values are normalised with."""
        found = self._vocabularies.get(k)
        if found is None:
            tokens, _ = subject.events.concepts()
            index = np.asarray([self.concepts.index(t) for t in tokens], dtype=np.int64)
            fitted = [self.concepts.stats.get(t, (0.0, 1.0)) for t in tokens]
            mean = np.asarray([m for m, _ in fitted], dtype=np.float64)
            std = np.asarray([d for _, d in fitted], dtype=np.float64)
            found = (index, mean, std)
            self._vocabularies[k] = found
        return found

    def _values_of(
        self, k: int, subject: SubjectHistory
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        """Per event version of one subject: its categorical value's index,
        and whether its numeric value is in the unit its concept was fitted in."""
        found = self._values.get(k)
        if found is None:
            events = subject.events
            tokens, codes = events.concepts()
            unit, text = events.column("unit"), events.column("value_text")
            n = len(events)
            index = np.zeros(n, dtype=np.int64)
            fits = np.zeros(n, dtype=bool)
            for i in range(n):
                token = tokens[codes[i]]
                index[i] = self.concepts.value_index(token, _text_at(text, i))
                fits[i] = self.concepts.fits_unit(token, _text_at(unit, i))
            found = (index, fits)
            self._values[k] = found
        return found

    def _events(self, row: RowView) -> dict[str, npt.NDArray[Any]]:
        """The admitted versions, in input order, as arrays (see the module
        notes on time): read from the subject's columns, not from objects."""
        subject = row.subject
        if subject is None:
            return _no_events()
        events = subject.events
        k = row.selected.astype(np.int64)
        _, codes = events.concepts()
        index, mean, std = self._vocabulary_of(int(row.subject_index or 0), subject)
        categories, fits = self._values_of(int(row.subject_index or 0), subject)
        local = codes[k]
        has_value = events.column("value_num_valid")[k]
        raw = events.column("value_num")[k]
        # Normalised only in the unit the statistics are in: a value in another
        # unit, or of a concept never fitted, is 0 with `unit` 1 --- present,
        # and not comparable --- rather than a raw number among standard scores.
        fitted = has_value & fits[k]
        value = np.where(fitted, (raw - mean[local]) / std[local], 0.0)
        unit = np.where(has_value, np.where(fitted, 2, 1), 0).astype(np.int64)
        # `eq` when absent and a value is valid (1.1 §5.1); 0 when no value.
        comparator = events.column("value_comparator_code")[k].astype(np.int64) + 1
        comparator = np.where(
            comparator > 0,
            comparator,
            np.where(has_value, COMPARATORS.index("eq") + 1, 0),
        )
        out: dict[str, npt.NDArray[Any]] = {
            "concept": index[local],
            "kind": events.column("kind_code")[k].astype(np.int64) + 1,
            "status": events.column("status_code")[k].astype(np.int64) + 1,
            "temporal_type": events.column("temporal_type_code")[k].astype(np.int64)
            + 1,
            "value": value.astype(np.float32),
            "has_value": np.asarray(has_value, dtype=bool),
            "unit": unit,
            "value_index": categories[k],
            "comparator": comparator,
            "missing": _present(events.column("missing_reason"), k),
        }
        for name, key in (
            ("effective_start", "start"),
            ("effective_end", "end"),
            ("available", "available"),
        ):
            ages, known = _ages(subject, k, name, row.cutoff_us)
            out[f"{key}_age_h"] = ages
            out[f"{key}_known"] = known
        out["tie_group"] = row.tie_group.astype(np.int64)
        out["plan"] = np.asarray(row.plan, dtype=bool)
        return out

    def _documents_of(self, row: RowView) -> list[tuple[int, int, int, str]]:
        """``(position, event, fragment, document_id)`` of every document an
        admitted ``document`` version owns, in input order: what a row may
        read as text (1.1 §7.3)."""
        subject = row.subject
        if subject is None:
            return []
        selected = row.selected
        kinds = subject.events.column("kind_code")[selected]
        found = []
        for position in np.flatnonzero(kinds == EVENT_KINDS.index("document")):
            event = int(selected[position])
            for fragment, document_id in subject.owned_documents(event):
                found.append((int(position), event, fragment, document_id))
        return found

    def _document_features(
        self, row: RowView, samples: dict[int, Sample]
    ) -> dict[str, npt.NDArray[Any]] | None:
        if self._documents is None and self._encoder is None:
            return None
        subject = row.subject
        owned = self._documents_of(row)
        features = []
        for _, event, fragment, document_id in owned:
            assert subject is not None
            if self._documents is not None:
                content_id = row.sources[fragment].content_id
                event_id = str(subject.events.column("event_id")[event])
                feature = self._documents.get().event_feature(content_id, event_id)
                if feature is None:
                    raise MEDH5ValidationError(
                        f"the document cache has no feature for event "
                        f"{event_id!r} of {row.sources[fragment].locator} "
                        f"at {content_id}: rebuild it for these sources",
                        "T403",
                    )
            else:
                # The document's own bytes: no other text, and not the events.
                feature = self._encoder.encode(
                    samples[fragment].document_text(document_id)
                )
            features.append(np.asarray(feature, dtype=np.float32))
        dim = self._feature_dim() if not features else int(features[0].shape[0])
        k = np.asarray([e for _, e, _, _ in owned], dtype=np.int64)
        out: dict[str, npt.NDArray[Any]] = {
            "features": np.stack(features)
            if features
            else np.zeros((0, dim), dtype=np.float32),
            "event": np.asarray([p for p, _, _, _ in owned], dtype=np.int64),
        }
        for name, key in (("effective_start", "start"), ("available", "available")):
            if subject is None:
                ages, known = (
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros(0, dtype=bool),
                )
            else:
                ages, known = _ages(subject, k, name, row.cutoff_us)
            out[f"{key}_age_h"] = ages
            out[f"{key}_known"] = known
        return out

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
        fills = row.slots
        needed = {f.fragment for f in fills.values() if f.fragment is not None}
        if self._encoder is not None:
            needed |= {fragment for _, _, fragment, _ in self._documents_of(row)}
        with contextlib.ExitStack() as stack:
            # Each source as the version the row pins: a handle cached from
            # before a replacement is reopened, a changed source refused.
            samples: dict[int, Sample] = {
                i: stack.enter_context(
                    CACHE.lease(
                        *self._source(row, i), content_id=row.sources[i].content_id
                    )
                )
                for i in sorted(needed)
            }
            images: dict[str, Any] = {}
            valid: dict[str, Any] = {}
            present: dict[str, Any] = {}
            label: dict[str, Any] = {}
            annotated: dict[str, Any] = {}
            ignore: dict[str, Any] = {}
            times: dict[str, Any] = {}
            visits: dict[str, Any] = {}
            for slot in self.slots:
                fill = fills[slot.name]
                sample = None if fill.fragment is None else samples[fill.fragment]
                read = self._slot(sample, slot, row)
                images[slot.name] = to_tensor(read["image"])
                valid[slot.name] = to_tensor(read["valid"])
                present[slot.name] = _scalar(read["present"], torch.bool)
                times[slot.name] = {
                    "start_age_h": to_tensor(read["time"]["start_age_h"]),
                    "start_known": _scalar(read["time"]["start_known"], torch.bool),
                    "available_age_h": to_tensor(read["time"]["available_age_h"]),
                    "available_known": _scalar(
                        read["time"]["available_known"], torch.bool
                    ),
                }
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
            target = row.target
            target_value = target.value if target.value is not None else 0.0
            subject = row.subject
            ids = (
                []
                if subject is None
                else [
                    str(subject.events.column("event_id")[int(k)]) for k in row.selected
                ]
            )
            item: dict[str, Any] = {
                "images": images,
                "valid": valid,
                "present": present,
                "image_time": times,
                "ignore": ignore,
                "events": events,
                "target": {
                    "value": _scalar(target_value, torch.float32),
                    "observed": _scalar(target.observed, torch.bool),
                },
                "meta": {
                    "row_id": row.row_id,
                    "subject_id": row.subject_id,
                    "partition": row.partition,
                    "cutoff_us": row.cutoff_us,
                    "fingerprint": row.fingerprint,
                    "status": row.status,
                    "target_status": target.status,
                    "event_ids": ids,
                    "visits": visits,
                },
            }
            if label:
                item["label"] = label
                item["annotated"] = annotated
            if reads_documents:
                docs = self._document_features(row, samples)
                assert docs is not None
                item["documents"] = {k: to_tensor(v) for k, v in docs.items()}
                item["meta"]["document_events"] = [ids[int(p)] for p in docs["event"]]
            if self._rows_cache is not None:
                feature = self._rows_cache.get().row_feature(row.row_id)
                if feature is None:
                    raise MEDH5ValidationError(
                        f"the row-feature cache has no entry for row {row.row_id!r}",
                        "T404",
                    )
                item["row_feature"] = to_tensor(np.asarray(feature))
            return item


def _no_time() -> dict[str, Any]:
    return {
        "start_age_h": np.zeros(2, dtype=np.float32),
        "start_known": False,
        "available_age_h": np.zeros(2, dtype=np.float32),
        "available_known": False,
    }


def _ages(
    subject: SubjectHistory, k: npt.NDArray[np.int64], name: str, cutoff_us: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Versions `k`'s `name` bounds as ages before the cutoff, in hours:
    ``[:, 0]`` the least, ``[:, 1]`` the most; zeros where unknown."""
    bounds = subject.events.column(name)[k]
    known = np.asarray(subject.events.column(f"{name}_known")[k], dtype=bool)
    ages = np.stack([cutoff_us - bounds[:, 1], cutoff_us - bounds[:, 0]], axis=1) / HOUR
    ages[~known] = 0.0
    return ages.astype(np.float32).reshape(-1, 2), known


def _present(column: Packed | None, k: npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Which cells of a packed column are not null (``None``: none are)."""
    if column is None:
        return np.zeros(len(k), dtype=bool)
    if column.valid is None:
        return np.ones(len(k), dtype=bool)
    return np.asarray(column.valid[k], dtype=bool)


def _no_events() -> dict[str, npt.NDArray[Any]]:
    out: dict[str, npt.NDArray[Any]] = {
        name: np.zeros(0, dtype=np.int64)
        for name in (
            "concept",
            "kind",
            "status",
            "temporal_type",
            "comparator",
            "tie_group",
            "unit",
            "value_index",
        )
    }
    out.update(
        value=np.zeros(0, dtype=np.float32),
        has_value=np.zeros(0, dtype=bool),
        missing=np.zeros(0, dtype=bool),
        plan=np.zeros(0, dtype=bool),
    )
    for key in ("start", "end", "available"):
        out[f"{key}_age_h"] = np.zeros((0, 2), dtype=np.float32)
        out[f"{key}_known"] = np.zeros(0, dtype=bool)
    return out


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
