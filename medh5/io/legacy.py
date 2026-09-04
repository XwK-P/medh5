"""0.x → 1.0 migration (spec Appendix B, plan §6).

The mapping is mostly mechanical; four steps are not, and each is reported per
file so a curator can audit a cohort rather than trust it:

1. **Voxel encoding.**  0.x stored one boolean volume per mask name.  1.0
   measures the overlap graph and picks an encoding (§7.6), which changes the
   size and nothing else.
2. **Box corners.**  0.x boxes were slice-like integers ``[min, max)``; 1.0
   boxes sit at voxel edges, so ``lo = min − 0.5`` and ``hi = max − 0.5``.  That
   is a real half-voxel shift in the numbers, and it is reported as one.
3. **Label set.**  0.x had names, not classes.  Mask names and ``bbox_labels``
   become keys with minted ids, written to a sidecar so a curator can review,
   edit and reapply them cohort-wide before converting the rest.
4. **Grouping.**  A 0.x file is study-scoped and carries no subject key, so the
   default is one sample per file with a single declared ``tp0``.  Nothing about
   time is invented.  ``--group-by subject`` merges files that share a key the
   curator names, and orders them by date when there is one, by mtime otherwise
   — and says which.

Instance correspondence is **never** inferred across merged files: each file's
objects keep independent ids, because asserting that lesion 2 at baseline is
lesion 2 at follow-up would fabricate the tracking ground truth §7.4 exists to
record.

The migration is one-way, and that is deliberate.  A 0.x reader opening a 1.0
file raises on the missing ``schema_version``, which is the correct loud failure.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from medh5.errors import MEDH5Error
from medh5.io._common import sanitize_key, sanitize_stem
from medh5.io._legacy_reader import LegacyMeta, LegacySample, is_legacy
from medh5.io._legacy_reader import read_meta as _read_meta
from medh5.io._legacy_reader import read_sample as _read_sample
from medh5.io.grouping import (
    Occasion,
    SubjectGroup,
    group_by_subject,
    note_instance_ids,
    output_name,
)
from medh5.io.report import ConversionReport

BOX_SHIFT = -0.5
"""0.x ``[min, max)`` integer boxes sit at voxel edges once shifted (§8.1)."""


def read_legacy(path: str | os.PathLike[str]) -> LegacySample:
    """Read a whole 0.x file (:mod:`medh5.io._legacy_reader`)."""
    return _read_sample(path)


def legacy_meta(path: str | os.PathLike[str]) -> LegacyMeta:
    """Read a 0.x file's metadata without its arrays."""
    return _read_meta(path)


def _subject_key(meta: LegacyMeta, key: str | None) -> str | None:
    """Pull a subject key out of 0.x ``extra`` by dotted path."""
    if not key:
        return None
    node: Any = {"extra": dict(meta.extra)}
    for part in key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return str(node) if node not in (None, "") else None


def build_label_set(
    paths: Sequence[str | os.PathLike[str]],
    *,
    report: ConversionReport | None = None,
) -> Any:
    """Mint one label set covering a whole cohort's mask names and box labels.

    Cohort-wide rather than per file: ids minted independently per file would
    make ``liver`` id 1 in one sample and id 2 in the next, which is exactly the
    inconsistency a label set exists to prevent.
    """
    from medh5.labels.labelset import LabelClass, LabelSet

    log = report
    names: list[str] = []
    reused: dict[str, int] = {}
    for path in paths:
        try:
            meta = legacy_meta(path)
        except MEDH5Error:
            continue  # reported by the pass that migrates it
        for name in meta.seg_names:
            if name not in names:
                names.append(name)
        extra = dict(meta.extra)
        nnunet = extra.get("nnunetv2") or {}
        for label, value in (nnunet.get("labels") or {}).items():
            if isinstance(value, int) and value > 0:
                reused[_key(label)] = int(value)
        sample = None
        try:
            sample = read_legacy(path)
        except MEDH5Error:
            sample = None  # likewise
        if sample is not None and sample.bbox_labels:
            for label in sample.bbox_labels:
                if label not in names:
                    names.append(label)

    classes: list[LabelClass] = []
    used: set[int] = set(reused.values())
    next_id = 1
    for name in names:
        key = _key(name)
        if key in reused:
            class_id = reused[key]
        else:
            while next_id in used:
                next_id += 1
            class_id = next_id
            used.add(class_id)
        classes.append(LabelClass(class_id, key, name))
    if log is not None:
        log.decision(
            "label_set",
            f"{len(classes)} class(es) were minted across {len(list(paths))} file(s); "
            + (
                f"{len(reused)} id(s) came from an existing extra.nnunetv2.labels "
                "mapping"
                if reused
                else "no existing id mapping was found, so ids are sequential"
            ),
            {"ids": {c.key: c.id for c in classes}, "reused": sorted(reused)},
        )
    return LabelSet("migrated", version="1.0.0", classes=classes)


def _key(name: str) -> str:
    return sanitize_key(name)


def _sample_key(subject_id: str) -> str:
    """A sample id from a subject id: §2.3's identifier rule, not a label key.

    ``pat-a`` stays ``pat-a`` --- the hyphen is legal in an identifier and in a
    filename, and a migrated cohort keeps the names its manifest already uses.
    """
    return (sanitize_stem(str(subject_id).strip(), limit=128) or "sample").lower()


def migrate(
    path: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    label_set: Any = None,
    codec: str = "balanced",
    report: ConversionReport | None = None,
) -> ConversionReport:
    """Migrate one 0.x file into one 1.0 sample (Appendix B)."""
    log = report or ConversionReport(converter="migrate")
    log.source = os.fspath(path)
    labels = label_set or build_label_set([path], report=log)
    group = SubjectGroup(
        subject_id=Path(os.fspath(path)).stem,
        occasions=[Occasion(key=os.fspath(path), payload=os.fspath(path))],
        ordered_by="given",
    )
    _write(group, Path(os.fspath(out)), labels, codec=codec, log=log)
    return log


def migrate_paths(
    paths: Sequence[str | os.PathLike[str]],
    outdir: str | os.PathLike[str],
    *,
    group_by: str = "study",
    subject_key: str | None = None,
    label_set: Any = None,
    codec: str = "balanced",
) -> ConversionReport:
    """Migrate a cohort, minting one label set for all of it."""
    directory = Path(os.fspath(outdir))
    directory.mkdir(parents=True, exist_ok=True)
    log = ConversionReport(converter="migrate", source=f"{len(paths)} file(s)")
    labels = label_set or build_label_set(paths, report=log)

    occasions = []
    for path in paths:
        text = os.fspath(path)
        try:
            meta = legacy_meta(text)
        except MEDH5Error as exc:
            # Not a 0.x file, or a broken one.  A cohort migration reports it
            # and carries on rather than abandoning the other files.
            log.warn("unreadable", f"{text}: {exc}", {"path": text})
            continue
        occasions.append(
            Occasion(
                key=text,
                subject_id=_subject_key(meta, subject_key),
                date=_date_of(meta),
                order_hint=Path(text).stat().st_mtime,
                payload=text,
            )
        )
    if group_by == "subject" and subject_key is None:
        log.warn(
            "grouping",
            "--group-by subject needs --subject-key: a 0.x file has no subject "
            "field of its own, and identity is never inferred from filenames",
            {},
        )
    groups = group_by_subject(occasions, mode=group_by, report=log)
    used: set[str] = set()
    for group in groups:
        target = directory / f"{output_name(group, used, safe=_sample_key)}.medh5"
        _write(group, target, labels, codec=codec, log=log)
    return log


def _date_of(meta: LegacyMeta) -> str | None:
    extra = dict(meta.extra)
    for key in ("study_date", "date", "acquisition_date"):
        value = extra.get(key)
        if value:
            return str(value)
    return None


def _write(
    group: SubjectGroup,
    target: Path,
    label_set: Any,
    *,
    codec: str,
    log: ConversionReport,
) -> None:
    import medh5

    note_instance_ids(group, log)
    if group.ordered_by == "order_hint":
        log.guess(
            "timepoint_order",
            f"{target.name}: timepoints were ordered by file mtime, which is a "
            "heuristic; supply dates in extra to make the order evidence",
            {"order": [o.key for o in group.occasions]},
        )
    days = group.days_from_baseline()
    with medh5.create(
        target,
        sample_id=_sample_key(group.subject_id),
        subject_id=group.subject_id,
        codec=codec,
    ) as writer:
        writer.label_set(label_set)
        tool = writer.software("medh5", medh5.__version__)
        for index in range(len(group.occasions)):
            writer.add_timepoint(
                f"tp{index}", index=index, days_from_baseline=days[index]
            )
        for index, occasion in enumerate(group.occasions):
            _migrate_one(
                writer,
                read_legacy(occasion.payload),
                str(occasion.payload),
                f"tp{index}",
                label_set,
                tool,
                log,
                single=len(group.occasions) == 1,
            )
    log.outputs.append(str(target))


def _migrate_one(
    writer: Any,
    sample: LegacySample,
    source: str,
    timepoint: str,
    label_set: Any,
    tool: Any,
    log: ConversionReport,
    *,
    single: bool,
) -> None:
    """One 0.x file into one timepoint of a 1.0 sample."""
    meta = sample.meta
    suffix = "" if single else f"_{timepoint}"
    activity = writer.activity(
        "import",
        agent=tool,
        tool="medh5 migrate",
        inputs=[f"medh5-0.x:{source}"],
    )
    grid_id = f"ref{suffix}"
    spatial = meta.spatial
    first = sample.images[sorted(sample.images)[0]]
    writer.add_grid(
        grid_id,
        shape=first.shape,
        spacing=spatial.spacing or [1.0] * first.ndim,
        origin=spatial.origin,
        direction=spatial.direction,
        coord_system=spatial.coord_system or "LPS",
        axis_names=spatial.axis_labels,
        timepoint=timepoint,
        patch_hint=meta.patch_size,
    )
    for name, array in sorted(sample.images.items()):
        writer.add_image(
            f"{name}{suffix}", array, grid=grid_id, modality="OT", prov=activity
        )

    if sample.seg:
        masks = {
            label_set[_key(name)].id: np.asarray(v, dtype=bool)
            for name, v in sorted(sample.seg.items())
        }
        kind, stats = writer.add_segmentation(
            f"seg{suffix}",
            grid=grid_id,
            masks=masks,
            annotated_classes=[label_set[_key(n)].id for n in sorted(sample.seg)],
            prov=activity,
        )
        log.decision(
            "encoding",
            f"{source}: {len(masks)} mask(s) were measured and stored as {kind!r}",
            {
                "source": source,
                "kind": kind,
                "overlapping_pairs": 0 if stats is None else len(stats.edges),
            },
        )
        log.guess(
            "coverage",
            f"{source}: annotated_class_ids was set to the migrated mask names --- "
            "the only defensible inference. Widen or narrow it if the curator "
            "knows which classes were actually searched for (§11.3)",
            {"source": source, "classes": sorted(sample.seg)},
        )

    if sample.bboxes is not None and len(sample.bboxes):
        boxes = np.asarray(sample.bboxes, dtype=np.float64) + BOX_SHIFT
        labels = sample.bbox_labels or ["object"] * boxes.shape[0]
        class_ids = [label_set[_key(name)].id for name in labels]
        writer.add_boxes(
            f"boxes{suffix}",
            boxes=boxes.astype(np.float32),
            class_ids=class_ids,
            scores=sample.bbox_scores,
            grid=grid_id,
            space="index",
            task="detection",
            prov=activity,
        )
        log.decision(
            "box_convention",
            f"{source}: {boxes.shape[0]} box(es) were shifted by -0.5 on every "
            "axis --- 0.x stored slice-like [min, max) integers, 1.0 stores voxel "
            "edges, and the numbers differ by half a voxel (§8.1)",
            {"source": source, "boxes": int(boxes.shape[0]), "shift": BOX_SHIFT},
        )

    if meta.label is not None:
        name = meta.label_name or str(meta.label)
        entry = label_set.get(_key(name))
        if entry is not None:
            writer.add_classification(
                f"label{suffix}",
                labels={entry.id: 1.0},
                scope="sample",
                timepoints=[timepoint],
                prov=activity,
            )
        else:
            log.warn(
                "label",
                f"{source}: sample label {name!r} is not in the label set and was "
                "not migrated",
                {"source": source, "label": name},
            )

    extra = dict(meta.extra)
    if extra:
        writer.extra("legacy", extra)
    review = extra.get("review")
    if isinstance(review, Mapping):
        _migrate_review(writer, review, suffix, log, source)


def _migrate_review(
    writer: Any,
    review: Mapping[str, Any],
    suffix: str,
    log: ConversionReport,
    source: str,
) -> None:
    """0.x ``extra.review`` into the provenance graph and a quality record."""
    reviewer = review.get("reviewer") or review.get("by")
    agent = writer.person(str(reviewer)) if reviewer else None
    writer.activity(
        "review",
        agent=agent,
        ended=_timestamp(review.get("date") or review.get("reviewed_at")),
        params={"verdict": str(review.get("status", "reviewed"))},
        outputs=[f"annotations/seg{suffix}"],
    )
    status = str(review.get("status", "reviewed")).lower()
    writer.set_quality(
        f"seg{suffix}",
        status=status
        if status
        in ("draft", "submitted", "reviewed", "approved", "rejected", "deprecated")
        else "reviewed",
        reviewed_by=[agent.id] if agent else [],
    )
    log.decision(
        "review",
        f"{source}: extra.review became a `review` activity plus a quality record; "
        "0.x kept review state in an ad-hoc dict that could not say what produced "
        "the data being reviewed (§11.1)",
        {"source": source, "status": status},
    )


def _timestamp(value: Any) -> str | None:
    """A 0.x date as RFC 3339, or ``None`` when it is not one."""
    if not value:
        return None
    text = str(value)
    if text.endswith("Z") and "T" in text:
        return text
    if len(text) == 10 and text[4] == "-":
        return f"{text}T00:00:00Z"
    return None


def write_sidecar(label_set: Any, path: str | os.PathLike[str]) -> Path:
    """Write the minted label set for review before a cohort-wide migration."""
    target = Path(os.fspath(path))
    target.write_text(json.dumps(label_set.to_json(), indent=2) + "\n")
    return target


def load_sidecar(path: str | os.PathLike[str]) -> Any:
    from medh5.labels.labelset import LabelSet

    return LabelSet.from_json(json.loads(Path(os.fspath(path)).read_text()))


__all__ = [
    "BOX_SHIFT",
    "is_legacy",
    "build_label_set",
    "legacy_meta",
    "load_sidecar",
    "migrate",
    "migrate_paths",
    "read_legacy",
    "write_sidecar",
]
