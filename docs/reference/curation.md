# Curation

Who produced this, how good is it, who is it about, and can it be shared.
Spec §11–§13.

## Provenance

An agents-and-activities graph, not a free-text comment:

```python
tool = w.software("nnU-Net", "2.4.2")
rad  = w.person("RAD-07")
site = w.organization("Site A")

act = w.activity("predict", agent=tool, tool="nnUNetv2_predict",
                 params={"fold": "all", "checkpoint": "final"},
                 inputs=["images/CT_tp0"],
                 started="2026-02-03T09:11:02Z", ended="2026-02-03T09:11:40Z")

w.add_segmentation("organs_pred", grid="ct_tp0", masks=masks, prov=act)
```

Activity types: `import`, `annotate`, `review`, `predict`, `resample`,
`register`, `derive`, `deidentify`, `transcode`, `other`.

```python
s.document.provenance.agents
s.document.provenance.activities
s.document.provenance.activities_by_type("review")
s.annotations["organs_pred"].prov      # the activity id
```

```
$ medh5 prov case.medh5
```

The question this answers is the one that comes up when a model behaves oddly
on a subset: *was this mask drawn or predicted, and by what?*

Ids are unique per graph. The writer refuses a taken id; a file that declares
one agent or activity id twice is refused when its document is read — the
validator reports it through the parse — because a reference to the id cannot
say which one it means. Before 1.4.2 the reader kept the last one silently, and
a no-op `amend` then deleted the other from the file.

## Quality

```python
from medh5 import Issue

w.set_quality("organs", status="reviewed", reviewed_by=[rad.id],
              confidence=0.8,
              issues=[Issue(code="boundary", severity="warning",
                            class_ids=[1], note="inferior liver edge")])
```

```python
q = s.document.quality["organs"]
q.status          # draft | submitted | reviewed | approved | rejected | deprecated
q.reviewed_by
q.issues
q.agreement
q.confidence, q.edit_effort_s
q.is_usable       # does this record mark data fit to train or evaluate on?
```

## Agreement

Between two annotations of the same sample — two readers, or a reader and a
model:

```python
from medh5 import compare_annotations

with medh5.open(path) as s:
    result = compare_annotations(s.annotations["organs_a"],
                                 s.annotations["organs_b"],
                                 metric="dice")
result.metric             # "dice"
result.per_class          # {"liver": 0.67} — keyed by class key, for reading
result.value              # the mean over compared classes; None when none was
result.skipped            # classes one side never examined — not scored as zero
result.to_record()        # the quality.agreement record, keyed by class id
```

`skipped` matters: a class the other annotation never examined is not a
disagreement, and scoring it as zero would punish a reader for a class nobody
asked them to draw.

```
$ medh5 agree case.medh5 organs_a organs_b --metric dice --record
```

`--record` prints the `quality.agreement` record the measurement produces, so a
number that gets quoted can be traced to the comparison that made it. The
record is keyed by class **id**, as §11.2 and the schema require, and it is
refused when nothing was comparable: `value` is then `None`, not 0 — two raters
who both found nothing agree, and a 0 in the file would say they did not.

Object-level agreement matches instances by id where both carry one, and by IoU
otherwise. A miss counts only where the other side looked: an unmatched object
whose class the other annotation never examined is left out, and its class is
listed under `skipped`. A matched pair always counts, even when the two classed
it differently — that is what `class_mismatches` reports. Both must be on one
grid for index boxes, or one frame for world boxes; comparing index coordinates
of two grids is refused (`E101`).

<!-- illustrative -->
```python
from medh5.curation.agreement import compare_instances

result = compare_instances(s.annotations["pred"], s.annotations["truth"],
                           threshold=0.5)
result.matched            # ((a_id, b_id, iou), ...)
result.only_in_a, result.only_in_b
result.matched_by         # "instance_id" where both carry one, else "iou"
result.mean_iou           # None when nothing matched
result.value              # object F1; None when neither side has an object
result.skipped            # classes not examined by both
result.class_mismatches   # matched objects whose classes disagree
```

`compare(a, b)` picks the comparison the kinds support — objects for two
`instances` or `boxes` annotations, voxels for two voxel annotations — and
refuses an argument the chosen one cannot use (`metric=` for objects,
`threshold=` for voxels) rather than ignoring it.

## Identity and cohort

```python
w.identity(sex="F", laterality="left", bodypart="breast")
w.cohort(dataset_id="abdomen-v3", site_id="site-B",
         scanner_id="SOMATOM-Force-042", group_id="family-7",
         acquisition_protocol="portal-venous 1mm")
```

`subject_id` is required and should be a pseudonym. `group_id` defaults to
`subject_id` and is the grouping key for leakage-free splits; it exists for the
coarser cases — a family, an enrolling site, a scanner batch.

Per-visit identifiers (`study_uid`, `series_uids`, dates, age) belong to the
**timepoint**, not to identity, because a sample has several of each.

## Split claims

```python
w.split(set_id="cv5-2026-02", partition="train", fold=1,
        assigned_by="pipeline@v3", manifest_sha256=manifest.sha256())
```

A claim in a file is a *claim*, not the authority. The manifest is the
authority, and `manifest_sha256` is what lets a reader notice the claim
predates the current split instead of quietly training on a stale partition.

Writing a claim for a `set_id` **replaces** any earlier claim for the same set:
two claims for one set is precisely the `W906` conflict the validator catches.

```
$ medh5 splits cohort/*.medh5        # conflicting claims and subject leakage
```

That check needs the whole cohort, which is why it is not part of `validate`.
See [Cohorts](../guides/cohorts.md).

## De-identification

```python
w.deidentification(method="dicom-psi-profile",
                   profile="DICOM PS3.15 E.1 basic + clean pixel",
                   date_shift_days=-117, id_mapping="external",
                   performed_by=rad.id, burned_in_annotation_checked=True)
```

**A file with no de-identification record must be treated as potentially
identifying.** Absence is never evidence.

`medh5 scrub` finds identifiers and writes this record for you. The procedure —
including, importantly, what it cannot do — is
[De-identify and publish](../guides/deidentify.md).

## Collections and integrity

Collections (`.medh5c`), per-object digests and `content_id` are container
concerns rather than curation ones. They are documented together in
[Storage](storage.md#collections).

## Related

- **[De-identify and publish](../guides/deidentify.md)** — the procedure.
- **[Build and split a cohort](../guides/cohorts.md)** — where split claims come from.
- **[Sample document schema](schema.md)** — the fields these calls write.
- **[Storage](storage.md)** — collections, digests, `content_id`.
