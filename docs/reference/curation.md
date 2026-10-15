# Curation records

Who produced this, how good is it, who is it about, and can it be shared.
Spec §11–§12.

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

Object-level agreement matches instances by id where both carry one and both
are in one sample, and by IoU otherwise. An `instance_id` names an object within
its own sample (§7.4), so two raters' files that each number their objects from
1 are matched by where the objects are, not by their numbers. A miss counts only
where the other side looked: an unmatched object whose class the other
annotation never examined is left out, and its class is listed under `skipped`.
A matched pair always counts, even when the two classed it differently — that
is what `class_mismatches` reports. Both must be on one grid for index boxes, or
one frame for world boxes; comparing index coordinates of two grids is refused
(`E101`), and so are boxes of different numbers of axes, which are not in one
space (`E414`), and a box whose bound is not finite or whose `lo` is above its
`hi` (`E406`).

<!-- illustrative -->
```python
from medh5.curation.agreement import compare_instances

result = compare_instances(s.annotations["pred"], s.annotations["truth"],
                           threshold=0.5, match_by="auto")
result.matched            # ((index in a, index in b, iou), ...): rows, not ids
result.only_in_a, result.only_in_b
result.matched_by         # "instance_id" within one sample where both carry ids, else "iou"
                          # (match_by="iou" or "instance_id" says which)
result.to_json()["threshold"]  # None for an id match, which pairs whatever the IoU
result.mean_iou           # None when nothing matched
result.value              # object F1; None when neither side has an object
result.skipped            # classes not examined by both
result.class_mismatches   # matched objects whose classes disagree
```

`compare(a, b)` picks the comparison the kinds support — objects for two
`instances` or `boxes` annotations, voxels for two voxel annotations — and
refuses an argument the chosen one cannot use (`metric=` for objects,
`threshold=` for voxels) rather than ignoring it. The IoU threshold lies in
(0, 1], as for `medh5 agree --threshold`; NaN, 0 or less, and more than 1 are
refused.

Two files compare where their grids share a declared frame and are one lattice
in it — two raters' copies of one scan. A class id is then a number within each
file's label set: an id both use that the two label sets give different
meanings, or a class name the two resolve to different ids, is refused (`C102`,
as `dataset check` refuses it in a cohort) rather than scored. Where a file has
no label set, its ids are compared as numbers.

`match_by` chooses how objects pair. `"auto"` is the rule above; `"iou"` pairs
by where the objects are even where both carry ids; `"instance_id"` pairs by id
and refuses two samples, because an id names an object only within its own
sample (§7.4). No id map across samples exists: the format defines no identity
across them.

### Two vocabularies

Two raters' files often number one vocabulary differently: liver is 1 in one
and 6 in the other. Compared by id that is refused (`C102`); `map_classes` says
which class of `b` each class of `a` is scored against.

<!-- illustrative -->
```python
from medh5.curation.agreement import compare_voxel

with medh5.open("rater_a.medh5") as a, medh5.open("rater_b.medh5") as b:
    seg = (a.annotations["organs"], b.annotations["organs"])
    by_key = compare_voxel(*seg, map_classes="key")
    stated = compare_voxel(*seg, map_classes={"liver": 6})
by_key.per_class          # {"liver": 0.94}: keyed by a's classes
by_key.b_class_ids        # {"liver": 6}: the class of b each was scored against
by_key.skipped            # a key b's label set lacks, and why: never scored 0
```

- `"id"` (the default): one id is one class on both sides.
- `"key"`: classes of one key are compared. Both sides need an inline label
  set, so a label set by reference is refused, and where both classes carry a
  code of one ontology the codes must agree: one key with two codes is two
  classes (`C102`).
- A mapping of `a`'s classes to `b`'s, by key or id: it must be one to one,
  since a class two classes map onto is counted twice. Classes it does not name
  are not compared.

`medh5 agree` takes the same choices as `--map-classes id|key`, `--map-class
A=B` (repeatable) and `--match-by auto|iou|instance-id`.

### Recording a comparison

A `quality.agreement` record's `against` is the path of an annotation of the
recording sample (§11.2), `annotations/<id>`. Within one sample, even opened
through two handles, that is the other annotation, and `to_record()` fills it
in. A comparison across two samples has `against =
None`, because no path of the recording sample names the other file's
annotation, and `to_record()` refuses it; copy the other annotation in and name
it with `to_record(against="annotations/rater_b")`.

## Tracking

`Sample.tracks()` joins `instance_id` across visits (§7.4) and returns a
`Tracking`: `{instance_id: Track}`, each absence with a state (`present`,
`resolved`, `unexamined`, `medh5.curation.STATES`) and a reason
(`medh5.curation.REASONS`, in precedence order) from `reason_at` and
`reasons`. `coverage` is class-level: the classes each visit examined, not
where. The [longitudinal guide](../guides/longitudinal.md) has the rule.
In Rust, `curation::tracking::locate_tracks` returns a `LocatedTracking`;
`Tracking`'s own `state_at`, `states`, `is_new` and `is_resolved` are the
class-level rule, deprecated in 2.1.

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

<!-- illustrative -->
```python
audit = medh5.audit_splits(paths)
audit.ok
audit.conflicts        # W906: one set_id, different manifest digests
audit.leaks            # a subject or group key in more than one partition, or fold
audit.unclaimed        # files carrying no split claim at all
```

That check needs the whole cohort, which is why it is not part of `validate`.
See [Build and split a cohort](../guides/cohorts.md).

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

## Editing a record

A record's dicts — `Identity.extra`, `Cohort.extra`,
`Deidentification.extra`, `Activity.params`, `Timepoint.series_uids`, an
agreement's `per_class` — are the record's own: an edit in place is kept, and
every rule is checked again on the record as edited whenever it is used. A
record the writer or a document hands out writes the edit through to the
document, so it reaches the file at `commit`:

```python
w.identity(sex="F").extra["consent"] = "research-only"
act = w.activity("review", agent=w.person("RAD-08"))
act.params["window"] = [-150, 250]
w.document.timepoints["tp1"].series_uids["CT"] = "1.2.840.99.1"

assert w.document.identity.extra == {"consent": "research-only"}
assert w.document.provenance.activity(act.id).params["window"] == [-150, 250]
```

A record given to the writer — `w.document.identity = record`,
`w.document.provenance.add_activity(record)` — is stored as it stands then:
edit what the document hands back, not the record given. An edit through a
record read before its member changed another way (`w.identity(...)` again,
say) raises `RuntimeError` rather than undoing that change; read it again and
edit that.

An image's own attributes are corrected the same way, without rewriting its
samples: `w.set_image_attrs("CT_tp0", modality="CT")` in an amendment, with
`None` to remove one. An image or grid that should not be there is removed
with `w.remove_image` and `w.remove_grid`, and the commit refuses what would
leave a reference dangling ([Writing](python-api.md#derived-data)).

A record added to an existing file goes through `medh5.amend`, which writes
the new version beside the file and renames it over the old
([Amending](python-api.md#amending)). Through a symbolic link, the file the
link names is rewritten and the link kept; a file with several hard links is
replaced at the path given only, and its other names keep the old version. A
link in a directory anyone can write to, such as `/tmp`, that neither you nor
the directory's owner owns is not followed: the amend is refused
(`MEDH5FileError`), as the system would refuse to open the file through it.
A file that changed after the amendment read it is not replaced either: the
commit is refused (`MEDH5FileError`), and the other writer's version is kept.

## Collections and integrity

Collections (`.medh5c`), per-object digests and `content_id` are container
concerns rather than curation ones. They are documented together in
[Storage](storage.md#collections).

## Related

- **[De-identify and publish](../guides/deidentify.md)** — the procedure.
- **[Build and split a cohort](../guides/cohorts.md)** — where split claims come from.
- **[Sample document schema](schema.md)** — the fields these calls write.
- **[Storage](storage.md)** — collections, digests, `content_id`.
