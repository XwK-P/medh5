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
`register`, `derive`, `deidentify`, `other`.

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
result.per_class          # {"liver": 0.67} — keyed by class key
result.value              # the mean over compared classes
result.skipped            # classes one side never examined — not scored as zero
result.to_record()        # the quality.agreement record
```

`skipped` matters: a class the other annotation never examined is not a
disagreement, and scoring it as zero would punish a reader for a class nobody
asked them to draw.

```
$ medh5 agree case.medh5 organs_a organs_b --metric dice --record
```

`--record` prints the `quality.agreement` record the measurement produces, so a
number that gets quoted can be traced to the comparison that made it.

Object-level agreement matches instances by id where both carry one, and by IoU
otherwise:

```python
from medh5.curation.agreement import compare_instances

result = compare_instances(s.annotations["pred"], s.annotations["truth"],
                           threshold=0.5)
result.matched            # ((a_id, b_id, iou), ...)
result.only_in_a, result.only_in_b
result.matched_by         # "instance_id" where both carry one, else "iou"
result.mean_iou
result.value              # object F1
result.class_mismatches   # matched objects whose classes disagree
```

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
See [Cohorts](cohorts.md).

## De-identification

```python
w.deidentification(method="dicom-psi-profile",
                   profile="DICOM PS3.15 E.1 basic + clean pixel",
                   date_shift_days=-117, id_mapping="external",
                   performed_by=rad.id, burned_in_annotation_checked=True)
```

**A file with no de-identification record must be treated as potentially
identifying.** Absence is never evidence.

### `medh5 scrub`

```
$ medh5 scrub out/*.medh5                                   # find, change nothing
$ medh5 scrub out/*.medh5 --apply --date-shift-days -117 --by RAD-07
```

```python
from medh5.curation import scrub

report = scrub.scan(path)                # find
report.actionable                        # what --apply would change
report.needs_review                      # what a person has to judge
print(report.format())

report = scrub.apply(path, date_shift_days=-117, salt="")   # act, and attest
```

Rules: identifying DICOM keywords anywhere in `extra` or `acquisition`, DICOM
person names, real DICOM UIDs in place of pseudonyms, unshifted dates, and free
text no rule can judge.

**UIDs are pseudonymised, not deleted.** A frame UID is how two files agree
they share a frame of reference (§3.4); deleting it breaks registration.
`pseudonymise(uid, salt)` is stable, so a cohort scrubbed file by file — even
on different machines — still joins. Only a *salted* run records
`id_mapping: external`; an unsalted hash is recoverable by anyone holding the
original UIDs, and claiming otherwise would be the overclaim this tool exists
to avoid.

**Dates shift rather than vanish**, so intervals survive. Running scrub twice
does not shift them twice.

**What it cannot do, and says so.** It reads metadata. It does not look at
voxels, so burned-in text, an identifiable face in a head CT and an accession
number photographed onto a film are all outside it. The record it writes sets
`burned_in_annotation_checked: false` and the report lists what was not
checked. A file this tool calls clean may still be identifying.

## Collections

One `.medh5c` shard holding many samples, for filesystems that dislike a
million small files:

```
$ medh5 pack cohort/*.medh5 -o shard.medh5c
$ medh5 ls shard.medh5c
$ medh5 unpack shard.medh5c -o restored/
```

```python
import medh5

medh5.pack(paths, "shard.medh5c")
with medh5.open_collection("shard.medh5c") as c:
    c["case_0001"].images["CT"].read()      # an ordinary Sample
```

Each member **is** a sample root, so every reader, validator and loader works
on it unchanged.

Packing is a container operation: chunks move as raw bytes, nothing is
decompressed, and `content_id` is preserved. Unpacking reproduces the original
files chunk for chunk — there is a test that compares them at that level,
because comparing through the value API would decompress and recompress and
prove nothing.

`sample ⊂ collection` is strict containment: a collection is not a sample and
does not pretend to be one. `medh5 validate` dispatches on the kind.

## Integrity

```python
s.verify().ok
s.verify(partial=["images/CT_tp0"])
s.content_id
```

Every object carries a SHA-256 over its **decompressed** content; the root
carries a Merkle `content_id` over those digests.

```
$ medh5 verify cohort/*.medh5
$ medh5 recompress cohort/*.medh5 --profile archive     # content_id survives
```

Because the root hashes *digests*, editing a dataset without restamping it
breaks the object digest and leaves the root matching — which is why `verify`
checks every object and not only the root. `medh5 fix --rewrite-digests` exists
for the case where an external tool made the edit, and it will not run without
a reason it can record.
