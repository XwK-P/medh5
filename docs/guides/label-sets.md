# Choose and map a label set

<!-- sequential -->

An annotation stores a class as a `uint16` id and never as a name, so the label set is the only
thing that says id 2 is a lesion. A cohort trains on what it means only when every file means the
same thing by an id. This page chooses a vocabulary, writes two sites' samples, finds where they
disagree (`C101`, `C102`), relabels one site to fix it, and compares two raters who numbered one
vocabulary differently. Its blocks run in order.

## Choose a vocabulary

Three vocabularies ship with the package, one for each shape a label set takes: one class
(`binary-foreground`), a small hierarchy (`brats-subregions`) and a flat set of organs
(`amos22-organs`).

```python
from medh5.labels import LabelClass, LabelSet, available, load

assert available() == ("amos22-organs", "binary-foreground", "brats-subregions")
organs = load("amos22-organs")
assert organs["liver"].key == "liver"
```

When none fits, write your own. A class is an id, a key that code uses, and a name people read;
`parents` says what it lies inside, and may name more than one, because a hierarchy of anatomy is
a graph rather than a tree:

```python
abdomen = LabelSet("abdomen", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "lesion", "Lesion", parents=[1], category="lesion"),
])
assert abdomen["lesion"].id == 2 and abdomen[2].key == "lesion"
assert abdomen.ancestors("lesion") == (1,)          # inside the liver
```

Ids `0` (background) and `65535` (ignore) are reserved. Three rules keep a vocabulary usable
across a cohort:

- **An id keeps its meaning.** A class that goes away leaves its id unused; a new one takes a new
  id.
- **A change is a new version.** Two versions that agree on every class they share do not
  clash, though `dataset check` still notes that a cohort carries both.
- **Keys are for code, names are for people.** Matching across vocabularies goes by key, so keep
  a key once it is in a file.

## Write two sites' samples

Each site writes its masks with its own label set. Site B drew one class more, a cyst, and gave it
id 2, which is site A's lesion:

```python
import os
from pathlib import Path

import numpy as np
import medh5

def write(path, label_set, masks, frame=None):
    with medh5.create(path, sample_id=Path(path).stem) as w:
        w.label_set(label_set)
        w.add_grid("ct", shape=(8, 16, 16), spacing=(2.0, 1.0, 1.0),
                   frame_uid=frame or "pseudo:" + path)
        w.add_image("CT", np.zeros((8, 16, 16), np.int16), grid="ct", modality="CT")
        w.add_segmentation("organs", grid="ct", masks=masks, annotated_classes="all")

def box(z, y, x, size=4):
    mask = np.zeros((8, 16, 16), bool)
    mask[z:z + size, y:y + size, x:x + size] = True
    return mask

site_b = LabelSet("abdomen-b", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "cyst", "Cyst", parents=[1], category="lesion"),
    LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
])
os.makedirs("sites", exist_ok=True)
write("sites/a1.medh5", abdomen, {1: box(0, 0, 0, 8), 2: box(2, 2, 2)})
write("sites/b1.medh5", site_b, {1: box(0, 0, 0, 8), 2: box(2, 8, 8), 3: box(2, 2, 2)})
```

## Find where a cohort disagrees

`medh5.labels.check` groups files by vocabulary, which is the first thing to ask of files that came
from more than one place:

```python
from medh5.labels import check

found = check(["sites/a1.medh5", "sites/b1.medh5"])
assert not found["ok"] and len(found["vocabularies"]) == 2
```

`medh5 dataset check` says what the difference costs. Two vocabularies in one cohort is `C101`, a
warning: they may agree. An id that means two things is `C102`, an error, because a model trained
on the union learns cysts as lesions:

```python
from medh5.dataset import check as check_cohort, scan

manifest, _ = scan("sites")
report = check_cohort(manifest)
codes = {f.code: f.severity for f in report.findings}
assert codes["C101"] == "warning" and codes["C102"] == "error"
print(report.errors[0].message)     # class id(s) [2] mean different things ...
```

The report also carries `C301`: site B examined a class site A never looked for, so site A's
samples are no negatives of it. That is coverage (§11.3), not vocabulary, and relabelling leaves
it. From the shell, `medh5 labels check sites/*.medh5` groups the files, and `medh5 dataset check`
reads a manifest saved with `manifest.save("cohort.json")`.

## Relabel to one vocabulary

Fix the clash by bringing site B into site A's vocabulary. Site A's set has no cyst, so the shared
vocabulary is a new version of it that adds one, under a new id:

```python
shared = LabelSet("abdomen", version="1.1.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "lesion", "Lesion", parents=[1], category="lesion"),
    LabelClass(3, "cyst", "Cyst", parents=[1], category="lesion"),
])
```

Then rewrite site B's annotation through `medh5.amend`, moving each class to the id its key has in
the shared set. Class keys carry the meaning across:

```python
with medh5.open("sites/b1.medh5") as s:
    organs = s.annotations["organs"]
    old = dict(zip((site_b[i].key for i in organs.class_ids), organs.dense()))

with medh5.amend("sites/b1.medh5") as w:
    w.remove_annotation("organs")
    w.label_set(shared)
    w.add_segmentation("organs", grid="ct", annotated_classes="all",
                       masks={shared[key].id: mask for key, mask in old.items()})

report = check_cohort(scan("sites")[0])
assert report.errors == [] and "C101" in [f.code for f in report.warnings]
```

The error is gone: 1.0.0 and 1.1.0 agree on every class they share. `C101` stays, because the
cohort still carries two versions. Updating site A's label set to 1.1.0 clears it, and needs no
relabel, since the ids it uses keep their meaning:

```python
with medh5.amend("sites/a1.medh5") as w:
    w.label_set(shared)

report = check_cohort(scan("sites")[0])
assert report.ok and "C101" not in [f.code for f in report.warnings]
```

Relabelling is the fix when the cohort is yours to rewrite. When it is not, measure each
vocabulary apart: the cohort's statistics refuse the clash too (`C102`), rather than count a cyst
as a lesion.

## Compare raters who numbered differently

Two raters' files of one scan often use two numberings: here rater B's tool numbers liver 6.
Compared by id, the two files are refused, because id 1 is not liver on both sides:

```python
rater_b = LabelSet("tool-b", version="1.0.0", classes=[
    LabelClass(6, "liver", "Liver", category="organ"),
    LabelClass(1, "lesion", "Lesion", parents=[6], category="lesion"),
])
frame = "pseudo:scan-17"                # one scan, so one frame
write("rater_a.medh5", abdomen, {1: box(0, 0, 0, 8), 2: box(2, 2, 2)}, frame=frame)
write("rater_b.medh5", rater_b, {6: box(0, 0, 0, 8), 1: box(2, 2, 3)}, frame=frame)
```

They are written on one grid, in one frame, which is what two readings of one scan share. Compare
by key with `map_classes="key"`, or state the pairs:

```python
from medh5.curation.agreement import compare_voxel
from medh5.errors import MEDH5ValidationError

with medh5.open("rater_a.medh5") as a, medh5.open("rater_b.medh5") as b:
    seg = (a.annotations["organs"], b.annotations["organs"])
    refused = None
    try:
        compare_voxel(*seg)
    except MEDH5ValidationError as refusal:
        refused = refusal.code
    by_key = compare_voxel(*seg, map_classes="key")
    stated = compare_voxel(*seg, map_classes={"liver": 6, "lesion": 1})

assert refused == "C102"
assert by_key.b_class_ids == {"liver": 6, "lesion": 1}      # what each was scored against
assert by_key.per_class["liver"] == 1.0 and by_key.per_class == stated.per_class
```

`"key"` needs both label sets inline, and where both classes of one key carry a code of one
ontology, the codes must agree. A stated map is one to one: a class of `b` two classes map onto
would be counted twice. From the shell, `medh5 agree` takes `--map-classes key` and
`--map-class A=B`.

A record of the comparison names an annotation of the sample that stores it (`against`, §11.2).
These are two files, so none of rater A's paths names rater B's reading, and `to_record()` refuses
until you say which one does:

```python
refused = None
try:
    by_key.to_record()
except MEDH5ValidationError as refusal:
    refused = refusal
assert refused is not None
```

Copy rater B's reading into rater A's file, in A's ids, and record against that copy:

```python
with medh5.open("rater_b.medh5") as b:
    theirs = b.annotations["organs"]
    planes = zip((rater_b[i].key for i in theirs.class_ids), theirs.dense())
    copied = {abdomen[key].id: mask for key, mask in planes}

with medh5.amend("rater_a.medh5") as w:
    w.add_segmentation("organs_b", grid="ct", masks=copied, annotated_classes="all")
    w.set_quality("organs", status="reviewed",
                  agreement=[by_key.to_record(against="annotations/organs_b")])
```

## Related

- [Annotations › Label sets](../reference/annotations.md#label-sets): the classes, the bundled
  vocabularies and the reserved ids.
- [Curation › Two vocabularies](../reference/curation.md#two-vocabularies): `map_classes` and
  `match_by` in full.
- [Cohort check codes](../reference/cohort-checks.md): `C101`, `C102` and `C103`.
- [Record review, provenance and agreement](review-and-provenance.md): comparing two raters in one
  file.
