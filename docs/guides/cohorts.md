# Build and split a cohort

Turn a directory of `.medh5` files into a manifest, a leakage-free split, the
normalisation constants for it, and a check that the whole thing is coherent.

A sample is self-describing, but a *cohort* has properties no single file can
carry: which label set everyone agrees on, which subject is in which partition,
what the intensity distribution is, whether a class was examined everywhere or
only where somebody got around to it.

Manifests, splits and the cross-file checks come **from metadata alone**, so
they cost milliseconds over a large cohort rather than a pass over every voxel.
Statistics are the exception and are priced differently: `medh5 dataset stats`
streams one pass per file, and reads class counts from the sampling index only
where one is current — otherwise it decompresses the annotations to count
them. Build the index first if you are running it over a large study.

## Index the directory

```
$ medh5 dataset index studies/ -o cohort.json
cohort.json: 412 sample(s), 289 subject(s), sha256 4f21c8ab90e1
```

```python
from medh5.dataset import scan, Manifest

manifest, failures = scan("studies/")     # metadata-only
manifest.save("cohort.json")
manifest = Manifest.load("cohort.json")

len(manifest), manifest.subjects
manifest.groups("site_id")                # {"site-A": [Entry, ...], ...}
manifest.filter(lambda e: "seg" in e.profiles)
manifest.stale()                          # files that changed since the scan
```

A scan that dies on one broken file has told you nothing about the other 411,
so failures come back alongside the manifest rather than as an exception. A
collection fans out into one entry per member.

The manifest's paths are relative to the manifest wherever they lie beneath its
directory, and absolute otherwise, so it opens from any directory and moves
with its cohort: index into the cohort's own directory
(`medh5 dataset index studies/ -o studies/cohort.json`) and copy the directory
to a cluster mount, and every path still resolves. `Manifest.save(path,
paths="absolute")` (or `--absolute`) writes every path absolute instead.
[The cohort manifest](../reference/cohort-manifest.md) is the format, and says
how a manifest written before 2.1 is read and upgraded
(`medh5 dataset upgrade`).

Each entry carries what splitting, stratifying and filtering need:

```python
e = manifest[0]
e.subject_id, e.group_id, e.site_id, e.scanner_id, e.sex, e.bodypart
e.timepoints, e.days_from_baseline
e.images, e.modalities
e.class_ids, e.annotated_class_ids
e.examined(2)                 # was class 2 looked for here?
e.label_set_id, e.label_set_digest
e.splits, e.quality, e.deidentified
e.field("cohort.site_id")     # dotted or bare — same value
```

### What the digest covers

```python
manifest.sha256()
```

Covers the cohort's **membership** — which samples, grouped how — and nothing
else. Not paths, sizes, mtimes, the generating version, and deliberately not
`content_id`.

Membership and grouping by `group_id` are what a default split is computed
from, which makes this the right thing for a claim to be checkable against.
Including content would make it useless for that: writing a claim into a file
changes the file, so every claim would be stale the moment it was written.
Content drift is a different question, and `dataset check` answers it
separately (`C401`, and `--deep`).

A split made with `--group-by` or `--stratify-by` on another field depends on
that field's values too, and the digest does not cover them: re-curate
`site_id` and a `--stratify-by site_id` split comes out differently while every
claim still matches the digest, so `C201` cannot see it. Re-split after changing
a field you split on.

Re-scanning the same cohort on another machine produces the same digest.
Filtering the manifest changes it, so a split made from a subset cannot be
mistaken for one made from the whole.

This is the link the spec's `SplitClaim.manifest_sha256` was designed around.

## Split it

```
$ medh5 dataset split cohort.json --group-by group_id --stratify-by site_id \
      --ratios train=0.7,val=0.15,test=0.15 --seed 0 -o split.json --write-claims
```

<!-- illustrative -->
```python
from medh5.dataset import make_splits, write_claims

split = make_splits(manifest, set_id="cv5", group_by="group_id",
                    stratify_by="site_id", ratios={"train": .7, "val": .15, "test": .15},
                    seed=0)
split.counts            # {"test": 62, "train": 288, "val": 62}
split.balance()         # achieved stratum counts per partition
split.underfilled       # partitions that were asked for and got nothing
split.leaks()           # groups or entries in more than one partition — structurally ()
split.paths("train")    # files, and `path::key` for a sample inside a collection

write_claims(split, manifest, assigned_by="pipeline@v3")
```

Three rules, and they are the whole module.

**Group before you split.** The unit is `cohort.group_id`, which defaults to
the subject — never a file. A subject with a baseline and two follow-ups is one
unit. Since a sample is already a subject, this is mostly free; `group_id`
exists for the coarser cases (a family, an enrolling site).

A `--group-by` field that *cuts across* a subject is refused (`C204`) rather
than split. Grouping by `site_id` when one subject was scanned at two sites puts
that subject's samples in two groups, and two groups can land in different
partitions — the leak the grouping exists to prevent, reintroduced by the
grouping key. `dataset check` reports the realised version of the same problem
as `C202`; `C204` catches it while the split is being made, which is before it
can be trained on. Both `C202` and `medh5 splits` look for leaks over subjects
and grouping keys together — two files are one unit if they share either — so
one subject whose visits were curated under two `group_id`s is found in two
partitions, not two unrelated groups each in one.

**Stratify on what you can see.** Groups are indivisible and a group's stratum
is its majority, so exact balance is not always reachable. `balance()` reports
what was actually achieved rather than what was asked for.

**A claim in a file is not the split.** `write_claims` stamps each sample with
its partition *and* the manifest digest it came from. A later reader can then
tell a current claim from one that predates a re-split. Writing a claim for a
`set_id` replaces any earlier claim for that set. A k-fold split's claims name
the validation fold (`fold=2`, `--fold 2`): that fold's samples are written as
`val` and the rest as `train`. Every refusal is met before the first file
changes, so a refused write leaves every file as it was: a fold the split does
not have (or a fold given for a split made with ratios), a split that places
something twice, a sample inside a collection, and each file an amendment would
refuse — a later minor version, a profile this engine does not implement, HDF5
references, a file the validator rejects, or a dataset whose bytes fail its
digest (E701). Only a failure while writing, such as a full disk, can stop it
part way.

Assignment is deterministic given the manifest's entries, the seed and the
parameters — a hash, not a shuffle — so two machines produce the same
partitions.

### Small cohorts

Groups are dealt by largest deficit against the target ratios, not sliced by
index. With 6 groups at 70/15/15, slicing gives train everything.

Where the arithmetic genuinely cannot work — 4 indivisible groups into three
partitions is 2.8/0.6/0.6, and no integer allocation fills all three — the
split says so:

```
WARNING: test got no groups --- 4 indivisible group(s) cannot be split in the
ratios train=0.7,val=0.15,test=0.15
```

An empty test set is the kind of thing noticed after the results are written
up.

### K-fold

```
$ medh5 dataset split cohort.json --k-folds 5 --set-id cv5 -o folds.json
$ medh5 dataset split cohort.json --k-folds 5 --set-id cv5 --write-claims --fold 0
```

`--fold N` says which fold is the validation set for the claims being written;
every other fold becomes `train`. Every fold gets at least one group, so more
folds than groups is refused rather than run fewer ways than it says, and
`--ratios` does not apply.

## Compute the statistics

```
$ medh5 dataset index studies/ -o cohort.json     # record the claims just written
$ medh5 dataset stats cohort.json --partition train --set-id cv5 --workers 8 -o stats.json
```

`--partition` reads the claims from the manifest, which holds what the files
carried when it was indexed. `--write-claims` writes into the files, not into
the manifest, so re-index after it — the digest is unchanged, and the claims
still match it.

```python
from medh5.dataset import compute_stats

stats = compute_stats(paths, images=["CT"], workers=8)
stats.normalization("CT")          # (mean, std) for a z-score
stats.class_weights(scheme="inverse_frequency")

ct = stats.images["CT"]
ct.mean, ct.std, ct.minimum, ct.maximum

lesion = stats.classes[3]
lesion.voxels, lesion.present_in, lesion.examined_in, lesion.prevalence

stats.failures
```

Streaming: one pass per file, constant memory per worker, and an **exact**
Welford merge (Chan–Golub–LeVeque), not an approximation.

The intensity moments are over **physical** values, `stored × slope +
intercept`, because that is what the datasets hand a model with
`physical=True`. A z-score computed over what a file *stores* normalises the
wrong distribution: a CT stored `int16` under slope 2 and intercept −1024 has a
stored mean near 100 and a physical mean near −824. `compute_stats(...,
physical=False)`, or `--stored` on the command line, measures the stored values
when those are what you want, and the result records which it did.

Two things it deliberately does not do.

**It does not average per-file means.** That weights a 40-slice scan the same
as a 900-slice one. The merge weights by voxel count.

**It does not treat an unexamined class as a zero.** `prevalence` is over the
samples that actually examined the class, so a class annotated in a tenth of
the cohort does not look ten times rarer than it is. `present_in` counts the
same samples — those that examined the class and found it — so a class drawn
where nobody looked for it adds to `voxels` but not to its prevalence, which
stays at most 1.

Class counts come from the §14.3 sampling index when it is current — a few
hundred bytes per annotation instead of decompressing every mask. A stale index
is not trusted.

A class is its id *and* the key its label set gives it (`lesion.key`). Two
samples whose label sets give one id two keys — id 1 is `liver` in one and
`spleen` in the other — are refused rather than counted as one class, whose
weight would feed `class_weights`; `medh5 dataset check` reports the same clash
as C102. Measure each vocabulary apart, or relabel to one.

A NaN or infinite voxel (PET, resampled data) is left out of the moments and
counted in `ct.nonfinite`, so one does not turn the mean into NaN. Saved
statistics with a moment written as `null` are refused on reading, not read as
0.

### Normalisation from the training split only

`--partition train` is how you compute normalisation constants without looking
at your test set. From Python, select the partition off the **split** — it is
what knows the assignment — using the same `split` object you made above:

```python
from medh5.dataset import compute_stats

train = split.paths("train")   # a collection's member as `path::key`
stats = compute_stats(train, images=["CT"], workers=8)
mean, std = stats.normalization("CT")
weights = stats.class_weights(scheme="inverse_frequency")
```

**Do not read the partition off `entry.splits` straight after `--write-claims`.**
Writing claims amends the *sample files*; it does not update the manifest you
loaded, or the `cohort.json` on disk. `entry.splits` still holds whatever the
scan saw, so the filter matches nothing and you silently compute statistics over
an empty set. Either use `split.paths()` as above, or re-scan the directory
first:

```python
from medh5.dataset import scan

manifest, _ = scan("studies/")     # now entry.splits reflects the claims
```

If you are re-deriving the split in a later session rather than reusing the
object, pass `make_splits` **exactly** the arguments that produced it — set id,
`group_by`, `stratify_by`, ratios or folds, and seed. It is deterministic given
those, and quietly a different split without them.

Constants derived from the whole cohort leak the test set into training, which
is the kind of leak that does not show up as a bug — it shows up as a model that
scores better than it deserves to.

## Check the cohort

```
$ medh5 dataset check cohort.json --deep
412 samples: FAILED (1 errors, 2 warnings)
  ERROR   C102 class id(s) [3] mean different things in different label sets
          (3 is 'cyst' or 'lesion'); training on the union labels one class
          with another's examples
```

A class id means its key in the label set, which the manifest records for every
id a sample's annotations name (`entry.class_keys`). So two versions of one
vocabulary that agree on a class do not clash (`C101` still says the cohort
uses two), and two vocabularies published under one name and version do.

```python
from medh5.dataset import check

report = check(manifest, set_id="cv5", deep=True)
report.ok, report.errors, report.warnings
report.coverage      # {class_id: {"examined_in": n, "named_in": m, "of": total}}
```

A manifest is metadata, so coverage says which samples looked for a class
(`examined_in`, §11.3) and which have an annotation that can express it
(`named_in`) — never which hold any of it. That is a voxel question:
`compute_stats` answers it (`present_in`, `prevalence`). `C302` reports a class
named somewhere and examined nowhere, which no sample is a negative example of.

Findings carry `C1xx`–`C5xx` codes; the full table is in
[Cohort check codes](../reference/cohort-checks.md), which also says why they
are a separate space from the format's `E`/`W` codes.

`--deep` re-reads each `content_id` instead of trusting size and mtime, which
is the difference between "somebody touched this file" and "somebody changed
it".

## Audit claims without a manifest

`medh5 splits` audits claims across files directly, for when you have the
cohort but not the manifest that produced it:

```
$ medh5 splits cohort/*.medh5
```

It reports conflicting claims (`W906`) and subject leakage. See
[Curation](../reference/curation.md#split-claims).

## Related

- **[Cohort check codes](../reference/cohort-checks.md)** — every `C1xx`–`C5xx` finding.
- **[Partial labels and coverage](partial-labels.md)** — what `C301` is telling you.
- **[Curation records](../reference/curation.md#split-claims)** — the `w.split()` API.
- **[`medh5 dataset`](../reference/cli.md#medh5-dataset-index)** — every flag.
