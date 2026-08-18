# Cohorts

Everything above the single file. A sample is self-describing, but a *cohort*
has properties no file can carry: which label set everyone agrees on, which
subject is in which partition, what the intensity distribution is, whether a
class was examined everywhere or only where somebody got around to it.

`medh5.dataset` computes those from metadata alone, so the answers cost
milliseconds rather than a pass over every voxel in the study.

## Manifests

```
$ medh5 dataset index studies/ -o cohort.json
studies/: 412 sample(s), 289 subject(s), sha256 4f21c8ab90e1
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

### The digest

```python
manifest.sha256()
```

Covers the cohort's **membership** — which samples, grouped how — and nothing
else. Not paths, sizes, mtimes, the generating version, and deliberately not
`content_id`.

Membership and grouping are exactly what a split is computed from, which makes
this the right thing for a claim to be checkable against. Including content
would make it useless for that: writing a claim into a file changes the file,
so every claim would be stale the moment it was written. Content drift is a
different question, and `dataset check` answers it separately (`C401`, and
`--deep`).

Re-scanning the same cohort on another machine produces the same digest.
Filtering the manifest changes it, so a split made from a subset cannot be
mistaken for one made from the whole.

This is the link the spec's `SplitClaim.manifest_sha256` was designed around.

## Splits

```
$ medh5 dataset split cohort.json --group-by group_id --stratify-by site_id \
      --ratios train=0.7,val=0.15,test=0.15 --seed 0 -o split.json --write-claims
```

```python
from medh5.dataset import make_splits, write_claims

split = make_splits(manifest, set_id="cv5", group_by="group_id",
                    stratify_by="site_id", ratios={"train": .7, "val": .15, "test": .15},
                    seed=0)
split.counts            # {"test": 62, "train": 288, "val": 62}
split.balance()         # achieved stratum counts per partition
split.underfilled       # partitions that were asked for and got nothing
split.leaks()           # groups in more than one partition — structurally ()
split.paths("train")

write_claims(split, manifest, assigned_by="pipeline@v3")
```

Three rules, and they are the whole module.

**Group before you split.** The unit is `cohort.group_id`, which defaults to
the subject — never a file. A subject with a baseline and two follow-ups is one
unit. Since a sample is already a subject, this is mostly free; `group_id`
exists for the coarser cases (a family, an enrolling site).

**Stratify on what you can see.** Groups are indivisible and a group's stratum
is its majority, so exact balance is not always reachable. `balance()` reports
what was actually achieved rather than what was asked for.

**A claim in a file is not the split.** `write_claims` stamps each sample with
its partition *and* the manifest digest it came from. A later reader can then
tell a current claim from one that predates a re-split. Writing a claim for a
`set_id` replaces any earlier claim for that set.

Assignment is deterministic given `(manifest digest, seed, parameters)` — a
hash, not a shuffle — so two machines produce the same partitions.

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
every other fold becomes `train`.

## Statistics

```
$ medh5 dataset stats cohort.json --partition train --set-id cv5 --workers 8 -o stats.json
```

```python
from medh5.dataset import compute_stats

stats = compute_stats(paths, images=["CT"], workers=8)
stats.normalization("CT")          # (mean, std) for a z-score
stats.class_weights(scheme="inverse_frequency")
stats.images["CT"].mean, .std, .minimum, .maximum
stats.classes[3].voxels, .present_in, .examined_in, .prevalence
stats.failures
```

Streaming: one pass per file, constant memory per worker, and an **exact**
Welford merge (Chan–Golub–LeVeque), not an approximation.

Two things it deliberately does not do.

**It does not average per-file means.** That weights a 40-slice scan the same
as a 900-slice one. The merge weights by voxel count.

**It does not treat an unexamined class as a zero.** `prevalence` is over the
samples that actually examined the class, so a class annotated in a tenth of
the cohort does not look ten times rarer than it is.

Class counts come from the §14.3 sampling index when it is current — a few
hundred bytes per annotation instead of decompressing every mask. A stale index
is not trusted.

`--partition train` is how you compute normalisation constants without looking
at your test set.

## Cross-file checks

```
$ medh5 dataset check cohort.json --deep
412 samples: FAILED (1 errors, 2 warnings)
  ERROR   C102 class id(s) [3] appear in more than one label set; check they
          mean the same thing before training on the union
```

```python
from medh5.dataset import check

report = check(manifest, set_id="cv5", deep=True)
report.ok, report.errors, report.warnings
report.coverage      # {class_id: {"examined_in": n, "present_in": m, "of": total}}
```

| Code | |
|---|---|
| `C101` | the cohort uses more than one label set |
| `C102` | a class id means different things in different label sets |
| `C103` | a sample declares no label set |
| `C201` | a split claim's manifest digest is not this manifest's |
| `C202` | one subject's samples claim different partitions of one split |
| `C203` | a sample carries no split claim |
| `C301` | a class is examined in only part of the cohort |
| `C302` | a class appears in no sample |
| `C401` | a file changed after the manifest was written |
| `C402` | a file in the manifest no longer exists |
| `C501` | the cohort mixes de-identified and non-de-identified samples |

These are **cohort** codes, deliberately distinct from the format's `E`/`W`
table: a file is not non-conforming because the cohort around it is
inconsistent.

`--deep` re-reads each `content_id` instead of trusting size and mtime, which
is the difference between "somebody touched this file" and "somebody changed
it".

## Split claims in files, without a manifest

`medh5 splits` audits claims across files directly, for when you have the
cohort but not the manifest that produced it:

```
$ medh5 splits cohort/*.medh5
```

It reports conflicting claims (`W906`) and subject leakage. See
[Curation](curation.md#split-claims).
