# Partial labels and coverage

Almost no real cohort is annotated everywhere for everything. This is how to
record what you actually looked for, so a model learns from your data and not
from your annotation schedule.

**The one distinction to hold on to.** `class_ids` is what an annotation
*contains*. `annotated_class_ids` is what was *looked for*. A class examined and
found absent is a usable negative example; a class nobody examined is not, and
collapsing the two is how a model learns that a site's scans have no spleens.

```python
ann.class_ids            # what is here
ann.annotated_class_ids  # what was examined
ann.is_annotated("spleen")
ann.is_fully_covered     # nothing appears that was not examined
```

## Which coverage claim do you want?

`annotated_classes` on the writer takes three forms, and the default is the
conservative one.

| Form | Claims | Use when |
|---|---|---|
| `"all_given"` *(default)* | only the classes you passed masks for | you annotated exactly these and looked at nothing else |
| an explicit list | exactly the classes you name, present or not | you searched for a set and some were absent |
| `"all"` | every class in the label set | a complete annotation pass over the whole vocabulary |

The explicit list is the one most cohorts need and the one most often skipped:

```python
w.add_segmentation("organs", grid="ct",
                   masks={"liver": liver, "lesion": lesion},
                   annotated_classes=["liver", "spleen", "lesion"])
```

That names the spleen although there is no spleen mask, which records "we looked
and found none". Leave it out and the default records only liver and lesion; the
spleen becomes a class nobody examined, and every downstream consumer is right to
ignore it.

## "I only annotated liver in half the cohort"

Write what is true per file. The files with liver annotations claim liver; the
files without do not claim it at all — they must not claim it and score zero.

Then let the cohort check find the asymmetry:

```bash
medh5 dataset check cohort.json      # C301: a class examined in only part of the cohort
```

`C301` is not an error to suppress. It is the fact you need before choosing a
loss: a class examined in 50 % of the cohort needs masking, or a per-sample
weight, or a decision to train it separately — but it needs to be a decision.

Class prevalence accounts for this already:

```python
stats = compute_stats(paths, images=["CT"])
lesion = stats.classes[3]
lesion.present_in, lesion.examined_in, lesion.prevalence
```

`prevalence` is over the samples that actually examined the class, so a class
annotated in a tenth of the cohort does not look ten times rarer than it is.

## "Some voxels I cannot label either way"

That is an **ignore** region — class `65535` — and it is different from
background. It marks voxels that must not contribute to a loss in *either*
direction:

```python
ann.has_ignore_region       # is there one at all?
```

**`dense([65535])` is refused.** The ignore id is not an ordinary class, so no
encoding can return a plane for it — `dense` raises `E404` rather than handing
back an all-zero mask indistinguishable from a class examined and found absent.
It used to return that mask, and the ignored voxels went into the loss.

**Where the region lives depends on the encoding**, so read it through both
routes. `labelmap` and `layers` carry it in band and expose `ignore_mask()`;
`bitmask` and `probmap` cannot represent a reserved id in band, so theirs is a
separate `mask` annotation named by `header.ignore_mask`. Ask a `bitmask` for
`ignore_mask()` and you get `AttributeError`, with `has_ignore_region` already
True:

```python
def ignore_region(sample, ann, roi=None):
    """The ignore region, wherever this encoding keeps it."""
    referenced = ann.header.ignore_mask
    if referenced:                                 # a separate mask annotation
        return sample.annotations[referenced].dense(roi=roi)
    reader = getattr(ann, "ignore_mask", None)     # in band: labelmap, layers
    return reader(roi=roi) if reader else None
```

Verified across all three: the same 64 ignored voxels from `labelmap`, `layers`
and `bitmask`.

Use it for a truncated field of view, an unreadable region, a structure a rater
declined to call. Do not use it for "background": background is a positive
statement that nothing is there, and it is a signal.

## What this buys you at training time

The batch does not carry coverage — it carries `path`, `sample_id`,
`subject_id` and the patch. Look coverage up from the manifest, which holds it
as metadata and therefore costs nothing to consult:

```python
from medh5.dataset import Manifest

ANN = "organs"          # the annotation the loader is reading

manifest = Manifest.load("cohort.json")
coverage = {
    e.path: set(e.annotations[ANN]["annotated_classes"])
    for e in manifest
    if ANN in e.annotations
}

for batch in loader:
    for path in batch["meta"]["path"]:
        examined = coverage[path]     # mask the loss to these class ids
```

**Take the coverage from the annotation you are training on, not from the
entry.** `Entry.annotated_class_ids` is the *union* across every annotation in
the sample, which is the right answer for a cohort-level question and the wrong
one for a loss mask. A sample whose `organs` examined liver and spleen and whose
`vessels` examined vessel reports `(1, 2, 4)` at entry level:

```python
e.annotated_class_ids                            # (1, 2, 4)
e.annotations["organs"]["annotated_classes"]     # [1, 2]
e.annotations["vessels"]["annotated_classes"]    # [4]
```

Mask with the union and class 4 becomes a negative for a model reading `organs`,
which never looked for it — the exact substitution this page exists to prevent.
`e.annotations[name]["timepoints"]` narrows it further when a sample carries
per-visit annotations.

`ManifestEntry`'s own pair — `has_class(class_id)` for what is present,
`examined(class_id)` for what was looked for — draws the same distinction as an
annotation does, at sample scope.

Straight off a sample, when you are not working from a manifest:

```python
ann = s.annotations["organs"]
ann.annotated_class_ids           # (1, 2)
ann.is_annotated("spleen")        # True  — examined, and absent
ann.is_annotated("lesion")        # False — nobody looked
```

The point of all of it: a per-sample loss mask that is *correct* rather than
assumed, so a partially annotated cohort trains as a partially annotated cohort.

## Related

- **[Segmentation](segmentation.md)** — where `annotated_classes` is passed.
- **[Annotation kinds](../reference/annotations.md#coverage)** — the coverage API.
- **[Cohort check codes](../reference/cohort-checks.md#c301)** — `C301` and `C302`.
- **[The data model](../explanation/data-model.md)** — why absence is not silence.
