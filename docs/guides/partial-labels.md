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
ann.has_ignore_region
ann.dense([65535])          # the mask to exclude from the loss
```

Use it for a truncated field of view, an unreadable region, a structure a rater
declined to call. Do not use it for "background": background is a positive
statement that nothing is there, and it is a signal.

## What this buys you at training time

The batch does not carry coverage — it carries `path`, `sample_id`,
`subject_id` and the patch. Look coverage up from the manifest, which holds it
as metadata and therefore costs nothing to consult:

```python
from medh5.dataset import Manifest

manifest = Manifest.load("cohort.json")
coverage = {e.path: e.annotated_class_ids for e in manifest}

for batch in loader:
    for path in batch["meta"]["path"]:
        examined = coverage[path]     # mask the loss to these class ids
```

`ManifestEntry` has the same distinction as an annotation does:
`has_class(class_id)` is what is present, `examined(class_id)` is what was looked
for.

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
