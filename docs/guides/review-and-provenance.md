# Record review, provenance and agreement

<!-- sequential -->

A mask in a file should be able to say who drew it, whether anyone checked it, and how far two
readers agree on it, so that a model trained on it, or a reader puzzled by it, can find out. This
page adds those records to an existing sample, `case.medh5`, which holds `organs` and two raters'
readings of the same organs, `organs_a` and `organs_b`. Its blocks run in order.

Every change to an existing file goes through `medh5.amend`, which writes a new version beside the
file and renames it over the old one when the block ends. Nothing is half-written if it fails.

## Record who did what

Provenance is a graph of **agents** (people, software, organisations) and **activities** (what
they did, when, to what), not a free-text comment:

```python
import medh5

with medh5.amend("case.medh5") as w:
    reviewer = w.person("RAD-11", role="reviewer")
    review = w.activity(
        "review",
        agent=reviewer,
        inputs=["annotations/organs"],
        started="2026-03-02T10:00:00Z",
        ended="2026-03-02T10:25:00Z",
    )

with medh5.open("case.medh5") as s:
    reviews = s.document.provenance.activities_by_type("review")
    assert [a.id for a in reviews] == [review.id]
    drawn_by = s.annotations["organs"].prov    # the activity that drew it
```

Activity types are a closed list: `import`, `annotate`, `review`, `predict`, `resample`,
`register`, `derive`, `deidentify`, `transcode`, `other`. An annotation's `prov` names the one
activity that produced it, so the question that comes up when a model fails on a subset (was this
mask drawn or predicted, and by what?) has an answer in the file.

## Record the review's outcome

A **quality** record belongs to one annotation: its status, who reviewed it, how confident they
were, and what they found.

```python
from medh5 import Issue

with medh5.amend("case.medh5") as w:
    w.set_quality(
        "organs",
        status="reviewed",
        reviewed_by=[reviewer.id],
        confidence=0.8,
        issues=[Issue(code="boundary", severity="warning", class_ids=[1],
                      note="inferior liver edge")],
    )

with medh5.open("case.medh5") as s:
    quality = s.document.quality["organs"]
    assert quality.status == "reviewed" and reviewer.id in quality.reviewed_by
    assert [i.code for i in quality.issues] == ["boundary"]
```

`status` runs `draft`, `submitted`, `reviewed`, `approved`, `rejected`, `deprecated`.
`quality.is_usable` says whether the record marks the data fit to train or evaluate on, which is
the question a training pipeline should ask before it reads a mask.

## Compare two raters

```python
from medh5 import compare_annotations

with medh5.open("case.medh5") as s:
    result = compare_annotations(s.annotations["organs_a"], s.annotations["organs_b"],
                                 metric="dice")

per_class = result.per_class       # keyed by class key, for reading
mean = result.value                # over the classes both examined
not_scored = result.skipped        # classes one side never examined
```

A class one rater never examined is **skipped**, not scored as zero: punishing a reader for a
class nobody asked them to draw would make every partial annotation look like a disagreement. When
nothing was comparable, `value` is `None`, never 0.

## Store the agreement

The measurement goes into the quality record of the annotation it qualifies, keyed by class id, so
a number that is quoted later can be traced to the comparison that made it.

```python
with medh5.amend("case.medh5") as w:
    w.set_quality("organs_b", status="reviewed", reviewed_by=[reviewer.id],
                  agreement=[result.to_record()])

with medh5.open("case.medh5") as s:
    stored = s.document.quality["organs_b"].agreement     # one record per comparison
    assert [a.metric for a in stored] == ["dice"]
```

`to_record()` refuses a comparison with nothing comparable rather than store a 0 that would say
two raters disagreed when nobody measured. From the shell, the same measurement and record:

```bash
medh5 agree case.medh5 organs_a organs_b --metric dice --record
medh5 prov case.medh5
```

## Related

- [Curation records](../reference/curation.md): every field of provenance, quality and agreement,
  and object-level agreement between instances or boxes.
- [Partial labels and coverage](partial-labels.md): why a class nobody examined is not a negative.
- [Python API › Amending](../reference/python-api.md#amending): what `amend` does to the file.
