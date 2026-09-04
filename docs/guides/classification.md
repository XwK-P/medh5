# Classification and change labels

Attach a label to a whole sample, one visit, one object or one slice — and, for
a label that describes a *difference*, to the interval between two visits.

## Choose a scope

A classification says what its label is *about*. Getting the scope wrong is the
common mistake, because every scope reads the same at the call site.

| Scope | The label is about | Example |
|---|---|---|
| `sample` | the subject | RECIST response across the study |
| `timepoint` | one visit | image quality at follow-up |
| `instance` | one object | this lesion is calcified |
| `slice` | one plane | this slice is non-diagnostic |

```python
w.add_classification("quality", {"non_diagnostic": 1.0},
                     scope="timepoint", timepoints=["tp1"], scope_ids=[1])
```

**A scope needs a `scope_ids` to point at.** `timepoints` records which visits
the annotation concerns; `scope_ids` is what keys each assertion to one of them,
by timepoint index. Give only `timepoints` and the assertion comes back with
`scope_id=None`, so `assertions()` and `by_scope_id()` cannot tell you which
visit it was about — which is the entire point of a timepoint scope.

The same applies to `instance` and `slice`: the scope names the *kind* of thing,
`scope_ids` names the thing.

## Change labels span an interval

A label that describes a *difference* is a classification naming both visits:

```python
w.add_classification("response", {"progressive_disease": 1.0},
                     scope="sample", timepoints=["tp0", "tp1"],
                     schemes=["RECIST 1.1"])
```

```python
c = s.annotations["response"]
c.timepoints        # ("tp0", "tp1") — a statement about the interval
c.labels
```

Because both visits are in one file, a change label has a referent. Split
across two files it would be a claim about a filename.

`timepoints` on a change label is not "which visits this applies to" but "which
interval this describes". A label naming `["tp0", "tp1"]` is a statement about
what happened *between* them, so it is neither `tp0`'s label nor `tp1`'s, and
reading it as either is how a progression label gets attributed to the baseline
scan that predates it.

**Pass them in acquisition order.** The writer stores the sequence you give it
verbatim — it does not sort or normalise — and `TimepointPairSampler` matches a
change label to a visit pair by comparing the tuples exactly. Pass
`["tp1", "tp0"]` and the label is stored reversed, the sampler's forward pair
finds nothing, and paired training silently sees no label at all:

<!-- illustrative -->
```python
w.add_classification("response", {...}, timepoints=["tp1", "tp0"])   # reversed
# sampler pair ("tp0", "tp1") -> label None.  No error, no warning.
```

Order the pair the way the sample declares its timepoints — `s.timepoints.ids`
is that order.

## Read them back

```python
c = s.annotations["response"]
c.scope             # "sample"
c.timepoints        # ("tp0", "tp1")
c.labels            # {"progressive_disease": 1.0}
c.schemes           # ("RECIST 1.1",)
```

`labels` maps a class key to a confidence, so a one-hot label and a soft label
have the same shape and nothing has to know which it is holding.

## Related

- **[Longitudinal studies](longitudinal.md)** — the task change labels belong to.
- **[Annotation kinds](../reference/annotations.md#classification)** — the full API.
- **[Specification §9](../spec/medh5-1.0.md)** — the normative model.
