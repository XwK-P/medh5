# Profiles and validation levels

Two independent dials on `medh5 validate`: **how much** to check, and **what to
hold the file to**.

```bash
medh5 validate case.medh5 --level strict --profile seg --profile det
```

## Levels — how much to check

Each level includes the ones before it, so `strict` runs everything.

| Level | Checks | Reads |
|---|---|---|
| `structural` | layout, required attributes, dtypes, shapes, identifier syntax, the [JSON Schema](schema.md) | metadata, plus bounded payload scans |
| `semantic` *(default)* | cross-references resolve, geometry consistency, class ids in the label set, encoding invariants, profile requirements | the same, plus layer data |
| `integrity` | per-object digests, `content_id`, sampling-index `source_digest` currency | every byte |
| `strict` | the same rules as `integrity`, with warnings promoted to errors | every byte |

**None of the levels is free.** Even `structural` decompresses voxels: it reads
an image to decide whether a float array would be lossless as `int16` (capped at
4 M values), and scans a labelmap or layers payload to find an in-band ignore
region (capped at 64 M). `semantic` additionally reads layer data to judge
encoding optimality, under the same 64 M cap. The caps bound the work on a large
volume; they do not make it metadata.

Measured on a 12.6 Mvox, 18.7 MB sample — against a metadata-only `open()` of
0.5 ms:

| | |
|---|---|
| `structural` | 62 ms |
| `semantic` | 62 ms |
| `integrity` | 143 ms |

So `semantic` is the right default and `integrity` is what to run after a file
has moved between machines — but neither belongs in a hot path, and a
per-iteration `validate` will cost you.

`strict` runs **no additional rules**. It changes what counts as failure: every
`W9xx` is reported as an error, and the promotion reaches the counts, not just
the verdict — a `strict` report never says `FAILED (0 errors, 2 warnings)`,
because that contradicts itself and a CI job gating on `errors == 0` would pass
a file the same payload called not-ok. Each diagnostic keeps its measured
severity, so [the table](diagnostic-codes.md) still tells you which were
warnings.

Use `strict` in CI, where a stale sampling index or an unbound class id should
stop a build; use `semantic` interactively, where the same warnings are
information.

A validation pass never raises on a bad file — it reports. Curation needs to see
everything wrong with a file at once, and a validator that stops at the first
problem turns one review cycle into ten.

## Profiles — what to hold the file to

A file declares which profiles it satisfies, and the validator can hold it to
them:

```python
s.profiles   # {"core", "seg", "det", "curation", "longitudinal"}
```

| Profile | Requires (spec §1.3) |
|---|---|
| `core` | the container, geometry and timepoints, at least one image, integrity — always required |
| `seg` | a label set and at least one voxel annotation (a bare `mask` does not count) |
| `det` | a label set and at least one annotation whose `task` is `detection` |
| `cls` | a label set and at least one classification annotation |
| `reg` | at least one transform |
| `curation` | a provenance graph, and `quality` on every annotation |
| `multiscale` | the §4.3 pyramid layout on every image |
| `training` | a sampling index |
| `longitudinal` | at least two declared timepoints, each grid bound to one |

`--profile` **overrides** what the file claims, which is the useful direction: a
tool can require `det` and get a diagnostic whether or not the file thought to
claim it.

A profile is coarser than a kind, though. `det` is satisfied by any annotation
whose task is `detection` — oriented boxes, keypoints and points as well as
boxes; contours and meshes default to `segmentation` — so requiring it does not
guarantee the annotation your code is about to read. Check the kind you need in
your own code as well.

A declared profile whose requirement is missing is `E009`. A stale sampling
index does not break `training` — it is `W905`, a cache to rebuild — and
`W909`–`W911` report the longitudinal properties a validator can only warn
about: stable instance ids, distinct frames per visit, a transform relating the
visits.

`w.infer_profiles()` sets them from what was actually written, so a writer
rarely declares them by hand.

## Related

- [Diagnostic codes](diagnostic-codes.md) — what a failure at any level reports.
- [Check a file before training on it](../guides/validate.md) — choosing a level for a job.
- [`medh5 validate`](cli.md#medh5-validate) — every flag.
