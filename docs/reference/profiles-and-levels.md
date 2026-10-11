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
| `structural` | layout, required attributes, dtypes, shapes, identifier syntax, the [JSON Schema](schema.md) | metadata, plus payload scans in bounded slabs |
| `semantic` *(default)* | cross-references resolve, geometry consistency, class ids in the label set, encoding invariants, profile requirements | the same, plus layer data |
| `integrity` | per-object digests, `content_id`, sampling-index `source_digest` currency, and the `probmap` value checks (values in [0, 1], `normalized` sums) made in the same full read | every byte |
| `strict` | the same rules as `integrity`, with warnings promoted to errors | every byte |

**None of the levels is free.** Even `structural` decompresses voxels: it reads
an image to decide whether a float array would be lossless as `int16` (only an
image of at most 4 M values), and scans a labelmap or layers payload to find an
in-band ignore region where the answer decides a finding --- partial coverage
(W904), or a `uint16` labelmap whose ids would fit `uint8` (E411). `semantic`
additionally reads layer data to judge encoding optimality. Both scans read the
whole payload, in slabs of at most 8 MiB: the slabs bound the memory a large
volume takes, not the work, which grows with its size.

Measured on a 12.6 Mvox, 25.2 MB sample (`medh5.bench.synthetic_sample()`: an
`int16` CT and eight classes in `layers`) — against a metadata-only `open()`
of 0.13 ms:

| | |
|---|---|
| `structural` | 1.4 ms |
| `semantic` | 287 ms |
| `integrity` | 513 ms |
| `strict` | 505 ms |

`structural` found nothing to scan on this sample --- its image is `int16` and
its annotation's coverage complete --- so its time is the metadata's; a float
image, or a labelmap or layers payload with partial coverage, adds a scan.
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
| `reg` | at least one transform whose `from_frame` and `to_frame` are each the `frame_uid` of one of the sample's grids |
| `curation` | a provenance graph recording at least one activity, and `quality` on every annotation |
| `multiscale` | the §4.3 pyramid layout on every image |
| `training` | a sampling index with at least one entry; a stale entry is `W905`, not a missing profile |
| `longitudinal` | at least two declared timepoints, each grid bound to one, and stable instance ids for objects seen at more than one visit (§7.4) |
| `clinical` | **format 1.1**: the `clinical/` group --- a canonical descriptor with the subject clock, at least one event, and the documents and links the events use ([1.1](../spec/medh5-1.1.md) §3–§7) |

`--profile` holds the file to a profile **besides** those it claims, which is the
useful direction: a tool can require `det` and get a diagnostic whether or not
the file thought to claim it. A requirement never waives a profile the file
declares, and a name that is not a profile is refused.

A profile is coarser than a kind, though. `det` is satisfied by any annotation
whose task is `detection` — oriented boxes, keypoints and points as well as
boxes; contours and meshes default to `segmentation` — so requiring it does not
guarantee the annotation your code is about to read. Check the kind you need in
your own code as well.

A declared profile whose requirement is missing is `E009`: no voxel annotation
other than a `mask` under `seg`, no `detection` task under `det`, no
classification under `cls`, no transform whose `from_frame` and `to_frame` are
each a grid's `frame_uid` under `reg`, an image that is not a §4.3 pyramid
under `multiscale`, no provenance activity or an annotation without `quality`
under `curation`, no sampling-index entry under `training`, and fewer than two
timepoints under `longitudinal` (the label set `seg`, `det` and `cls` need is
`E301`). Not every requirement can be checked that way:

- **`training`** — an index without an entry is `E009`. A stale entry is
  `W905`, not a profile violation: readers ignore it and fall back to scanning,
  so the file stays readable and the index needs a rebuild
  (`medh5 fix --rebuild-index`). Under `strict` it fails like an error, which
  is why `strict` is the level for a readiness check.
- **`longitudinal`** — whether one physical object kept its `instance_id` across
  visits is not something a validator can see; that is the curator's to get
  right. What it can see is a symptom: `W909` fires when one `instance_id`
  carries two different class ids. Alongside it, `W910` reports grids at
  different visits sharing a `frame_uid`, and `W911` a multi-timepoint sample
  with no transform relating any two visits.

- **`clinical`** — declaring it in a 1.0 file, or declaring it without the
  group or without events, is `E009`; the group without the declaration is
  `E803`. The profile's own rules are the `E8xx` codes, and a numeric value
  without a unit is `W914`. A file of a later minor than this package
  implements is validated as a *projection*: what is known is checked, what is
  not is `W913`.

`w.infer_profiles()` returns the profiles the content written so far
satisfies, together with any declared; commit writes that set, so a writer
rarely declares them by hand. Commit writes the version from it too: a new
sample is 1.0 unless it carries `clinical`.

## Related

- [Diagnostic codes](diagnostic-codes.md) — what a failure at any level reports.
- [Check a file before training on it](../guides/validate.md) — choosing a level for a job.
- [`medh5 validate`](cli.md#medh5-validate) — every flag.
