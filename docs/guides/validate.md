# Check a file before training on it

Two different questions, two different commands.

**`medh5 validate` asks whether the file conforms** to the specification — is
the geometry consistent, do the class ids exist, does the coverage make sense.

**`medh5 verify` asks whether the bytes are the bytes** — does every object still
match the digest stamped on it.

A file can pass one and fail the other, which is why they are separate.

```bash
medh5 validate case.medh5
medh5 verify case.medh5
```

## Choose a level

```bash
medh5 validate case.medh5 --level strict
```

| Level | Reads | Use for |
|---|---|---|
| `structural` | metadata + bounded payload scans | a fast "is this even a medh5 file" |
| `semantic` *(default)* | the same, plus layer data | day-to-day |
| `integrity` | every byte | after a file has moved between machines |
| `strict` | every byte | CI, where a warning should stop a build |

**None of them is free.** Even `structural` decompresses voxels — it reads an
image to test int16-losslessness and scans annotation payloads for an ignore
region, both capped, but neither is metadata. On a 12.6 Mvox sample: 62 ms for
`structural` and `semantic`, 143 ms for `integrity`, against 0.5 ms for a
metadata-only `open()`. Do not put one in a hot path.

`strict` runs no extra rules — it promotes warnings to errors. Full table and
measurements in
[Profiles and validation levels](../reference/profiles-and-levels.md).

## Read the output

```
case.medh5: FAILED [strict] profiles=core,seg (1 errors, 0 warnings)
  ERROR   E102 grids/ct: `direction` is not orthonormal to 1e-4
```

Every diagnostic carries a stable code. Look it up in
[Diagnostic codes](../reference/diagnostic-codes.md) — a code's meaning never
changes and codes are never reused, so it is safe to branch on one in a
pipeline.

A validation pass never raises on a bad file; it reports everything wrong at
once. A validator that stopped at the first problem would turn one review cycle
into ten.

## Require what you actually need

```bash
medh5 validate case.medh5 --profile seg --profile det
```

`--profile` overrides what the file claims, which is the useful direction: hold
a file to `det` whether or not it thought to declare it.

## Fix what is fixable

```bash
medh5 fix case.medh5                    # diagnose, change nothing
medh5 fix case.medh5 --rebuild-index    # rebuild stale sampling indices
```

`--rebuild-index` is ordinary repair. Restamping digests is **not** repair: a
digest that no longer matches is *evidence that the bytes changed*, and
recomputing it destroys the evidence. That is why it needs a reason it can
record — see [`medh5 fix`](../reference/cli.md#medh5-fix).

## Check the cohort too

Every file can be valid while the cohort is unusable — two label sets, or a
subject in both train and test. No per-file check can see either:

```bash
medh5 dataset check cohort.json --deep
```

## Related

- **[Diagnostic codes](../reference/diagnostic-codes.md)** — all 71.
- **[Profiles and validation levels](../reference/profiles-and-levels.md)** — the two dials.
- **[Cohort check codes](../reference/cohort-checks.md)** — the `C1xx` space.
