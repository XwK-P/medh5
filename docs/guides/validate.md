# Check a file before training on it

Two different questions, two different commands.

**`medh5 validate` asks whether the file conforms** to the specification — is
the geometry consistent, do the class ids exist, does the coverage make sense.

**`medh5 verify` asks whether the bytes are the bytes** — does every object still
match the digest stamped on it.

A file can pass `validate` at its default level and fail `verify`, which is
why they are separate. From `--level integrity` up they agree on the bytes:
`verify` fails a file exactly when that level reports an attestation finding
— a digest that does not match (E701), a `content_id` that does not (E702), a
digest that does not parse (E703), a path the root does not bind (E704), an
undigested clinical dataset (E818), or a dataset under a `content_id` that
carries no digest at all (W915, a warning since 2.1: §13.1 asks for the digest
but does not require it, so `strict` fails the file and `integrity` does not).

```bash
medh5 validate case.medh5
medh5 verify case.medh5
```

## Choose a level

```bash
medh5 validate case.medh5 --level integrity
```

| Level | Reads | Use for |
|---|---|---|
| `structural` | metadata + payload scans in bounded slabs | a fast "is this even a medh5 file" |
| `semantic` *(default)* | the same, plus layer data | day-to-day |
| `integrity` | every byte | after an import, or after a file has moved between machines |
| `strict` | every byte | a dataset you publish, or CI, where a warning should stop a build |

**None of them is free.** Even `structural` decompresses voxels — it reads a
float image of up to 4 M values to test int16-losslessness, and may scan a
labelmap or layers payload for an ignore region: the whole payload, in bounded
slabs, so memory is bounded and the work is not. On a 12.6 Mvox sample: 1.4 ms
for `structural`, which found nothing there to scan, 287 ms for `semantic` and
513 ms for `integrity`, against 0.13 ms for a metadata-only `open()`. Do not
put one in a hot path.

`strict` runs no extra rules — it promotes warnings to errors. So it fails a
file that conforms but is not ready to publish: one with no de-identification
record (`W903`), or whose classes carry no ontology code (`W912`), as every
freshly written or imported sample is until [de-identification](deidentify.md)
and the label set's codes have dealt with both. Check a new file at
`integrity`, and gate what you publish at `strict`. Full table and
measurements in
[Profiles and validation levels](../reference/profiles-and-levels.md).

A `strict` gate in CI should not change its verdict because medh5 was
upgraded. Each release can add warnings, in a numbered revision of the code
table, so pin the revision the gate was written against and raise it when you
choose to:

```
$ medh5 validate --level strict --codes-revision 1 case.medh5
```

A revision the table does not have (`0`, or one past the newest) is refused,
since it would pass any file.

## Read the output

```
case.medh5: FAILED [strict] profiles=core,seg (1 errors, 0 warnings)
  ERROR   E102 /grids/ct: `direction` is not orthonormal (max residual 0.4)
```

A warning `strict` promotes keeps its code and still prints as `WARNING`; the
first line counts it as an error. `FAILED [strict] … (2 errors, 0 warnings)`
above two `WARNING` lines is two promoted warnings.

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

`--profile` adds to what the file claims, which is the useful direction: hold
a file to `det` whether or not it thought to declare it, and to everything it
does declare.

## Fix what is fixable

```bash
medh5 fix case.medh5                    # diagnose, change nothing
medh5 fix case.medh5 --rebuild-index    # rebuild stale sampling indices
```

`--rebuild-index` is ordinary repair. Restamping digests is **not** repair: a
digest that no longer matches is *evidence that the bytes changed*, and
recomputing it destroys the evidence. That is why it needs a reason it can
record — see [`medh5 fix`](../reference/cli.md#medh5-fix).

A file reporting `E702` at `/` was changed outside medh5 after its
`content_id` was written; one reporting `E703` holds a digest, or declares a
`digest_algo`, nothing can check. No command amends either — a scrub, a split
claim and an added annotation each refuse it, naming the repair — because the
commit would restamp the change with no record of it. Restore the file from a
copy you trust, or re-attest it deliberately:

```bash
medh5 fix case.medh5 --rewrite-digests --reason "spacing corrected by hand, checked against the DICOM"
```

## Check the cohort too

Every file can be valid while the cohort is unusable — two label sets, or a
subject in both train and test. No per-file check can see either:

```bash
medh5 dataset check cohort.json --deep
```

## Related

- **[Diagnostic codes](../reference/diagnostic-codes.md)** — all 95 codes.
- **[Profiles and validation levels](../reference/profiles-and-levels.md)** — the two dials.
- **[Cohort check codes](../reference/cohort-checks.md)** — the `C1xx`–`C5xx` space.
