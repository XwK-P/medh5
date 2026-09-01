# De-identify and publish

Remove identifiers from a cohort, record what was done, and check the result
before it leaves your control.

**Read this first: the tool reads metadata, and does not look at voxels.**
Burned-in text, an identifiable face in a head CT, an accession number
photographed onto a film — all are outside `medh5 scrub`, and all survive it
untouched. **A file this tool calls clean may still be identifying.** Pixel
de-identification is a separate job, and this page assumes you have done it or
established that you do not need to.

## 1. Look before you change anything

`scrub` with no `--apply` finds and reports; it writes nothing.

```bash
medh5 scrub out/*.medh5
```

```python
from medh5.curation import scrub

report = scrub.scan(path)
report.actionable        # what --apply would change
report.needs_review      # what a person has to judge
print(report.format())
```

It looks for identifying DICOM keywords anywhere in `extra` or `acquisition`,
DICOM person names, real DICOM UIDs where a pseudonym belongs, unshifted dates,
and free text no rule can judge. That last category is why the read-only pass
exists: `needs_review` is the part no tool should decide for you.

The exit code is non-zero when anything is actionable, so this doubles as a
pipeline gate before a publication step:

```bash
medh5 scrub out/*.medh5 || { echo "identifiers remain"; exit 1; }
```

## 2. Apply, with a salt

```bash
medh5 scrub out/*.medh5 --apply --date-shift-days -117 --salt "$SALT" --by RAD-07
```

```python
report = scrub.apply(path, date_shift_days=-117, salt=SALT)
```

Two behaviours worth knowing before you run it:

**UIDs are pseudonymised, not deleted.** A frame UID is how two files agree they
share a frame of reference, so deleting it breaks registration.
`pseudonymise(uid, salt)` is stable, which means a cohort scrubbed file by file
— even on different machines — still joins afterwards. Only a *salted* run
records `id_mapping: external`; an unsalted hash is recoverable by anyone
holding the original UIDs, and claiming otherwise would be exactly the overclaim
this tool exists to avoid. **Keep the salt, and keep it separately from the
data.**

**Dates shift rather than vanish**, so intervals — and therefore
`days_from_baseline`, and therefore any longitudinal model — survive. Running
scrub twice does not shift them twice.

## 3. Record what was done

The record is part of the file, not a note in a README:

```python
w.deidentification(method="dicom-psi-profile",
                   profile="DICOM PS3.15 E.1 basic + clean pixel",
                   date_shift_days=-117, id_mapping="external",
                   performed_by=rad.id, burned_in_annotation_checked=True)
```

`--apply` writes one for you. Set `burned_in_annotation_checked` yourself, and
only if you actually checked — `scrub` sets it `false`, because it did not.

**A file with no de-identification record must be treated as potentially
identifying.** Absence is never evidence.

## 4. Check the cohort, not just the files

A cohort that mixes de-identified and non-de-identified samples is the failure
this catches, and no per-file check can see it:

```bash
medh5 dataset check cohort.json        # C501 if the cohort is mixed
```

## Related

- **[Curation records](../reference/curation.md#de-identification)** — the `deidentification` API.
- **[`medh5 scrub`](../reference/cli.md#medh5-scrub)** — every flag.
- **[Cohort check codes](../reference/cohort-checks.md#c501)** — what `C501` means.
