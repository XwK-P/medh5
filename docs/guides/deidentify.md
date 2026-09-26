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

**What it examines is the whole file, not a list of places.** Every string in
`/meta`, every object name, every HDF5 attribute and every string dataset —
including objects and attributes written by a newer version, which `amend`
carries through without understanding them. JSON stored inside a string (an
attribute holding a serialized mapping, a box's `attributes` column) is decoded
and examined as a mapping. The few fields it skips are skipped by name, with the
reason in the code: the writer's own timestamp and digests, which every commit
rewrites; provenance `started` and `ended`, which say when a tool ran rather
than when the subject was seen; and closed vocabularies such as `sex`.

The rules it applies everywhere: identifying DICOM keywords as mapping keys *or*
attribute names (`PatientName` stored as an HDF5 attribute is caught like one in
`extra`), DICOM person names in any script (`Müller^Hans`, `山田^太郎`), real
DICOM UIDs where a pseudonym belongs — including inside provenance references
such as `dicom:<SeriesInstanceUID>` — filesystem paths in provenance, unshifted
dates, ages over 89, the names of person and organization agents, and free text
no rule can judge. That last category is why the read-only pass exists:
`needs_review` is the part no tool should decide for you.

Some findings are reported and never acted on, at any profile, because
rewriting them would break something that refers to them: ids (object names,
timepoint and activity ids, a split's `set_id`) and the label vocabulary, which
is shared across the cohort and pinned by its digest. Their detail says so.

The exit code is non-zero when the scan found **anything at all** — not only
what `--apply` could fix — so this works as a pipeline gate before a publication
step:

```bash
medh5 scrub out/*.medh5 || { echo "review before publishing"; exit 1; }
```

That includes `needs_review` findings, which no `--apply` run will clear: free
text a rule cannot judge stays a finding until the metadata itself changes.

**There is no approval state.** A report records findings, not decisions about
them, so a benign free-text field keeps this gate red on every future scan no
matter who has looked at it. Only editing or removing the flagged metadata makes
it green.

If you want a gate that fails on what can be fixed and merely reports the rest,
build it on `actionable` rather than on the exit code:

```python
from medh5.curation import scrub

reports = [scrub.scan(p) for p in paths]
blocking = [r for r in reports if r.actionable]
for r in reports:
    for f in r.needs_review:
        print(f"review: {r.path}: {f}")
raise SystemExit(1 if blocking else 0)
```

## 2. Apply, with a salt

```bash
medh5 scrub out/*.medh5 --apply --date-shift-days -117 --salt "$SALT" --by RAD-07
```

```python
report = scrub.apply(path, date_shift_days=-117, salt=SALT)
```

Three behaviours worth knowing before you run it:

**It re-scans its own output.** After amending, `--apply` runs the same rules
over the file it just wrote, records `remaining` and `remaining_actionable` in
the de-identification activity's parameters, and **exits non-zero if anything
actionable is left** — and, under `--profile strict`, while the sample's own ids
are identifiers (below). The attestation and the exit code therefore agree with a
re-scan by construction, rather than reporting what the tool set out to do. In
Python, `report.remaining` holds those findings and `report.ok` is the same
answer as the exit code.

`--apply` acts on everything the scan calls actionable, which under
`--profile strict` includes a person or organization agent's name (replaced by
a stable pseudonym, so the graph still says two activities were the same
person), an age over 90 (recorded as 90, the Safe Harbor "90 or older"
category), and flagged free text (removed). Provenance paths keep only their
file name, and under `strict` lose that too.

**The sample's own ids are reported with where they came from.** The DICOM
importer keys a sample by `PatientID` — usually the medical record number — and
records that in `identity.id_source`, so the scan reports
`identity.sample_id` and `identity.subject_id` as copied from DICOM rather than
guessing from their shape. `--apply` does not rewrite them by default, because
every manifest, split claim and cross-file join names the sample by them; under
`--profile strict` it **exits non-zero while they are open**, since a file whose
own name for the subject is the record number is not de-identified. Two ways to
close them:

```bash
medh5 scrub out/*.medh5 --apply --profile strict --salt "$SALT" --pseudonymise-ids
```

```python
report = scrub.apply(path, profile="strict", salt=SALT, pseudonymise_ids=True)
for finding in report.open_identity:
    print(finding)      # e.g. the file is still named after the old sample id
```

`--pseudonymise-ids` replaces both ids — and a `cohort.group_id` equal to
either — with salted stable pseudonyms, so every file of one subject still
agrees, and it records the change in `id_source`. It **requires** `--salt`: an
unsalted hash of a record number is reversed by hashing every record number.
Or re-mint the ids yourself with `w.identity(sample_id=..., subject_id=...)`;
the writer drops the recorded DICOM source of an id that changes. Either way,
`scrub` cannot rename files: a file still named after the old id is reported as
`file_name`, and renaming it is yours to do.

**UIDs are pseudonymised, not deleted.** A frame UID is how two files agree they
share a frame of reference, so deleting it breaks registration.
`pseudonymise(uid, salt)` is stable, which means a cohort scrubbed file by file
— even on different machines — still joins afterwards. Only a *salted* run
records `id_mapping: external`; an unsalted hash is recoverable by anyone
holding the original UIDs, and claiming otherwise would be exactly the overclaim
this tool exists to avoid. **Keep the salt, and keep it separately from the
data.** The same goes for `report.uid_map` (`uid_map` in `--json` output): it
maps every original value the run replaced — UIDs, and the ids under
`--pseudonymise-ids` — to its pseudonym, so it is identifying.

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

## 4. Re-index, then check the cohort

A cohort that mixes de-identified and non-de-identified samples is the failure
this catches, and no per-file check can see it. **Re-index first**, or it will
not see it either:

```bash
medh5 dataset index out/ -o cohort.json    # scrub changed the files; re-read them
medh5 dataset check cohort.json            # C501 if the cohort is mixed
```

`dataset check` reads the manifest, not the samples: `deidentified` is recorded
when the directory is indexed, and `--apply` amends the files without touching
`cohort.json`. Check against a manifest built before the scrub and every entry
still says "not de-identified", so a *partial* pass looks uniform and `C501` never
fires. Measured on three samples with two scrubbed:

| | |
|---|---|
| stale manifest | `C401` — a file changed since the scan |
| after re-indexing | **`C501`** — the cohort is partly de-identified |

`C401` is the only hint you get, and it says the files moved, not that half of
them are still identifying.

## Related

- **[Curation records](../reference/curation.md#de-identification)** — the `deidentification` API.
- **[`medh5 scrub`](../reference/cli.md#medh5-scrub)** — every flag.
- **[Cohort check codes](../reference/cohort-checks.md#c501)** — what `C501` means.
