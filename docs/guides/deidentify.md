# De-identify and publish

Remove identifiers from a cohort, record what was done, and check the result
before it leaves your control.

**Read this first: the tool reads metadata, and does not look at voxels.**
Burned-in text, an identifiable face in a head CT, an accession number
photographed onto a film — all are outside `medh5 scrub`, and all survive it
untouched. **A file this tool calls clean may still be identifying.** Pixel
de-identification is a separate job, and this page assumes you have done it or
established that you do not need to.

**Nor does it de-identify the clinical profile** (format 1.1). Reports, the
text of every event, coded values and the subject clock can carry anything
an identifier can, and a date shift applied to `/meta` would not move the
clinical clock — images and history would tell two times (1.1 §3, §6). So a
scan of a sample that carries the profile reports it as **not examined** (it is
never clean), and `--apply` refuses the file rather than attest a
de-identification it did not do. Share the imaging projection instead —
`medh5 clinical strip case.medh5 --out case.imaging.medh5` writes the sample
without its clinical records, as a different file — or de-identify the
records with a tool that covers free text and dates, and record that.

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
`/meta`, the de-identification record included; the name of every link — a
second hard link to an object, a soft link and the path it holds, a committed
datatype; every HDF5 attribute, on groups, datasets and committed datatypes
alike; and every string dataset — including objects and attributes written by a
newer version, which `amend` carries through without understanding them. JSON
stored inside a string (an attribute holding a serialized mapping, a box's
`attributes` column) is decoded and examined as a mapping. The few fields it
skips are skipped by name, with the reason in the code: the writer's own
timestamp and digests, which every commit rewrites; provenance `started` and
`ended`, which say when a tool ran rather than when the subject was seen; and
closed vocabularies such as `sex`.

**What it does not read, it reports.** Strings inside a compound, array or
variable-length type are not decoded, so a dataset whose type holds them, at any
depth of nesting, is reported as `unreadable`. Numbers are not read as text. In
the imaging groups (`images`, `annotations`, `grids`, `transforms`, `index`)
they are what the format puts there; anywhere else, every non-string dataset is
named in the report's `not_checked`, and one that can carry text — a byte
buffer, or a type the tool cannot decode — is reported as `unreadable` too.

The rules it applies everywhere: identifying DICOM keywords as mapping keys *or*
attribute names, at any level (`PatientName` stored as an HDF5 attribute is
caught like one in `extra`, and `extra.PatientID` like `extra.dicom.PatientID`),
DICOM person names in any script (`Müller^Hans`, `山田^太郎`), real DICOM UIDs
where a pseudonym belongs — alone, inside a provenance reference such as
`dicom:<SeriesInstanceUID>`, or inside any longer string — filesystem paths in
provenance, unshifted dates and date-times (ISO 8601 or DICOM, alone or inside
a longer string such as `CT_2020-01-31.nii`; a value that is wholly a date is
one under any key, `RadiopharmaceuticalStartDateTime` as surely as
`StudyDate`), ages over 90, the names of person and organization agents, a copy
of the sample document outside `/meta` or `/meta` itself under a second name
(either keeps every identifier `--apply` replaces in the document, so `--apply`
removes it), and free text no rule can judge. That last category is why the
read-only pass exists: `needs_review` is the part no tool should decide for
you.

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

If you want a gate that fails on what still has to be done and merely reports
the rest, build it on the report rather than on the exit code — but not on
`actionable` alone, which passes three things no `--apply` clears and no
publication should: the sample's own ids while they are identifiers
(`open_identity`: `--apply` never rewrites them by itself), a sample carrying
the clinical profile (rule `clinical`: not examined, and refused by `--apply`),
and whatever the scan could not read (`unreadable`, `too_deep`). Block on all
of them:

```python
from medh5.curation import scrub

NOT_EXAMINED = {"clinical", "unreadable", "too_deep"}

reports = [scrub.scan(p) for p in paths]
blocking = [
    r
    for r in reports
    if r.actionable
    or r.open_identity
    or any(f.rule in NOT_EXAMINED for f in r.findings)
]
for r in reports:
    for f in r.needs_review:
        print(f"review: {r.path}: {f}")
raise SystemExit(1 if blocking else 0)
```

What is reported as not examined stays blocking until a person has dealt with
it: removed it, or decided outside this gate that it can go.

## 2. Apply, with a salt

```bash
medh5 scrub out/*.medh5 --apply --date-shift-days -117 --salt "$SALT" --by RAD-07
```

```python
report = scrub.apply(path, date_shift_days=-117, salt=SALT)
```

`--date-shift-days`, `--salt` and `--by` act only with `--apply`, and without
it they are refused rather than ignored: a scan meant as a salted, shifted
scrub would otherwise exit as a plain scan. `scrub.scrub_paths` refuses its
apply options without `apply_changes=True` the same way. Like the command line,
it processes every path even when one is refused — a clinical sample always is
— and then raises the first refusal, carrying every path's report in order as
`reports` (`None` for a refused path), so the `uid_map` of a file already
rewritten is never lost.

`--apply` writes the scrubbed version beside the file and renames it over the
original. Through a symbolic link, the file the link names is rewritten and the
link kept. A file with several hard links is replaced at the path given only,
and its other names keep the original, identifiers and all: remove them. A link
in a directory anyone can write to, such as `/tmp`, that neither you nor the
directory's owner owns is not followed: the file is refused (`MEDH5FileError`),
as the system would refuse to open it through the link.

Four behaviours worth knowing before you run it:

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
person — salted with `--salt` and recorded in `uid_map` like every other
pseudonym, since an unsalted hash of a name is reversed by hashing candidate
names), an age over 90 (recorded as 90: Safe Harbor aggregates every age over
89 into one "90 or older" category, so 90 itself is the category and stays),
and flagged free text (removed). Provenance paths keep only their
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
The old id is replaced wherever else a string holds it as a word of its own —
a timepoint label, a note in `extra`, an attribute — since there it would
outlive its pseudonym, and a later scan cannot know it was an id. An object
name holding it is never renamed, so it is reported as an open identity finding
by the run that knows to look. An id shorter than four characters is replaced
in `identity` and not looked for elsewhere: `s1` is in too many words.
Or re-mint the ids yourself with `w.identity(sample_id=..., subject_id=...)`;
the writer drops the recorded DICOM source of an id that changes. Either way,
`scrub` cannot rename files: a file still named after the old id is reported as
`file_name`, and renaming it is yours to do.

**UIDs are pseudonymised, not deleted.** A frame UID is how two files agree they
share a frame of reference, so deleting it breaks registration. A UID inside a
longer string — `derived from 1.2.840…` — is replaced by the same pseudonym it
gets everywhere else. `pseudonymise(uid, salt)` is stable, which means a cohort
scrubbed file by file — even on different machines — still joins afterwards.
Only a *salted* run records `id_mapping: external`; an unsalted hash is
recoverable by anyone holding the original UIDs, and claiming otherwise would
be exactly the overclaim this tool exists to avoid. **Keep the salt, and keep
it separately from the data.** The same goes for `report.uid_map` (`uid_map` in
`--json` output): it maps every original value the run replaced — UIDs, the ids
under `--pseudonymise-ids`, the names `strict` pseudonymises — to its
pseudonym, so it is identifying.

**Dates shift rather than vanish**, so intervals — and therefore
`days_from_baseline`, and therefore any longitudinal model — survive. A
date-time keeps its time, and a value holding two dates moves both. What is
shifted is every date `--apply` acts on: a timepoint's `date`, a date attribute
(`StudyDate`, `AcquisitionDateTime`, …) and any value that is wholly a date,
whatever its key. A date inside free text is reported for review and left as
written, since no rule can tell what a sentence means by it; so is a software
version that reads as a date, which is a build's and not the subject's. A date
`--apply` cannot shift — written in another script's digits, or moved off the
calendar — is removed.

A file is shifted once. Its record keeps the shift, and a later run leaves its
dates alone: with no `--date-shift-days`, or the same one, it re-attests the
recorded shift, and a different one is refused, since the record would then be
false of some of the dates. A shift of `0` is refused too — it moves nothing,
and a record of it would pass every later date over; a record that says `0` —
an earlier version wrote one when asked to — is read as no shift at all. A
timepoint whose `date` was
added after the shift is reported: its `days_from_baseline` puts it apart from
the dates that moved together, and a person has to decide what it should say.

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
