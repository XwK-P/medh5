# Import from DICOM

Turn a directory of studies into one sample per patient, with every visit in the
same file.

```bash
pip install "medh5[dicom]"
```

## 1. See what is there

```bash
medh5 convert from-dicom /studies out/ --group-by subject
```

That is the whole command for the common case. `--group-by subject` is the
default: it resolves patient identity across studies and writes one
multi-timepoint sample per patient, which is what makes a later split
subject-safe.

To narrow it first:

```bash
medh5 convert from-dicom /studies out/ --modality CT
medh5 convert from-dicom /studies out/ --series 1.2.840...
```

## 2. Read the report — especially the guesses

```bash
medh5 convert from-dicom /studies out/ --report import.json
```

A conversion report separates what the converter **decided** from the data from
what it **guessed** because the source did not say. A guess is not a failure; it
is the thing to go back and check.

The guess that matters most here is **subject grouping**. When identity cannot
be established — usually because a de-identification pass randomised
`PatientID` — the import falls back to one sample per study, warns, and names
the affected inputs. It never infers identity from filenames, dates or accession
numbers.

If that fallback happened, you have one file per *study*, not per subject, and
splitting by file is no longer subject-safe.

**`--group-by group_id` does not fix this**, although it looks as though it
should. The fallback gives each output a distinct synthetic subject
(`study:<StudyInstanceUID>`) and sets no `cohort.group_id`, and `group_id`
falls back to `subject_id` when unset — so grouping by it groups by values that
are already distinct, and two visits of one patient can still land on opposite
sides of the split. Silently: nothing warns.

Stamp the grouping you know, then re-scan:

```python
import medh5

for path, patient in known_identities.items():     # from your own records
    with medh5.amend(path) as w:
        w.cohort(group_id=patient)
```

```bash
medh5 dataset index out/ -o cohort.json   # re-scan, so the manifest sees it
medh5 dataset split cohort.json --group-by group_id   # C204 if a group splits a subject
medh5 dataset check cohort.json           # C202 if the written claims disagree
```

`C204` comes from **`split`**, not from `check` — it is refused while the split
is being made, which is before anything is written. `check` reads what is on
disk: it catches a subject whose samples claim different partitions (`C202`),
not a grouping that was wrong to begin with.

Neither can recover an identity DICOM has lost. If the mapping you stamped above
was wrong, both commands will agree with it.

Amending rewrites each file, so this is not free — but it is the only thing that
makes the split subject-safe, and there is no way to recover the identity from
the files alone once DICOM has lost it.

## 3. Expect refusals, and read them

The importer would rather stop than write a file whose geometry is invented. A
series is refused when its slices disagree about orientation, spacing or rescale;
when slice gaps are irregular; or when a required tag is missing or malformed.
The message names the offending `SOPInstanceUID`.

These refusals carry **no diagnostic code**, because a DICOM series is not a
MEDH5 file yet. [The reasoning for each is here](../explanation/refusals.md#dicom).

## 4. Check the result

```bash
medh5 info out/case_0001.medh5
medh5 timeline out/case_0001.medh5     # did the visits land as you expect?
medh5 validate out/*.medh5 --level strict
```

Values are stored as the scanner wrote them, with the modality LUT recorded
rather than applied:

```python
s.images["CT"].read()                  # stored counts
s.images["CT"].read(physical=True)     # HU
```

## Related

- **[Converters](../reference/converters.md#dicom)** — options and Python entry points.
- **[What the converters refuse, and why](../explanation/refusals.md#dicom)** — every DICOM refusal.
- **[Build and split a cohort](cohorts.md)** — the next step for a directory of samples.
