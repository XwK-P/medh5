# Import a clinical history from MEDS or FHIR

You have the images as `.medh5` samples and the history somewhere else: a
[MEDS](https://github.com/Medical-Event-Data-Standard/meds) dataset, or FHIR R4
resources exported from a hospital system. The importers read one subject's
rows or resources, map them onto the clinical profile's events (1.1 §5), and
add them to the sample with `augment` (1.1 §10). The sample becomes 1.1; its
images and annotations are copied as stored.

The importers invent nothing the source does not say. No unit, no time zone,
no origin, no availability and no standard code is made up: where a rule would
need a guess, the import is refused and the message says what to give. So the
conversion report holds decisions and warnings, never a `guess`.

## From MEDS

MEDS needs `pyarrow`: `pip install "medh5[meds]"`.

<!-- illustrative -->
```python
from medh5.clinical import Clock
from medh5.io.meds import augment_from_meds

result, report = augment_from_meds(
    "P-0001.medh5",
    "meds/",                       # the dataset root: data/*.parquet
    subject_id=1001,               # whose rows this sample holds
    clock=Clock.utc("ehr"),
    timezone="America/New_York",   # MEDS times carry no zone
    code_systems={"LOINC": "http://loinc.org"},
    kinds={"LOINC": "observation", "DX": "diagnosis"},
    day_precision=["DX"],
)
print(report.format(verbose=True))
```

`read_meds` takes the same options and returns the records and the report
without writing anything.

| MEDS | Event |
|---|---|
| `subject_id` | only the rows of `subject_id=`; which sample they belong to is your mapping, recorded as a `decision` note, and MEDS's id is not written |
| `time` null | `static` |
| `time` | `point` `[t, t]`. Times without a zone are read in `timezone`, which a `utc` or `shifted_utc` clock needs. A `relative` clock needs `origin`, which is subtracted. A code under a `day_precision` prefix spans its whole day in `timezone` |
| `code` | under a prefix of `code_systems`, that system (`LOINC/2160-0` is `http://loinc.org`, `2160-0`); otherwise verbatim, in `code_system` (default `meds:<dataset_name>` from `metadata/dataset.json`) |
| `numeric_value`, `text_value` | `value_num`, `value_text`. A row holding both is refused unless `both_values` is `"numeric"` or `"text"`. `unit` comes only from a column you declare (`columns={"unit": …}`), else it is null (W914) |
| no MEDS column | `kind` from a prefix of `kinds`, else `other`; `status` is `unknown` unless `status=`; `available_us` is null unless `columns={"available": …}` |
| ids | `event_id = record_id = "meds." +` the first 24 hex digits of the SHA-256 of the row's canonical JSON, declared `unit` and `available` columns included; a repeated row gets `.2`, `.3`. A re-extraction keeps the ids, so importing the same rows twice is refused |
| `MEDS_BIRTH` | left out (`exclude_codes`), with a note |
| text | rows whose code is in `document_codes` become `document` events; they own their `text_value` only with `texts=True` |

MEDS records no availability. Without an `available` column every event's is
null, and the report warns: such events can be a task's targets, never its
inputs, until their availability is known (1.1 §5.2, §9.1). A dataset older
than MEDS 0.3 names its subject column `patient_id`; pass
`columns={"subject_id": "patient_id"}`.

The dataset is scanned once per call. For a whole cohort, use
`augment_cohort_from_meds` (see [A cohort](#a-cohort)), which reads it once.

## From FHIR R4

The FHIR reader needs nothing beyond the standard library. It takes a Bundle,
one resource or a list of resources, as a JSON file or as data already read:

```python
from medh5.clinical import Clock
from medh5.io.fhir import augment_from_fhir

loinc = {"system": "http://loinc.org", "code": "2160-0"}
bundle = {"resourceType": "Bundle", "type": "collection", "entry": [
    {"resource": {"resourceType": "Observation", "id": "creat", "status": "final",
                  "subject": {"reference": "Patient/p1"}, "code": {"coding": [loinc]},
                  "effectiveDateTime": "2020-03-01T08:15:00-05:00",
                  "issued": "2020-03-01T09:15:00-05:00",
                  "valueQuantity": {"value": 1.1, "unit": "mg/dL",
                                    "system": "http://unitsofmeasure.org",
                                    "code": "mg/dL"}}},
]}
result, report = augment_from_fhir(
    "case.medh5", bundle, out="case-fhir.medh5",
    subject="Patient/p1", clock=Clock.utc("ehr"),
)
assert result["events"] == 1 and not report.guesses
```

| FHIR R4 | Event |
|---|---|
| `id`, `meta.versionId` | `record_id` is `obs.<id>` or `dr.<id>`. With several versions of one resource, `event_id` adds `.v<versionId>` and each version `supersedes` the one before; versions that cannot be ordered (by numeric `versionId`s naming distinct numbers, else by `meta.lastUpdated`) are refused |
| `status` | `preliminary`, `final`, `cancelled`, `unknown` kept; `partial` is `preliminary`; `amended`, `corrected` and `appended` are `amended` (a `decision` note); `entered-in-error` is `entered_in_error`; `registered` is left out |
| `code.coding` | the first coding in `prefer_systems` order, else the first, gives `code_system`, `code` and `code_version`. Without a coding the resource is left out, with a warning: no code is guessed from `code.text` |
| `effectiveDateTime`, `effectiveInstant` | a `point`, its bounds the value's precision (year, month, day or instant). A time without an offset needs `timezone`; a date with no known zone spans every instant it covers anywhere |
| `effectivePeriod` | an `interval` from `start` to `end`; an open end is null |
| `effectiveTiming`, or none | `unknown`, with a warning for a Timing |
| `issued` | `available_us = [t, t]`; null when absent |
| `valueQuantity` | `value_num`; `unit` the UCUM `code`, else the `unit` string; comparators `<`, `<=`, `>`, `>=` become `lt`, `le`, `gt`, `ge`, and `ad` is left out |
| `valueString`, `valueBoolean`, `valueCodeableConcept` | `value_text`; for a concept, the chosen coding's `code` |
| `valueInteger` | `value_num` |
| any other `value[x]` | left out, with a warning |
| `dataAbsentReason` | `missing_reason`, its code; both values null |
| `component[]` | one event per component, `record_id` `…c<i>`, sharing the resource's times and status |
| `subject`, `encounter` | only resources of `subject=`; the others are counted. `encounter` is `encounter_id` |
| `DiagnosticReport` | a `document` event. Its `text/plain` `presentedForm`, else its `conclusion`, is the document's text, only with `texts=True`. An `imagingStudy` becomes a `describes` link to an image only through `image_studies={reference: image_id}`; `result` is not linked |

## A cohort

A MEDS dataset or a FHIR Bulk Data export holds many subjects. The cohort
importers read it once, rather than once per sample, and write each subject
into the sample a mapping names:

<!-- illustrative -->
```python
from medh5.clinical import Clock
from medh5.io.fhir import augment_cohort_from_fhir
from medh5.io.meds import augment_cohort_from_meds

cohort = augment_cohort_from_meds(
    "meds/",
    {1001: "P-0001.medh5", 1002: "P-0002.medh5"},   # subject_id -> sample
    out_dir="imported/",           # else each sample in place
    timezone="America/New_York",   # the clock: each sample's own, unless clock=
    code_systems={"LOINC": "http://loinc.org"},
)
for subject in cohort.refused:
    print(subject.subject, subject.error)

bulk = augment_cohort_from_fhir(
    "export/",                     # Observation.ndjson, DiagnosticReport.ndjson, ...
    {"Patient/p1": "P-0001.medh5"},
    clock=Clock.utc("ehr"),
)
```

Each subject is mapped exactly as the per-sample importer maps it alone, so
its sample is the one `augment_from_meds` or `augment_from_fhir` would write,
and its entry in `cohort.subjects` carries the same augmentation and report.
The batch changes the cost, never the answer:

- **MEDS.** The dataset is opened once and read twice, a Parquet row group at
  a time: the subject column to count each mapped subject's rows, then the
  mapped subjects' rows, skipping row groups whose statistics hold none. A
  subject is imported as soon as its last row is read, so a dataset in MEDS
  order holds one row group and one subject in memory at a time.
- **FHIR.** The NDJSON files are read once to find where each mapped
  subject's Observations and DiagnosticReports are, then once per subject to
  read them, in file and line order, as a Bundle of them would be read.

A subject that refuses (no rows, a sample without a clock, ids already there)
is listed in `cohort.refused` with its reason, and the others are still
written. Subjects the mapping does not name are counted in an `info` note of
`cohort.report`, and never read. `cohort.to_json()` is the whole result.

## The report

Each import returns a conversion report, as every converter does (`format`,
`to_json`). Its notes are of three kinds:

- **decision**: something the data determined, such as the subject mapping, a
  status mapped to `amended`, or times known to the day spanning the day.
- **info**: something left out on purpose, such as `MEDS_BIRTH`, free text
  without `texts=True`, numeric values without a unit, or an unlinked study.
- **warning**: something the source held that has no form here, or availability
  that stays unknown. `report.ok` is false when there is one.

What is left out and reported (`interpretation`, `referenceRange`, MEDS code
modifiers) could only be kept by a later format minor. The importer discloses
it rather than folding it into a field that means something else.

## Refusals

Each refusal raises `MEDH5ValidationError` and writes nothing:

- the subject has no rows or resources;
- a time without a zone is read for a `utc` or `shifted_utc` clock and no
  `timezone` is given;
- a wall-clock time falls in a daylight-saving gap, so it names no instant (one
  that falls twice is bounded by both readings);
- a relative clock has no `origin`;
- a MEDS row holds both values and `both_values` is `"refuse"`;
- FHIR versions cannot be ordered;
- anything `augment` refuses: an existing `--out`, a clock other than the
  sample's, or an id already in the sample (so a second import of the same rows
  is refused).

A missing `pyarrow` is an `ImportError` that names the extra to install.

## From the command line

```bash
medh5 clinical augment P-0001.medh5 --from-meds meds/ --meds-subject 1001 \
    --clock-id ehr --clock utc --timezone America/New_York \
    --code-system LOINC=http://loinc.org --kind LOINC=observation \
    --day-precision DX --report P-0001.import.json
medh5 clinical augment P-0001.medh5 --from-fhir bundle.json --fhir-subject Patient/p1 \
    --with-text -o P-0001.fhir.medh5
```

Without `--clock-id` and `--clock` the records go on the sample's own clock,
and a sample with none is refused. `--with-text` imports free text. The
command prints the augmentation and the report; `--report` keeps the report as
JSON, and `--json` prints both. It exits 1 when the report holds a warning, as
every converter does. The file is still written: read the warnings. A MEDS
import without an `available` column always warns.

A cohort takes a mapping as CSV, with the header `subject,sample` (a relative
sample path is the CSV's):

```bash
medh5 clinical augment-cohort --from-meds meds/ --mapping cohort.csv \
    --timezone America/New_York --code-system LOINC=http://loinc.org \
    --out-dir imported/ --report cohort.import.json
medh5 clinical augment-cohort --from-fhir-bulk export/ --mapping cohort.csv \
    --clock-id ehr --clock utc
```

It prints each subject's result and report; a refused subject is listed and
the others written. It exits 1 when a subject was refused or a report holds a
warning.

The importers run in the Python package. The standalone binary hands them to
Python, as it hands over `medh5 convert`.

## De-identification

`medh5 scrub` examines the clinical profile: it moves a `utc` clock's times by
the date shift, re-mints the ids from the salt, and finds names, dates, UIDs
and record numbers in the text. `--redact-text` replaces what those rules find
with placeholders, as new documents; the rest of a document must be
de-identified by a tool that handles free text and attested with
`--attest-text`, or `--apply` refuses the sample (see
[De-identify and publish](deidentify.md#4-a-clinical-sample)). This is why
the importers leave out free text and birth instants unless asked. If you
import them, de-identify the text before importing, or keep the samples
private.
