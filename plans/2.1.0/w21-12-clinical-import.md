# W21-12 MEDS and FHIR importers

| | |
|---|---|
| Branch | `w21/clinical-import` |
| Size | L overall: PR 1 M, PR 2 M, PR 3 S–M |
| Lands after | W21-00 PR 4, PR 5; W21-11 PR 2 (rounding) and PR 4 (clocks, `day_bounds`) |
| Blocks | nothing |
| Owns | `medh5/io/{meds.py,fhir.py}` (new), the `meds` extra, the `test-meds` CI job, `docs/guides/clinical-import.md` (new); touches `crates/medh5-cli/src/{app.rs,clinical.rs}` and `medh5/cli.py` (`clinical augment`) |
| Decisions | D18 |
| Backport | none |

## Session prompt

> Implement workstream W21-12 (MEDS and FHIR importers) of medh5 2.1.0 in this
> repository. First read `plans/README.md`, `plans/2.1.0/README.md` and
> `plans/2.1.0/decisions.md` (D18), then `plans/2.1.0/w21-12-clinical-import.md`
> in full. Check that W21-11 has merged (you need its rounding, `Clock.utc` and
> `day_bounds`). Work from the latest `main` on `w21/clinical-import`; land the
> three items in order as one pull request. The importers follow 1.1 §5 and §10
> to the letter: invent no unit, no time zone, no code, no availability and no
> absolute date for a relative clock; where a rule would need a guess, refuse
> instead, so a report never holds a `guess` note. `pyarrow` is an optional
> dependency: its tests `importorskip` it and a CI job installs it. Write tests
> first, run the local gate from `CLAUDE.md`, add a fragment per item, open the
> `W21-12: …` pull request, and do not merge.

## Pull requests

| PR | Items | Size |
|---|---|---|
| 1 | 12.1 A MEDS reader, the `meds` extra and its CI job | M |
| 2 | 12.2 A FHIR R4 reader | M |
| 3 | 12.3 `clinical augment --from-meds / --from-fhir`; 12.4 the guide | S–M |

## Shared rules

- **State at 2.0.2.** 1.1 §10: "**MEDS** rows map from `subject_id`, the clock
  time, `code`, `value_num` and `value_text`; MEDS subject ids need an explicit
  mapping, and availability bounds, revisions, units and imaging references need
  extension columns. Relative times are never given invented absolute dates.";
  "**FHIR** effective time, issued time, coded values and absent-result reasons
  are useful import semantics; … a converter discloses what it transforms or
  omits." The only way in is `medh5 clinical augment PATH RECORDS [-o OUT]
  [--json]` with a hand-built logical-record bundle (both positionals required);
  `augment` refuses an existing `out`, a different clock, and any id already
  present (`clinical/augment.rs`). `scrub` reports the clinical profile as "not
  examined by this tool" (until W21-10 PR 3), so importers leave out free text
  and birth instants unless asked. `pyarrow` is in no extra.
- **Rules every importer meets.** 1.1 §5.1: no invented unit (W914 when absent);
  `<5` written as `lt 5`; never both `value_num` and `value_text`;
  `missing_reason` only when both values are null; no guessed standard code; ids
  match `[A-Za-z0-9_.-]{1,128}`. 1.1 §5.2: unknown availability stays null; a day
  spans all of its instants. Files are written at the lowest version their
  content needs (1.1).
- **Shape.** Python readers in `medh5.io` return `ClinicalRecords` and a
  `ConversionReport`; the engine's `augment` writes the file; the native binary
  hands over through `Host`, as the other converters do. Refusals raise
  `MEDH5ValidationError`: `pyarrow` missing (naming the extra), no rows for the
  subject, a naive time without `timezone`, a relative clock without `origin`,
  both values present under `refuse`, anything `augment` refuses.

## Items

### 12.1 A MEDS reader (PR 1, D18)

- **Change.**
  - `medh5.io.meds.read_meds(source, *, subject_id: int, clock: Clock,
    timezone=None, origin=None, code_system=None, code_systems={}, kinds={},
    status="unknown", day_precision=(), document_codes=(),
    exclude_codes=("MEDS_BIRTH",), texts=False, both_values="refuse",
    columns={}) -> tuple[ClinicalRecords, ConversionReport]`, and
    `augment_from_meds(path, source, *, out=None, **options) -> tuple[dict,
    ConversionReport]`.
  - Mapping (MEDS 0.3; a pre-0.3 dataset names its subject column `patient_id`
    and needs `columns={"subject_id": "patient_id"}`):

    | MEDS | medh5 |
    |---|---|
    | `subject_id` (int64) | only the rows of `subject_id=`; the sample's identity is the caller's mapping, recorded as a `decision` note |
    | `time` null | `static` |
    | `time` | `point` `[t, t]`; naive times read in `timezone`, required on `utc` and `shifted_utc` clocks; on a `relative` clock `origin` is subtracted, and required; a code under a `day_precision` prefix gets its whole day in `timezone` (`day_bounds`) |
    | `code` | a declared prefix maps to a system (`code_systems={"LOINC": "http://loinc.org"}`: `LOINC/2160-0` becomes that system, code `2160-0`); otherwise verbatim, with `code_system` (default `meds:<dataset_name>` from `metadata/dataset.json`) |
    | `numeric_value`, `text_value` | `value_num`, `value_text`; both present: `both_values=` decides (`"refuse"` default, `"numeric"`, `"text"`); `unit` only from a declared column, else null (W914) |
    | (no MEDS column) | `kind` from a declared prefix (`kinds=`), else `other`; `status` `unknown` unless `status=`; `available_us` null unless `columns={"available": …}`, and the report warns that such events can serve as targets but never as inputs until their availability is known |
    | ids | `event_id = record_id = "meds." +` the first 24 hex digits of SHA-256 over the canonical row; repeats get `.2`, `.3`: stable across a re-extraction, so importing the same rows again is refused as repeated ids |
    | `MEDS_BIRTH` | skipped (`exclude_codes`), with an `info` note |
    | text | only with `texts=True`: rows whose code is in `document_codes` become a `document` event owning their `text_value` |

  - The 2.1 reader scans the dataset once per sample (a batch import across a
    cohort is 2.2.0).
  - `pyproject.toml`: `meds = ["pyarrow>=14"]`; a `test-meds` CI job installing
    `[dev,meds]` and running `tests/integrations/test_meds.py` (gating; in `ci-ok`
    and `test_packaging`'s lists). The README's list of extras.
  - The floor is tested too: `test-min-deps` installs `"pyarrow==14.0.*"` and
    imports it in its import check, and
    `test_every_declared_floor_is_installed_by_ci`
    (`tests/project/test_packaging.py`) gains `pyarrow` in its expected set;
    the test fails until both are done.
  - D18's 1.1 §10 erratum and Appendix A row.
- **Tests.** `tests/integrations/test_meds.py` (module-level
  `importorskip("pyarrow")`): `test_S10_meds_naive_times_need_a_zone`,
  `test_S10_a_relative_clock_needs_an_origin`,
  `test_S5_2_meds_availability_stays_unknown`, `test_S5_1_no_unit_is_invented`,
  `test_S5_1_both_values_are_refused`, `test_S10_birth_and_text_are_opt_in`,
  `test_S10_ids_survive_reextraction`, `test_S10_a_reimport_is_refused`. A `Meds`
  kit in `tests/kits.py`.
- **Changelog.** Added, area `clinical`.

### 12.2 A FHIR R4 reader (PR 2)

- **Change.**
  - `medh5.io.fhir.read_fhir(source, *, subject: str, clock: Clock,
    timezone=None, origin=None, texts=False, prefer_systems=(), image_studies={})
    -> tuple[ClinicalRecords, ConversionReport]` and `augment_from_fhir(...)`.
    Standard library only (a Bundle or a list of resources, JSON).
  - Mapping:

    | FHIR R4 | medh5 |
    |---|---|
    | `id`, `meta.versionId` | `record_id = obs.<id>` or `dr.<id>`; with several versions in a Bundle, `event_id = …v<versionId>`, each version `supersedes` the one before |
    | `status` | `preliminary`, `final`, `cancelled`, `unknown` kept; `partial` → `preliminary`; `amended`, `corrected`, `appended` → `amended` (a `decision` note); `entered-in-error` → `entered_in_error`; `registered` skipped (`info`) |
    | `code.coding` | the first coding in `prefer_systems` order, else the first coding, gives `code_system`, `code`, `code_version`; no coding: skipped (`warning`) |
    | `effectiveDateTime`, `effectiveInstant` | bounds follow the precision (year, month, day, instant); a time without an offset needs `timezone`; a date with no known zone spans all its possible instants |
    | `effectivePeriod` | `interval` from `start` and `end`; an open end gives a null end |
    | `effectiveTiming`, or none | `unknown`, with a `warning` for Timing |
    | `issued` | `available_us = [t, t]`; null when absent |
    | `valueQuantity` | `value_num`; `unit` the UCUM `code`, else the `unit` string; comparators `<`, `<=`, `>`, `>=` become `lt`, `le`, `gt`, `ge`; `ad` skipped |
    | `valueString`, `valueBoolean`, `valueCodeableConcept` | `value_text` (for a concept, the chosen coding's `code`) |
    | `valueInteger` | `value_num` |
    | other `value[x]` | skipped (`warning`) |
    | `dataAbsentReason` | `missing_reason` its code, both values null |
    | `component[]` | one event per component (`record_id = …c<i>`), sharing the parent's times and status |
    | `subject`, `encounter` | only resources of `subject=`; others counted in an `info` note; `encounter` → `encounter_id` |
    | `DiagnosticReport` | a `document` event; its `text/plain` `presentedForm` (else `conclusion`) its text, only with `texts=True`; `imagingStudy` becomes a `describes` link only through `image_studies={reference: image_id}`; `result` not linked (`info`) |

  - What is dropped and reported (`interpretation`, `referenceRange`, MEDS code
    modifiers) could only be kept by a later format minor; the guide says so.
- **Tests.** `tests/integrations/test_fhir.py`: `test_S5_2_fhir_precision_becomes_bounds`,
  `test_S5_2_issued_is_availability`, `test_S5_1_comparators_are_kept`,
  `test_S5_1_data_absent_reason_is_missing_reason`,
  `test_S5_2_versions_supersede`, `test_S10_registered_is_skipped`,
  `test_S10_images_link_only_through_the_map`. A `Fhir` kit.
- **Changelog.** Added, area `clinical`.

### 12.3 `clinical augment --from-meds / --from-fhir` (PR 3)

- **Change.** `medh5 clinical augment SAMPLE [RECORDS] [--from-meds DIR
  --meds-subject N | --from-fhir FILE --fhir-subject REF] [--clock-id ID --clock
  {utc,shifted_utc,relative}] [--origin ISO] [--timezone TZ] [--code-system
  PREFIX=URI]... [--kind PREFIX=KIND]... [--day-precision PREFIX]...
  [--document-code CODE]... [--with-text] [--both-values {refuse,numeric,text}]
  [--report FILE] [-o OUT] [--json]`. `RECORDS`, `--from-meds` and `--from-fhir`
  form an `ArgGroup` of which exactly one is given. The handler hands over through
  `Host::convert(argv, "clinical-augment", …)`, with `_HANDLERS["clinical-augment"]`
  in `medh5/cli.py`; `--report` writes through `_core.write_atomic`.
- **Tests.** CLI `test_clinical_augment_from_meds`, `…_from_fhir`, and a test of
  the native binary's hand-over; every documented flag is checked by
  `test_docs_examples.py`.

### 12.4 The guide (PR 3)

- **Change.** `docs/guides/clinical-import.md`: both mapping tables, the report,
  the refusals, what `scrub` covers on `main` when this lands and what it does
  not (W21-10 PR 3 updates that paragraph when it makes `scrub` cover the
  clinical profile); nav entry;
  `docs/reference/cli.md`.
- **Changelog.** Documentation.
