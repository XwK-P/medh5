# MEDH5 format specification 1.1 — clinical context

**Status: Final.** The reference engine reads, writes, validates and amends the
`clinical` profile defined here, and the conformance corpus (§11.3) holds at least one case per
diagnostic code the profile adds. The task and feature-cache contract that training uses is a
separately versioned companion, [Task and cache contract 1](task-cache-1.md); nothing in it is a
`.medh5` payload requirement.

This document is a **delta to [MEDH5 1.0](medh5-1.0.md)**. Every 1.0 clause continues to apply
unless a section below names it. Keywords are those of 1.0 §1.2.

---

## 1. Scope

MEDH5 1.1 adds **optional clinical context to an imaging sample**: source documents, asynchronous
observations and interventions, typed relationships between them and the existing imaging objects,
and explicit **information-availability** times. A sample still holds exactly one subject, at least
one grid and one image, and at least one imaging timepoint. Geometry, annotation encodings, image
value semantics, transforms and sample-relative addressing are unchanged.

The optional `clinical` profile requires `core` and the structures of §§3–7. A clinical event need
not coincide with an imaging timepoint; `longitudinal` keeps its 1.0 meaning (at least two imaging
timepoints), so a sample with one CT and years of laboratory history is `core, clinical` without
`longitudinal`. Several imaging modalities alone do not need 1.1.

What 1.1 deliberately does not do, and which version would:

| Requirement | Where it belongs |
|---|---|
| Reports, labs, diagnoses, medications and assessments beside images | This profile |
| Prediction cutoffs, irregular sequences, missing-modality batches | The companion [task and cache contract](task-cache-1.md) |
| Subjects with no image and no grid (EHR-, text- or waveform-only) | A 2.0: it relaxes the mandatory core |
| A record assembled from mandatory external payloads | A 2.0: it changes the self-contained sample |
| Replacing grid-owned imaging time, or annotation and value semantics | A 2.0 |
| A clinical vocabulary larger than the voxel class space | This profile: clinical codes are strings, never voxel class ids (§5.1) |

## 2. Compatibility

### 2.1 Versions written

A writer writes the **lowest version its content needs**: `1.0` when a sample holds imaging alone,
`1.1` when it declares `clinical`. An amendment never lowers the version a file already declares
(§2.3), except one that removes the clinical profile: that is the §10 imaging projection ---
separately requested, and a different sample with its own `content_id` --- whether it is written to
a new file (`medh5 clinical strip`) or in place of its source (`drop_clinical` in an amendment), and
it is written at the lowest version what remains needs.

### 2.2 Read, validate and amend are separate capabilities

An implementation distinguishes:

1. **reading a supported projection** --- for example the 1.0 imaging objects of a 1.1 file;
2. **validating declared profiles**, which requires implementing their semantics;
3. **amending a file**, which requires preserving all supported and opaque content, its
   declarations and its integrity contract.

This refines 1.0 §16 ("readers **MUST** accept a higher MINOR"; see 1.0 Appendix C.1). For a file
of a **higher minor** version than it implements, an implementation:

- **MUST NOT** reject the file solely for its minor version, and **MAY** expose the supported
  projection;
- reports every item of the newer version it does not know --- a profile, a `/meta` member, an
  enumerated value, a column, an attribute --- as **W913**, not as the error the same item would be
  in a file of a version it implements, and reports the version itself as **W913** at `/`;
- **MUST NOT** claim conformance to the newer version, or clinical-training support, because the
  file opened and its projection validated;
- reports `content_id` it cannot recompute as **W913** ("unsupported, not verified"), never as
  **E702**: a later minor may cover root attributes this implementation does not know. Dataset
  digests (1.0 §13.1) are still checked and still report **E701**;
- **MUST** refuse to amend, recompress or pack the file (§2.3).

An unknown **major** version remains a refusal (**E002**).

### 2.3 Amendment

A writer **MUST** refuse an amendment it cannot make losslessly: a file of another major or a higher
minor, or one declaring a profile the writer does not implement (**E007**). The refusal happens
before anything is written; the source is never modified.

A writer that implements every declared profile re-derives the **known** profiles from the amended
content, as in 1.0: an edit that removes the last voxel annotation removes the `seg` claim rather
than keeping a false one. A no-op amendment therefore preserves every declaration of a valid file.
A recognised `clinical` group (§3) carries its profile through an amendment untouched, byte for
byte, until the amendment changes it (§8). A `clinical` group that is **not** the profile's --- a
1.0 file's own extension --- is copied through and never reinterpreted.

### 2.4 Subject grouping

One sample **MUST NOT** span subjects; one subject **MAY** occupy several samples (1.0 §2.2), so
file-level partitioning alone does not separate subjects. Dataset splits **MUST** group every sample,
collection member and clinical fragment of one known subject together. Identity namespaces and any
cross-source reconciliation are recorded by the dataset manifest or the task manifest
([task-cache-1](task-cache-1.md) §3.3); matching local ids across institutions is not sufficient.
Packing is physical packaging, not a merge of patient records. `instance_id` stays sample-scoped.

## 3. Layout and the clinical descriptor

Paths are relative to the sample root, inside a collection too.

| Path | Representation | Requirement |
|---|---|---|
| `meta`, `grids`, `images`, `annotations`, `transforms` | 1.0 objects | Unchanged |
| `index/<ann_id>` | 1.0 sampling cache | Unchanged; not a clinical cache |
| `clinical/meta` | Scalar variable-length UTF-8 string dataset, canonical JSON | Required by `clinical` |
| `clinical/events/` | Column table (§4), one row per immutable event version | Required; at least one row |
| `clinical/documents/` | Column table of source text | Present when a document exists |
| `clinical/links/` | Column table of typed relationships | Present when a relationship exists |

The root declares `medh5_version = "1.1"` and includes `clinical` in `medh5_profiles`. A `clinical`
group is **recognised** as the profile's when its `meta` is a scalar string dataset holding a JSON
object whose `schema` is `medh5.clinical/1`. A recognised group without the declaration is **E803**,
whatever the file's version. In a file declaring 1.1 or later the name is reserved: any object named
`clinical` without the declaration is **E803**. In a 1.0 file a `clinical` object that is not
recognised is the file's own extension (1.0 §2.3). The declaration without the group, without
events, or in a file declaring `1.0`, is **E009**. A member of `clinical/` the profile does not
define is **E804**, and a clinical group or dataset carrying an attribute other than a dataset's
`digest` is **E819**: the profile's semantics live in its columns and its descriptor, never in
attributes, which keeps the 1.0 `content_id` construction unchanged (§8).

The 1.0 `/meta` document and its timepoint field set are unchanged. A clinical-only visit is
**not** inserted into `meta.timepoints` to simulate a scan.

The descriptor:

```json
{"clock":{"id":"subject-clock-01","origin_description":"acquisition start of the baseline CT (tp0)","reference":"relative","unit":"us"},"schema":"medh5.clinical/1"}
```

`clinical/meta` is serialized in the canonical JSON of 1.0 §5.1 --- sorted keys, no whitespace,
UTF-8 --- so equal descriptors digest equally (**E801** otherwise). Its schema is
`medh5-clinical-1.schema.json` (`#/$defs/descriptor`), closed (**E802**):

- `schema` **MUST** be `medh5.clinical/1`.
- `clock.id` is a non-empty opaque identifier, scoped by the dataset's subject identity namespace.
- `clock.unit` **MUST** be `us`.
- `clock.reference` is `relative`, `utc` or `shifted_utc`. Relative times are signed microseconds
  from a documented, subject-specific origin, and a relative clock **MUST** carry a non-empty
  `origin_description` (**E802**). UTC is signed microseconds from the Unix epoch; shifted UTC uses
  the same epoch after the de-identification shift `/meta` declares, and a `shifted_utc` clock in a
  sample that declares no `deidentification` is **E802**.

An origin need not be birth or the first image; negative times are valid, including events before
baseline imaging. Fragments of a subject that share a clock **MUST** share its origin and unit. No
conversion between clocks is defined: a task's preflight joins a subject's fragments only when they
declare one clock --- the same `clock` object --- and refuses the temporal join otherwise (**T304**,
[task-cache-1](task-cache-1.md) §3.3). Ingestion time is never a clinical origin by default. Date
shifts preserve one timeline across images, reports and events.

## 4. Column encoding

One primitive column layout --- no HDF5 compound records, no pickles, no group per event. Every column
of a table has the same row count `N`.

| Logical column | HDF5 representation |
|---|---|
| Integer, floating point | Dataset `(N,)` of the column's dtype (§5–§7): `int64`, `uint64` or `float64`, little-endian |
| UTF-8 string | Group `<column>/` holding `data: uint8[B]` and `offsets: uint64[N+1]`, little-endian |
| Nullable column | Optional `valid/<column>: uint8[N]`, values exactly 0 or 1 |

A column of another dtype, rank or row count is **E805**. For a packed string, `offsets[0] = 0`,
offsets never decrease, the last equals `B`, and row `i` is `data[offsets[i]:offsets[i+1]]`, which
**MUST** decode as UTF-8 (**E806**). Offsets count bytes, not code points.

Required columns have no validity dataset. A nullable column without one is entirely valid, and an
**omitted** optional column is entirely null --- not zero. A null numeric cell **MUST** hold `0` and
a null string cell **MUST** be empty (equal adjacent offsets), so stale bytes cannot leak a source
value (**E807**); a mask with values other than 0 and 1, a mask for a required or absent column, or a
mask of the wrong length is **E807** too. A zero-length valid string is distinct from a null.
**Identifier columns** are never empty where valid (**E809**): `event_id`, `record_id` and
`document_id`, which also match the 1.0 id syntax (1.0 §2.3); the events' `timepoint_id`,
`encounter_id`, `code_system`, `code`, `code_version`, `unit`, `missing_reason` and `prov`; the
documents' `language` and `source_type`; and the links' `source_id`, `target_id`,
`target_annotation_id` and `asserted_by_event_id`. Every valid floating-point value is finite; NaN
and infinity are forbidden with or without a mask (**E808**).

The set of columns is **closed** at 1.1: an unknown column is **E804** (W913 in a higher minor's
projection, §2.2). Large numeric columns and UTF-8 buffers **SHOULD** be chunked and compressed with
a 1.0 §14.2 codec; offsets and small tables **MAY** be contiguous; `clinical/meta` stays an
uncompressed scalar. Physical row order is not clinical chronology and not an identifier. Writers
**SHOULD** store events by known effective start, then event id, with unknown-time rows last --- the
reference writer does --- and derived lookup orderings never change identities.

## 5. Events and temporal semantics

### 5.1 Event columns

An event row is one **immutable version** of information about the subject.

| Column | Type | Presence | Meaning |
|---|---|---|---|
| `event_id` | UTF-8 | required | Unique immutable event-version id, 1.0 id syntax |
| `record_id` | UTF-8 | required | Logical record shared by the versions of one record, 1.0 id syntax |
| `kind` | UTF-8 | required | `imaging`, `document`, `observation`, `diagnosis`, `medication_order`, `medication_administration`, `procedure`, `assessment`, `other` |
| `temporal_type` | UTF-8 | required | `point`, `interval`, `static`, `unknown` |
| `effective_start_lo_us`, `effective_start_hi_us` | int64 | optional | Inclusive bounds on when the event started or occurred |
| `effective_end_lo_us`, `effective_end_hi_us` | int64 | optional | Inclusive bounds on the end of an interval |
| `available_lo_us`, `available_hi_us` | int64 | optional | Inclusive bounds on when **this whole version** became available in the source workflow |
| `status` | UTF-8 | required | `unknown`, `preliminary`, `final`, `amended`, `entered_in_error`, `planned`, `in_progress`, `completed`, `cancelled` |
| `timepoint_id` | UTF-8 | optional | The imaging occasion the event belongs to, if any |
| `encounter_id` | UTF-8 | optional | Opaque source encounter key; not an imaging timepoint |
| `code_system`, `code`, `code_version` | UTF-8 | optional | Source clinical concept and terminology version |
| `value_num` | float64 | optional | Recorded numeric value, before any model-specific binning |
| `value_comparator` | UTF-8 | optional | `eq`, `lt`, `le`, `gt`, `ge`; `eq` when absent and `value_num` is valid |
| `unit` | UTF-8 | optional | Unit of `value_num`, UCUM preferred; `1` is dimensionless |
| `value_text` | UTF-8 | optional | Categorical or short textual result, in its source meaning |
| `missing_reason` | UTF-8 | optional | Source-supported reason an expected result is absent |
| `prov` | UTF-8 | optional | An activity id of the 1.0 provenance graph (1.0 §11.1) |

A value outside a vocabulary above is **E810**; an id that is empty, malformed or repeated is
**E809**. Event and record ids survive reordering, repacking and no-op amendment, are scoped by
subject identity, and **MUST NOT** collide when fragments of a subject are joined. A model token id is
not an event id.

Values (**E812** unless stated):

- `code_system` and `code` are both valid or both null. A missing terminology version stays null.
- `value_comparator` and `unit` are null when `value_num` is null. A numeric value **SHOULD** carry a
  unit (**W914** when it does not), and a converter **MUST NOT** invent one. `<5` is
  `value_comparator = lt`, `value_num = 5` --- never an exact 5.
- `value_num` and `value_text` are never both valid. A `missing_reason` requires both null: a
  missing expected result is absence plus its reason, never a negative finding.
- An `observation` carrying a value names what was measured (`code_system` and `code`; a declared
  local concept will do). A converter **MUST NOT** guess a standardized code.

Clinical concepts are strings. They **MAY** be dictionary-encoded in training caches, but **MUST
NOT** reuse the segmentation `uint16` class space, its background and ignore sentinels, or a
tokenizer's vocabulary: a cohort can carry far more than 65 534 clinical concepts beside any label
set (conformance case `clinical-wide-vocabulary`).

### 5.2 Time bounds, durations and revisions

Each pair of bounds is wholly valid or wholly null, and `lo ≤ hi` (**E811**). An exact instant has
`lo = hi`. Day-only information spans the day's possible instants after clock conversion; it never
implies midnight or an order within the day. Temporal uncertainty (the bounds) and clinical duration
(start to end) are different things.

| `temporal_type` | Start bounds | End bounds |
|---|---|---|
| `point` | required | absent |
| `interval` | required | optional (null: ongoing or unknown); valid ends admit a start-before-end assignment, `start_lo ≤ end_hi` |
| `static` | absent | absent --- availability is still meaningful |
| `unknown` | absent | absent --- never silently treated as `static` |

Any other combination is **E811**. Source precision is represented by the bounds; the microsecond
unit does not claim microsecond precision. Availability need not follow occurrence: an order can be
known before its execution. Orders and administrations remain distinct kinds when the source
distinguishes them.

Availability covers **every field of a version**. A medication course whose actual end was learned
later is a new version; so are a corrected measurement, a revised report and a retrospectively
assigned diagnosis. The versions of one record form one acyclic, unambiguous `supersedes` chain (§7);
independent conflicting claims use distinct record ids rather than an invented revision order.

Unknown availability **MUST** stay null. A writer **MUST NOT** substitute ingestion time, the file's
`created`, or clinical occurrence time; a task that assumes one records it as an assumption in its
own policy. Static demographics need availability too when a task reads them.

### 5.3 Imaging events and timepoints

Imaging timepoints stay observation occasions. An `imaging` event adds precise acquisition and
availability bounds to an existing image, which it links by `describes` (§7). If it names a
`timepoint_id`, that **MUST** be the timepoint of the image's grid (**E814**); clinical timestamps do
not override grid membership. A grid's within-acquisition `time_values` remain distinct from the
patient clock, and channel axes or acquisition parameters are never elapsed visits.

## 6. Source documents

`clinical/documents` is a §4 table with required UTF-8 columns `document_id` (unique, 1.0 id
syntax), `media_type` and `text`, and optional `language` (BCP 47 where known) and `source_type`. In
this profile `media_type` **MUST** be `text/plain` (**E810**); the text may be in any language.

Each document is **owned by exactly one** `document` event, which links it by `describes` and owns its
timing, status and provenance, and a `document` event owns **at most one** document (**E815**): one
text, one event version, one availability --- an event whose text is not stored owns none. A revision
is a new document *and* a new event version; the `record_id` persists. The text is the canonical,
de-identified source content --- not token ids, a summary or an embedding. Imported whitespace and
Unicode are preserved after documented de-identification; any change to the stored bytes is a new
document and a new event version with explicit provenance, and is not a newly issued report: its source
availability is retained when the same information is being represented. Model summaries belong in
separately identified derived records or caches.

Links may cite spans of a document as half-open UTF-8 byte intervals `[start, end)` whose endpoints
lie on code-point boundaries within that exact revision (**E814**). De-identification covers the body
as well as metadata: a date-shift field alone claims nothing about free text.

No text needs to be read to read the rest of the profile. The `text` offsets alone give every
document's byte length, so a reader can open the profile, select at a cutoff and know which documents
a row may read without decompressing a report, and read one document's bytes when it is asked for ---
checking that row's UTF-8 then (**E806**), as every row is checked on its own. A validator still checks
every row, which it can do a bounded slab of the buffer at a time; a span's endpoints need only the
bytes at them.

PDFs, raw waveforms, whole-slide tile pyramids and arbitrary external assets are not standardized by
this profile, and an extension **MUST NOT** replace a required local payload with an unresolvable
external dependency.

## 7. Typed links and assessments

### 7.1 Links

`clinical/links` is a §4 table. Required UTF-8 columns: `source_type`, `source_id`, `relation`,
`target_type`, `target_id`. Endpoint types: `event`, `document`, `image`, `grid`, `annotation`,
`transform`, `timepoint`, `instance`. Relations: `describes`, `compares_with`, `measures`,
`derived_from`, `supersedes`, `assesses` (**E810** otherwise). These links do not change the 1.0
annotation `derived_from` field.

Optional columns: `source_start` and `source_end` (`uint64`, a text span, valid only for a `document`
source), `target_annotation_id` (UTF-8, valid only for an `instance` target) and
`asserted_by_event_id` (UTF-8, the event version that supplies the relationship as evidence).

- Every endpoint **MUST** resolve within the sample root --- never against a working directory or an
  absolute HDF5 path --- and every `asserted_by_event_id` to a local event version (**E813**). Each
  sample therefore holds the local closure of its links, including the predecessors `supersedes`
  names.
- `instance` ids are the canonical decimal string of a sample-scoped `instance_id`, and the instance
  **MUST** occur in the `instance_ids` dataset of an annotation of the sample, whatever its kind
  (**E813** otherwise, as for any endpoint that does not resolve). A `target_annotation_id`
  **MUST** name an annotation of the sample (**E813**) whose `instance_ids` hold the instance
  (**E814**). A disappearance therefore links the known instance, not an absent follow-up row.
- Spans: both columns valid or both null, `0 ≤ start ≤ end ≤` the document's byte length, endpoints on
  code-point boundaries (**E814**).

A report can describe several images, and a comparison can name different visits without forcing
their grids to coincide.

### 7.2 Revisions

For `supersedes` the source is the **newer** event version and the target the older; both are events
of one `record_id`. Different versions keep their own documents and availability. A version that
supersedes itself, two successors of one version (a branch), two predecessors of one version (a
merge), a cycle, a record whose versions the links do not order into one chain, or availability
bounds that definitely contradict the direction --- the newer version known available before the
older could have been --- are **E816**. When availability cannot order eligible revisions, the chain
does. Versions held by several fragments of a subject are merged by the task contract before
selection ([task-cache-1](task-cache-1.md) §3.3): a version two fragments hold has one content,
which the manifest records, and anything else --- a difference, an unrecorded duplicate, chains each
sound but contradicting once merged --- is **T305**, the subject's rows in error. No order is
invented to resolve it.

### 7.3 Attribution

A relationship used as a model input **MUST** be attested at the cutoff (§9): either it is the
immutable **structural** link from an event to its own payload --- a `document` event `describes`
its document, an `imaging` event `describes` its image --- or its `asserted_by_event_id` names an
event version eligible at the cutoff. Adding or replacing an owned payload needs a new event
version. An unattributed link supports navigation, never a claim about what was known when: a
report-to-lesion grounding drawn later is not available with the original report.

A single file cannot show when a link was written, so the rule is enforced where it can be: at the
amendment boundary. A writer amending a sample **MUST** refuse a structural link from an event
version the sample already held --- **E809**, as for a repeated event id: either would change a
version already written. Attached later, the payload would be an input at every cutoff after the old
version's availability, before the payload existed. A new version that supersedes the old one owns
it instead (§7.2), with its own availability.

### 7.4 Lesion assessments

An optional lesion assessment is a `kind = assessment` event with the local concept
`code_system = org.medh5.assessment`, `code = lesion_presence`, a `value_text` of `present`,
`absent`, `not_assessed`, `outside_fov`, `uncertain` or `resolved`, the assessed imaging
`timepoint_id`, and an `assesses` link to the instance (**E817**). Its provenance **SHOULD** name the
assessor. `resolved` requires an explicit source assessment: a missing follow-up mask or class-level
coverage is never exported as a clinical resolution, and this does not redefine the 1.0 tracking
helper's derived states.

## 8. Integrity, packing and amendment

Every dataset under `clinical/` --- byte buffers, offsets, validity masks and `clinical/meta` ---
**MUST** carry its 1.0 §13.1 digest (**E818**). `clinical/meta` is digested as a variable-length
string dataset; only the root `meta` keeps its special 1.0 §13.2 treatment.

The 1.0 §13.2 Merkle construction and its covered attributes are **unchanged**: clinical objects
define no attribute but the dataset `digest`, so clinical semantics are attested through dataset
content, the descriptor included. Checking the stored root alone is insufficient --- a verifier
recomputes the relevant datasets (**E701**); the conformance cases `E701-clinical-text-edited` and
`E701-clinical-descriptor-edited` change bytes under an unchanged stored root. Where the profile is
declared, `clinical/` is attested as the four 1.0 groups are: every path through it is its object's
own, the one its line names (1.0 §13.2, **E704**).

Declaring `1.1` and `clinical` changes `content_id`, so augmenting a 1.0 sample gives it a new
address, while its image and annotation payload digests are unchanged: their paths and decompressed
data did not change. Recompressing, packing and unpacking an unchanged 1.1 sample preserve its
`content_id`. All clinical references are sample-relative, so extraction from a collection rewrites
nothing. Copy-on-write amendment and atomic replacement (1.0 §14.4) remain the write model; live
in-place appending of a timeline is not introduced.

A collection holding a 1.1 member **MUST** declare an outer `medh5_version` at least that member's
(**E011**); the reference packer declares the newest member's. Mixed 1.0 and 1.1 members are
permitted, each keeping its own version, profiles and `content_id`; validation uses each member's
declarations, and extracting a 1.0 member does not promote it. A member of a higher minor is not
packed (§2.2).

## 9. Information availability and prospective selection

Selection is part of the profile's meaning --- what a record says about what was known when --- so
it is defined here, normatively, and implemented once, in the engine: the same rules decide what a
cutoff admits from Rust, Python and the command line. Tasks name a policy
([task-cache-1](task-cache-1.md) §3.4); this section defines what each policy admits.

### 9.1 Strict prospective selection

For cutoff `c` (`selection = strict_prospective`, the default):

1. **Availability.** Only event versions with known `available_hi ≤ c` are candidates.
2. **Versions.** Along each record's `supersedes` chain, the newest candidate is selected. A newer
   version definitely available after `c` (`available_lo > c`) leaves the earlier one usable. A
   newer version whose availability is unknown, or straddles `c`, makes the selection
   **uncertifiable**: the earlier version is still reported, but the row cannot be certified and a
   task excludes it with that reason --- the earlier record is never silently turned into an ordinary
   missing feature, because that missingness would itself depend on the later history. A selected
   version marked `entered_in_error` withdraws the record.
3. **Kind, plan, context.** The task's kinds filter. A `static` event is admitted unless the policy
   excludes static events, and is unordered. Otherwise the order time is the effective start (or the
   availability, with `order_by = available`); an event with no order time is excluded
   (`unknown_time`) --- unknown time is never placed on the timeline. An event that is `planned`, or
   whose effective start is not definitely at or before `c`, is a **plan**: excluded unless the
   policy admits plans, and never a completed outcome. A context window of width `w` admits order
   times within `[c − w, c]` (`closed`) or `(c − w, c]` (`open`); `contained` requires the whole
   bounds inside it, `overlaps` any part. The window bounds **every** admitted event's order time
   from below, plans included; its upper edge, `c`, is not a plan's, since a plan may lie after the
   cutoff. An ongoing interval is read with the fields of its eligible version, never its eventual
   end.
4. **Order and ties.** Admitted events are ordered by their order bounds. Events whose bounds overlap
   form one **tie group**: an uncertain or tied time is never fabricated into distinct exact times,
   and stable ids break ties only for storage. An event-count limit keeps the latest (or earliest)
   **timed** events, and a tie group straddling the boundary is kept or dropped **whole**, as the
   policy says; a limit of 0 keeps none. Static events are unordered under every ordering, so a limit
   never counts or drops them.
5. **Payloads.** Only payloads the selected versions attest are admitted (§7.3): the structural
   targets of selected `document` and `imaging` events, and the endpoints --- with any
   `target_annotation_id` --- of links asserted by a selected event whose event and document endpoints
   are themselves admitted. A future label, annotation, later-visit registration or whole-history
   embedding cannot enter through a secondary reference. Eligibility covers every input dependency,
   including the choice of image, region and crop centre.

Deterministic preprocessing may be computed later from exclusively eligible inputs with a fixed
recipe; its wall-clock time is not clinical availability. Dependence on a future visit, a future
label or held-out information is what makes a derived input ineligible.

### 9.2 `latest_provable`

The one named alternative keeps, per record, the newest version provably available at `c` and
ignores later revisions of unknown or straddling availability. Its status is `provable`, never
`certified`: it **MUST NOT** claim to be the newest source version at the cutoff. Any other
treatment of unknown or coarse times is a task policy of its own, recorded as such; a retrospective
assumption is not a prospective guarantee. An event's absence from the data never establishes its
absence in the patient.

### 9.3 Worked example

Hours from one subject origin (the file stores microseconds); conformance case
`clinical-worked-example`, and `docs/examples/clinical_longitudinal.py` builds a cohort of the same
shape:

| Event version | Effective | Available | Payload or relation | At hour 24 |
|---|---|---|---|---|
| `lab0` | −48 | −47 | Coded creatinine, numeric value with unit | Admitted (and inside any context of ≥ 72 h) |
| `ct0` | 0 | 1 | `describes` image `CT_tp0` on grid `ct_tp0` | Admitted, with `CT_tp0` |
| `report0_v1` | 0 | 4 | `describes` document `report0_text_v1`, record `report0` | Admitted, with its text |
| `report0_v2` | 0 | 48 | New document; `supersedes` `report0_v1` | Not yet available; v1 remains the record's version |
| `ct1` | 2160 | 2161 | `describes` image `CT_tp1` | Not available |
| `response1` | 2160 | 2184 | Lesion assessment at `tp1`, `assesses` instance 1 | Not available; a target only if a task says so |

The early laboratory value needs no artificial imaging timepoint. Reading the current final report
by the scan date would leak `report0_v2`. A CT crop centred on a lesion found only in `ct1` would leak
future evidence while returning only `CT_tp0` voxels; the task contract centres crops on eligible
annotations only. An absent follow-up is censoring under the task's policy, never a negative label.

## 10. Migration and interoperability

- **1.0 files stay 1.0.** Nothing migrates them automatically; a writer keeps writing 1.0 when no
  clinical capability is needed.
- **Augmentation** (`medh5 clinical augment`, `medh5.clinical.augment`) copies a sample's objects
  as stored --- paths and compressed payloads unchanged, so their digests are unchanged --- adds
  source-backed events, documents and links, and computes the new 1.1 identity. It never resamples.
  It reports what the records leave unknown (availability, precision); missing data stays unknown.
  If `clinical/` already exists and is not this profile's, augmentation refuses rather than
  overwrite or reinterpret an object 1.0's extension rules permitted.
- **Visit dates** (`days_from_baseline`) may inform day-precision imaging events when their meaning
  is known. They supply no report availability and no precise clock:
  `imaging_events_from_timepoints` leaves availability unknown and says so.
- **Downgrade** is a separately requested imaging projection (`medh5 clinical strip`): removing
  clinical evidence is reported as a loss, and the result is a different sample with its own
  `content_id`, written at the lowest version it needs.
- **MEDS** rows map from `subject_id`, the clock time, `code`, `value_num` and `value_text`; MEDS
  subject ids need an explicit mapping, and availability bounds, revisions, units and imaging
  references need extension columns. Relative times are never given invented absolute dates.
- **FHIR** effective time, issued time, coded values and absent-result reasons are useful import
  semantics; this profile is not a FHIR or OMOP serialization, and a converter discloses what it
  transforms or omits.

## 11. Validation

### 11.1 Levels

The 1.0 levels apply. `structural` adds the declaration, layout, descriptor and column rules (§3–§4:
E009 and E803 for declarations, E011, E801, E802, E804–E808, E819); `semantic` adds the record rules
(§5–§7: E809–E817, W914) and the one descriptor rule that reads `/meta`, a `shifted_utc` clock
without a declared de-identification (E802, §3); `integrity` adds E818 beside 1.0's digest codes,
E701–E704. A profile violation in a file of a version the validator implements is an error; the same
item in a higher minor is W913 (§2.2).

### 11.2 Diagnostic codes

1.1 adds these codes to the 1.0 §15.2 table, which they extend without redefining any code. The
complete table, generated from the engine's registry, is
[Diagnostic codes](../reference/diagnostic-codes.md).

| Range | Domain | Codes |
|---|---|---|
| `E0xx` | container | `E011` a collection's `medh5_version` is lower than a member's |
| `E8xx` | clinical | `E801` `clinical/meta` absent, not a scalar UTF-8 string, not JSON, or not canonical JSON; `E802` the descriptor fails its schema or its clock rules; `E803` clinical content present without the declared profile; `E804` a required table or column is absent, or a member is not one the profile defines; `E805` a column of the wrong dtype, rank or row count; `E806` malformed offsets or invalid UTF-8; `E807` a malformed or misplaced validity mask, or a null cell holding a value; `E808` a NaN or infinite value; `E809` an empty, malformed or repeated identifier; `E810` a value outside its vocabulary; `E811` time bounds incomplete, inverted or inconsistent with `temporal_type`; `E812` inconsistent value, code, comparator, unit or missing-reason fields; `E813` a reference that does not resolve in the sample; `E814` a span, instance-annotation or imaging-timepoint constraint violated; `E815` a document not owned by exactly one `document` event, or a `document` event owning more than one; `E816` a broken, branching, cyclic or availability-contradicted revision chain; `E817` a lesion assessment without its timepoint, its instance link or a permitted value; `E818` a clinical dataset without a digest; `E819` an attribute the profile does not define |
| `W9xx` | warnings | `W913` a higher minor version: only the supported projection was validated, or an unsupported item was ignored; `W914` a numeric clinical value without a unit |

### 11.3 Conformance

The corpus ([Conformance suite](conformance.md)) holds the 1.0 cases unchanged and 36 cases for this
profile: eight valid --- the §9.3 worked example, a one-visit history with years of laboratory values,
time uncertainty, UTF-8 text, a vocabulary wider than the class space, a mixed-version collection, a
higher-minor projection, and a value without a unit --- and twenty-eight invalid, at least one per new
error code. The task-and-cache contract is not part of the format corpus; the published suite carries
its own fixtures beside it ([task-cache-1](task-cache-1.md) §10), whose valid task states the selection
§9.3 describes for every row.

---

## Appendix A — Changes from the reviewed draft, and why

The draft this version implements left choices open or stated rules an implementation could not
honour as written. Each change below is normative, and was made because implementing the draft
showed it was needed.

| Clause | Change | Why |
|---|---|---|
| §2.1 | A writer writes the lowest version its content needs: 1.0 for imaging, 1.1 with `clinical`. | The draft said writers "SHOULD continue using 1.0" and that production defaults must not emit 1.1 before implementation; one rule states both, and keeps every 1.0 reader working for imaging-only data. |
| §2.2 | Unknown items of a higher minor are **W913**, the version itself is W913 at `/`, and an unrecomputable `content_id` is W913, not E702. | The draft required "unsupported features reported explicitly" and "integrity unsupported, not verified" but allocated no code. Reporting an item a later minor may define as an error would make the projection unusable; reporting nothing would claim conformance. |
| §2.3 | Known profiles are re-derived from the amended content; unknown profiles and higher minors refuse the amendment. | "A no-op amend MUST NOT silently remove an unsupported profile" was read against 1.0's rule that profiles are derived, not inherited. Keeping every declaration through any edit would keep false claims (an amendment removing the last voxel annotation would still declare `seg`); refusing the unknown is what the draft's safety requirement needs. |
| §3 | The descriptor is canonical JSON, checked as such (E801); `shifted_utc` requires a declared de-identification (E802). | §8 of the draft required canonical serialization without saying what a validator does with a non-canonical one; equal descriptors must digest equally. A shifted clock is meaningless without the shift it is relative to. |
| §3 | The declaration without the group, without events, or in a 1.0 file is E009; the group without the declaration E803. | The draft said "A recognized `clinical/meta` descriptor without that profile declaration is invalid" and that events need "at least one row", but gave no codes. |
| §4 | Each numeric column has one dtype (`int64` times, `float64` values, `uint64` spans); the column set is closed (E804) except in a higher minor's projection (W913). | "A fixed-width numeric dtype" let two writers store the same time as `int32` and `int64`, which digest differently; a closed set is what lets a 1.1 validator tell an unknown column from a misspelled one. |
| §4 | A mask for a required or omitted column, or with values other than 0/1, is E807. | The draft defined mask values but not where a mask may appear. |
| §4 | A UTF-8 column's `offsets` are little-endian, as the numeric columns are (E805). | Found by the re-audit of 2.0: the table said it of the numeric columns only, and a column whose offsets were stored big-endian validated --- correct to a reader converting through HDF5, and not to one reading the buffer as the format lays it out. |
| §5.1 | `W914` for a numeric value without a unit. | The draft's "SHOULD carry a unit" needed a reportable form that is not an error. |
| §5.3 | An imaging event's `timepoint_id` must equal its image's grid timepoint (E814). | Stated in the draft without a code. |
| §6 | A document is owned by **exactly one** `document` event, and a `document` event owns **at most one** document (E815). | "MUST be linked from an immutable `document` event" allowed two owners, and then the document's availability would be ambiguous. The converse was found implementing the cache contract: two texts under one event would share one availability, status and revision chain --- revising either would force a copy of the other --- and an event-level feature (task contract §7) would no longer name one text. |
| §7.2 | Merging chains (two predecessors) and a record whose versions the links do not order into one chain are E816, beside branches, cycles and contradicted availability. | The draft required "an acyclic, unambiguous `supersedes` chain"; a record with two unrelated versions is ambiguous in exactly the way selection cannot resolve. |
| §7.3 | A writer amending a sample refuses a structural link from an event version the sample already held (E809). | Found by the fourth audit of 2.0: "adding or replacing an owned payload needs a new event version" had no point of enforcement, and one file cannot show when a link was written. Text attached in an amendment to an inherited text-less document event was owned by a version available a day before the text existed, and selection admitted it at that cutoff. |
| §8 | A collection's outer version must be at least each member's (E011); the packer declares the newest member's, and refuses a higher-minor member. | The draft required `1.1` outside a 1.1 member but gave no code, and said nothing of a member newer than the packer. |
| §9 | Strict selection, the plan rule, tie groups, the event-limit boundary and payload attestation are defined here, normatively, and the task contract only names a policy. | The draft placed them in a companion "not an additional payload requirement". They define what a record *means* about what was known when; two implementations of one selection must agree, so they belong to the profile. |
| §9.1 | An event with no order time (`unknown`, or without the time `order_by` names) is excluded as `unknown_time`. | "Unknown/coarse times may be excluded or handled by an explicitly named alternative policy": strict excludes them, and the exclusion is counted. |
| §9.1 | A selected `entered_in_error` version withdraws the record; an uncertain later revision makes the row *uncertifiable* while still reporting the earlier version. | Draft step 2 said to exclude the row and also that an earlier version "remains usable"; the preflight needs both facts --- why the row is excluded, and what it would otherwise have read. |
| §9.1 | The context window bounds every admitted event's order time from below, plans included; a plan may lie after the cutoff. | Found by the fourth audit of 2.0: read literally, `[c − w, c]` excluded every future plan under `order_by = effective`, while the implementation applied no window to plans at all --- so a stale order from 100 days before a 7-day window was admitted beside the week. The lower edge is what a window is for; the upper is the cutoff, which plans exist to lie beyond. |
| §9.1 | An event-count limit counts **timed** events only, and a limit of 0 keeps none of them; a static event is unordered under every `order_by`. | Found by the 2.0 audit: ordered by availability, a static fact became a timed event a limit could drop, though step 3 calls it unordered and the task contract limits timed events; and 0, which the task schema admits, had no defined boundary. |
| §11 | Diagnostic codes E011, E801–E819, W913 and W914 are allocated, each with a conformance case. | The draft did not allocate codes for unimplemented rules; they are now implemented. |
| §11.1 | `semantic` checks E802's rule for a `shifted_utc` clock, which needs `/meta`'s de-identification record; the rest of E802 stays `structural`. `integrity` adds E818 beside every 1.0 digest code, E704 included. | Found by the review of 2.0 before its release: the clause placed all of E802 at `structural`, though that rule compares the descriptor with `/meta`, a cross-reference the reference validator checks at `semantic` with the record rules --- so a validator built from the text reported at `structural` a defect the reference reports only from `semantic` on; and it named 1.0's integrity codes E701–E703, written before 2.0 allocated E704. |
| §2.1 | An amendment that removes the clinical profile lowers the version: it is the §10 imaging projection, written to a new file or in place of its source. | Found by the review of 2.0 before its release: "an amendment never lowers the version" contradicted the reference writer, whose `drop_clinical` in an amendment rewrites a 1.1 sample in place as 1.0 --- the projection §10 describes, a different sample with its own `content_id`. |
| §3 | A `clinical` group is *recognised* when its `meta` holds a JSON object whose `schema` is `medh5.clinical/1`. Undeclared, a recognised group is E803 in any version, and from 1.1 on so is any object named `clinical`; in a 1.0 file an unrecognised one is an extension. | E803 and the §2.3 amendment rule turned on "a recognised `clinical/` group", which nothing defined, so a validator could not tell a 1.0 file's own extension from an undeclared profile; the reference validator's rule --- the name reserved from 1.1 on, the descriptor deciding before --- was written nowhere. |
| §3, §7.2 | No conversion between clocks is defined: a subject's fragments join only on one clock (T304), and the versions they share are merged by the task contract, which refuses what does not agree (T305). | The text deferred to "a verified conversion" the task contract never defined, and called cross-fragment revisions "resolved" before selection, where the contract refuses them; an implementation looked for a conversion and a resolution that do not exist. |
| §4, §5.1 | The identifier columns are listed, table by table, and `record_id` takes the 1.0 id syntax, as `event_id` and `document_id` do. | "Identifier columns are non-empty" named no column, and only two ids had a syntax, while the reference writer and validator refuse an empty `unit` or `prov` and a `record_id` of `rec 1` (E809): which empty strings and which record ids were errors could not be read from the text. |
| §7.1 | An `instance` endpoint resolves through the `instance_ids` dataset of an annotation of any kind, and is E813 when none holds it; a `target_annotation_id` that does not hold it is E814. | "An instance-bearing annotation" was undefined, and the clause gave E814 for an instance found nowhere, which the reference validator reports as the unresolved endpoint it is (E813). |

## Appendix B — Schemas

| Schema | Published at | Validates |
|---|---|---|
| `medh5-clinical-1.schema.json` | the engine (`crates/medh5/data/`), the published conformance suite | `clinical/meta` (`#/$defs/descriptor`) and the logical-record bundle `augment` reads and `export` writes (`#/$defs/records`) |
| `medh5-task-1.schema.json` | as above | Task manifests, [task-cache-1](task-cache-1.md) §3 |
| `medh5-cache-1.schema.json` | as above | Cache dependency manifests, [task-cache-1](task-cache-1.md) §7 |
