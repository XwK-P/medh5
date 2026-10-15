# Task and cache contract 1 — `medh5.task/1`, `medh5.cache/1`

**Status: Final**, implemented by the engine (`medh5::companion`), the Python package
(`medh5.task`, `medh5.cache`, `medh5.torch.clinical`) and the command line (`medh5 task`,
`medh5 cache`). This contract is **separately versioned from the MEDH5 format**: a `.medh5` file is
readable without any of it, and nothing here is a payload requirement of
[MEDH5 1.1](medh5-1.1.md).

A sample is the patient source. Training needs three things the format deliberately does not hold,
each versioned on its own and each **pinned to the exact source versions it read**:

- a **task manifest** (`medh5.task/1`, one JSON document): the sources, the subjects and their
  partitions, the rows (a subject at a cutoff), what a row may read, and the target (§3);
- the **row views** a manifest's preflight produces: per row, what the selection policy admits, the
  image filling each modality slot, and the label or why there is none (§4–§6);
- **feature caches** (`medh5.cache/1`, one HDF5 file): derived features with their dependency
  manifest and checksums (§7–§8).

Findings carry **T-codes** (§9), distinct from the format's E/W codes: a task can be wrong about
perfectly valid files. Keywords are those of 1.0 §1.2; times are microseconds on the subject clock
(1.1 §3).

## 1. Scope

In scope: addressing sources, splitting subjects, choosing what a row may see at its cutoff, labelling
forecasting targets with explicit censoring, filling stable modality slots, and caching features
without letting a cache redefine, outlive or leak past its sources.

Out of scope: model architectures, tokenizers and encoders (a cache names its encoder; it does not
standardize one), resampling (a slot reads an image on its own grid; geometry is never invented), and
any change to a `.medh5` file — none of these operations writes to a source.

## 2. Source references

A **source reference** names one sample — a standalone file or a member of a `.medh5c` collection
— at one version:

| Field | Meaning |
|---|---|
| `source_id` | The id the manifest and reconciliation records use for this source (1.0 id syntax, plus `@` and `:`) |
| `uri` | Where the bytes are: a path, absolute or relative to the manifest's directory, or a `file:` URI (RFC 8089) — its host empty or `localhost`, its path percent-decoded |
| `sample_key` | The member of a collection, or null for a standalone file |
| `content_id` | **The pin**: the 1.0 §13.2 address of the sample version the reference names |
| `local_subject_id` | The sample's own `identity.subject_id`, as the manifest maps it to a subject; when absent, the subject's own `subject_id` |

A URI is a locator, not identity. A reference resolves to a sample, and the sample counts only while:

1. its stored `content_id` is the pin;
2. every dataset of an object the root speaks for — `grids/`, `images/`, `annotations/`,
   `transforms/`, and `clinical/` where the profile is declared — carries its digest, and is
   reached there only at its own path, the one its line names (1.0 §13.1, §13.2; 1.1 §8);
3. its `content_id` **recomputes** to the pin from its stored digests;
4. the actual bytes of its clinical datasets match their digests — every dataset's, with `--deep`.

Steps 2 and 4 are not redundant with steps 1 and 3: `content_id` is a Merkle root over *stored*
digests, so a dataset added without one — a column the file did not have — changes no root, and
neither does an edit that leaves the stored digests alone. A dataset whose bytes cannot be read —
a damaged chunk — does not match. A failure of any step is **T302**; a source that does not open,
or names a member that does not exist, **T301**. Re-pinning is explicit: a changed source is never
silently accepted (`SourceRef.pin`), and a sample that fails step 2 cannot be pinned at all.

**Opening a source.** An implementation **MUST NOT** open a source that is not a regular file: a
pipe or a device is no sample, and opening one can block. A source that fails any of steps 1–4 is
read no further: it adds no identity, clock, event, document or link to its subject (§3.3), and no
finding but its own T301 or T302 speaks of its content — T302 names the pin, not what the file
stores. `reconcile` refuses a source that fails them (Appendix A).

## 3. The task manifest

### 3.1 The document

One JSON document, validated by `medh5-task-1.schema.json` (**T101** when it is not JSON or fails its
schema):

| Member | Content |
|---|---|
| `schema` | `"medh5.task/1"` |
| `task` | `{"id", "version", "description"?}` — the task's identity |
| `identity_namespace` | The namespace subject ids, and clock ids, are scoped by |
| `policy` | The selection policy (§3.4); strict prospective when absent |
| `slots` | The modality slots (§3.5) |
| `target` | The target (§5), or null for an unlabelled task |
| `split` | `{"set_id", "partitions"}` — **the training partition first** |
| `subjects` | `[{"subject_id", "partition"?, "clock_id"?, "sources": [...], "reconciled"?: [...]}]` |
| `rows` | `[{"row_id", "subject_id", "cutoff_us"}]` |
| `fingerprint` | Optional: the manifest fingerprint (§3.2); when present it **MUST** match (**T103**) |

An implementation normalises a manifest by filling every default, so two spellings of one task are
one task. The **normal form** of a manifest writes every member, null where a nullable one is
absent, and keeps every list in the manifest's order:

- `schema`; `task` as `{"id", "version"}`, with `description` only when it is given;
  `identity_namespace`;
- `policy` with all eleven members of §3.4, whether or not the manifest has a `policy`;
- `slots`, `[]` when absent, each slot `{"name", "modality", "required", "patch", "roi",
  "classes"}` (§3.5);
- `target`, null when absent, or `{"id", "version", "event", "positive", "negative", "horizon_us",
  "min_follow_up_us", "censoring", "exclude_prevalent"}` (§5), whose `event` is `{"code_system",
  "code"}` with `kind` only when it is given;
- `split` only when it is given;
- `subjects`, each `{"subject_id", "clock_id", "partition", "sources", "reconciled"}` with
  `reconciled` `[]` when absent, each source `{"source_id", "uri", "sample_key", "content_id",
  "local_subject_id"}` (§2) and each record `{"event_id", "digest", "sources"}`;
- `rows`, each `{"row_id", "subject_id", "cutoff_us"}`;
- `fingerprint` only when the manifest declares one.

Times, voxel counts, class ids and `max_events` stay integers.

### 3.2 Identity and fingerprints

A fingerprint is `"sha256:" + hex(sha256(c))`, `hex` lowercase, where `c` is the UTF-8 canonical
JSON (1.0 §5.1) of the value the table gives. Members are written as the normal form (§3.1) writes
them, and "sorted" is by code point.

| Fingerprint | The value hashed | Identifies |
|---|---|---|
| **task** | `{"schema", "task": {"id", "version"}, "identity_namespace", "policy", "slots", "target"}` — `task` without its `description` | The task *definition* — not its rows, not its sources |
| **manifest** | The normal form without `fingerprint` | One frozen task instance, rows and pins included |
| **row** | `{"task": <task fingerprint>, "namespace": <identity_namespace>, "subject_id", "sources": <the pinned content_ids of every source the manifest gives the subject, sorted>, "cutoff_us"}` | One example. A cohort-membership digest or a URI alone pins no data |
| **subjects digest** | The array of `[identity_namespace, subject_id]` arrays, one per subject in the partition, sorted | Who a partition holds — what learned preprocessing records (§7.3) |

### 3.3 Subjects, fragments and splits

**Patient splits precede window construction.** A partition belongs to a *subject*, and every row of
the subject inherits it, so no window puts one patient on both sides of a split. With a `split`,
every subject names one of its partitions (**T202**); without one, none does. A subject is declared
once and every row names a declared subject (**T201**); row ids and source ids are unique (**T204**).
One source belongs to one subject, and two subjects never pin one sample — an overlapping snapshot
would cross the split (**T203**).

A subject may have several **fragments** — files or members, one per site or per export — on one
clock. Preflight (§4) opens them all, checks each source's identity against the subject it is mapped
to (**T303**) and their clocks against each other and the subject's `clock_id` (**T304**), and merges
their events. An event version present in several fragments **MUST** have one content, and the
manifest records it — `reconciled: [{"event_id", "digest", "sources"}]`, the digest being the
fingerprint (§3.2) of the version as `medh5-clinical-1.schema.json#/$defs/event` spells it, all
nineteen members present (null where the column is null, each pair of bounds `[lo, hi]`, `value_num`
a float), and `sources` the `source_id`s of exactly the fragments that hold it — before selection; a difference, a duplicate with no record, a record whose `sources` are not the holders, a
second record of one event, or a record of an event no two fragments hold is **T305**. Documents
present in several fragments **MUST** be identical (**T305**). A `document` event's content includes the
document it owns: its holders **MUST** own one text — the same media type, text, language and source
type, whatever id each fragment gives it — and the merged version owns that one document (1.1 §6;
a document event owns at most one). A version several fragments hold is merged from the one whose
pinned `content_id` is smallest, never from the first the manifest lists: the row fingerprint (§3.2)
sorts the pins, so nothing a row reads may depend on their order. `medh5 task reconcile` writes the
records.

Preflight checks each fragment's revision chains as part of its clinical tables, and the merged
history's once: chains each sound but contradicting once merged — a version superseded in one
fragment by one version and in another by a different one — are the subject's **T305**, its rows in
error, never the whole preflight's failure.

### 3.4 Selection policy

The policy names how a row's inputs are chosen; what each choice *means* is defined by MEDH5 1.1 §9,
and implemented once, by the engine.

| Member | Default | Values |
|---|---|---|
| `selection` | `strict_prospective` | `strict_prospective`, `latest_provable` (1.1 §9.2) |
| `order_by` | `effective` | `effective` (clinical time), `available` |
| `context_us` | null | A window length before the cutoff; null reads the whole history |
| `context_boundary` | `closed` | `closed` (`[c − w, c]`), `open` (`(c − w, c]`) |
| `uncertainty` | `contained` | `contained` (all of an event's bounds in the window), `overlaps` (any of them) |
| `kinds` | null | The event kinds a row may read; null admits every kind |
| `plans` | `false` | Whether known future orders and plans are read, as plans |
| `static` | `true` | Whether `static` events are read (they still need availability) |
| `max_events` | null | A limit on timed events |
| `keep` | `latest` | `latest`, `earliest`: which end of the history a limit keeps |
| `ties` | `keep_group` | `keep_group`, `drop_group`: a tie group straddling the limit is kept or dropped whole |

An inconsistent policy, slot or target — a slot declared twice, a value both positive and negative,
a follow-up longer than the horizon — is **T102**; a value outside a member's vocabulary, such as an
unknown `roi`, fails the schema (**T101**).

### 3.5 Modality slots

| Member | Default | Meaning |
|---|---|---|
| `name` | — | Stable across the cohort: never a filename or a visit-specific image id |
| `modality` | — | The image `modality` that fills it |
| `required` | `false` | A row with no eligible image for a required slot is excluded (`missing_required_slot`) |
| `patch` | null | The region read around the centre, in voxels of the image's **own** grid; null reads the whole volume |
| `roi` | `center` | `center` (voxel `⌊n/2⌋` on each spatial axis of `n` voxels) or `eligible_instances` (the centre of the first eligible instance) |
| `classes` | `[]` | Class ids read as label supervision from voxel annotations on the image's grid (§6) |

On a spatial axis of `n` voxels, a `patch` of `p` around centre `c` starts at `c − ⌊p/2⌋`, shifted
inside the axis — to `0`, or to `n − p` — so the patch keeps its size wherever the axis allows; an
axis no longer than the patch is read whole and padded, `⌊(p − n)/2⌋` voxels before it and the rest
after. Padding is never valid (§6).

A slot is filled per row by the **newest eligible image** of its modality: an image an admitted
`imaging` event `describes` (1.1 §7.3). Newest is by that event's order bounds (1.1 §9.1): the
greater upper bound, then the greater lower bound — a `static` version, which has none, is older
than any timed one — then the greater event id; then — one version held by several fragments, or
one describing several images — the smallest pinned `content_id` of the source holding the image,
then the smallest image id. With the merge rule of §3.3, a row's selection and fills are a function
of its fingerprint. An eligible image of another visit is a different fill, not an error; no image
is resampled onto another's grid, and no registration is invented.

With `roi = eligible_instances` the centre comes from the **eligible** (1.1 §7.3, §9.1 step 5)
`instances` and `boxes` annotations, a mask asserted by its own eligible procedure included, on the
image's grid that hold an object — an annotation drawn
after the cutoff never chooses a crop. The first of them, in byte order of annotation id, gives its
first object by `instance_id` (a `boxes` annotation without `instance_ids` numbers its boxes from 0,
in row order), and the centre is voxel `⌊(lo + hi)/2 + ½⌋` on each spatial axis of that object's box
in the grid's index space (1.0 §8.1) — a box stored in world space is carried there first, and an
annotation whose boxes cannot be is passed over. `center_fallback` records that there was none, and
the crop is then centred as `center` is.

### 3.6 Rows

A row is `{"row_id", "subject_id", "cutoff_us"}`. Any number of rows per subject, at any cutoffs, is
permitted; each is one example under the subject's partition.

## 4. Preflight

Preflight reads no voxel and uses no report: text is read only to verify it — its digest (§2) and its
UTF-8 (1.1 §4), a bounded slab at a time — and to compare a document two fragments both hold
(§3.3). It decides, per row, before any batch is built:

1. the manifest's own findings (T1xx, T2xx); a manifest with any leaves every row in `error`;
2. per subject: every source opened and checked (§2), identities, clocks and duplicates reconciled
   (§3.3), and every source a **full** implementation of its version — a higher minor's projection
   or invalid clinical tables cannot supply certified inputs (**T306**);
3. per row: the selection at the cutoff over the subject's merged history (1.1 §9), the slot fills
   (§3.5) and the target label (§5).

Each row gets one status and its reasons:

| Status | When |
|---|---|
| `eligible` | Certified (or `provable` under `latest_provable`) inputs, every required slot filled, a usable target |
| `uncertifiable` | A later revision's availability is unknown or straddles the cutoff: `uncertain_revision:<record>` |
| `excluded` | `missing_required_slot:<slot>`, `prevalent_target`, `censored` (with `censoring = exclude`), or `no_clinical_source` |
| `error` | A finding about the row's sources or the manifest; its findings are its reasons |

Any finding makes the task unfit to train on as a whole (`ok = false`); a training frontend refuses
it unless told to train on the unaffected rows only.

A row's view holds: the selection (status, admitted versions with their order bounds and tie groups,
attested links and payloads, uncertain records, exclusion counts), the selected event versions with
the fragment each came from, each slot's fill, and the target label. The rows of a subject share its
merged history: a view may name its versions by position in that history rather than copy them, which
is what keeps a preflight of many cutoffs linear in the history, not in rows times history. A source
that cannot be read — a damaged chunk in its clinical tables, or in the grid or object table a slot
reads (never a voxel or a mask) — is that source's finding (**T302**, **T306**), its subject's rows
in error, never the whole preflight's failure.

## 5. Targets

| Member | Default | Meaning |
|---|---|---|
| `id`, `version` | — | The target's own identity, part of the task fingerprint |
| `event` | — | `{"kind"?, "code_system", "code"}`: the concept the target reads |
| `positive` | — | `value_text`s that are the outcome |
| `negative` | `[]` | `value_text`s that are a verified absence of it |
| `horizon_us` | — | The window `(c, c + h]` an outcome must fall in |
| `min_follow_up_us` | `horizon_us` | A negative needs an observation at least this long after the cutoff (rule 4). With the default, only an observation whose effective start is exactly `c + h` can be one, so nearly every row without a positive is censored: give it whenever negatives matter |
| `censoring` | `censor` | `censor` keeps an unlabelled row with its target unobserved; `exclude` drops it |
| `exclude_prevalent` | `true` | Exclude rows whose outcome had already occurred by the cutoff |

The target is read from the **full history** — the final version of each record, `entered_in_error`
withdrawn — because it lies after the cutoff by definition; it can live in the same file as the
inputs, and it never enters them. It reads **occurrences**: a final version whose `status` is
`cancelled` did not happen, and is neither a positive nor a negative; one that is `planned` is not
known to have happened, so a planned positive makes no row positive or prevalent, and a planned
negative observes nothing. In order:

1. **prevalent** — a positive whose effective start is definitely at or before `c`
   (`exclude_prevalent`);
2. **positive** — a positive whose effective start is definitely inside `(c, c + h]` — its bounds
   `[lo, hi]` have `lo > c` and `hi ≤ c + h` — the earliest such naming the outcome: value `1.0`;
3. **censored** — a positive whose time straddles either edge of the window, or is unknown, or a
   planned positive whose time may fall inside the window: the outcome cannot be placed, and is not
   a negative;
4. **negative** — a negative whose effective start is definitely inside the window and definitely
   at or after `c + min_follow_up` — its bounds `[lo, hi]` have `lo > c`, `hi ≤ c + h` and
   `lo ≥ c + min_follow_up` — the latest such naming the observation: value `0.0`;
5. **censored** otherwise — insufficient follow-up is not a negative label.

## 6. Input views

What a row admits is what **this manifest's** preflight says. A preflight is of one manifest
(`manifest_fingerprint`), not of a task definition: two instances of one definition with different
splits, rows or pins share the task fingerprint, and building one's views from the other's preflight
would put one's validation subjects in the other's training rows. A frontend refuses such a preflight
(**T404**) — for its views and for anything it fits on them (§7.3).

A frontend reads **only** what a row admits, and only when it builds the batch:

- **events** — the selected versions, in clinical order, as the row's sequence; their count varies
  per row;
- **documents** — the text (or a cached feature, §7) of documents the selection admits, i.e. owned
  by a selected `document` event. A cached document feature comes from an **event-level** cache,
  validated against the task (§8: a cache fitted on another split is **T405**); a patient-level
  cache is a row's feature and is read only as that row's (**T404**);
- **images** — each slot's fill, read as the region of §3.5 on the image's own grid, with the
  region's validity: inside the image and inside its 1.0 §4.4 valid region, never the padding;
- **labels** — supervision for a slot's classes, read from the voxel annotations on the slot
  image's grid that label or examined one of them. Like the target, supervision may come from after
  the cutoff, and it never enters an input; coverage (1.0 §11.3) comes with it, so a class nobody
  examined is not a negative.

Masks stay distinct, because they mean different things to a loss: sequence **padding**, modality
**availability** (was the slot filled), voxel **validity** (field of view), annotation **coverage**
(was the class examined), and target **observation** (is the label observed). A zero-filled missing
modality is not an observed negative.

Time keeps the meaning 1.1 §5.2 gives it. A frontend **MUST NOT** collapse an input's time to one
instant chosen from its bounds: each time an input carries — effective start, effective end,
availability — is given as both bounds (relative to the cutoff, in whatever unit), with whether it is
known; an unknown time is marked unknown, never zero. A `static` event stays distinct from one of
`unknown` time, events of one **tie group** — a connected component of the overlap relation on their
order bounds (1.1 §9.1) — stay marked as such (their order in a sequence is a storage tie-break, not
evidence), and a plan admitted by `plans` stays marked as a plan. Missingness stays
apart from measurement: an absent value, a value known only as a bound (`value_comparator`) and a
`missing_reason` are each represented, never read as a measured value. Every availability an input
carries is at or before the cutoff — what strict selection admits — and only what a version
available at the cutoff itself recorded about later (a plan's start, a course's recorded end) lies
after it.

A frontend reads each source as the version its row pins: a source whose `content_id` is no longer the
pin when a batch is built — the file was replaced after preflight — is refused (**T302**), never
read in its place. A read already in progress finishes on the bytes it opened; every new one
re-checks the source. That check is §2's step 1 alone: it compares the *stored* `content_id`, and
re-hashes nothing. The bytes are verified by preflight (step 4: the clinical datasets always, every
dataset with `--deep`), so an edit made in place after preflight that leaves the stored digests and
`content_id` as they were is found by running preflight again, not by building a batch.

## 7. Feature caches

### 7.1 Layout

One HDF5 file, conventionally `*.medh5cache`:

```text
/                    medh5_companion = "medh5.cache/1", manifest_digest
├── manifest         scalar UTF-8: the dependency manifest, canonical JSON
└── entries/<id>     one feature array per entry, with its 1.0 §13.1 `digest`
```

`manifest_digest` is `"sha256:" + hex(sha256(b))`, `hex` lowercase, where `b` is the bytes of the
`manifest` dataset as stored (UTF-8). The manifest is validated by `medh5-cache-1.schema.json`:

| Member | Content |
|---|---|
| `schema` | `"medh5.cache/1"` |
| `level` | `event` or `patient` |
| `encoder` | `{"name", "revision", "tokenizer": {"name", "revision"}?}` — an **immutable** revision: a content hash or a commit, never a moving tag |
| `preprocessing` | Everything else that determines the output |
| `output` | `{"dtype", "shape", "pooling"?, "chunking"?}`; every entry has this dtype and shape (**T402**) |
| `task_fingerprint` | The task a patient-level cache was built for (required at that level) |
| `selection` | The `selection` it read under (§3.4) — `strict_prospective` or `latest_provable` — or null |
| `fitted_on` | For learned preprocessing: `{"task_fingerprint", "set_id", "partition", "subjects_digest"}` |
| `entries` | `[{"entry_id", "sources", "event_id"?, "document_id"?, "row_id"?, "row_fingerprint"?, "cutoff_us"?, "event_versions"?, "digest"}]` |

Every entry names **every source version it read**, as `{uri, sample_key, content_id}`. The full
sample `content_id` is the source key: a conservative, sufficient one — any change to the sample
makes its entries stale. A finer dependency key would need its own specification.

An entry copies its task's source reference, so a relative `uri` is relative to the **task
manifest's directory**: a validator given the task resolves it there, and one given none against
the base it is given, by default the cache's own directory.

Entry ids are unique, and so is what a reader looks an entry up by (§7.2): in an event-level cache
the `content_id` of its (first) source with its `event_id` — one feature per event version — and in
a patient-level one its `row_id` — one feature per row. A manifest repeating any of them is corrupt
(**T401**): a lookup would serve one entry, and the other's checksum would never be compared. A
writer refuses an entry that would repeat one, or make the manifest fail its schema.

### 7.2 Levels

- **Event level**: one feature per event version of a pinned source (or the document it owns),
  conventionally entry `e` + 24 hex digits of `sha256(content_id + "\n" + event_id)`; every entry names
  its `event_id`, and **no row**: no `row_id`, `row_fingerprint`, `cutoff_us` or `event_versions`.
  An event version is immutable, so its feature is free of any cutoff and shared by every row —
  each row's selection decides which of them it may read; an entry declaring a row's history would
  say otherwise, and a reader looking it up by event would serve it to every row. The `event_id`
  **MUST** be an event version of the entry's source, and a `document_id` the document that version
  owns through its structural `describes` link ([1.1](medh5-1.1.md) §7.3) (**T407**): a row admits a
  version, so the feature it reads is that version's, and a reader of a document feature checks the
  entry's `document_id` against the document the selected version owns before using it.
- **Patient level**: one feature per row, pinning the row — its id and its fingerprint
  (`row_fingerprint`, §3.2) — its cutoff and the exact event versions it encodes
  (`event_versions`), under the task it was built for (`task_fingerprint`). Every entry names all of
  them, and its sources are **its row's subject's**: event ids are local to a sample, so two
  subjects' rows can admit the same ids at one cutoff, and only the pins say whose history a feature
  encodes. The fingerprint is what binds a feature to its inputs: a source added to the subject
  changes what a slot reads without changing the cutoff or an admitted version, and changes the
  fingerprint; reordering the sources changes neither (§3.3, §3.5).

A schema-valid manifest holds what its level requires (**T401** otherwise). Each level answers its own
question: a reader looks event features up by `(content_id, event_id)` in an event-level cache only,
and row features by row in a patient-level one — a patient-level entry that also names an event
encodes its row's history, never that event alone.

### 7.3 Learned preprocessing

Normalisation statistics, code bins, vocabularies and anything else fitted to data record `fitted_on`:
the task fingerprint, the split, the partition and that partition's subjects digest (§3.2). They are
fitted on the **training partition** — the split's first — and nothing else, and every artifact
recording `fitted_on` is held to the one comparison of §8 (T405), a vocabulary as a cache.

## 8. Cache validation

Validation never modifies a source. A path that names no file is refused, as opening it is: there
is no cache to report on. Two failures are told apart, because they need different fixes:

- **stale** — **T403**: an entry's source no longer has the content it pins, or cannot be reached.
  The cache is right about a sample that changed; rebuild the entry.
- **corrupt** — **T401** (the manifest: not the companion, checksum mismatch, not JSON, fails its
  schema) or **T402** (an entry's payload fails its own checksum, or has the wrong dtype or shape).
  The cache is wrong about itself; rebuild it.

Given a task (and its preflight), validation also checks that the cache belongs to it:

- **T404** — built for another task (`task_fingerprint`), or a patient-level entry for a row the
  task does not have, at another cutoff, for another version of its row (`row_fingerprint`: a source
  of its subject added, removed or re-pinned), or reading a source version its row's subject does
  not pin (compared by `content_id`, so a relocated copy is the same version); and a patient-level
  entry of a row the task's preflight leaves in `error`, whose admitted versions are unknown — one
  T404, never T406;
- **T405** — learned preprocessing fitted on another task, another split (`set_id`), another
  membership, or a partition other than the task's training partition;
- **T406** — a patient-level entry encodes event versions other than those its row admits at its
  cutoff, compared as a set (the order an entry lists them in is no part of it): a whole-history
  embedding, or one built under another policy, is inadmissible.

With or without a task, an event-level entry is checked against its source:

- **T407** — the entry names an event version its source does not hold, or a document that
  version does not own (§7.2), or its source's clinical tables cannot be read to say: a pin compares
  digests only, and an entry that cannot be verified is not accepted. Joined to another version's
  text — a later revision's — the feature would be read by every row that admits the version it
  names, before that text existed.

A stale or unverifiable cache is rejected and rebuilt; it never redefines or invalidates the source.

A verdict is about the bytes validated. A reader that admits a cache by validating it reads that
cache only while its `manifest_digest` is the one validated, compared whenever it opens the file
again — a first read, a forked worker, an unpickled dataset — and refuses another (**T404**):
a cache rebuilt at the same path for another task is otherwise read under the first one's verdict.

## 9. Finding codes

Twenty codes, distinct from the format's E/W codes: a task can be wrong about perfectly valid files.
The engine's `CODES` table (`crates/medh5/src/companion/mod.rs`) is their source, and a test holds
this table to it.

| Code | Meaning |
|---|---|
| <a id="t101"></a>`T101` | the manifest is not JSON, or fails its JSON Schema |
| <a id="t102"></a>`T102` | the manifest's policy, slot or target definition is inconsistent |
| <a id="t103"></a>`T103` | a declared fingerprint does not match the manifest |
| <a id="t201"></a>`T201` | a subject is declared twice, or a row names an undeclared subject |
| <a id="t202"></a>`T202` | a subject's partition is not one of the split's partitions |
| <a id="t203"></a>`T203` | one source belongs to two subjects, or two subjects pin one sample |
| <a id="t204"></a>`T204` | a row id or a source id is not unique |
| <a id="t301"></a>`T301` | a source cannot be opened, or names a collection member that does not exist |
| <a id="t302"></a>`T302` | a source's content no longer matches its pinned content_id |
| <a id="t303"></a>`T303` | a source's identity.subject_id is not the subject the manifest maps it to |
| <a id="t304"></a>`T304` | a subject's fragments disagree about their clock, or it is not the subject's |
| <a id="t305"></a>`T305` | an event or document present in several fragments differs, or is not reconciled |
| <a id="t306"></a>`T306` | a source is a version or profile this engine cannot certify, or its clinical tables are invalid |
| <a id="t401"></a>`T401` | a cache's manifest is corrupt, or fails its schema |
| <a id="t402"></a>`T402` | a cache entry's payload is corrupt, or has the wrong dtype or shape |
| <a id="t403"></a>`T403` | a cache entry's source changed or cannot be reached: the entry is stale |
| <a id="t404"></a>`T404` | a cache or preflight is of another task, manifest, row or row version, or a validated cache changed |
| <a id="t405"></a>`T405` | learned preprocessing was fitted on a split other than the task's training partition |
| <a id="t406"></a>`T406` | a cache entry is not admissible as input at the row's cutoff |
| <a id="t407"></a>`T407` | an event-level cache entry names an event its source lacks, or a document that event does not own |

## 10. Schemas and implementations

| Artifact | Where |
|---|---|
| `medh5-task-1.schema.json`, `medh5-cache-1.schema.json` | The engine (`crates/medh5/data/`) and the published conformance suite |
| Fixtures | `companion/` in the published [conformance suite](conformance.md#the-task-and-cache-fixtures): one invalid manifest per T-code from T101 to T306, two valid tasks whose every row's selection, slot and label is stated, and under `companion/caches/` nine feature caches: one per T-code from T401 to T407, and a valid cache of each level. `medh5 conformance score --companion` scores an implementation's reports on them |
| Engine | `medh5::companion` — `task`, `view` (preflight), `cache`, `source` |
| Python | `medh5.task`, `medh5.cache`, `medh5.torch.clinical` ([Train on clinical tasks](../guides/clinical-training.md)) |
| Command line | `medh5 task validate`, `medh5 task preflight`, `medh5 task reconcile`, `medh5 cache validate` ([Command line](../reference/cli.md)) |

## Appendix A. Errata

Corrections to `medh5.task/1` and `medh5.cache/1` that add no field and change no normal form,
fingerprint or default. Each says what an implementation must refuse to open or must not report,
never what a manifest must contain.

| Clause | Change | Why |
|---|---|---|
| §2 | A source that is not a regular file is not opened; a source that fails any of steps 1–4 is read no further, and T302 names only the pin. `reconcile` refuses such a source. | Found after 2.0.2: once a source failed its pin, preflight kept reading the file — T303 printed its subject id, T305 its event ids, `--json` its whole event table — and T302 printed the `content_id` it stores, so a manifest naming a file its author does not hold reported that file's contents; `reconcile` checked no pin; and a named pipe given as a source blocked `task preflight`, `task reconcile` and `cache validate` inside `H5Fopen`. |
| §3.5 | A mask its own eligible procedure asserts it describes is an eligible annotation (1.1 §7.3). | 1.1 §7.3 required the asserter to be admitted when it was also the link's endpoint, so under `order_by = effective` a procedure of unknown time never attested its mask, and `roi = eligible_instances` fell back to `center_fallback`. |
