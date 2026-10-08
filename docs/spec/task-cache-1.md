# Task and cache contract 1 — `medh5.task/1`, `medh5.cache/1`

**Status: Candidate, implemented** by the engine (`medh5::companion`), the Python package
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
any change to a `.medh5` file --- none of these operations writes to a source.

## 2. Source references

A **source reference** names one sample --- a standalone file or a member of a `.medh5c` collection
--- at one version:

| Field | Meaning |
|---|---|
| `source_id` | The id the manifest and reconciliation records use for this source (1.0 id syntax, plus `@` and `:`) |
| `uri` | Where the bytes are: a path, absolute or relative to the manifest's directory; `file://` accepted |
| `sample_key` | The member of a collection, or null for a standalone file |
| `content_id` | **The pin**: the 1.0 §13.2 address of the sample version the reference names |
| `local_subject_id` | The sample's own `identity.subject_id`, as the manifest maps it to a subject |

A URI is a locator, not identity. A reference resolves to a sample, and the sample counts only while:

1. its stored `content_id` is the pin;
2. its `content_id` **recomputes** to the pin from its stored digests;
3. the actual bytes of its clinical datasets match their digests --- every dataset's, with `--deep`.

Step 3 is not redundant with steps 1–2: `content_id` is a Merkle root over *stored* digests, so an
edit that leaves the stored digests alone changes no root. A dataset whose bytes cannot be read ---
a damaged chunk --- does not match. A failure of any step is **T302**; a source that does not open,
or names a member that does not exist, **T301**. Re-pinning is explicit: a changed source is never
silently accepted (`SourceRef.pin`).

## 3. The task manifest

### 3.1 The document

One JSON document, validated by `medh5-task-1.schema.json` (**T101** when it is not JSON or fails its
schema):

| Member | Content |
|---|---|
| `schema` | `"medh5.task/1"` |
| `task` | `{"id", "version", "description"?}` --- the task's identity |
| `identity_namespace` | The namespace subject ids, and clock ids, are scoped by |
| `policy` | The selection policy (§3.4); strict prospective when absent |
| `slots` | The modality slots (§3.5) |
| `target` | The target (§5), or null for an unlabelled task |
| `split` | `{"set_id", "partitions"}` --- **the training partition first** |
| `subjects` | `[{"subject_id", "partition"?, "clock_id"?, "sources": [...], "reconciled": [...]}]` |
| `rows` | `[{"row_id", "subject_id", "cutoff_us"}]` |
| `fingerprint` | Optional: the manifest fingerprint (§3.2); when present it **MUST** match (**T103**) |

An implementation normalises a manifest by filling every default, so two spellings of one task are
one task.

### 3.2 Identity and fingerprints

A fingerprint is `"sha256:" + hex(sha256(canonical JSON))`, with 1.0 §5.1's canonical JSON.

| Fingerprint | Over | Identifies |
|---|---|---|
| **task** | `schema`, `task` id and version, `identity_namespace`, the normalised `policy`, `slots`, `target` | The task *definition* --- not its rows, not its sources |
| **manifest** | The whole normalised manifest without `fingerprint` | One frozen task instance, rows and pins included |
| **row** | The task fingerprint, the namespace, the subject, the **sorted pinned `content_id`s** of the subject's sources, the cutoff | One example. A cohort-membership digest or a URI alone pins no data |
| **subjects digest** | The sorted `(namespace, subject_id)` pairs of one partition | Who a partition holds --- what learned preprocessing records (§7.3) |

### 3.3 Subjects, fragments and splits

**Patient splits precede window construction.** A partition belongs to a *subject*, and every row of
the subject inherits it, so no window puts one patient on both sides of a split. With a `split`,
every subject names one of its partitions (**T202**); without one, none does. A subject is declared
once and every row names a declared subject (**T201**); row ids and source ids are unique (**T204**).
One source belongs to one subject, and two subjects never pin one sample --- an overlapping snapshot
would cross the split (**T203**).

A subject may have several **fragments** --- files or members, one per site or per export --- on one
clock. Preflight (§4) opens them all, checks each source's identity against the subject it is mapped
to (**T303**) and their clocks against each other and the subject's `clock_id` (**T304**), and merges
their events. An event version present in several fragments **MUST** have one content, and the
manifest records it --- `reconciled: [{"event_id", "digest", "sources"}]`, the digest being the
fingerprint of the event's logical record --- before selection; a difference, or a duplicate with no
record, is **T305**. Documents present in several fragments must be identical (**T305**).
`medh5 task reconcile` writes the records.

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

An inconsistent policy, slot or target --- a slot declared twice, an unknown `roi`, a value both
positive and negative, a follow-up longer than the horizon --- is **T102**.

### 3.5 Modality slots

| Member | Default | Meaning |
|---|---|---|
| `name` | --- | Stable across the cohort: never a filename or a visit-specific image id |
| `modality` | --- | The image `modality` that fills it |
| `required` | `false` | A row with no eligible image for a required slot is excluded (`missing_required_slot`) |
| `patch` | null | The region read, in voxels of the image's **own** grid; null reads the whole volume |
| `roi` | `center` | `center` (the grid's centre) or `eligible_instances` (the first eligible instance) |
| `classes` | `[]` | Class ids read as label supervision from voxel annotations on the image's grid (§6) |

A slot is filled per row by the **newest eligible image** of its modality: an image an admitted
`imaging` event `describes` (1.1 §7.3), ordered by that event's order time, ties broken by event id.
An eligible image of another visit is a different fill, not an error; no image is resampled onto
another's grid, and no registration is invented. With `roi = eligible_instances` the centre is that of
the first instance (by `instance_id`) of an **eligible** instance-bearing annotation on the image's
grid --- an annotation drawn after the cutoff never chooses a crop --- and `center_fallback` records
that there was none.

### 3.6 Rows

A row is `{"row_id", "subject_id", "cutoff_us"}`. Any number of rows per subject, at any cutoffs, is
permitted; each is one example under the subject's partition.

## 4. Preflight

Preflight reads no voxel and uses no report: text is read only to verify it --- its digest (§2) and its
UTF-8 (1.1 §4), a bounded slab at a time --- and to compare a document two fragments both hold
(§3.3). It decides, per row, before any batch is built:

1. the manifest's own findings (T1xx, T2xx); a manifest with any leaves every row in `error`;
2. per subject: every source opened and checked (§2), identities, clocks and duplicates reconciled
   (§3.3), and every source a **full** implementation of its version --- a higher minor's projection
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
that cannot be read --- a damaged chunk in its clinical tables --- is that source's finding (**T302**,
**T306**), never the whole preflight's failure.

## 5. Targets

| Member | Default | Meaning |
|---|---|---|
| `id`, `version` | --- | The target's own identity, part of the task fingerprint |
| `event` | --- | `{"kind"?, "code_system", "code"}`: the concept the target reads |
| `positive` | --- | `value_text`s that are the outcome |
| `negative` | `[]` | `value_text`s that are a verified absence of it |
| `horizon_us` | --- | The window `(c, c + h]` an outcome must fall in |
| `min_follow_up_us` | `horizon_us` | A negative needs an observation at least this long after the cutoff |
| `censoring` | `censor` | `censor` keeps an unlabelled row with its target unobserved; `exclude` drops it |
| `exclude_prevalent` | `true` | Exclude rows whose outcome had already occurred by the cutoff |

The target is read from the **full history** --- the final version of each record, `entered_in_error`
withdrawn --- because it lies after the cutoff by definition; it can live in the same file as the
inputs, and it never enters them. In order:

1. **prevalent** --- a positive whose effective start is definitely at or before `c`
   (`exclude_prevalent`);
2. **positive** --- a positive whose effective start is definitely inside `(c, c + h]`: value `1.0`;
3. **censored** --- a positive whose time straddles either edge of the window, or is unknown: the
   outcome cannot be placed, and is not a negative;
4. **negative** --- a negative observed inside the window at or after `c + min_follow_up`: value
   `0.0`;
5. **censored** otherwise --- insufficient follow-up is not a negative label.

## 6. Input views

A frontend reads **only** what a row admits, and only when it builds the batch:

- **events** --- the selected versions, in clinical order, as the row's sequence; their count varies
  per row;
- **documents** --- the text (or a cached feature, §7) of documents the selection admits, i.e. owned
  by a selected `document` event;
- **images** --- each slot's fill, read as the region of §3.5 on the image's own grid, with the
  region's validity: inside the image and inside its 1.0 §4.4 valid region, never the padding;
- **labels** --- supervision for a slot's classes, read from the voxel annotations on the slot
  image's grid. Like the target, supervision may come from after the cutoff, and it never enters an
  input; coverage (1.0 §11.3) comes with it, so a class nobody examined is not a negative.

Masks stay distinct, because they mean different things to a loss: sequence **padding**, modality
**availability** (was the slot filled), voxel **validity** (field of view), annotation **coverage**
(was the class examined), and target **observation** (is the label observed). A zero-filled missing
modality is not an observed negative.

Time keeps the meaning 1.1 §5.2 gives it. A frontend **MUST NOT** collapse an input's time to one
instant chosen from its bounds: each time an input carries --- effective start, effective end,
availability --- is given as both bounds (relative to the cutoff, in whatever unit), with whether it is
known; an unknown time is marked unknown, never zero. A `static` event stays distinct from one of
`unknown` time, events of one **tie group** stay marked as such (their order in a sequence is a storage
tie-break, not evidence), and a plan admitted by `plans` stays marked as a plan. Missingness stays
apart from measurement: an absent value, a value known only as a bound (`value_comparator`) and a
`missing_reason` are each represented, never read as a measured value. Every availability an input
carries is at or before the cutoff --- what strict selection admits --- and only what a version
available at the cutoff itself recorded about later (a plan's start, a course's recorded end) lies
after it.

A frontend reads each source as the version its row pins: a source whose `content_id` is no longer the
pin when a batch is built --- the file was replaced after preflight --- is refused (**T302**), never
read in its place. That check is §2's step 1 alone: it compares the *stored* `content_id`, and
re-hashes nothing. The bytes are verified by preflight (step 3: the clinical datasets always, every
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

`manifest_digest` is `sha256` over the manifest's bytes. The manifest is validated by
`medh5-cache-1.schema.json`:

| Member | Content |
|---|---|
| `schema` | `"medh5.cache/1"` |
| `level` | `event` or `patient` |
| `encoder` | `{"name", "revision", "tokenizer": {"name", "revision"}?}` --- an **immutable** revision: a content hash or a commit, never a moving tag |
| `preprocessing` | Everything else that determines the output |
| `output` | `{"dtype", "shape", "pooling"?, "chunking"?}`; every entry has this dtype and shape (**T402**) |
| `task_fingerprint` | The task a patient-level cache was built for (required at that level) |
| `selection` | The selection policy it read under |
| `fitted_on` | For learned preprocessing: `{"task_fingerprint", "set_id", "partition", "subjects_digest"}` |
| `entries` | `[{"entry_id", "sources", "event_id"?, "document_id"?, "row_id"?, "cutoff_us"?, "event_versions"?, "digest"}]` |

Every entry names **every source version it read**, as `{uri, sample_key, content_id}`. The full
sample `content_id` is the source key: a conservative, sufficient one --- any change to the sample
makes its entries stale. A finer dependency key would need its own specification.

### 7.2 Levels

- **Event level**: one feature per event version of a pinned source (or the document it owns),
  conventionally entry `e` + 24 hex digits of `sha256(content_id + "\n" + event_id)`. An event version
  is immutable, so its feature is free of any cutoff and shared by every row --- each row's selection
  decides which of them it may read.
- **Patient level**: one feature per row, pinning the row, its cutoff and the exact event versions it
  encodes (`event_versions`), under the task it was built for.

### 7.3 Learned preprocessing

Normalisation statistics, code bins, vocabularies and anything else fitted to data record `fitted_on`:
the task fingerprint, the split, the partition and that partition's subjects digest (§3.2). They are
fitted on the **training partition** --- the split's first --- and nothing else.

## 8. Cache validation

Validation never modifies a source. Two failures are told apart, because they need different fixes:

- **stale** --- **T403**: an entry's source no longer has the content it pins, or cannot be reached.
  The cache is right about a sample that changed; rebuild the entry.
- **corrupt** --- **T401** (the manifest: not the companion, checksum mismatch, not JSON, fails its
  schema) or **T402** (an entry's payload fails its own checksum, or has the wrong dtype or shape).
  The cache is wrong about itself; rebuild it.

Given a task (and its preflight), validation also checks that the cache belongs to it:

- **T404** --- built for another task (`task_fingerprint`), or a patient-level entry for a row the
  task does not have, or at another cutoff;
- **T405** --- learned preprocessing fitted on another task, another membership, or a partition
  other than the task's training partition;
- **T406** --- a patient-level entry encodes event versions other than those its row admits at its
  cutoff: a whole-history embedding, or one built under another policy, is inadmissible.

A stale or unverifiable cache is rejected and rebuilt; it never redefines or invalidates the source.

## 9. Finding codes

<!--@companion-codes-->

## 10. Schemas and implementations

| Artifact | Where |
|---|---|
| `medh5-task-1.schema.json`, `medh5-cache-1.schema.json` | The engine (`crates/medh5/data/`) and the published conformance suite |
| Fixtures | `companion/` in the published [conformance suite](conformance.md#the-task-and-cache-fixtures): one invalid manifest per T-code from T101 to T306, and a valid task whose every row's selection, slot and label is stated |
| Engine | `medh5::companion` --- `task`, `view` (preflight), `cache`, `source` |
| Python | `medh5.task`, `medh5.cache`, `medh5.torch.clinical` ([Train on clinical tasks](../guides/clinical-training.md)) |
| Command line | `medh5 task validate`, `medh5 task preflight`, `medh5 task reconcile`, `medh5 cache validate` ([Command line](../reference/cli.md)) |
