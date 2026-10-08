# CLI reference

```
medh5 COMMAND [args] [--json]
```

The command line is a native program over the format engine. `pip install
medh5` puts it on the path; so do `cargo install medh5-cli`, the binaries
attached to every GitHub Release, and `brew install XwK-P/medh5/medh5`. Each is
the same code, with the same output and exit codes. The converters
(`medh5 convert …`, `medh5 migrate`) are Python integrations: the standalone
binary runs them through a Python that has the package installed (`python3`,
or the interpreter `MEDH5_PYTHON` names), and says what to install when there
is none.

Exit codes are Unix-conventional: **0** success, **1** a handled error or a
failed check, **2** a usage error. Every inspection command takes `--json` and
writes a machine-readable document to stdout.

## Inspecting

### `medh5 info`

```
medh5 info PATH [--key K] [--json]
```

Grids, images, annotations, coverage, quality, codecs and content id.

**Collections.** `info`, `tree`, `verify`, `timeline` and `track` all take a
`.medh5c` shard with `--key`, which names one sample inside it — a member of a
shard is a sample root, so every per-sample view works on one. `info` and
`tree` without a key describe the shard itself; the commands that report on one
sample ask for a key and list the ones available.

### `medh5 tree`

```
medh5 tree PATH [--key K] [--json]
```

An annotated object listing — `h5ls` with each object's role in the spec.

### `medh5 validate`

```
medh5 validate PATH... [--level L] [--profile P] [-v] [--json]
```

Check against the specification. `--level` is `structural`, `semantic`
(default), `integrity` or `strict`; each includes the ones before it.
`--profile` overrides the profiles the file declares, so you can hold a file to
`det` even if it does not claim it. Diagnostics carry stable codes (spec §15.2).

```
$ medh5 validate case.medh5 --level strict
case.medh5: FAILED [strict] profiles=core,seg (1 errors, 0 warnings)
  ERROR   E102 grids/ct: `direction` is not orthonormal to 1e-4
```

### `medh5 verify`

```
medh5 verify PATH... [--key K] [--partial OBJ] [--json]
```

Recompute every object's digest and the root `content_id`. `--partial` limits
it to named objects — useful when you only want to know whether one image
changed.

In a file that declares a `content_id`, a dataset inside a grid, image,
annotation or transform that carries no digest fails the check (`UNSIGNED`):
the root covers the digests present, so an undigested dataset added to an
object — `instance_ids` on a boxes annotation, say — changes what it means
without changing the root. The writer digests every dataset, so none of its
files has one.

### `medh5 fix`

```
medh5 fix PATH... [--rebuild-index] [--rewrite-digests --reason WHY] [--by WHO] [--json]
```

With no flags: diagnose and change nothing. Exit 1 if anything needs attention.

`--rebuild-index` recomputes stale sampling indices. Ordinary repair.

`--rewrite-digests` is not repair. A digest that no longer matches is *evidence
that the bytes changed*, and recomputing it destroys the evidence. It therefore
requires `--reason`, which is recorded in the file's provenance along with the
fact that the tool did not verify the content it just re-attested. `--by` names
who made the change, in the same record.

## Longitudinal

### `medh5 timeline`

```
medh5 timeline PATH [--key K] [--json]
```

Timepoints, their intervals, and what belongs to each visit.

### `medh5 track`

```
medh5 track PATH [--class KEY] [--key K] [--json]
```

Join instance ids across visits: per object, its volume at each timepoint, the
relative change, and whether it is `present`, `resolved` or `unexamined`.

## Annotations

### `medh5 seg stats`

```
medh5 seg stats PATH ANNOTATION [--json]
```

Per-class voxel counts, the class overlap graph, and what the encoding
auto-selector would cost for each candidate encoding.

### `medh5 seg convert`

```
medh5 seg convert PATH ANNOTATION --to KIND [--dry-run] [--drop-identity] [--json]
```

Losslessly re-encode a voxel annotation: `labelmap`, `layers`, `bitmask`,
`instances`, `probmap`. `--dry-run` prints the size change without writing.
From `instances` to a dense encoding the voxels survive and the object ids do
not, so it is refused unless `--drop-identity` says to drop them; the file then
records a `transcode` activity saying so.

### `medh5 index build`

```
medh5 index build PATH... [--max-coords N] [--occupancy K] [--seed S] [--json]
```

Build or refresh sampling indices, which make foreground patch sampling O(1) in
the volume. `--max-coords` bounds the stored coordinates per class (default
4096); `--occupancy` sets the block size of the coarse per-class occupancy map,
in voxels per axis (default 8, so the map is 1/8 resolution); `--seed` (default
0) fixes the coordinate subsample, so rebuilding an index reproduces it.

### `medh5 labels`

```
medh5 labels show PATH [--json]
medh5 labels check PATH... [--json]
medh5 labels registry list [--json]
```

Inspect a file's label set, check a cohort's label sets against each other, and
list the bundled vocabularies.

## Curation

### `medh5 prov`

```
medh5 prov PATH [--json]
```

Agents, activities, quality records and the de-identification record.

### `medh5 agree`

```
medh5 agree PATH A B [--metric dice|iou] [--threshold T] [--record] [--json]
```

Per-class Dice or IoU between two voxel annotations, and object F1 between two
`instances` or `boxes` annotations of the same sample — two readers, or a reader
and a model. `--metric` applies to the first and `--threshold` to the second;
the other is refused rather than ignored. Only classes both sides examined are
scored (§11.3), and a comparison with nothing to score says so rather than
reporting 0. The two must share a grid (for index boxes) or a frame (for world
boxes). `--record` prints the `quality.agreement` record the measurement
produces, keyed by class id, and exits 1 when there is nothing to record.

### `medh5 splits`

```
medh5 splits PATH... [--json]
```

Cross-file audit: conflicting split claims (W906) and subject leakage between
partitions. Needs the whole cohort, which is why it is not part of `validate`.
A leak is found over subjects *and* grouping keys together, so one subject
curated under two `group_id`s in two partitions is a leak.

### `medh5 scrub`

```
medh5 scrub PATH... [--profile basic|strict] [--apply] [--date-shift-days N] [--salt S] [--by WHO] [--pseudonymise-ids] [--json]
```

Find identifiers in the container. Without `--apply` nothing is written and the
exit code is 1 if anything was found, so it works as a pipeline gate.

The scan covers every string in the file — all of `/meta`, every object name,
attribute and string dataset, including ones a newer version wrote — and skips
only fields named in the code with a reason (see the
[de-identification guide](../guides/deidentify.md#1-look-before-you-change-anything)).
`--apply` removes identifying attributes wherever the scan reports them,
pseudonymises UIDs (so files still join on a shared frame of reference) including
inside provenance references, reduces provenance paths to a file name, shifts
or drops dates, and writes a §11.4 de-identification record saying **what was
and was not checked**. It does not look at pixels: burned-in text and
identifiable anatomy are outside its reach, and the record it writes says so
rather than claiming a clean file.

**`--apply` re-scans what it wrote**, records the result in the
de-identification activity's parameters, and **exits non-zero if anything
actionable is left** — and, under `--profile strict`, while `sample_id` or
`subject_id` is still an identifier (the DICOM importer's are the PatientID).
An apply that exits 0 is a file the same rules now find nothing to fix in.

`--pseudonymise-ids` (with `--apply` and `--salt`) replaces `sample_id` and
`subject_id` — and a `cohort.group_id` equal to either — with salted stable
pseudonyms. A file still named after the old id is reported: `scrub` does not
rename files.

```
$ medh5 scrub out/*.medh5 --apply --date-shift-days -117 --by RAD-07
```

Running it twice does not shift dates twice.

## Collections

### `medh5 pack`

```
medh5 pack PATH... -o SHARD.medh5c [--key K ...] [--json]
```

Bundle samples into one `.medh5c`. Chunks move as raw bytes, so packing is
byte-identical and `content_id` is preserved. Each member's key is its file
stem unless `--key` is repeated once per source, in order.

### `medh5 unpack`

```
medh5 unpack SHARD.medh5c -o DIR [--key K ...] [--json]
medh5 ls SHARD.medh5c [--json]
```

Extract — every member, or only the keys named — or list what is inside.

## Cohorts

### `medh5 dataset index`

```
medh5 dataset index ROOT -o manifest.json [--strict] [--json]
```

Metadata-only scan of a directory tree. Reports what would not open instead of
aborting; `--strict` makes the first failure fatal.

### `medh5 dataset split`

```
medh5 dataset split manifest.json [options]
```

```
--set-id ID          name of this split (default "default")
--group-by FIELD     grouping key (default group_id — never a file)
--stratify-by FIELD  balance a field across partitions
--ratios train=0.7,val=0.15,test=0.15
--k-folds N          k-fold instead of ratios
--seed N             deterministic given the manifest, seed and parameters (default 0)
-o, --out split.json
--write-claims       stamp each sample with its partition and the manifest digest
--fold N             with --k-folds --write-claims: which fold is validation
--assigned-by WHO    recorded on each claim as who assigned it
--json
```

Groups, not files: a subject's baseline and follow-up cannot land on opposite
sides. If the ratios cannot be met by an indivisible set of groups, the command
says which partition got nothing rather than leaving you to notice later.

### `medh5 dataset stats`

```
medh5 dataset stats manifest.json [--image K ...] [--annotation A ...] [--workers N]
                                  [--stride S] [--stored] [--partition P] [--set-id ID]
                                  [--out FILE] [--json]
```

Streaming intensity moments and class frequencies. Reads class counts from the
sampling index when it is current. `--partition train` restricts the pass to
one partition of `--set-id` (default `default`), which is how you compute
normalisation constants without looking at your test set. `--stride N` reads
every Nth slab along the first axis — faster, approximate, and opt-in.

Intensity moments are over **physical** values: each image's rescale is applied
first, which is what the loaders read with `physical=True`. `--stored` measures
the values the files store instead.

### `medh5 dataset check`

```
medh5 dataset check manifest.json [--set-id ID] [--deep] [--json]
```

Cross-file consistency. Findings carry
[`C1xx`–`C5xx` codes](cohort-checks.md).

`--deep` re-reads each `content_id` instead of trusting size and mtime.

## Converting

Every `convert` command except `to-nifti` writes a report of what it
**decided** (determined from the data) and where it **guessed** (assumed
something it could not read). `--report FILE` keeps it as JSON and `--json`
prints it; without either, guesses and warnings still print.

```
medh5 convert from-nifti OUT --image NAME=PATH [--image ...] [--mask NAME=PATH ...]
                             [--modality NAME=CODE ...] [--coord-system LPS|RAS]
                             [--fourth-axis auto|time|channel] [--assume-geometry]
                             [--sample-id ID] [--subject-id ID]
medh5 convert to-nifti PATH IMAGE OUT [--annotation A --class K] [--stored]

medh5 convert from-dicom ROOT OUT [--group-by subject|study]
                                  [--modality M ...] [--series UID ...]

medh5 convert from-dicom-seg SEG SAMPLE [--id ANN] [--grid G]
medh5 convert to-dicom-seg PATH ANNOTATION OUT --source DICOM [--source DICOM ...]

medh5 convert from-rtstruct RTSTRUCT SAMPLE [--id ANN] [--grid G] [--rasterize]
medh5 convert to-rtstruct PATH ANNOTATION OUT --source DICOM [--source DICOM ...]

medh5 convert from-nnunet ROOT OUT [--case ID ...]
medh5 convert to-nnunet OUT PATH... [--dataset-name NAME] [--annotation A]
```

All but `to-nifti` also take `[--report FILE] [--json]`. Options marked `...`
repeat, one value per occurrence.

`from-nifti --fourth-axis` says what a 4-D series' extra axis is — `time` for
cine, DCE and 4-D CT, `channel` for multi-b-value DWI and multi-echo; `auto`
(the default) reads the file's sidecars and header and records a guess when they
do not say. `--assume-geometry` imports a file that declares no spatial mapping,
recorded as a guess; without it such a file is refused.

`from-dicom --group-by subject` (the default) merges a patient's studies into
one multi-timepoint sample. Identity is never inferred from filenames, dates or
accession numbers; when it cannot be established the command falls back to one
sample per study, warns, and records the fallback.

`from-dicom-seg` writes annotation `seg` and `from-rtstruct` writes `contours`
unless `--id` names another; `to-nnunet` exports annotation `seg` unless
`--annotation` names another, as dataset `Dataset001_medh5` unless
`--dataset-name` does.

See [Converters](converters.md).

### `medh5 migrate`

```
medh5 migrate PATH... -o OUTDIR [options]
```

0.x files to 1.0 (spec Appendix B).

```
--group-by subject|study   default study — a 0.x file has no subject key
--subject-key extra.patient_id
--write-labels FILE        mint the cohort's label set for review, then stop
--label-set FILE           reuse a reviewed label set
--report FILE
--json
```

See [Migrate from 0.x](../guides/migrate-0x.md) for the two-pass procedure.

## Storage

### `medh5 recompress`

```
medh5 recompress PATH... --profile P [--rechunk] [--json]
medh5 recompress PATH --profile P --out FILE [--rechunk] [--json]
```

Re-encode bulk data under `training`, `balanced`, `archive` or `portable`.
Every stored byte changes; no `content_id` does, because the digest is over
content and not over its encoding.

The output is **verified** against the digests it carries before the command
returns, and the exit code follows: a file whose bytes had already been
corrupted before you re-encoded it now fails here rather than reporting a
preserved `content_id` and exiting 0.

### `medh5 bench`

```
medh5 bench [PATH] [--patch N] [--repeats N] [--workers N] [--annotation A] [--no-throughput] [--json]
```

Reproduce the performance targets on your hardware. With no path it builds a
synthetic sample first, and a synthetic two-visit sample related by a
displacement field for the `paired_center_ms` row — one patch centre moved
between visits, the read a paired dataset does once per item — and a 63-class
sample for the `foreground_sample_many_ms` row, which holds the foreground draw
to its target where the class count is large. A longitudinal sample given as
`PATH` gets the paired row when a transform relates its first two visits.
Progress goes to stderr, so `--json` output is only the document.

## Clinical history (format 1.1)

### `medh5 clinical show`

```
medh5 clinical show PATH [--key K] [--json]
```

The subject clock, every event version with its effective and available times,
the documents (metadata, not text) and the links. See
[Clinical history beside the images](../guides/clinical.md).

### `medh5 clinical select`

```
medh5 clinical select PATH --cutoff-hours H [--policy POLICY] [--context-us W] [--key K] [--json]
medh5 clinical select PATH --cutoff-us T [--policy-file POLICY.json] [--json]
```

What strict prospective selection (1.1 §9) admits at the cutoff: the event
versions, the payloads they attest, the records a later revision of unknown
availability makes uncertifiable, and how many events each rule excluded.
`--policy latest_provable` names the one alternative; `--policy-file` takes a
whole policy (contract §3.4).

### `medh5 clinical export`

```
medh5 clinical export PATH [--key K] [--out RECORDS.json]
```

The logical-record bundle --- clock, events, documents with their text, links ---
as JSON, checked against `medh5-clinical-1.schema.json`.

### `medh5 clinical augment`

```
medh5 clinical augment PATH RECORDS.json [--out OUT] [--json]
```

Add a record bundle to a sample, in place or into `--out`: images and
annotations are copied as stored (their digests do not change), the sample
becomes 1.1 with a new `content_id`, and what the records leave unknown is
reported. A `clinical` group the profile did not write is refused, never
reinterpreted.

### `medh5 clinical strip`

```
medh5 clinical strip PATH --out OUT [--json]
```

The imaging projection: a new file without the clinical profile, written as
1.0, with the loss reported. Never in place.

## Tasks and caches

The [task and cache contract](../spec/task-cache-1.md): task manifests
(`medh5.task/1`) and feature caches (`medh5.cache/1`). See
[Train on clinical tasks](../guides/clinical-training.md).

### `medh5 task validate`

```
medh5 task validate MANIFEST [--json]
```

Everything wrong with a manifest that opening no file can find: its schema,
its policy, slots and target, its fingerprint, and its subjects, partitions,
rows and source ids (T1xx, T2xx). Prints the task and manifest fingerprints.

### `medh5 task preflight`

```
medh5 task preflight MANIFEST [--base DIR] [--deep] [--json]
```

Open and check every source --- pins, identities, clocks, duplicated events
--- and report every row: `eligible`, `uncertifiable`, `excluded` or `error`,
with the reason, the event count, the image filling each slot and the target
label. Exits 1 when anything was found. `--deep` re-verifies every dataset of
every source, not only the clinical ones.

### `medh5 task reconcile`

```
medh5 task reconcile MANIFEST [--base DIR] [--out OUT]
```

Record, per subject, the event versions several of its fragments hold and the
digest they share --- what preflight checks duplicates against.

### `medh5 cache validate`

```
medh5 cache validate CACHE [--task MANIFEST] [--base DIR] [--json]
```

Check a feature cache's checksums and every source pin it records: *stale*
entries (T403) are told apart from *corrupt* ones (T401, T402). With `--task`,
also that it was built for this task and its cutoffs (T404), fitted on its
training partition (T405), and encodes only versions its rows admit (T406).

## Conformance

### `medh5 conformance list`

```
medh5 conformance list [--json]
```

Every corpus case, the clause it tests, and the codes it expects.

### `medh5 conformance build`

```
medh5 conformance build OUTDIR [--case NAME]
```

Write the corpus files and `expected.json`, and nothing else. Use this when you
want the cases as fixtures; use `publish` when you want the whole distributable
suite.

### `medh5 conformance run`

```
medh5 conformance run OUTDIR [--case NAME] [--json]
```

Build the corpus and check *this* validator against it.

### `medh5 conformance publish`

```
medh5 conformance publish OUTDIR [--case NAME]
```

Write the distributable suite: cases, `expected.json`, the code table, the JSON
Schemas (`/meta`, the clinical profile, and the task and cache contract's),
`SHA256SUMS` and a README.

### `medh5 conformance score`

```
medh5 conformance score SUITE RESULTS.json [--json]
```

Score any implementation's results against a published suite. See
[Conformance](../spec/conformance.md).

## Related

- **[Check a file before training on it](../guides/validate.md)** — `validate` versus `verify`.
- **[Diagnostic codes](diagnostic-codes.md)** — every code these commands report.
- **[Cohort check codes](cohort-checks.md)** — what `medh5 dataset check` reports.
- **[Task and cache contract](../spec/task-cache-1.md#9-finding-codes)** — the T-codes `medh5 task` and `medh5 cache` report.
- **[How-to guides](../guides/index.md)** — these commands in the tasks they belong to.
