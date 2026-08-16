# CLI reference

```
medh5 COMMAND [args] [--json]
```

Exit codes are Unix-conventional: **0** success, **1** a handled error or a
failed check, **2** a usage error. Every inspection command takes `--json` and
writes a machine-readable document to stdout.

## Inspecting

### `medh5 info PATH [--json]`

Grids, images, annotations, coverage, quality, codecs and content id.

### `medh5 tree PATH`

An annotated object listing — `h5ls` with each object's role in the spec.

### `medh5 validate PATH... [--level L] [--profile P] [-v] [--json]`

Check against the specification. `--level` is `structural`, `semantic`
(default), `integrity` or `strict`; each includes the ones before it.
`--profile` overrides the profiles the file declares, so you can hold a file to
`det` even if it does not claim it. Diagnostics carry stable codes (spec §15.2).

```
$ medh5 validate case.medh5 --level strict
case.medh5: FAILED [strict] profiles=core,seg (1 errors, 0 warnings)
  ERROR   E102 grids/ct: `direction` is not orthonormal to 1e-4
```

### `medh5 verify PATH... [--partial OBJ] [--json]`

Recompute every object's digest and the root `content_id`. `--partial` limits
it to named objects — useful when you only want to know whether one image
changed.

### `medh5 fix PATH... [--rebuild-index] [--rewrite-digests --reason WHY] [--json]`

With no flags: diagnose and change nothing. Exit 1 if anything needs attention.

`--rebuild-index` recomputes stale sampling indices. Ordinary repair.

`--rewrite-digests` is not repair. A digest that no longer matches is *evidence
that the bytes changed*, and recomputing it destroys the evidence. It therefore
requires `--reason`, which is recorded in the file's provenance along with the
fact that the tool did not verify the content it just re-attested.

## Longitudinal

### `medh5 timeline PATH [--json]`

Timepoints, their intervals, and what belongs to each visit.

### `medh5 track PATH [--class KEY] [--json]`

Join instance ids across visits: per object, its volume at each timepoint, the
relative change, and whether it is `present`, `resolved` or `unexamined`.

## Annotations

### `medh5 seg stats PATH ANNOTATION [--json]`

Per-class voxel counts, the class overlap graph, and what the encoding
auto-selector would cost for each candidate encoding.

### `medh5 seg convert PATH ANNOTATION --to KIND [--dry-run] [--json]`

Losslessly re-encode a voxel annotation: `labelmap`, `layers`, `bitmask`,
`instances`. `--dry-run` prints the size change without writing.

### `medh5 index build PATH... [--max-coords N] [--occupancy K] [--seed S]`

Build or refresh sampling indices, which make foreground patch sampling O(1) in
the volume. `--max-coords` bounds the stored coordinates per class.

### `medh5 labels show PATH [--json]` · `labels check PATH...` · `labels registry list`

Inspect a file's label set, check a cohort's label sets against each other, and
list the bundled vocabularies.

## Curation

### `medh5 prov PATH [--json]`

Agents, activities, quality records and the de-identification record.

### `medh5 agree PATH A B [--metric dice|iou] [--threshold T] [--record] [--json]`

Per-class Dice or IoU and object F1 between two annotations of the same sample
— two readers, or a reader and a model. `--record` prints the
`quality.agreement` record the measurement produces.

### `medh5 splits PATH... [--json]`

Cross-file audit: conflicting split claims (W906) and subject leakage between
partitions. Needs the whole cohort, which is why it is not part of `validate`.

### `medh5 scrub PATH... [--profile basic|strict] [--apply] [--date-shift-days N] [--salt S] [--by WHO] [--json]`

Find identifiers in the container. Without `--apply` nothing is written and the
exit code is 1 if anything was found, so it works as a pipeline gate.

`--apply` removes identifying attributes, pseudonymises UIDs (so files still
join on a shared frame of reference), shifts or drops dates, and writes a §11.4
de-identification record saying **what was and was not checked**. It does not
look at pixels: burned-in text and identifiable anatomy are outside its reach,
and the record it writes says so rather than claiming a clean file.

```
$ medh5 scrub out/*.medh5 --apply --date-shift-days -117 --by RAD-07
```

Running it twice does not shift dates twice.

## Collections

### `medh5 pack PATH... -o SHARD.medh5c [--key K] [--json]`

Bundle samples into one `.medh5c`. Chunks move as raw bytes, so packing is
byte-identical and `content_id` is preserved.

### `medh5 unpack SHARD.medh5c -o DIR [--key K] [--json]` · `medh5 ls SHARD.medh5c [--json]`

Extract, or list what is inside.

## Cohorts

### `medh5 dataset index ROOT -o manifest.json [--strict] [--json]`

Metadata-only scan of a directory tree. Reports what would not open instead of
aborting; `--strict` makes the first failure fatal.

### `medh5 dataset split manifest.json [options]`

```
--set-id ID          name of this split (default "default")
--group-by FIELD     grouping key (default group_id — never a file)
--stratify-by FIELD  balance a field across partitions
--ratios train=0.7,val=0.15,test=0.15
--k-folds N          k-fold instead of ratios
--seed N             deterministic given the manifest, seed and parameters
--out split.json
--write-claims       stamp each sample with its partition and the manifest digest
--fold N             with --k-folds --write-claims: which fold is validation
```

Groups, not files: a subject's baseline and follow-up cannot land on opposite
sides. If the ratios cannot be met by an indivisible set of groups, the command
says which partition got nothing rather than leaving you to notice later.

### `medh5 dataset stats manifest.json [--image K] [--annotation A] [--workers N] [--stride S] [--partition P] [--out FILE] [--json]`

Streaming intensity moments and class frequencies. Reads class counts from the
sampling index when it is current. `--partition train` restricts the pass to
one partition of `--set-id`, which is how you compute normalisation constants
without looking at your test set.

### `medh5 dataset check manifest.json [--set-id ID] [--deep] [--json]`

Cross-file consistency, under `C1xx` codes:

| Code | |
|---|---|
| `C101` / `C102` | more than one label set; a class id meaning two things |
| `C103` | a sample with no label set |
| `C201` | a claim whose manifest digest is not this manifest's |
| `C202` | a subject in two partitions of one split |
| `C203` | samples with no claim |
| `C301` / `C302` | a class examined in only part of the cohort; a class never seen |
| `C401` / `C402` | a file changed, or vanished, since the scan |
| `C501` | a partly de-identified cohort |

`--deep` re-reads each `content_id` instead of trusting size and mtime.

## Converting

Every `convert` command writes a report of what it **decided** (determined from
the data) and where it **guessed** (assumed something it could not read).
`--report FILE` keeps it as JSON; without it, guesses and warnings still print.

```
medh5 convert from-nifti OUT --image NAME=PATH [--mask NAME=PATH]
                             [--modality NAME=CODE] [--coord-system LPS|RAS]
                             [--sample-id ID] [--subject-id ID]
medh5 convert to-nifti PATH IMAGE OUT [--annotation A --class K] [--stored]

medh5 convert from-dicom ROOT OUT [--group-by subject|study]
                                  [--modality M] [--series UID]

medh5 convert from-dicom-seg SEG SAMPLE [--id ANN] [--grid G]
medh5 convert to-dicom-seg PATH ANNOTATION OUT --source DICOM...

medh5 convert from-rtstruct RTSTRUCT SAMPLE [--id ANN] [--grid G] [--rasterize]
medh5 convert to-rtstruct PATH ANNOTATION OUT --source DICOM...

medh5 convert from-nnunet ROOT OUT [--case ID]
medh5 convert to-nnunet OUT PATH... [--dataset-name NAME] [--annotation A]
```

`from-dicom --group-by subject` merges a patient's studies into one
multi-timepoint sample. Identity is never inferred from filenames, dates or
accession numbers; when it cannot be established the command falls back to one
sample per study, warns, and records the fallback.

See [Converters](converters.md).

### `medh5 migrate PATH... -o OUTDIR [options]`

0.x files to 1.0 (spec Appendix B).

```
--group-by subject|study   default study — a 0.x file has no subject key
--subject-key extra.patient_id
--write-labels FILE        mint the cohort's label set for review, then stop
--label-set FILE           reuse a reviewed label set
--report FILE
```

## Storage

### `medh5 recompress PATH... --profile P [--out DIR] [--rechunk] [--json]`

Re-encode bulk data under `training`, `balanced`, `archive` or `portable`.
Every stored byte changes; no `content_id` does, because the digest is over
content and not over its encoding.

### `medh5 bench [PATH] [--patch N] [--repeats N] [--workers N] [--annotation A] [--no-throughput] [--json]`

Reproduce the performance targets on your hardware. With no path it builds a
synthetic sample first.

## Conformance

### `medh5 conformance list [--json]`

Every corpus case, the clause it tests, and the codes it expects.

### `medh5 conformance run OUTDIR [--case NAME] [--json]`

Build the corpus and check *this* validator against it.

### `medh5 conformance publish OUTDIR [--case NAME]`

Write the distributable suite: cases, `expected.json`, the code table, the JSON
Schema, `SHA256SUMS` and a README.

### `medh5 conformance score SUITE RESULTS.json [--json]`

Score any implementation's results against a published suite. See
[Conformance](conformance.md).
