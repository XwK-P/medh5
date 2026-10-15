# CLI reference

```
medh5 COMMAND [args] [--json]
```

The command line is a native program over the format engine. `pip install
medh5` puts it on the path; so do `cargo install --locked medh5-cli`, the binaries
attached to every GitHub Release, and `brew install medh5-project/medh5/medh5`. Each is
the same code, with the same output and exit codes. The converters
(`medh5 convert …`) and the dataloader row of `medh5 bench` are Python
integrations: the standalone binary runs them through a Python that has the
package installed (`python3` — `python` on Windows — or the interpreter
`MEDH5_PYTHON` names). That package must be a medh5 2.x; when it is not, or
will not import, the command says why — the version found, or the last line
Python printed — and what to install. `medh5 migrate` is native.

Exit codes are Unix-conventional: **0** success, **1** a handled error or a
failed check, **2** a usage error. Every inspection command takes `--json` and
writes a machine-readable document to stdout. Help you ask for (`-h`,
`--help`) goes to stdout and exits 0; a usage error goes to stderr and exits 2,
including `medh5` or a command group given no command, which prints its help
there.

A long option can be shortened to any prefix that names only one of the
command's options — `--dry` for `--dry-run` — as the 1.x parser allowed.
Spell options in full in scripts: a later release that adds an option can make
a prefix name two.

A command given several files reports on each of them. One it cannot process
is named on stderr — and, with `--json`, is an entry of its own,
`{"path": …, "ok": false, "error": …}`, its `error` what stderr says — and
the files after it are still processed; the command then exits with 1. Output
that cannot be written (a full disk) fails the command too.

A command that changes a file in place — `fix`, `index build`, `seg convert`,
`scrub --apply`, `recompress` without `--out`, `dataset split --write-claims`,
`clinical augment` without `--out` — writes the new version beside it and
renames it over the old (§14.4). Through a symbolic link, the file the link
names is rewritten and the link kept; a file with several hard links is
replaced at the path given only, and its other names keep the old version —
after `scrub --apply`, remove them. A link in a directory anyone can write to,
such as `/tmp`, that neither you nor the directory's owner owns is not
followed: the command refuses the file, as the system would refuse to open it
through the link.

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
medh5 validate PATH... [--level L] [--profile P] [--codes-revision N] [-v] [--json]
```

Check against the specification. `--level` is `structural`, `semantic`
(default), `integrity` or `strict`; each includes the ones before it.
`--profile` adds to the profiles the file declares, so you can hold a file to
`det` even if it does not claim it. Diagnostics carry stable codes (spec §15.2).
`--codes-revision` reports only the codes that revision of the code table
allocated, so a `strict` gate keeps its verdict when a later revision adds a
warning; a revision the table does not have is refused.
Every diagnostic is printed; `-v` adds, under each, what its code means.

```
$ medh5 validate case.medh5 -v
case.medh5: FAILED [semantic] profiles=core (1 errors, 0 warnings)
  ERROR   E102 /grids/ct: `direction` is not orthonormal (max residual 0.4)
          -> `direction` is not orthonormal to 1e-4
```

### `medh5 verify`

```
medh5 verify PATH... [--key K] [--partial OBJ] [--json]
```

Recompute every object's digest and the root `content_id`. `--partial` limits
it to datasets named by path (`images/CT`, `annotations/seg/data`) — useful
when you only want to know whether one image changed. A digest that is not
`ALGO:HEX`, or a `--partial` name with no digest at that path (a group, a
typo), is reported as `MALFORMED` and fails the check. A dataset carrying a
digest its type cannot have (§13.1) — variable-length data other than strings,
references — can be neither checked nor refuted: it is named on an `UNCHECKED`
line (`unverifiable` in `--json`) and does not fail the check.

In a file that declares a `content_id`, a dataset inside a grid, image,
annotation or transform that carries no digest fails the check (`UNSIGNED`):
the root covers the digests present, so an undigested dataset added to an
object — `instance_ids` on a boxes annotation, say — changes what it means
without changing the root. So does a path there that is not its object's own
— an alias sorting before it, a soft link — because a line binds bytes to the
path it names, and such a name could be relinked to other covered bytes under
the same root (`E704`). The writer digests every dataset and makes no links,
so none of its files has either.

### `medh5 fix`

```
medh5 fix PATH... [--rebuild-index] [--rewrite-digests --reason WHY [--by WHO]] [--json]
```

With no flags: diagnose and change nothing. `fix` exits 0 only when nothing is
left needing attention once it has done what it was asked, and 1 otherwise —
a diagnosis that finds something, or an index entry left stale: an entry pins
its annotation's digests (§13.3), so restamping them without `--rebuild-index`
leaves it stale. Each file's report names what is left — a digest that is not
`ALGO:HEX` as malformed — and its notes, which say why, are printed whatever
was done.

`--rebuild-index` recomputes stale sampling indices. Ordinary repair.

`--rewrite-digests` is not repair. A digest that no longer matches is *evidence
that the bytes changed*, and recomputing it destroys the evidence. It therefore
requires `--reason`, which is recorded in the file's provenance along with the
fact that the tool did not verify the content it just re-attested. `--by` names
who made the change, in the same record. Only the restamp records them, so
`--reason` or `--by` without `--rewrite-digests` is a usage error.

It is also the one amendment of a file whose attestation broke. Every other
command that amends a file (`fix` without it, `scrub --apply`, a split claim,
`clinical augment` and `strip`, `seg`) refuses one whose `content_id` no longer
recomputes (`E702`) or that holds a digest, or declares a `digest_algo`, it
cannot read (`E703`), and names this repair. A file declaring an algorithm
§2.1 does not permit is restamped in `sha256`, and the activity's parameters
record the change (`"digest_algo": {"from": "blake3", "to": "sha256"}`).
Without `--rewrite-digests`, such a file is diagnosed as `E703`, naming the
repair.

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

Join instance ids across visits: per object, its volume at each timepoint in
mm³ (in px on an uncalibrated grid), the relative change, and whether it is
`present`, `resolved` or `unexamined`. `resolved` means the class was examined
at this visit, outside any ignore region, and the object was not found;
`unexamined`, that nobody looked for that class at this visit, or that an
ignore region (§7.7) may hide it. Masks, boxes, oriented boxes and points
join; an annotation of another kind that carries `instance_ids` is named as
skipped, and the rest still join.

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

Re-encode a voxel annotation: `labelmap`, `layers`, `bitmask`, `instances`,
`probmap`. `--dry-run` prints the size change without writing. A conversion
to the encoding the annotation already has re-encodes nothing and leaves the
file as it was (`"applied": false` with `--json`). Every conversion is
lossless but two. A `probmap` is lossless only under its `threshold` (§7.5), so
another encoding keeps the voxels at or above it as the class and loses the
probabilities, as §7.6 allows. From `instances` to a dense encoding the voxels
survive and the object ids do not, so it is refused unless `--drop-identity`
says to drop them; the file then records a `transcode` activity saying so. A
refused conversion leaves the file untouched.

### `medh5 index build`

```
medh5 index build PATH... [--max-coords N] [--occupancy K] [--seed S] [--json]
```

Build or refresh sampling indices, which make foreground patch sampling O(1) in
the volume. `--max-coords` bounds the stored coordinates per class (default
4096, at least 1); `--occupancy` sets the block size of the coarse per-class
occupancy map, in voxels per axis (default 8, so the map is 1/8 resolution;
`0` builds no map); `--seed` (default 0) fixes the coordinate subsample, so
rebuilding an index reproduces it.

An entry that already holds what the build would write — current, and drawn
with the same `--max-coords`, `--occupancy` and `--seed` — is kept, and a file
whose entries all hold it is left untouched (`index current for …; the file is
unchanged`). A file with nothing to index — no voxel annotation with classes;
a `mask` has none — is done, not a failure. With `--json`, the document
maps each path to the annotations it now indexes, or to the entry of a file it
could not process.

### `medh5 labels`

```
medh5 labels show PATH [--json]
medh5 labels check PATH... [--json]
medh5 labels registry list [--json]
```

Inspect a file's label set, check a cohort's label sets against each other, and
list the bundled vocabularies. With `--json`, `check` gives every file an entry
under `files` — its vocabulary key, or why it could not be read — beside the
files grouped by vocabulary (`vocabularies`) and the failures by path
(`unreadable`). Files share a vocabulary when their label sets have one digest,
the rule `dataset check` applies for `C101`: an empty label set is a vocabulary
of its own (`id@version#` and 16 hex digits, as any other), and `<none>` holds
only the files that declare no label set. `medh5.labels.check(paths)` returns
the same document.

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
the other is refused rather than ignored. Two objects match at an IoU of at
least the threshold, which lies in (0, 1] (default 0.5); any other value is a
usage error. Only classes both sides examined are scored (§11.3), and a
comparison with nothing to score says so rather than reporting 0. The two must
share a grid (for index boxes) or a frame (for world boxes). `--record` prints
the `quality.agreement` record the measurement produces, keyed by class id, and
exits 1 when there is nothing to record.

### `medh5 splits`

```
medh5 splits PATH... [--json]
```

Cross-file audit: conflicting split claims (W906) and subject leakage between
partitions. Needs the whole cohort, which is why it is not part of `validate`.
A leak is found over subjects *and* grouping keys together, so one subject
curated under two `group_id`s in two partitions is a leak. A file it cannot
read is named on stderr (and is an entry under `unreadable` with `--json`), and
fails the audit.

### `medh5 scrub`

```
medh5 scrub PATH... [--profile basic|strict] [--apply [--date-shift-days N] [--salt S] [--by WHO] [--pseudonymise-ids]] [--json]
```

Find identifiers in the container. Without `--apply` nothing is written and the
exit code is 1 if anything was found, so it works as a pipeline gate.

The scan covers every string in the file — all of `/meta`, every link name,
attribute (committed types included) and string dataset, including ones a newer
version wrote — and skips only fields named in the code with a reason (see the
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

Running it twice does not shift dates twice: a second run with a different
shift is refused, as is a shift of zero. `--date-shift-days`, `--salt` and
`--by` act only with `--apply`, and are refused without it.

## Collections

### `medh5 pack`

```
medh5 pack PATH... -o SHARD.medh5c [--key K ...] [--json]
```

Bundle samples into one `.medh5c`. Chunks move as raw bytes, so packing is
byte-identical and `content_id` is preserved. Each member's key is its file
stem unless `--key` is repeated once per source, in order. An output that is
one of the samples being packed is refused, as is a directory.

### `medh5 unpack`

```
medh5 unpack SHARD.medh5c -o DIR [--key K ...] [--json]
medh5 ls SHARD.medh5c [--json]
```

Extract — every member, or only the keys named — or list what is inside.

## Cohorts

### `medh5 dataset index`

```
medh5 dataset index ROOT -o manifest.json [--strict] [--absolute] [--json]
```

Metadata-only scan of a directory tree. Reports what would not open instead of
aborting; `--strict` makes the first failure fatal. Follows symlinked
directories (each once), matches suffixes in any case, and skips AppleDouble
`._*` files.

The manifest's paths are [relative to the manifest](cohort-manifest.md) where
they lie beneath its directory, and absolute otherwise, so it opens from any
directory and moves with its cohort; `--absolute` writes every path absolute.

### `medh5 dataset upgrade`

```
medh5 dataset upgrade manifest.json -o upgraded.json [--base DIR] [--json]
```

Rewrites a manifest written before 2.1, whose paths are relative to the
directory its scan ran in, as a [`medh5.dataset/1`](cohort-manifest.md)
manifest: every path beneath the manifest's directory is written relative to
it, and the rest absolute. `--base` names the
directory the old paths are relative to (by default the working directory, as
2.0 read them). Refuses, writing nothing, when any entry's file does not exist
under that directory. The cohort commands read such a manifest as 2.0 did, and
warn where 3.0 will read it otherwise.

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
--algorithm 1|2      medh5.split/1 (the 2.x default) or medh5.split/2 (the 3.0 default)
--previous SPLIT     continue a split JSON: its groups keep their places
-o, --out split.json
--write-claims       stamp each sample with its partition and the manifest digest
--fold N             with --k-folds --write-claims: which fold is validation
--assigned-by WHO    with --write-claims: recorded on each claim as who assigned it
--json
```

Groups, not files: a subject's baseline and follow-up cannot land on opposite
sides. If the ratios cannot be met by an indivisible set of groups, the command
says which partition got nothing rather than leaving you to notice later.
`--fold` and `--assigned-by` describe the claims `--write-claims` stamps, so
either without it is a usage error.

`--algorithm` names how groups are dealt
([the two algorithms](../guides/cohorts.md#the-algorithm-and-continuing-a-split));
left out, the split is `medh5.split/1`, with a `warning:` on stderr where
`medh5.split/2` would place some group differently. `--previous` continues
a split: every group it placed keeps its partition or fold, and new groups are
dealt by `medh5.split/2`; its parameters must be the previous split's, and
`--algorithm 1` is refused with it. The split JSON records `algorithm` and,
for a continued split, `previous` (the old manifest's digest, how many groups
were kept, and those added and dropped).

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
`--image` or `--annotation` naming an id that no sample in the manifest (or the
partition) has is an error, not an empty table. Exits 1 if any file would not
open, naming each on stderr; the class table shows each class's label-set key.

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

medh5 convert from-dicom-seg SEG SAMPLE [--id ANN] [--grid G] [--frame-salt S]
medh5 convert to-dicom-seg PATH ANNOTATION OUT --source DICOM [--source DICOM ...]
                                               [--frame-salt S]

medh5 convert from-rtstruct RTSTRUCT SAMPLE [--id ANN] [--grid G] [--rasterize]
                                            [--frame-salt S]
medh5 convert to-rtstruct PATH ANNOTATION OUT --source DICOM [--source DICOM ...]
                                              [--frame-salt S]

medh5 convert from-nnunet ROOT OUT [--case ID ...]
medh5 convert to-nnunet OUT PATH... [--dataset-name NAME] [--annotation A]
                                    [--class K ...] [--unlabeled refuse|test]
```

All but `to-nifti` also take `[--report FILE] [--json]`. A report never
replaces a file the command reads or writes: a `--report` naming one — through
`..` or a link included — or lying inside the tree or nnU-Net dataset the
command reads or the nnU-Net dataset it writes is refused before anything is
converted, and so is an output naming one of the command's inputs. Options
marked `...` repeat, one value per occurrence. `PATH` and `SAMPLE` are sample
files: a collection is refused before anything is converted, and
`medh5 unpack` extracts its members.

`from-nifti --fourth-axis` says what a 4-D series' extra axis is — `time` for
cine, DCE and 4-D CT, `channel` for multi-b-value DWI and multi-echo; `auto`
(the default) reads the file's sidecars and header and records a guess when they
do not say. `--assume-geometry` imports a file that declares no spatial mapping,
recorded as a guess; without it such a file is refused.

`to-nifti` writes `OUT`, a `.nii` or `.nii.gz` file. With `--annotation` it
writes that annotation's labelmap, each ignored voxel as its ignore id; with
`--class` too, one class as a 0/1 mask, refused for an annotation with an ignore
region, which a mask has no value for.

`from-dicom --group-by subject` (the default) merges a patient's studies into
one multi-timepoint sample. Identity is never inferred from filenames, dates or
accession numbers; when it cannot be established the command falls back to one
sample per study, warns, and records the fallback. `OUT` is the file the one
sample is written to, or a directory of samples named for their subjects: when
several samples result, when it is an existing directory, or when it ends in a
path separator (`out/`).

`--frame-salt` is for a sample `medh5 scrub` pseudonymised: given the salt the
scrub used, the DICOM side's frame of reference (the SEG's or the RTSTRUCT's on
import, the source images' on export) is compared with the sample's as the
scrub wrote it, rather than refused as another frame.

`from-dicom-seg` writes annotation `seg` and `from-rtstruct` writes `contours`
unless `--id` names another; `to-nnunet` exports annotation `seg` unless
`--annotation` names another, as dataset `Dataset001_medh5` unless
`--dataset-name` does, with every class the cases carry unless `--class` names
the ones to export — each of them examined by every case. A case without the
annotation is refused unless `--unlabeled test` writes it to `imagesTs`.

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
Every re-encoded dataset's stored bytes change (one whose type a rebuild would
lose is copied as stored, and printed as `KEPT`); no `content_id` does, because
the digest is over content and not over its encoding.

The output is **verified** against the digests it carries before the command
returns, and the exit code follows: a file whose bytes had already been
corrupted before you re-encoded it now fails here rather than reporting a
preserved `content_id` and exiting 0. What the check cannot read — the output
has replaced the input by then — fails it too, named on an `UNCHECKED` line
(`unverified` in `--json`).

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
`--repeats` (default 20) is how many times each measurement runs, its median
reported, and `--patch` (default 64) the side of the patches it reads. Each is
at least 1, in the Python API too: no run has no median, and a patch of side 0
reads nothing.

The `patch_throughput` row runs the PyTorch dataloader, so it needs
`medh5[torch]`; the standalone binary hands it to Python as it does the
converters. Without PyTorch the row is skipped with the reason on stderr, and
`--no-throughput` skips it outright.

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
whole policy (contract §3.4). A later minor version, read as a projection, is
refused (1.1 §2.2).

### `medh5 clinical export`

```
medh5 clinical export PATH [--key K] [--out RECORDS.json]
```

The logical-record bundle — clock, events, documents with their text, links —
as JSON, checked against `medh5-clinical-1.schema.json`. `--out` never names
the file being exported, by any path or link; that is refused before anything
is written.

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

Open and check every source — pins, identities, clocks, duplicated events
— and report every row: `eligible`, `uncertifiable`, `excluded` or `error`,
with the reason, the event count, the image filling each slot and the target
label. Exits 1 when anything was found. `--deep` re-verifies every dataset of
every source, not only the clinical ones.

With `--json`, each subject's merged history is listed once, under
`subjects` (its sources, event versions, links and owned documents), and each
row names what it admits in `selection`: the admitted versions in input order
(`events`, with their order bounds, tie groups and plan flags), the attested
links by position in its subject's `links`, and the payloads.

Both forms take the subjects one at a time and hold neither the cohort's
histories nor its document: `--json` writes each subject's history as soon as
it is merged, with the members in the order they become known — the two
fingerprints, `subjects`, `rows`, then `ok`, `counts` and `findings` — and
the table keeps one line per row. Memory follows the largest subject and the
rows, not the cohort.

### `medh5 task reconcile`

```
medh5 task reconcile MANIFEST [--base DIR] [--out OUT]
```

Record, per subject, the event versions several of its fragments hold and the
digest they share — what preflight checks duplicates against. The manifest is
replaced whole, in place or at `--out`; an `--out` that is one of the task's
sources, by any path or link, is refused before anything is written.

### `medh5 cache validate`

```
medh5 cache validate CACHE [--task MANIFEST] [--base DIR] [--json]
```

Check a feature cache's checksums and every source pin it records: *stale*
entries (T403) are told apart from *corrupt* ones (T401, T402), and an
event-level entry must name a version its source holds and the document that
version owns (T407). With `--task`,
also that it was built for this task and its cutoffs (T404), fitted on its
training partition (T405), and encodes only versions its rows admit (T406) —
the task is preflighted a subject at a time, keeping only each row's cutoff
and admitted versions. Entries copy their task's source references, so their
relative URIs resolve against `--base`, by default the task manifest's
directory with `--task` and the cache's directory without.

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
suite. `--case`, here and in `run` and `publish`, repeats to name several
cases; a name the corpus does not have is refused as a usage error.

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
[Conformance](../spec/conformance.md). A suite whose files no longer match its
`SHA256SUMS` fails the score, whatever the cases say — a score over files
that are not the published ones is not a score — and `--json` names those
files under `changed`.

## Related

- **[Check a file before training on it](../guides/validate.md)** — `validate` versus `verify`.
- **[Diagnostic codes](diagnostic-codes.md)** — every code these commands report.
- **[Cohort check codes](cohort-checks.md)** — what `medh5 dataset check` reports.
- **[Task and cache contract](../spec/task-cache-1.md#9-finding-codes)** — the T-codes `medh5 task` and `medh5 cache` report.
- **[How-to guides](../guides/index.md)** — these commands in the tasks they belong to.
