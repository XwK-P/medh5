# Runnable examples

Eight scripts. Four are standalone: the benchmarks behind the measured claims
in the [specification](../spec/medh5-1.0.md) (§7.0, §14.2, §14.3, §14.5) and the
[design rationale](../explanation/design-rationale.md), and a reference writer
that builds a complete file from the specification alone. None of those imports
`medh5` — they measure and exercise the *format*, with h5py and nothing between
the reader and the bytes.

The other four exercise [format 1.1](../spec/medh5-1.1.md) and the
[task and cache contract](../spec/task-cache-1.md) through the package's public
API, end to end, and the test suite runs each of them
(`tests/integrations/test_clinical_training.py`):

| Script | Does |
|---|---|
| [`clinical_longitudinal.py`](clinical_longitudinal.py) | The worked example of 1.1 §9.3 as a cohort: two CT visits, an intervening lab, a report and its revision, follow-up responses. Writes, reopens and validates every file, preflights a progression task, and prints a training batch whose rows see different visits, missing modalities and histories of different lengths |
| [`clinical_collection.py`](clinical_collection.py) | A subject split across two members of a `.medh5c` shard: refused until its duplicated event is reconciled, then trained on through worker processes |
| [`clinical_cache.py`](clinical_cache.py) | Event- and patient-level feature caches, then every way one stops being usable --- a source amended (stale), bytes rotted (corrupt), a whole-history embedding (inadmissible), statistics fitted on the wrong split --- and the explicit re-pin |
| [`bench_clinical.py`](bench_clinical.py) | Write, select, read, batch, cache and amend costs on a synthetic cohort ([results below](#clinical-paths-format-11)) |

```bash
pip install "medh5[torch]"
python docs/examples/clinical_longitudinal.py out/   # ~5 s
python docs/examples/clinical_collection.py out2/    # ~5 s
python docs/examples/clinical_cache.py out3/         # ~5 s
python docs/examples/bench_clinical.py --json bench.json   # ~1 min at the defaults
```

| Script | Produces |
|---|---|
| [`bench_encodings.py`](bench_encodings.py) | Multi-label voxel encodings on a 160³, 200-class phantom: size, write time, all-class patch read (spec §7.0) |
| [`bench_query.py`](bench_query.py) | Codec-matched single-class versus all-class patch reads across per-class, `layers` and `bitmask` (spec §7.0) |
| [`bench_io.py`](bench_io.py) | Codec profiles, `int16 + rescale` versus `float32`, and the foreground index versus `argwhere` (spec §14.2, §14.3) |
| [`reference_writer.py`](reference_writer.py) | A complete MEDH5 1.0 file exercising `core+seg+det+cls+reg+curation+training+longitudinal` — one subject at two timepoints — then validates `/meta` against the JSON Schema and runs the §15 semantic and integrity checks |

```bash
pip install numpy h5py hdf5plugin jsonschema
cd docs/examples
python bench_encodings.py     # ~15 s; run first, bench_io.py reads its files
python bench_query.py         # ~30 s
python bench_io.py            # ~20 s
python reference_writer.py    # ~10 s, writes case_0001.medh5 to the current directory
```

The benchmarks write their scratch `*.h5` files and JSON results beside
themselves; those are ignored by git and by the documentation build.

## Recorded results

macOS on Apple silicon, Python 3.12, h5py 3.16, hdf5plugin 6.0, local SSD,
medians over 10–50 repetitions. Absolute timings are hardware-dependent; the
ratios between encodings and codecs are what the decisions rest on.

### Multi-label encodings

160³ phantom, 200 classes (24 mutually exclusive organs and 176 overlapping
structures, 0.25 labels per voxel), greedy colouring `L = 5` layers, `P = 4`
bitmask planes:

| Codec | Encoding | Size | One-class 64³ read | All-class 64³ read |
|---|---|---|---|---|
| lz4 L1 + shuffle | per-class `bool` (0.x) | 3.57 MiB | 0.51 ms | 116.89 ms |
| lz4 L1 + shuffle | **`layers`** | **0.55 MiB** | **0.09 ms** | **6.69 ms** |
| lz4 L1 + shuffle | `bitmask` | 0.68 MiB | 2.79 ms | 10.14 ms |
| zstd L5 + bitshuffle | per-class `bool` (0.x) | 3.01 MiB | 0.33 ms | 139.27 ms |
| zstd L5 + bitshuffle | **`layers`** | **0.20 MiB** | 0.18 ms | 13.48 ms |
| zstd L5 + bitshuffle | `bitmask` | **0.15 MiB** | 10.31 ms | 37.06 ms |

`instances` on the same data is 0.08 MiB — 45× smaller than per-class dense,
because its cost tracks object volume rather than image volume.

### Codec profiles

192×256×256 `int16` CT, 32×64×64 chunks, read with h5py, a new 64³ window
each read:

| Codec | Write | Size | Ratio | 64³ read | Full read |
|---|---|---|---|---|---|
| lz4 L1 (`training`) | 0.07 s | 12.80 MiB | 1.9× | 1.3 ms | 0.02 s |
| lz4hc L8 (the 0.x default) | 0.54 s | 12.33 MiB | 1.9× | 1.0 ms | 0.02 s |
| zstd L9 + bitshuffle (`archive`) | 5.37 s | 9.53 MiB | 2.5× | 2.3 ms | 0.03 s |
| gzip L4 (`portable`) | 0.48 s | 9.72 MiB | 2.5× | 8.9 ms | 0.08 s |

The same volume stored as `float32` rather than `int16` plus a rescale is
36.75 MiB against 12.33 MiB at lz4hc L8 — three times the disk for no
information, which is why spec §4.2 recommends `int16` HU and `W907` flags the
alternative.

### Foreground sampling

160³, one class, 33 533 foreground voxels:

| Path | Time | Resident memory |
|---|---|---|
| 0.x: full mask and `np.argwhere`, cached per file and class | 9.2 ms | 0.8 MiB, O(volume) |
| 1.0: read a 4096-coordinate index | **0.52 ms** | **48 KiB**, O(1) |

The package's own sampler now reads one coordinate per draw rather than the
whole subsample: see [Tune performance](../guides/performance.md#the-numbers)
for its current numbers, and `medh5 bench` to reproduce them.

### Access pattern

`d[k][roi]` against `d[(k, *roi)]` on a 160³ `uint64` bit plane: **32.8 ms
against 0.8 ms**. The first form materialises the whole plane before slicing,
which is why spec §14.5 requires one-call slicing of the reference reader.

### Clinical paths (format 1.1)

`bench_clinical.py` at its defaults: 24 subjects, each with two `64×128×128`
`int16` CT visits (stored in `64×32×32` chunks, `training` codec), an MR at
follow-up for every second subject, 500 coded laboratory values over four
years, a report and its revision (≈2 KiB of text each) and two responses; a
task of 48 rows (two cutoffs per subject) with a CT slot of `32³` and an MR slot
of `16³`. Linux x86_64 container, 4 vCPUs, Python 3.13, NumPy 2.5, local disk;
medians of 20 repetitions where a measurement repeats. One machine, a
synthetic cohort: these validate the path and show where its costs are, and are
not clinical-scale performance.

| Measurement | Result |
|---|---|
| Write one sample, 1.1 with its history / 1.0 without | 116 ms / 87 ms |
| Bytes per sample, 1.1 / 1.0 | 2.88 MB / 2.76 MB (+4 %) |
| Open, read the clinical tables, select at a cutoff (cold) | 4.6 ms |
| Select again on the open sample (507 events) | 1.5 ms |
| One document's text | 0.02 ms |
| A `32³` slot window: from the 1.1 file / from its 1.0 imaging projection | 0.06 ms / 0.08 ms |
| Preflight of the whole task (24 sources, 48 rows; every pin and clinical digest re-verified) | 0.86 s |
| Build / validate the event-level report cache (48 entries) | 0.47 s / 0.27 s |
| Batches of 8, documents encoded on the fly: 0 / 2 workers | 171 / 158 rows/s |
| Batches of 8, documents from the cache: 0 / 2 workers | 505 / 187 rows/s |
| Amend one sample by one event (copy-on-write of the whole file) | 54 ms |
| Peak resident memory of the measuring process | 604 MiB |

The imaging path is unchanged: a window reads the same chunks from a 1.1 file
as from its 1.0 projection, and opening a sample reads no clinical table until
`Sample.clinical` is asked for. A history adds a few percent to a sample whose
images dominate it. Selection is metadata-only and milliseconds per subject;
preflight is dominated by re-verifying every pinned sample's clinical digests,
which is what lets it detect an edit under an unchanged stored root. At this
size two workers do not beat one process --- each worker pays its own opens for
48 small rows --- so measure your own shard and worker layout before choosing.

## The reference writer

`reference_writer.py` is the executable proof that the specification is
self-consistent: it follows the spec literally and its output passes JSON Schema
validation, cross-reference checks (E1xx–E6xx), per-object digests and
`content_id` (E7xx), plus reader-side round trips for the affine, the box↔slice
convention, instance mask decoding and lossless `layers ↔ bitmask` transcoding.
CI runs it on every push.

The sample it writes is longitudinal: baseline CT and PET sharing one frame of
reference, a follow-up CT on its own grid with shorter z coverage and its own
frame, organ and lesion annotations at both visits, a RECIST response label
spanning them, and the registration relating the two. It therefore also
exercises the timepoint rules (§3.7 — E106/E107/E108), cross-timepoint instance
tracking (§7.4 — persisted / resolved / new), change labels (§9 — E409) and the
longitudinal warnings W909–W911.

> **Note.** Reading a Blosc2-compressed MEDH5 file requires `import hdf5plugin`
> before the read, or HDF5 raises a confusing plugin-path error rather than a
> missing-filter error. This is the reason the spec defines the `portable` codec
> profile (§14.2).
