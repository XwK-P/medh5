# Tune performance

Four levers decide how fast a training loop reads: the **sampling index**, the
**chunk shape**, the **codec profile** and the **worker count**. This page is
what each is worth, how to tell which one you are missing, and how to reproduce
the numbers on your own hardware.

The short version, in payoff order:

| Lever | Worth | Do it when |
|---|---|---|
| **Build a sampling index** | foreground sampling goes from O(volume) to O(1) — 63 ms → 0.03 ms on a 12 Mvox volume, 650 ms → 0.03 ms at 512³ | always, unless you only sample uniformly |
| **Set `patch_hint` on the grid** | sizes chunks to what you will actually read | at write time, if you know your patch size |
| **`--profile training`** | decompresses fastest | the cohort is read far more often than written |
| **`num_workers > 0`** | ~130 → ~270 patches/s with four workers on four cores | always, with `worker_init_fn` |
| **`FileGroupedSampler`** | one open per file per epoch instead of one per item | shuffled training over more files than the handle cache holds |

## Build the index first

This is the one that changes the complexity class rather than the constant.

```bash
medh5 index build cohort/*.medh5 --max-coords 4096
```

Per voxel annotation, per class, it stores a bounded sample of foreground
coordinates, the exact voxel count and a tight bounding box. Foreground patch
sampling becomes a lookup instead of a scan, and `medh5 dataset stats` gets its
class counts for a few hundred bytes rather than a decompression pass.

**How to tell you are missing it.** Nothing fails — the sampler falls back to
scanning the labels and records what it did:

```python
patch = batch["meta"]["patch"]
patch["used_index"]     # True, False, or None
```

Three states, because there are three answers. `True` means the index answered
the foreground query; `False` means the foreground had to be scanned because no
current index was present; **`None` means the question did not arise** — a
uniform draw consults no index at all.

`False` is the one that costs you:

```python
scanning = [p for p in patches if p["used_index"] is False]
```

An index carries the digest of the annotation it derives from, so it goes
**stale** when the annotation changes. Readers must then ignore it, and the
validator raises `W905`:

```bash
medh5 fix cohort/*.medh5 --rebuild-index
```

A stale index is not a file error. It is a cache that needs rebuilding, and the
format says so rather than making the file invalid.

## Size the chunks to the patch

The chunk is the real unit of I/O: reading one voxel reads a whole chunk. Two
forces pull against each other — sizing to the L3 cache keeps a patch in cache
after decompression, sizing to the training patch keeps read amplification low
— and the optimiser resolves them by starting at the patch, growing toward the
cache budget, and stopping before the chunk is much larger than the patch.

<!-- illustrative -->
```python
w.add_grid("ct", shape=..., spacing=..., patch_hint=(96, 96, 96))
```

`patch_hint` is how you say what you will read. Without one it assumes a
reasonable default and you get a reasonable answer. L3 is detected per core
where the platform allows and falls back to ~1.375 MiB; chunks are held between
512 KiB and 4 MiB.

For an existing cohort, `--rechunk` re-derives the chunk shape as well as the
codec, by the writer's own rule — from each grid's `patch_hint`, one plane per
chunk for `layers`, `bitmask` and `probmap` — so a re-chunked file is chunked
the way a fresh write would be:

```bash
medh5 recompress cohort/*.medh5 --profile training --rechunk
```

(Before 1.4.2 it let h5py choose, and h5py's guess spanned the stacked axis the
spec keeps at 1, which the validator reported as `W902`.)

## Choose a codec profile

Storage is a training parameter. `training` decompresses fastest; `archive` is
smallest. The full table is in [Storage](../reference/storage.md#codec-profiles).

```bash
medh5 recompress cohort/*.medh5 --profile training
```

Every stored byte changes and no `content_id` does — the digest is over content,
not over its encoding — so a recompressed cohort is still verifiably the same
data.

## Use workers

```python
loader = DataLoader(dataset, batch_size=2, num_workers=8,
                    worker_init_fn=worker_init_fn, collate_fn=collate)
```

`worker_init_fn` drops handles inherited across a `fork`. It is recommended
rather than required — the cache is PID-keyed and resets on first use in a
forked worker either way. See [PyTorch and MONAI](../reference/torch.md#the-dataloader).

## Keep a file's patches together

Each worker keeps its 32 most recently used files open. `shuffle=True`
permutes *items*, so with `samples_per_volume` items per file and more files
than that, consecutive items rarely share a file and most of them open it
again: at 100 small files and 4 items each, 304 opens an epoch and 0.77 ms per
item.

`FileGroupedSampler` shuffles files and yields each file's items together —
100 opens, one per file, and 0.30 ms per item on the same run:

```python
from medh5.torch import FileGroupedSampler

loader = DataLoader(ds, batch_size=4, sampler=FileGroupedSampler(ds, seed=0),
                    num_workers=8, worker_init_fn=worker_init_fn, collate_fn=collate)
for epoch in range(epochs):
    ds.set_epoch(epoch)     # new patches, and a new file order
```

With workers, make `samples_per_volume` a multiple of `batch_size`: a batch is
read by one worker, so a file whose items straddle two batches is opened by
both. `set_cache_size(n)` is the other lever — raise it toward the number of
files a worker cycles through, lower it when descriptors are scarce.

## On a network filesystem

HDF5's file locking is unreliable on NFS, Lustre and GPFS, where training
clusters usually keep their data, and a lock that hangs or fails there looks
like a slow or broken file. Reading is safe without it, so turn it off for the
training job:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
```

Set it in the job's environment before Python starts, so every `DataLoader`
worker inherits it. medh5 never sets it for you (§14.4): it also disables the
lock that keeps two writers apart, and whether that is safe is a fact about your
cluster, not about the file.

## The numbers

Measured with 2.0 on one four-core machine, on a 192×256×256 synthetic CT with
eight classes; 1.4.4 on the same machine was as fast or slower on every row
([changelog](../changelog.md)). The 0.x column is the layout 1.0 replaced, as
measured then.

| Metric | Target | 0.x | Measured |
|---|---|---|---|
| 64³ patch, multi-class labels only | ≤ 10 ms | 117 ms | **7.8 ms** |
| Foreground centre sampling *(indexed)* | ≤ 1 ms, O(1) memory | 9.2 ms, O(volume) | **0.03 ms** |
| … at 63 classes | ≤ 1 ms | — | **0.05 ms** |
| Metadata-only read | ≤ 2 ms | ~1.5 ms | **0.19 ms** |
| Full `open()` → first patch | ≤ 15 ms | ~120 ms | **5.7 ms** |
| Sustained 96³ throughput | ≥ 400 patches/s | ~60 | **~270** (4 workers, 4 cores) |

```
$ medh5 bench                       # builds a synthetic sample and measures
$ medh5 bench case.medh5 --patch 64 --repeats 20 --workers 4 --json
```

Two things to know before quoting these.

**The sampling rows need an index.** `bench` calls `build_index()` on the
samples it builds, so 0.03 ms is the indexed path — the one you get after
`medh5 index build`, not the one you get by default. Unindexed, the same draw
scans the labels: 63 ms on this volume, 650 ms at 512³, growing with the volume
while the indexed draw stays flat. `used_index` in the batch metadata says which
you measured. The 63-class row exists because the class count is the other
axis: before 1.4.2 each draw re-read the index's class table once per class,
0.9 ms at eight classes and 7.6 ms at 63.

**`bench` does not check the throughput target.** The rows with a target are
verified and reported against; throughput depends on worker count, so it is
measured and printed without one. `medh5 bench` with no `--workers` runs
single-process and reports around 130 patches/s here — below the 400 in the
table, and still followed by *all targets met*, which is a statement about the
checked rows. Pass `--workers 4` to reproduce the number above. It is the row
that depends most on the machine: here the four workers share four cores with
the main process, and 1.4.4 measured the same.

Two decisions are behind the label-read number: each stacked plane is chunked
separately, so one layer reads without the others; and a multi-class `dense()`
reads **by plane rather than by class**, so a 200-class annotation packed into
four layers is four reads and not two hundred.

## Related

- **[Storage](../reference/storage.md)** — codec profiles, chunking and the index in full.
- **[PyTorch and MONAI](../reference/torch.md)** — the datasets and samplers these levers act on.
- **[`medh5 bench`](../reference/cli.md#medh5-bench)** — every flag.
