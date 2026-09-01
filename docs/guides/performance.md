# Tune performance

Four levers decide how fast a training loop reads: the **sampling index**, the
**chunk shape**, the **codec profile** and the **worker count**. This page is
what each is worth, how to tell which one you are missing, and how to reproduce
the numbers on your own hardware.

The short version, in payoff order:

| Lever | Worth | Do it when |
|---|---|---|
| **Build a sampling index** | foreground sampling goes from O(volume) to O(1) — 30 ms → 0.9 ms on a 12 Mvox volume, 312 ms → 0.9 ms at 512³ | always, unless you only sample uniformly |
| **Set `patch_hint` on the grid** | sizes chunks to what you will actually read | at write time, if you know your patch size |
| **`--profile training`** | decompresses fastest | the cohort is read far more often than written |
| **`num_workers > 0`** | ~330 → 600–850 patches/s | always, with `worker_init_fn` |

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
batch["meta"]["used_index"]        # False -> you are paying for a scan
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

```python
w.add_grid("ct", shape=..., spacing=..., patch_hint=(96, 96, 96))
```

`patch_hint` is how you say what you will read. Without one it assumes a
reasonable default and you get a reasonable answer. L3 is detected per core
where the platform allows and falls back to ~1.375 MiB; chunks are held between
512 KiB and 4 MiB.

For an existing cohort, `--rechunk` re-derives the chunk shape as well as the
codec:

```bash
medh5 recompress cohort/*.medh5 --profile training --rechunk
```

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

`worker_init_fn` is **required** for `num_workers > 0`: HDF5 handles must not
cross a `fork`. See [PyTorch and MONAI](../reference/torch.md#the-dataloader).

## The numbers

Measured on a 192×256×256 synthetic CT with eight classes.

| Metric | Target | 0.x | Measured |
|---|---|---|---|
| 64³ patch, multi-class labels only | ≤ 10 ms | 117 ms | **4.0 ms** |
| Foreground centre sampling *(indexed)* | ≤ 1 ms, O(1) memory | 9.2 ms, O(volume) | **0.90 ms** |
| Metadata-only read | ≤ 2 ms | ~1.5 ms | **0.21 ms** |
| Full `open()` → first patch | ≤ 15 ms | ~120 ms | **2.4 ms** |
| Sustained 96³ throughput | ≥ 400 patches/s | ~60 | **600–850** (4 workers) |

```
$ medh5 bench                       # builds a synthetic sample and measures
$ medh5 bench case.medh5 --patch 64 --repeats 20 --workers 4 --json
```

Two things to know before quoting these.

**The sampling row needs an index.** `bench` calls `build_index()` on the
sample it builds, so 0.90 ms is the indexed path — the one you get after
`medh5 index build`, not the one you get by default. Unindexed, the same draw
scans the labels: 30 ms on this volume, 312 ms at 512³, growing with the volume
while the indexed draw stays flat. `used_index` in the batch metadata says which
you measured.

**`bench` does not check the throughput target.** The first four rows carry a
target it verifies and reports against; throughput depends on worker count, so
it is measured and printed without one. `medh5 bench` with no `--workers` runs
single-process and reports around 330 patches/s — below the 400 in the table,
and still followed by *all targets met*, which is a statement about the four
checked rows. Pass `--workers 4` to reproduce the number above.

Two decisions are behind the label-read number: each stacked plane is chunked
separately, so one layer reads without the others; and a multi-class `dense()`
reads **by plane rather than by class**, so a 200-class annotation packed into
four layers is four reads and not two hundred.

## Related

- **[Storage](../reference/storage.md)** — codec profiles, chunking and the index in full.
- **[PyTorch and MONAI](../reference/torch.md)** — the datasets and samplers these levers act on.
- **[`medh5 bench`](../reference/cli.md#medh5-bench)** — every flag.
