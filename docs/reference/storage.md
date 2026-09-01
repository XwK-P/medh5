# Storage

The on-disk layout, how bytes get chosen, and how the file proves it is intact.
The normative statement is [spec §2, §13 and §14](../spec/medh5-1.0.md); this page
is the operational half.

## Layout

An ordinary HDF5 file. `h5ls`, HDFView, h5py, MATLAB and Julia all open it.

```
case_0001.medh5                      (root attrs: medh5_version, medh5_kind,
│                                     medh5_profiles, content_id, digest_algo)
├── meta                             the sample document: JSON, UTF-8, one dataset
├── grids/<grid_id>                  empty groups; geometry lives in attributes
├── images/<image_id>                arrays, chunked and compressed
├── annotations/<ann_id>/            per-kind datasets, plus a header in attributes
├── transforms/<transform_id>/
└── index/<ann_id>/                  derived: foreground coords, counts, bboxes
```

```
$ medh5 tree case_0001.medh5      # the same listing, annotated with spec roles
```

`meta` is a **scalar variable-length UTF-8 string**, always. HDF5 filters do not
apply to variable-length data — it lives in the global heap — so a compressed
`meta` is not expressible. A vocabulary large enough for that to matter uses a
referenced label set (`form: "ref"`, §5.1) instead.

Everything a reader needs to interpret an array is in `meta` plus the grid
attributes. There is no sidecar and no dataset-wide schema to keep in sync.

## Codec profiles

```
$ medh5 recompress cohort/*.medh5 --profile training
$ medh5 recompress cohort/*.medh5 --profile archive --out cold/
```

| Profile | Images | Labels | For |
|---|---|---|---|
| `training` | blosc2 lz4:1 shuffle | blosc2 lz4:1 shuffle | fastest decompression; the hot dataloader path |
| `balanced` | blosc2 zstd:3 shuffle | blosc2 zstd:3 bitshuffle | general use — the default |
| `archive` | blosc2 zstd:9 bitshuffle | blosc2 zstd:9 bitshuffle | smallest on disk; cold storage and distribution |
| `portable` | gzip:4 | gzip:4 | readable without `hdf5plugin` |

Labels get `bitshuffle` where images get `shuffle`: label planes are low-entropy
integers, and bit-level transposition compresses them much better than
byte-level does.

`portable` exists because a collaborator with a plain h5py install should be
able to open the file at all. Blosc2 needs the filter plugin; gzip is in every
HDF5 build.

Datasets below a size threshold are stored **contiguous**: chunking and a filter
pipeline cost more than they save at that size, and a contiguous read is one
seek.

```python
from medh5.storage.codecs import PROFILES, resolve_profile, dataset_kwargs
PROFILES["archive"].description
```

## Chunking

The chunk is the real unit of I/O: reading one voxel reads a whole chunk. Two
forces pull against each other — sizing to the L3 cache keeps a patch inside
cache after decompression, sizing to the training patch keeps read
amplification low — and the optimiser resolves them by starting at the patch,
growing toward the cache budget, and stopping before the chunk is much larger
than the patch.

```python
w.add_grid("ct", shape=..., spacing=..., patch_hint=(96, 96, 96))
```

`patch_hint` is how you tell it what you will read. Without one it assumes a
reasonable default and you get a reasonable answer.

L3 size is detected per-core where the platform allows and falls back to
~1.375 MiB otherwise. Chunks are held between 512 KiB and 4 MiB.

**Stacked encodings chunk per plane.** A `layers` or `bitmask` annotation is
chunked `(1, *spatial_chunk)`, so reading one layer does not decompress the
others. Combined with reading a multi-class `dense()` **by plane rather than by
class**, a 200-class annotation packed into four layers is four reads and not
two hundred — which is where the 64³ patch time went from 117 ms to 4 ms.

```
$ medh5 recompress case.medh5 --profile training --rechunk
```

`--rechunk` re-derives the chunk shape as well as the codec.

## Integrity

Every object carries a SHA-256 over its **decompressed** content, with the
object's sample-root-relative path, dtype and shape fed into the hash first, so
two arrays with the same bytes and different shapes do not collide.

The root carries `content_id`: a Merkle digest over those object digests plus
the metadata attributes that define the sample.

```
$ medh5 verify cohort/*.medh5
$ medh5 verify case.medh5 --partial images/CT_tp0
```

```python
s.verify().ok
s.verify(partial=["images/CT_tp0"])
s.content_id
s.compute_content_id()          # recompute rather than read
```

Two properties follow from digesting content rather than encoding:

**Recompression preserves `content_id`.** Every stored byte changes; no digest
does. A cohort re-encoded for training is still verifiably the same data.

**The root is not a substitute for the objects.** Because `content_id` hashes
*digests*, editing a dataset without restamping it breaks that object's digest
and leaves the root matching. `verify` checks both, which is why it reports
per-object mismatches rather than a single yes/no.

```
$ medh5 fix case.medh5                       # diagnose
$ medh5 fix case.medh5 --rebuild-index
$ medh5 fix case.medh5 --rewrite-digests --reason "rebuilt by an external tool"
```

`--rewrite-digests` is not repair — see [CLI](cli.md#medh5-fix).

## Atomic writes

`medh5.create` and `medh5.amend` build a temporary file beside the target and
`os.replace` it into position on a clean exit. A file appears complete or not
at all; an exception aborts and leaves nothing behind.

`amend` is copy-on-write, and it copies through **every object it does not
understand** — including ones written by a future minor version. Amending never
silently drops what this reader cannot read.

## The sampling index

```
$ medh5 index build cohort/*.medh5 --max-coords 4096
```

Per voxel annotation, per class: a bounded sample of foreground coordinates,
the exact voxel count, and a tight bounding box. That makes foreground patch
sampling O(1) in the volume instead of a scan, and gives `dataset stats` its
class counts for a few hundred bytes instead of a decompression pass.

An index carries the digest of the annotation it derives from. When they
disagree the index is **stale**, readers must ignore it, and the validator
raises `W905`:

```
$ medh5 fix cohort/*.medh5 --rebuild-index
```

A stale index is not a file error. It is a cache that needs rebuilding, and the
format says so rather than making the file invalid.

## Collections

```
$ medh5 pack cohort/*.medh5 -o shard.medh5c
```

A `.medh5c` holds many sample roots under `samples/<key>`. Each member **is** a
sample root, so every reader works on it unchanged.

Packing copies chunks as raw bytes — nothing is decompressed, and `content_id`
is preserved. Unpacking reproduces the original files chunk for chunk. See
[Curation](curation.md#collections).

## Reading it without medh5

The point of plain HDF5:

```python
import h5py, json

with h5py.File("case_0001.medh5") as f:
    doc = json.loads(f["meta"][()])
    doc["identity"]["subject_id"]
    dict(f["grids"]["ct_tp0"].attrs)       # spacing, origin, direction
    f["images"]["CT_tp0"][10:20]           # needs hdf5plugin for blosc2
```

`import hdf5plugin` before opening if the file uses a blosc2 profile;
`--profile portable` avoids that requirement entirely.
