# File format

`medh5` is plain HDF5 with a small, fixed schema. Any HDF5 reader can inspect
a `.medh5` file — use `h5ls -v sample.medh5` or `h5dump -A sample.medh5`.

## On-disk layout

```
sample.medh5
├── images/            (group, required, >= 1 entry)
│   ├── <name1>        (dataset, N-D, Blosc2-compressed, chunked)
│   ├── <name2>        (dataset, N-D, Blosc2-compressed, chunked)
│   └── ...            (all arrays share the same shape)
├── seg/               (group, optional)
│   ├── <mask1>        (dataset, N-D bool, Blosc2-compressed, chunked)
│   └── ...            (all masks share the image shape)
├── bboxes             (dataset, [n, ndims, 2], optional)
├── bbox_scores        (dataset, [n], optional)
└── bbox_labels        (dataset, [n] variable-length string, optional)
```

### Root attributes

| Attribute         | Type                | Notes                                                          |
|-------------------|---------------------|----------------------------------------------------------------|
| `schema_version`  | str                 | Currently `"1"`. Readers must refuse unknown versions.         |
| `image_names`     | JSON list of str    | Declared modality order. Must match `images/` children exactly.|
| `label`           | int \| str          | Image-level label. Optional.                                   |
| `label_name`      | str                 | Human-readable name for `label`. Optional.                     |
| `has_seg`         | bool                | True iff `seg/` has at least one entry.                        |
| `seg_names`       | JSON list of str    | Names of masks in `seg/`.                                      |
| `has_bbox`        | bool                | True iff `bboxes` dataset exists.                              |
| `extra`           | JSON string         | User-defined metadata dict.                                    |
| `checksum_sha256` | str                 | Optional SHA-256 over images + seg + bboxes + critical attrs.  |

### `images/` group attributes

Spatial metadata lives on the `images` group (not on the root), because it
describes the shared spatial grid of the image datasets.

| Attribute      | Type          | Notes                                                     |
|----------------|---------------|-----------------------------------------------------------|
| `shape`        | int array     | The shared voxel shape of all image datasets.             |
| `spacing`      | float array   | Voxel spacing per axis (physical units).                  |
| `origin`       | float array   | Origin of voxel (0, 0, …) in physical space.              |
| `direction`    | float array   | `ndim × ndim` direction cosines, flattened row-major.     |
| `axis_labels`  | string array  | Human labels (e.g. `["z", "y", "x"]`) — length `ndim`.    |
| `coord_system` | str           | `"RAS"`, `"LPS"`, or similar. Informational.              |
| `patch_size`   | int array     | Suggested patch size (informational; used for chunking).  |

Both `direction` and `axis_labels` are dimension-checked on read; malformed
values raise `MEDH5SchemaError`.

## Compression presets

The `compression` parameter on `write()` picks a Blosc2 codec/level pair:

| Preset       | Codec  | Level | Use case                             |
|--------------|--------|-------|--------------------------------------|
| `"fast"`     | lz4    | 3     | Quick writes, moderate ratio         |
| `"balanced"` | lz4hc  | 8     | **Default** — good ratio + throughput |
| `"max"`      | zstd   | 9     | Maximum ratio, slower writes         |

Bounding-box arrays smaller than 64 entries are written uncompressed to avoid
per-chunk filter overhead.

## Chunk optimization

Image and segmentation datasets are chunked with sizes computed by
`medh5.optimize_chunks()`, ported from DKFZ `mlarray`. Targets ~1.4 MiB per
chunk to fit comfortably in L3 cache during patch-based training reads.

```python
from medh5 import optimize_chunks

chunks = optimize_chunks(
    image_shape=(128, 256, 256),
    patch_size=192,
    bytes_per_element=4,      # float32
)
```

Chunk shape is chosen so that reads of the advertised patch size hit the
minimum number of chunks possible.

## Integrity

When `write(..., checksum=True)`, a SHA-256 digest is stored in the
`checksum_sha256` attribute. The digest covers:

- all datasets under `images/`
- all masks under `seg/`
- `bboxes`, `bbox_scores`, `bbox_labels` (if present)
- the attribute set in `meta._ROOT_META_ATTRS` / `_IMAGE_META_ATTRS`

`MEDH5File.verify(path)` recomputes and compares. `MEDH5File.update()` and
its shortcuts (`update_meta`, `add_seg`, `set_review_status`) **verify the
stored checksum before mutating**, so an already-corrupted file cannot
silently get a fresh checksum stamped on top. Use `force=True` for
intentional repairs.

## Atomic writes

`MEDH5File.write()` writes to a sibling temp file, `fsync`s, and
`os.replace`s into place. A crash, OOM, or Ctrl-C mid-write leaves the
pre-existing file at the destination untouched — never a truncated `.medh5`
at the target path.

## Trust model

`extra` is arbitrary JSON. When reading `.medh5` files from untrusted
sources, be aware that very large or deeply nested JSON can consume
significant memory. Arrays are still bounded by your compression and chunk
layout, not by user input.

## Forward compatibility

Readers see `schema_version` at the root and must refuse unknown values with
`MEDH5SchemaError`. Additional attributes on the root or `images/` group that
the current schema does not describe are ignored on read but preserved on
`update_meta`.

## Multi-modality invariants

- `images/` is a group with one dataset per modality, all sharing the same
  shape.
- `seg/` is a group with one boolean dataset per mask, all matching the image
  shape.
- `bboxes` shape is `[n, ndims, 2]` — `[z_min, z_max], [y_min, y_max], ...`
  ranges. `bbox_scores` length and `bbox_labels` length both equal `n`.

These invariants are enforced at write time and re-checked by `validate()`.
