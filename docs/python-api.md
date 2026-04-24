# Python API

All of medh5's public surface is re-exported from the top-level package:

```python
from medh5 import (
    MEDH5File,
    MEDH5Sample,
    SampleMeta,
    SpatialMeta,
    ValidationReport,
    ReviewStatus,
    optimize_chunks,
    MEDH5Error, MEDH5FileError, MEDH5SchemaError, MEDH5ValidationError,
)
```

The central type is `MEDH5File`. It exposes two styles:

- **Static methods** (`write` / `read` / `read_meta` / `verify` / `validate`
  / `update` / `update_meta` / `add_seg`) for one-shot operations.
- **Context-manager instance** (`with MEDH5File(path) as f`) for lazy typed
  access via `f.images`, `f.seg`, `f.meta`.

## Writing

```python
MEDH5File.write(
    path,
    images: dict[str, np.ndarray],
    *,
    seg: dict[str, np.ndarray] | None = None,
    bboxes: np.ndarray | None = None,
    bbox_scores: np.ndarray | None = None,
    bbox_labels: list[str] | None = None,
    label: int | str | None = None,
    label_name: str | None = None,
    spacing: Sequence[float] | None = None,
    origin: Sequence[float] | None = None,
    direction: Sequence[Sequence[float]] | None = None,
    axis_labels: Sequence[str] | None = None,
    coord_system: str | None = None,
    patch_size: int | Sequence[int] | None = None,
    extra: dict | None = None,
    compression: str = "balanced",    # "fast" | "balanced" | "max"
    checksum: bool = False,
)
```

All image arrays must share the same shape; all seg masks must match that
shape. Writes are atomic (see [File format](file-format.md)).

## Reading

Eager — load everything into a `MEDH5Sample` dataclass:

```python
sample = MEDH5File.read("sample.medh5")

sample.images          # dict[str, np.ndarray]
sample.seg             # dict[str, np.ndarray] | None
sample.bboxes          # np.ndarray | None
sample.meta            # SampleMeta
```

Lazy — open the file once and slice h5py datasets directly:

```python
with MEDH5File("sample.medh5") as f:
    patch = f.images["CT"][10:42, 20:84, 20:84]   # only this chunk is decompressed
    if f.seg is not None:
        mask_patch = f.seg["tumor"][10:42, 20:84, 20:84]
    meta = f.meta        # no array reads
```

Metadata-only — read attributes without touching arrays:

```python
meta = MEDH5File.read_meta("sample.medh5")
meta.label
meta.image_names
meta.spatial.spacing
```

## `SampleMeta` and `SpatialMeta`

```python
@dataclass
class SampleMeta:
    image_names: list[str]
    label: int | str | None
    label_name: str | None
    has_seg: bool
    seg_names: list[str]
    has_bbox: bool
    extra: dict
    spatial: SpatialMeta
    ...

@dataclass
class SpatialMeta:
    shape: list[int]
    spacing: list[float]
    origin: list[float]
    direction: list[list[float]]   # ndim × ndim
    axis_labels: list[str]
    coord_system: str | None
    patch_size: list[int] | None
    ...
```

Both serialize to HDF5 attributes round-trip. `SampleMeta.validate()` checks
dimension consistency (`direction` is `ndim × ndim`, `axis_labels` length
equals `ndim`, `patch_size` length and element types).

## Validation

```python
report = MEDH5File.validate("sample.medh5")

report.ok()                # bool, no errors
report.ok(strict=True)     # bool, no errors AND no warnings
report.errors              # list[ValidationIssue(code=..., message=...)]
report.warnings
```

Or, the one-call shortcut when you just want a boolean answer:

```python
MEDH5File.is_valid("sample.medh5")                # False if any errors
MEDH5File.is_valid("sample.medh5", strict=True)   # also False on warnings
```

`validate()` intentionally does not take `strict` — strictness lives on the
returned report, keeping validation policy out of the report-building layer.

## Integrity

```python
MEDH5File.verify("sample.medh5")    # True iff stored checksum matches recomputed
```

If no checksum was stored, `verify()` raises `MEDH5ValidationError`.

## In-place updates

The unified `update()` entry point handles metadata, segmentation, and bbox
mutations in one call. It verifies any stored checksum before mutating (so
a pre-corrupted file cannot silently get a fresh checksum stamped on top),
re-syncs derived attributes (`image_names`, `shape`, `has_seg`, `seg_names`,
`has_bbox`), and recomputes the checksum at the end when one is present.

```python
MEDH5File.update(
    "sample.medh5",
    meta={"label": 2, "extra": {"reviewed": True}, "spacing": [1.0, 0.5, 0.5]},
    seg_ops={"add": {"organ": organ_mask}, "remove": ["old_mask"]},
    bbox_ops={"bboxes": new_bboxes, "bbox_labels": ["tumor"]},
    force=False,         # set True to skip pre-mutation checksum verify
)
```

Convenience shortcuts delegate to `update()`:

```python
MEDH5File.update_meta("sample.medh5", label=2, extra={"reviewed": True})
MEDH5File.add_seg("sample.medh5", "new_mask", mask_array)
```

## Exceptions

```
MEDH5Error
├── MEDH5FileError        # IO / HDF5 open failures
├── MEDH5SchemaError      # unknown schema version, malformed spatial attrs
└── MEDH5ValidationError  # content fails validate() or integrity check
```

Catch `MEDH5Error` at the boundary to handle all medh5 failures uniformly.

## Chunk optimizer

```python
from medh5 import optimize_chunks

chunks = optimize_chunks(
    image_shape=(128, 256, 256),
    patch_size=192,
    bytes_per_element=4,
)
```

Called automatically by `write()`; exposed for callers that want to build
HDF5 datasets manually outside `MEDH5File`.
