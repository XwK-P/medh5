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

### Choosing the right read API

| API | Returns | Reads arrays? | Holds a file handle? | Use when |
|-----|---------|---------------|----------------------|----------|
| `MEDH5File.read(path)` | `MEDH5Sample` dataclass with every image/mask materialised in memory | Yes, all of them | No (opens, reads, closes) | You want the whole sample in RAM for training/preprocessing. |
| `MEDH5File.read_meta(path)` | `SampleMeta` only | No | No (opens, reads attrs, closes) | You need labels / shape / spatial metadata to build a manifest, without paying for array I/O. |
| `with MEDH5File(path) as f:` | Context-manager `MEDH5File` instance; access via `f.images`, `f.seg`, `f.meta`, `f.bbox_arrays()` | Only the slices you index | Yes, until the `with` block exits | You want lazy slicing (viewer-style), repeated partial reads, or patch sampling on a large volume. |

Rule of thumb: `read_meta` for catalog/index builders, the context manager
for viewers and patch pipelines, `read` when you truly need the full sample
as numpy arrays.

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
report.errors              # list[ValidationIssue(code, message, location)]
report.warnings
```

Each `ValidationIssue` carries an optional `location` string (e.g.
`"images/CT"`, `"seg/tumor"`, `"bboxes"`, `"checksum_sha256"`,
`"extra.nnunetv2.labels"`). UIs can use it to highlight the offending
dataset without re-parsing `message`. `ValidationIssue.to_dict()` omits
the key when `location is None`, so the JSON shape stays compact for
issues that don't have a natural location.

Or, the one-call shortcut when you just want a boolean answer:

```python
MEDH5File.is_valid("sample.medh5")                # False if any errors
MEDH5File.is_valid("sample.medh5", strict=True)   # also False on warnings
```

`validate()` intentionally does not take `strict` — strictness lives on the
returned report, keeping validation policy out of the report-building layer.

## Integrity

`MEDH5File.verify()` returns a tri-state `VerifyResult` so callers can
distinguish "no checksum was ever stored" from "checksum verified
successfully" — the two cases previously both returned `True`, which
made trustworthy audit UIs impossible to build.

```python
from medh5 import MEDH5File, VerifyResult

match MEDH5File.verify("sample.medh5"):
    case VerifyResult.OK:        ...   # stored checksum matches data
    case VerifyResult.MISSING:   ...   # no checksum was stored (opt-in)
    case VerifyResult.MISMATCH:  ...   # data has diverged from stored digest
```

Checksums are written opt-in via `MEDH5File.write(..., checksum=True)`.

## Concurrent reads: `open_shared`

HDF5 refuses to reopen a file already open elsewhere in the same
process. Lazy-read consumers (napari plugins, dashboards, viewers) that
need to hand out independent "handles" while keeping the underlying
file single-open should use `open_shared`:

```python
from medh5 import open_shared

with open_shared("sample.medh5") as f:
    patch = f["images/CT"][10:42, 20:84, 20:84]
```

`open_shared` is a ref-counted context manager keyed by
`Path.resolve()`: the first caller opens the file, subsequent callers
(in any thread of the same process) receive the *same* `h5py.File`,
and the handle closes when the last `with` block exits. Callers must
treat the returned object as read-only.

## Post-write callbacks: `on_reopened`

Every mutating entry point — `MEDH5File.update`, `update_meta`,
`add_seg`, and `set_review_status` — accepts `on_reopened`, invoked
with the file `Path` after the HDF5 write handle has closed *and only
when the operation succeeded*. This is the hook for lazy-read
consumers to re-acquire handles or rebind cached views without
inventing their own event system:

```python
def reopen(path):
    # re-issue any cached dask arrays / layer.data bindings here
    ...

MEDH5File.update("sample.medh5", meta={"label": 2}, on_reopened=reopen)
```

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
    on_reopened=None,    # callback for lazy-read consumers (see above)
)
```

Convenience shortcuts delegate to `update()`:

```python
MEDH5File.update_meta("sample.medh5", label=2, extra={"reviewed": True})
MEDH5File.add_seg("sample.medh5", "new_mask", mask_array)
```

`update()` (and therefore every shortcut) requires **exclusive write
access**: close any open `MEDH5File(...)` context managers and drop
lazy views first, or HDF5 raises an "already open" error. `medh5`
detects that specific error and raises `MEDH5FileError` with a clear
message pointing at the constraint.

## Spatial affine composition

Viewer-style consumers can ask `SpatialMeta` for a composed homogeneous
affine matrix in one call:

```python
from medh5 import MEDH5File

meta = MEDH5File.read_meta("sample.medh5")
affine = meta.spatial.as_affine(ndim=3)
if affine is None:
    # Rotation is effectively identity — use simpler scale+translate
    ...
else:
    # (ndim+1, ndim+1) matrix: direction · diag(spacing) + origin
    assert affine.shape == (4, 4)
```

`as_affine` returns `None` when `direction` is absent or numerically
close to identity, so consumers can pick the cheap path when a full
affine isn't needed.

## Bbox clamping

`validate_bboxes` clamps out-of-range boxes to the sample's spatial
bounds and reports every adjustment it made:

```python
from medh5 import validate_bboxes

clamped, issues = validate_bboxes(boxes, sample_shape=(128, 256, 256))
for i, axis, reason in issues:
    # reason ∈ {"min<0", "max>shape", "min>max"}
    print(f"box {i} axis {axis}: {reason}")
```

The input array is not mutated; `clamped` is always a fresh `int64`
array shaped `(n, ndim, 2)`. Shape mismatches raise
`MEDH5ValidationError`.

## `extra` subsystem conventions

Three `extra` sub-keys are well-known to medh5 and get light validation
on read:

| Key | Producer | Notes |
|-----|---------|-------|
| `extra["review"]` | `set_review_status` | Stamped with `schema_version: 1`. `status` must be a string. |
| `extra["nnunetv2"]` | `medh5.io.from_nnunetv2` | Stamped with `schema_version: 1`. `labels` must be `dict[str, int]`. |
| `extra["checksum"]` | reserved | Reserved for future structured checksum metadata. |

`read_meta` emits a `UserWarning` for malformed shapes and for any
`schema_version` newer than this library understands; the raw payload
is preserved in `meta.extra` either way.

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
