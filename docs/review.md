# Review / QA workflow

Annotation review state lives inside the `.medh5` file itself — no sidecar
database, no external tracker. State is stored under `extra["review"]`, so
no schema change is required and plain HDF5 tools can inspect it.

## Status values

| Status     | Meaning                                             |
|------------|-----------------------------------------------------|
| `pending`  | Not yet reviewed (implicit default).                |
| `reviewed` | Reviewer signed off.                                |
| `flagged`  | Needs further attention, not yet rejected.          |
| `rejected` | Annotation is incorrect and should not be used.     |

## Python API

```python
from medh5 import MEDH5File, ReviewStatus

MEDH5File.set_review_status(
    "sample.medh5",
    status="reviewed",
    annotator="puyang",
    notes="bbox coverage ok",
)

review: ReviewStatus = MEDH5File.get_review_status("sample.medh5")
review.status        # "reviewed"
review.annotator     # "puyang"
review.timestamp     # ISO 8601 UTC
review.notes
review.history       # list of prior states, oldest first
```

Each call to `set_review_status` appends the previous state to `history`,
so the full review trail is preserved in-file.

`set_review_status` uses the same integrity discipline as any other
`update()`: the stored SHA-256 is verified before mutating, and the new
checksum is recomputed after.

## CLI

```bash
medh5 review set sample.medh5 --status reviewed --annotator puyang --notes "ok"
medh5 review get sample.medh5
medh5 review get sample.medh5 --json
medh5 review list data/ --status pending
```

`review list` walks a directory and reports review status for every
`.medh5` file — useful for "what still needs review?" dashboards.

## Importing edited masks

Round-trip an edited seg mask from 3D Slicer / ITK-SNAP back into the
`.medh5` without rewriting image data:

```bash
medh5 review import-seg sample.medh5 --name tumor --from edited.nii.gz
medh5 review import-seg sample.medh5 --name tumor --from edited.nii.gz --resample --replace
```

`--resample` reslices onto the file's grid if affines differ; `--replace`
overwrites an existing mask of that name. The equivalent Python API is
`medh5.io.import_seg_nifti`.
