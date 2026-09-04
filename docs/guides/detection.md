# Detection and boxes

Write boxes, and crop the voxels they refer to without an off-by-one.

## The half-voxel, once

A box sits at **voxel edges**; an integer index is a **voxel centre**. So a box
`[a, b]` covers the numpy slice `a + 0.5 : b + 0.5`.

That is the whole convention, and it is where every off-by-one in medical
detection comes from. The library does the arithmetic for you — the point of
writing it down is so you never do it by hand:

```python
from medh5.geometry import box_to_slices

box_to_slices([[1.5, 5.5]])        # (slice(2, 6),)
```

The extent and the voxel count agree, which is the property the convention buys:
the box is `5.5 − 1.5 = 4.0` wide and the slice is 4 voxels, covering centres 2,
3, 4 and 5. Round the corners yourself instead and a one-voxel box on integer
edges comes out empty or two voxels wide depending on its parity.

## Write them

```python
w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct",
            space="index", scores=[0.91], instance_ids=[7])
```

`space="index"` or `space="world"`. `instance_ids` is what joins a box to the
same physical object at another visit — see
[Longitudinal studies](longitudinal.md).

## Read them, and crop

```python
b = s.annotations["lesions"]
b.boxes[0]              # [[1.5, 5.5], [3.5, 8.5], [4.5, 10.5]]
b.as_slices()[0]        # (slice(2, 6), slice(4, 9), slice(5, 11))
b.class_ids, b.scores, b.instance_ids
```

`as_slices()` is the round trip you want. Use it rather than rounding yourself:

```python
crop = s.images["CT"].read(b.as_slices()[0])     # exactly the boxed voxels
```

## A lesion drawn on one slice

The common radiology annotation is 2-D on a plane of a 3-D study: a box with a
degenerate axis (`lo == hi`) plus `slice_index`:

<!-- illustrative -->
```python
w.add_boxes("lesions", boxes, class_ids=["lesion"], grid="ct",
            space="index", slice_index=[37, 41])
```

`slice_index` is a **per-box column** — one plane per box, shape `(N,)`, each
plane inside the grid. All three rules are enforced by the writer, by
`as_slices()` on read, and by `medh5 validate`, under `E405`. The same rule in
all three places means a file written before the check existed fails the same
way rather than reading differently.

## Check it

```bash
medh5 validate case.medh5 --level strict --profile det
```

`--profile det` holds the file to having **some** detection annotation — the
check is `task='detection'`, which keypoints, points, contours and meshes satisfy
as well as boxes. A file carrying only keypoints passes `--profile det`, so a
pipeline that specifically needs boxes still gets its `KeyError`. Check for the
annotation you are going to read:

```python
with medh5.open(path) as s:
    ann = s.annotations.get("lesions")
    if ann is None or ann.kind != "boxes":
        raise SystemExit(f"{path}: no box annotation named 'lesions'")
```

## Related

- **[Annotation kinds](../reference/annotations.md#boxes)** — oriented boxes, keypoints, contours, meshes.
- **[Diagnostic codes](../reference/diagnostic-codes.md#e405)** — what `E405` means.
- **[Specification §8.1](../spec/medh5-1.0.md)** — the normative box↔slice rule.
