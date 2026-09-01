# Registration between visits

Relate two frames of reference so you can move points, boxes and predictions
between them.

Two visits are two frames. A transform relates them, and — because both visits
live in the same file — it is an object in that file rather than a convention
between two filenames.

## Write one

```python
w.add_transform("tp0_to_tp1", kind="affine",
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                matrix=matrix, invertible=True)
```

Kinds: `affine`, `displacement`, `bspline`, `composite`.

```python
t = s.transform_between("tp0", "tp1")
t.kind, t.from_frame, t.to_frame, t.is_invertible
t.transform_points(points)         # world -> world, in mm
t.inverse()                        # the *stored* inverse, when the file has one
```

`is_invertible` says the mapping is invertible; `inverse()` returns another
transform only when the file stores one under `inverse_id`. An affine computes
its own inverse (`AffineTransform.inverse_matrix()`, `inverse_points()`); a
displacement field does not, which is why the distinction exists.

`transform_between` searches the frame graph. It composes chains, uses an
inverse where a transform declares one, and returns `None` when no path exists.
It never fabricates a transform to make a call succeed.

```python
from medh5.transforms.apply import target_registration_error
target_registration_error(t, fixed_points, moving_points)   # {"mean", "max", ...}
```

A registration with no landmark pair has no TRE, and the file says so rather
than reporting zero.

## When there is no path

`transform_between` returns `None` rather than raising, and rather than
composing something plausible. But `None` is **two different answers**, and they
call for opposite responses:

- **The frames are already the same** — nothing to apply, because the arrays are
  co-registered by construction. Two images acquired at one visit on one scanner
  usually land here.
- **No path exists** — the frames are unrelated in this file, and any downstream
  number that treats them as comparable is invented.

The return value cannot tell them apart, so compare the frames first:

```python
t = s.transform_between("ct", "pet")
if t is None:
    if s.grids["ct"].frame_uid == s.grids["pet"].frame_uid:
        pass       # already aligned — use the arrays directly
    else:
        ...        # register them yourself, or work per-frame
```

For timepoints rather than grids, `frames_of_timepoint` gives the frames a visit
spans; a timepoint whose grids share a frame with the other visit's needs no
transform either.

Three things make the difference between a file that resolves and one that does
not: whether a transform was stored at all, whether its `from_frame` and
`to_frame` name frames that exist, and — for a chain — whether each link
declares `invertible=True` or stores an explicit `inverse_id`.

## When to store an inverse

`invertible=True` is the file's *claim*. What resolution needs is an inverse it
can actually **evaluate**, and the two are not the same set.

A forward chain resolves whatever you declare. Going the other way, an affine or
identity is inverted analytically, so nothing needs storing. **A displacement or
B-spline is not** — inverting one is an optimisation, not an algebraic step — so
`invertible=True` on a displacement field contributes no reverse edge at all, and
the reverse direction comes back `None`:

| Transform | `invertible=True`, no `inverse_id` | forward | reverse |
|---|---|---|---|
| affine | analytic inverse | resolves | resolves |
| displacement / B-spline | claim only | resolves | **`None`** |

So if you need the reverse direction for a non-affine registration, compute it at
write time and store it under `inverse_id`. Otherwise every reader that needs it
either recomputes it or silently goes without.

One thing to know when you do: a stored inverse is itself an edge in the frame
graph, so it can give two equally short paths between the same pair of frames.
Resolution refuses to choose (`E501`) rather than assert an alignment nobody
picked — select the one you want by id from `sample.transforms`.

## Related

- **[Longitudinal studies](longitudinal.md)** — the task this fits into.
- **[Python API](../reference/python-api.md)** — `transform_between`, the transform classes.
- **[Specification §10](../spec/medh5-1.0.md)** — the normative model.
