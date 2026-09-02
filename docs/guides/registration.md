# Registration between visits

Relate two frames of reference so you can move points, boxes and predictions
between them.

Two visits are two frames. A transform relates them, and — because both visits
live in the same file — it is an object in that file rather than a convention
between two filenames.

## Write one

A transform relates two **frames**, and a timepoint has a frame only because
its grids say so. Set `frame_uid` when you write the grids, and use those same
values on the transform:

```python
w.add_grid("ct_tp0", shape=..., spacing=..., timepoint="tp0",
           frame_uid="pseudo:frame-tp0")
w.add_grid("ct_tp1", shape=..., spacing=..., timepoint="tp1",
           frame_uid="pseudo:frame-tp1")

w.add_transform("tp0_to_tp1", kind="affine",
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                matrix=matrix)
```

Without `frame_uid` on the grids, a timepoint resolves to no frames at all and
`transform_between("tp0", "tp1")` returns `None` however carefully the transform
was written — the transform names endpoints nothing else refers to.

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
inverse where one can be **evaluated** — not merely where a transform declares
`invertible=True`; see [below](#when-to-store-an-inverse) — and returns `None`
when no path exists. It never fabricates a transform to make a call succeed.

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

What decides whether a path resolves depends on the direction.

**Forward**, along the way the transforms were written, needs only that they
exist and that their `from_frame` and `to_frame` name frames that exist. No link
has to declare anything about inversion.

**Backwards**, across any link traversed against its direction, needs an inverse
that can be *evaluated* — analytic for an affine or identity, otherwise a stored
`inverse_id`. `invertible=True` on its own is not enough; see below.

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
write time and **store it as its own transform** — a second `add_transform` with
`from_frame` and `to_frame` swapped. It is then an ordinary forward edge, and
both directions resolve:

```python
w.add_transform("tp0_to_tp1", kind="displacement", field=fwd,
                from_frame="frame-tp0", to_frame="frame-tp1", field_grid="ct_tp0")
w.add_transform("tp1_to_tp0", kind="displacement", field=back,
                from_frame="frame-tp1", to_frame="frame-tp0", field_grid="ct_tp1")
```

**Do not link them with `inverse_id`.** That makes `can_invert()` true for the
forward transform, so the graph gains an `InverseTransform` edge *in addition to*
the sibling's own edge — two distinct one-hop paths between the same frames, and
resolution refuses to choose between them:

| what you write | forward | reverse |
|---|---|---|
| `invertible=True` only | resolves | `None` |
| **two independent transforms** | **resolves** | **resolves** |
| linked with `inverse_id` | resolves | raises `E501` |

`inverse_id` is for `t.inverse()` — retrieving the stored inverse directly from a
transform you already hold. It is not a way to make `transform_between` work
backwards, and using it for that makes the reverse direction worse than leaving
it out.

## Related

- **[Longitudinal studies](longitudinal.md)** — the task this fits into.
- **[Python API](../reference/python-api.md)** — `transform_between`, the transform classes.
- **[Specification §10](../spec/medh5-1.0.md)** — the normative model.
