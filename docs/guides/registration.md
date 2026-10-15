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

<!-- illustrative -->
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

A transform maps coordinates in its `units`, which are those of the grids in
its frames: written without `units`, it takes theirs, and a `units` they are
not in is refused (`E506`), as are frames whose grids are in two units. Nothing
converts between units; the paired loader refuses to apply a transform to a
grid in other units rather than move every point by the wrong factor.

A file whose transform is in other units than its grids — the 1.x writer
stamped `mm` whatever the grids were in — fails validation with `E506`, and
so would any amendment that left the transform as it is, so that amendment is
refused. The one that mends it removes the transform and adds it again, its
numbers in its grids' units:

```python
with medh5.open("case.medh5") as s:
    t = s.transforms["tp0_to_tp1"]
    frames, matrix = (t.from_frame, t.to_frame), t.matrix
with medh5.amend("case.medh5") as w:
    w.remove_transform("tp0_to_tp1")
    w.add_transform("tp0_to_tp1", kind="affine", from_frame=frames[0],
                    to_frame=frames[1], matrix=matrix)   # units: the grids'
```

Kinds: `identity`, `affine`, `displacement`, `bspline`, `composite`.

```python
t = s.transform_between("tp0", "tp1")
t.kind, t.from_frame, t.to_frame, t.is_invertible
t.transform_points(points)         # world -> world, in t.units
t.inverse()                        # the stored inverse, or a chain's inverted
```

`is_invertible` is the file's *claim* that the mapping is invertible;
`InverseTransform.can_invert(t)` says whether an inverse can be *evaluated*.
`inverse()` returns the transform the file stores under `inverse_id`; for a
chain, or a composite without a usable `inverse_id`, it returns the chain of
its steps' inverses in reverse order, when every step can be inverted, and
`None` otherwise. An affine computes its own inverse
(`AffineTransform.inverse_matrix()`, `inverse_points()`); a displacement field
does not, which is why the distinction exists: a composite of two affines
inverts, one holding a displacement field without a stored inverse does not.

```python
from medh5.transforms import ChainTransform, InverseTransform

with medh5.open("case.medh5") as s:
    t = s.transforms["tp0_to_tp1"]
    chain = ChainTransform([t])
    back = chain.inverse()                     # t's analytic inverse, as a chain
    assert InverseTransform.can_invert(chain)
    assert (back.from_frame, back.to_frame) == (t.to_frame, t.from_frame)
```

A composite's inverse is not an edge of the frame graph: resolving backwards
across a composite goes through its components' own inverses, as it always
has, so an answer that resolved before does not change.

`transform_between` searches the frame graph. It composes chains, uses an
inverse where one can be **evaluated** — not merely where a transform declares
`invertible=True`; see [below](#when-to-store-an-inverse) — and returns `None`
when no path exists. It never fabricates a transform to make a call succeed.

```python
from medh5.transforms import target_registration_error
target_registration_error(t, fixed_points, moving_points)   # {"mean", "max", ...}
```

A registration with no landmark pair has no TRE: `mean`, `median` and `max`
are NaN and `n` is 0, rather than a zero that reads as a perfect registration.
`weights` are one per landmark, finite and non-negative, and not all zero;
anything else is refused rather than averaged.

A displacement field's `jacobian_determinant(roi)` and `folding_fraction(roi)`
check it for folding (`det(I + du/dx) <= 0`). A region's determinant is the
whole field's at the voxels it selects — its differences reach one voxel past
the region — so tiles of a large field add up to the field, a strided region
such as `np.s_[::2]` keeps a fold narrower than its step, and only the voxels
the region needs are read.

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
spans, and a visit with several grids may span several. Then no one transform
relates the visit to another, and `transform_between("tp0", "tp1")` raises
`MEDH5ValidationError` naming each visit's grids and frames, rather than
answering for whichever pair of grids it met first. Ask about the grids of the
images you will read (`by="grid"`), or about their frames
(`resolve_frames`), as the
[longitudinal preflight](longitudinal.md#train-on-the-pairs) does. A visit
whose grids carry no `frame_uid` names no frame, and answers `None`.

```python
with medh5.open("case.medh5") as s:
    t = s.transform_between("ct_tp0", "ct_tp1", by="grid")   # the grids' frames
    same = s.resolve_frames(s.grids["pet"].frame_uid, s.grids["ct_tp1"].frame_uid)
    assert t.transform_id == same.transform_id == "tp0_to_tp1"
```

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
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                field_grid="ct_tp0")
w.add_transform("tp1_to_tp0", kind="displacement", field=back,
                from_frame="pseudo:frame-tp1", to_frame="pseudo:frame-tp0",
                field_grid="ct_tp1")
```

**Linking them with `inverse_id` is fine, and says more.** Declare each as the
other's inverse and both directions still resolve, each to its own stored
transform: a transform and the stored inverse its `inverse_id` names are one
route, not two (§10.1). The link is what lets `t.inverse()` hand you the stored
inverse of a transform you already hold, and what the validator checks for
consistency (`E505`: the inverse must map the other way, and a declaration must
be mutual where both sides make one).

```python
w.add_transform("tp0_to_tp1", kind="displacement", field=fwd,
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                field_grid="ct_tp0", inverse_id="tp1_to_tp0")
w.add_transform("tp1_to_tp0", kind="displacement", field=back,
                from_frame="pseudo:frame-tp1", to_frame="pseudo:frame-tp0",
                field_grid="ct_tp1", inverse_id="tp0_to_tp1")
```

| what you write | forward | reverse |
|---|---|---|
| `invertible=True` only | resolves | `None` |
| two independent transforms | resolves | resolves |
| two transforms linked with `inverse_id` | resolves | resolves |

Before 1.4.2 the linked form raised `E501` in both directions: the resolver
counted the forward transform's delegated inverse as a second route beside the
stored one, and refused to choose between them.

## Storing a field in float16

A displacement field is stored as `float32` unless you ask otherwise. §10.3
permits `float16`, and recommends it below about 64 voxels of displacement,
where it costs roughly 5·10⁻⁴ of relative precision and halves the field's size.
`field_dtype=` chooses:

```python
w.add_transform("warp", kind="displacement", field=field, field_grid="ct_tp0",
                from_frame="pseudo:frame-tp0", to_frame="pseudo:frame-tp1",
                field_dtype="auto")
```

| `field_dtype` | stores |
|---|---|
| `"float32"` (default) | `float32`, whatever the field holds |
| `"float16"` | `float16`; a component beyond 65504, which `float16` cannot hold, is refused |
| `"auto"` | `float16` when every displacement is finite, within float16's range and below 64 voxels (in the field grid's smallest spacing, for a `world` field), else `float32` |

Only a displacement field has a field to store, so another kind given a
`field_dtype` other than `"float32"` is refused.

## Related

- **[Longitudinal studies](longitudinal.md)** — the task this fits into.
- **[Python API](../reference/python-api.md)** — `transform_between`, the transform classes.
- **[Specification §10](../spec/medh5-1.0.md#10-registration)** — the normative model.
