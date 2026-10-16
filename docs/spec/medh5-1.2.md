# MEDH5 format specification 1.2 — open curves

**Status: Final.** The reference engine reads, writes, validates and amends the `polylines` kind
defined here, and the conformance corpus (§4) holds a valid case and an invalid case for each
diagnostic code the kind reports beyond the codes every annotation reports.

This document is a **delta to [MEDH5 1.1](medh5-1.1.md)**, which is a delta to
[MEDH5 1.0](medh5-1.0.md). Every 1.0 and 1.1 clause continues to apply unless a section below
names it. Keywords are those of 1.0 §1.2.

---

## 1. Scope

MEDH5 1.2 adds **one annotation kind, `polylines`: open curves** — vessel centrelines, catheter
and applicator paths, needle tracks, and DICOM RTSTRUCT `OPEN_PLANAR` and `OPEN_NONPLANAR`
contours. It adds nothing else: no profile, attribute, object, code or change to an existing kind.

Open curves could not be stored before. `contours` (1.0 §8.6) are closed polygons, whose
combination rule reads every polygon as the boundary of a region; giving a polygon a third role,
"open", would change what an existing kind means, which 1.0 §16 forbids, and a 1.1 reader would
read the curve as a region. `points` (1.0 §8.5) are unordered. A new kind is the one addition a
1.1 reader already handles: its projection leaves an annotation of a kind it does not know out,
as **W913**, and validates the rest (1.1 §2.2).

What 1.2 deliberately does not include, and what it would need:

| Not in 1.2 | Why, and what it would need |
|---|---|
| Clinical extension columns (FHIR `interpretation` and `referenceRange`, MEDS code modifiers) | 1.1 §4 closes a clinical table's column set (E804); whether a 1.1 projection can drop an unknown column of a known table without misreading the rest needs its own study. If it cannot, they need a new table, an object a projection can drop. |
| Annotation pyramids | reserved (1.0 §16); `nearest` under an even factor and `max` over class ids are undefined |
| A `bspline` transform kind | an ITK B-spline is imported, where it can be exactly, as a sampled displacement field; a kind of its own needs a case the field cannot represent |
| `rle` | reserved (1.0 §16); a minor could add it as `polylines` is added here, but it needs a measured case `instances` serves badly (1.0 §7.0) |
| Subjects with no image | E111 and E201 are core errors a 1.1 reader still applies to a newer file's projection, so no 1.x minor can relax them: a 2.0 |

## 2. Compatibility

### 2.1 Versions written

A writer writes the **lowest version its content needs**: `1.0` when a sample holds imaging
alone, `1.1` when it declares `clinical`, and `1.2` when it holds a `polylines` annotation — `1.2`
whether or not it declares `clinical`. An amendment never lowers the version a file already
declares (1.1 §2.3), except one that removes the clinical profile (1.1 §2.1) or the last
`polylines` annotation of a file that held one: what remains is written at the lowest version it
needs. Either way the result is a different sample, with its own `content_id`, since the version
is part of what that address covers (1.0 §13.2).

A `polylines` annotation in a file that declares `1.0` or `1.1` is **E401**: it is no kind those
versions define, and a writer of it writes `1.2`.

### 2.2 A 1.2 file read by a 1.1 implementation

1.1 §2.2 applies unchanged. An implementation of 1.1 reads a 1.2 file as its supported projection:
it reports each `polylines` annotation as **W913** (an unknown `kind`, 1.0 §16) and the version as
**W913** at `/`, validates everything else, never claims conformance to 1.2, and **MUST NOT** amend,
recompress or pack the file (1.1 §2.3).

1.2 extends none of the attributes 1.0 §13.2 covers, so a 1.1 implementation recomputes a 1.2
file's `content_id` and finds it matches: the dataset digests and the address of a 1.2 file
verify under 1.1's rules.

### 2.3 Diagnostic codes

1.2 allocates no code. A `polylines` annotation's defects are reported under the codes that report
the same defect for the kinds before it (1.0 §15.2): what a code applies to widens, what it means
does not. **E408**'s summary names polyline offsets beside contour and mesh offsets.

## 3. The `polylines` kind

**`polylines`: open curves.** Ordered vertices, each joined to the next; the last vertex does not
join the first, and a polyline encloses nothing.

| Dataset | Shape, dtype | |
|---|---|---|
| `vertices` | `(V, S)` `float32` | in the annotation's `space` |
| `polyline_offsets` | `(M+1,)` `int64` | polyline *m* spans `[o[m], o[m+1])`; non-decreasing, `o[0] = 0`, `o[M] = V` (E408) |
| `polyline_class_ids` | `(M,)` `uint16` | each polyline's class (E402) |
| `polyline_plane` | `(M, 2)` `int32`, OPTIONAL | the `(axis, index)` of the grid plane a planar polyline lies in; `axis = −1` for none |
| `instance_ids` | `(M,)`, OPTIONAL | as for other kinds (1.0 §7.4) |

Each polyline has at least two vertices (E404). A polyline is never rasterised, and a polylines
annotation does not satisfy `seg` or `det`. Its `task` is `other` unless the writer states another.

The rest is as 1.0 §8 gives it for every geometric kind:

- `space` is `index` or `world` (1.0 §8.1), and coordinates convert through the grid's affine.
  In `index` space, `S` is the grid's spatial rank (E405).
- `polyline_plane`, when present, names for each polyline a plane of the annotation's `grid`, as
  `contour_plane` does for a polygon (1.0 §8.6); without it, every polyline has `axis = −1`.
- `instance_ids` are `uint32` or `uint64`; equal ids name one physical object across annotations
  and timepoints (1.0 §7.4). Without them a reader **MAY** number polylines by row, and such
  numbers name nothing.
- `class_ids` is every class a polyline holds or may hold, and `annotated_class_ids` the classes
  searched for (1.0 §6.2, §11.3): a centreline annotation that looked for the left anterior
  descending artery and found none is a verified negative for it.
- A `task` other than `other` describes the annotation and changes no profile: `det` reads
  detections, and a curve is none (1.0 §1.3).

## 4. Conformance

The corpus ([Conformance suite](conformance.md)) holds the 1.0 and 1.1 cases unchanged and three
for this version: `shapes-polylines`, valid, with open curves in index and world space, a planar
and a non-planar curve and instance ids; `E408-polyline-offsets`, offsets that do not delimit the
vertices; and `E404-polyline-one-vertex`, a polyline of one vertex. Each states `format_version`
`1.2`. The higher-minor case declares `1.3`.

---

## Appendix A — Changes from the planned text, and why

The text this version implements was settled before it was implemented. Each change below is
normative, and was made because implementing it showed it was needed.

| Clause | Change | Why |
|---|---|---|
| §2.1 | A `polylines` annotation in a file declaring 1.0 or 1.1 is E401. | A validator that knew the kind accepted it in any file, so a 1.1 file holding one validated here and failed in every 1.1 implementation: the version the file declares decides which kinds it may hold. |
| §3 | A `task` other than `other` changes no profile. | `det` reads an annotation's `task` (1.0 §6.3), so `task = "detection"` would have made a curve satisfy `det`, which the kind's definition says it does not. |
