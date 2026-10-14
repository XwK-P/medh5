# Segmentation

Write masks, let the encoding be chosen for you, and read them back in the shape
a loss wants.

## Write the masks

Hand `add_segmentation` a mapping of class to boolean volume. You do not choose
an encoding:

```python
w.add_segmentation("organs", grid="ct",
                   masks={"liver": liver, "lesion": lesion},
                   annotated_classes=["liver", "spleen", "lesion"])
```

It measures the class overlap graph and picks an encoding that can represent it,
returning which one and the statistics behind the choice:

<!-- illustrative -->
```python
kind, stats = w.add_segmentation(...)     # ("layers", OverlapStats(...))
```

`liver` and `lesion` overlap — a lesion is inside the liver — so a single
labelmap cannot hold both, and the writer picks among the encodings that can:
`instances` for sparse localized objects, `layers` while the classes pack into
few planes, `bitmask` beyond that. Disjoint classes get a `labelmap`, which is
smaller. This is a storage decision: what you read through the class contract
— `dense()`, `contains()`, `labelmap()` and `voxel_counts()`, the ignore
region included — does not depend on it. Only `instances()` does, since a
dense encoding keeps no objects to return.

**`annotated_classes` is the argument to think about**, not the encoding. Above
it names the spleen although there is no spleen mask, recording "we looked and
found none" — a usable negative. See
[Partial labels and coverage](partial-labels.md) for which form to use.

## Read them back

One API, whatever the encoding:

```python
organs = s.annotations["organs"]
organs.kind                            # "layers" — informational
organs.dense(["liver", "lesion"])      # (2, *shape) bool, one plane per class
organs.labelmap()                      # (*shape) of class ids
organs.voxel_counts()                  # {1: 75000, 2: 0, 3: 600}
```

`dense()` is the one a loss wants: `(C, *spatial)`, in the class order you asked
for, so the channel index is yours and not the file's. Ask for a class the
annotation does not contain and you get a zero plane — which is correct when the
class was examined, and is why coverage is a separate question.

Read a patch rather than a volume by passing an ROI:

<!-- illustrative -->
```python
roi = (slice(10, 42),) * 3
organs.dense(["liver"], roi=roi)       # only the chunks that ROI touches
```

## Overlap you did not expect

```bash
medh5 seg stats case.medh5 organs
```

That prints per-class counts, the overlap graph and what each encoding would
cost. It is the fastest way to find out that two classes you thought were
disjoint are not — usually a rater including a lesion in the organ it sits in.

## Change the encoding later

Without touching the annotation's meaning — losslessly, with two exceptions
(§7.6):

```bash
medh5 seg convert case.medh5 organs --to bitmask --dry-run
```

- **A `probmap`** is lossless only under its `threshold` (§7.5): another
  encoding keeps the voxels at or above it as the class, and the
  probabilities are gone.
- **`instances` to a dense encoding** — `labelmap`, `layers`, `bitmask` or
  `probmap` — keeps every voxel and drops every `instance_id`, the id that
  [joins an object across visits](longitudinal.md#join-the-objects). It is
  refused unless you ask for it, and the transcode is then recorded in the
  provenance:

```bash
medh5 seg convert case.medh5 lesions_tp0 --to bitmask --drop-identity
```

In Python, `w.transcode_annotation("lesions_tp0", "bitmask", drop_identity=True)`.

Not every conversion is possible: an encoding that cannot represent overlap
refuses a transcode from one that does, rather than dropping voxels.
`--dry-run` says what would happen, refusals included.

## Check it

```bash
medh5 validate case.medh5 --level integrity
```

`--level integrity` runs every check, every byte against its digest included,
and fails on errors. `--level strict` also fails on warnings, so it fails the
sample written above: it carries no de-identification record (`W903`), and its
classes no ontology code (`W912`). Strict is the gate for a dataset you
publish, once [de-identification](deidentify.md) and the label set's codes
have dealt with both; see [Choose a level](validate.md#choose-a-level).

## Related

- **[Partial labels and coverage](partial-labels.md)** — `annotated_classes` in full.
- **[Annotation kinds](../reference/annotations.md#voxel-annotations)** — the five encodings and the transcode table.
- **[Tune performance](performance.md)** — making `dense()` on a patch fast.
