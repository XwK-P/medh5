# Longitudinal studies

You have a baseline and one or more follow-ups for the same subject. This is how
to get them into one file, join a lesion across the visits, and train on the
pairs.

The reason all of this is straightforward is the one decision underneath the
format: a sample is a *subject*, not a scan. Both visits are already in scope,
so a change label has a referent, a registration is an object rather than a
convention between two filenames, and assigning whole files to train and test
cannot leak a patient between them.

## Declare the timepoints

Declared once on the sample, in acquisition order:

<!-- illustrative -->
```python
w.add_timepoint("tp0", label="baseline", days_from_baseline=0,
                date="2026-02-03", study_uid="pseudo:study-a")
w.add_timepoint("tp1", index=1, label="fu1", days_from_baseline=92)
```

Indices are dense and start at zero, and `days_from_baseline` must not
decrease. A grid names its timepoint; images and annotations inherit it:

<!-- illustrative -->
```python
w.add_grid("ct_tp1", shape=..., spacing=..., timepoint="tp1")
w.add_image("CT_tp1", array, grid="ct_tp1", modality="CT")   # tp1, by inheritance
```

Binding time to the **grid** rather than to each object is what stops a
follow-up CT and its segmentation disagreeing about which visit they are.

Reading:

```python
s.timepoints["tp1"].days_from_baseline    # 92
s.timepoints[0].label                     # "baseline"
s.timepoints.interval_days("tp0", "tp1")  # 92.0 — None where either lacks days
s.is_longitudinal                         # True

view = s.at("tp1")
view.images                               # only tp1
view.annotations
```

```
$ medh5 timeline case.medh5
```

## Relate the visits

Two visits are two frames of reference, and a transform between them is an
object in the file. See **[Registration between visits](registration.md)** for
writing one, resolving it through the frame graph, and what to do when
`transform_between` returns `None`.

## Join the objects

`instance_id` is **sample-scoped**: object 7 at baseline and object 7 at
follow-up are the same lesion, asserted by whoever wrote the file.

```python
from medh5.annotations.voxel import InstanceInput

w.add_segmentation("lesions_tp0", grid="ct_tp0", encoding="instances",
                   instances=[InstanceInput(class_id=3, instance_id=7, mask=m0)],
                   annotated_classes=["lesion"])
w.add_segmentation("lesions_tp1", grid="ct_tp1", encoding="instances",
                   instances=[InstanceInput(class_id=3, instance_id=7, mask=m1)],
                   annotated_classes=["lesion"])
```

Masks carry no identity, so the writer asserts none. A segmentation given as
`masks=` and stored as `instances` — chosen by measurement or asked for —
holds one object per class, each minted an id above every id the sample already
holds, so it joins nothing. Correspondence is the caller's to assert: pass
`instances=` with the ids.

Then the join is a method:

```python
tracking = s.tracks("lesion")

for instance_id, track in tracking.items():
    track.volumes                        # {"tp0": 812.4, "tp1": 1140.0}: the grids' units cubed (mm³ here)
    track.relative_change("tp0", "tp1")  # 0.403
    track.at("tp1")                      # the Observation, or None
```

```
$ medh5 track case.medh5 --class lesion
```

Each visit answers with one observation: the object's mask, where an
`instances` annotation saw it, or else its box — axis-aligned or oriented
(`obb`, whose volume its rotation leaves alone). A detection beside a
segmentation of the same lesion is the common case, and a box's volume is its
bounding box's, so the mask is the measurement. A landmark (`points`, with the
`instance_ids` §10.6 asks of trackable points) marks the object without
measuring it, and answers only where nothing else saw it. Two masks of one
visit — two raters — are not one answer: `at`, `volume`, `volumes` and
`relative_change` refuse rather than pick a rater, and `observations_at()`
lists every observation of the visit (each with its annotation's `kind`) for
the caller to choose from. An annotation of another kind that carries
`instance_ids` is left out and named in `tracking.skipped`.

Volumes are in their grid's units cubed (`Observation.units`), and
`relative_change` compares two visits in millimetres (`volume_mm`), so a
follow-up on a grid in metres reads as the change it is. Visits with no
physical measure in common — an uncalibrated (`px`) grid beside a calibrated
one, an area beside a volume — have no change, and it is `None`. `medh5
track` prints volumes in mm³.

### Three states, not two

The reason this is not a dictionary lookup:

| State | Meaning |
|---|---|
| `present` | the object was found at this visit |
| `resolved` | the object **was looked for** at this visit and not found |
| `unexamined` | nobody looked for it there: the class was not examined, the object lies outside what was imaged, or an ignore region (§7.7) may hide it |

```python
tracking.state_at(7, "tp1")     # "resolved"
tracking.states(7)              # {"tp0": "present", "tp1": "resolved"}
tracking.reason_at(7, "tp1")    # "located": why the absence is, or is not, a finding
tracking.reasons(7)             # {"tp0": None, "tp1": "located"}
tracking.is_new(7)
tracking.is_resolved(7)
tracking.is_resolved(7, located=True)   # only where the object was located
tracking.is_persistent(7)
tracking.unexamined()           # {timepoint: instance ids unexamined there}
tracking.coverage               # {timepoint: class ids examined, class-level}
```

A lesion that responded to treatment and a lesion nobody segmented at the
follow-up look identical if you only check whether the id is present. They are
completely different findings, and the state is derived from what the
annotator committed to (`annotated_class_ids`, the ignore region) and from
where the object lies, rather than from absence.

Every unseen (object, visit) carries one reason (`medh5.curation.REASONS`).
Tracking maps the object's boxes from the visits that saw it into each
annotation of the visit that did not, through the shared frame or a path of
affine transforms, and rounds the result out to whole voxels: the object's
*located extent* there.

| Reason | State | When |
|---|---|---|
| `located` | `resolved` | the located extent lies in the grid, in its images' acquired voxels (`valid_mask`, §4.4), and outside the ignore region |
| `class_examined` | `resolved` | no geometry relates the visits, and the class was examined with nothing ignored: 1.0's class-level rule, which a registration would decide |
| `outside_fov` | `unexamined` | the located extent leaves the grid or the acquired voxels |
| `ignored` | `unexamined` | the located extent holds an ignored voxel |
| `unrelated` | `unexamined` | no geometry relates the visits, and the annotation ignores some voxels, which may be where the object is |
| `no_identity` | `unexamined` | the class was examined by an annotation without `instance_ids` (a semantic mask) |
| `not_examined` | `unexamined` | nothing at the visit examined the class |

Where a visit's annotations disagree, the first reason in that order wins. A
displacement field or a B-spline does not locate the object, since a warped
box's corners do not bound it; nor does a path where any observation fails to
map. An ignore region away from the object leaves its absence a finding. The
baseline is read the same way: `is_new` asks that the baseline looked for the
object where it later appeared, so a lesion first seen at a follow-up is not
new where the baseline never imaged that place, or ignored it.
`located=True` on `is_new` and `is_resolved` refuses the class-level fallback.

### Class conflicts

```python
tracking.class_conflicts()      # {instance_id: (every class id it carries, sorted)}
```

One object with two classes across visits is either a reclassification worth
knowing about or a mistake worth finding. The validator raises `W909` for it,
sample-scoped, because the costly case is a disagreement *between* two visits'
annotations rather than within one.

### Two readers of one follow-up

An `instance_id` joins an object across the visits of one sample; it names
nothing in another file. Comparing two readers' files of one visit therefore
pairs objects by where they are, and `match_by="instance_id"` is refused there.
Where the readers used two numberings of one vocabulary, `map_classes="key"`
scores liver against liver, and the comparison is recorded only against an
annotation of the recording sample
([agreement](../reference/curation.md#two-vocabularies)).

## Train on the pairs

`PairedPatchDataset` draws the same anatomical location at two visits, mapping
the patch centre through the registration:

```python
from medh5.torch import PairedPatchDataset
from medh5.sampling import PatchSampler, TimepointPairSampler

dataset = PairedPatchDataset(
    paths,
    PatchSampler((96, 96, 96), strategy="foreground", foreground_classes=["lesion"]),
    pair_sampler=TimepointPairSampler("consecutive"),   # or baseline_vs_all, all_pairs
    align="transform",                                  # or "none"
    annotation="lesions_tp0",   # drawn from at tp0; other visits pick their own
)
```

`annotation` names an annotation **in the file**, not a family across visits,
and windows are drawn from it at its own visit only: at every other visit the
dataset selects that visit's own annotation, as it does when `annotation` is
omitted. The example above wrote `lesions_tp0` and `lesions_tp1`, so passing `"lesions"`
raises `KeyError` on the first item. Omitting it lets the dataset select the
one belonging to each visit, which is usually what you want.

`align="transform"` means the second patch is centred where the registration
says the first patch's centre went — a +4-voxel shift moves the paired patch by
exactly 4 voxels. `align="none"` takes the same index in both, which is right
only when the visits are already resampled onto a common grid.

`dataset.report` counts files, pairs and the cross-sectional files that
contributed none. It does **not** check registrations: it resolves no
transforms, so a file that needs one and lacks it is counted as a perfectly good
pair.

The failure is also deferred. `align="transform"` on a pair whose visits share
no transform at all raises `MEDH5ValidationError` from `__getitem__` — part way
into an epoch rather than at construction.

A pair whose two visits hold several frames each is the case worth checking by
hand. The loader resolves between the frames the images are actually on and
raises when nothing relates them, so a PET dataset is no longer moved by a CT
registration — but it raises when the item is read, which is a slow way to find
out that a cohort was never registered. Run the preflight first:

```python
import medh5

def frame_of(sample, timepoint, image):
    """The frame the dataset will read on at this visit, for this image."""
    grid_id = sample.at(timepoint).images[image].grid_id
    return sample.grids[grid_id].frame_uid

for path in paths:
    with medh5.open(path) as s:
        for pair in TimepointPairSampler("consecutive").pairs(s):
            first = frame_of(s, pair.first, f"CT_{pair.first}")
            second = frame_of(s, pair.second, f"CT_{pair.second}")
            if first is not None and first == second:
                continue                      # same frame, none needed
            if first and second and s.transform_between(first, second) is not None:
                continue                      # a transform relates those frames
            print(f"{path}: {first} -> {second} has no transform")
```

Two details decide whether this check is worth running.

**Resolve between the frames, not the timepoints.** A visit may hold a CT grid
and a PET grid on different frames, and then no one transform relates it to
another visit: `transform_between("tp0", "tp1")` raises
`MEDH5ValidationError`, naming each visit's grids and their frames. It answers
for two visits only when each visit's grids share one frame. Asking about the
two grids works when you say that the keys are grids,
`transform_between(first_grid, second_grid, by="grid")`; without `by=`, a key
is read as a timepoint before it is read as a grid, and grid ids and timepoint
ids are separate namespaces (spec §2.3), so a grid may legitimately be called
`tp0` and the question goes back to the whole visit. A frame uid, as the recipe
above passes, answers for that frame alone.

**Compare the frames first.** `transform_between` answers `None` both when
nothing relates the two and when they are already the same frame and need
nothing. Checking for `None` alone reports already-aligned visits as
unregistered, which is how you end up registering data that is already aligned.

## Label the change

A label that describes a *difference* names both visits it spans. See
**[Classification and change labels](classification.md)**.

## Related

- **[Registration between visits](registration.md)** — transforms and the frame graph.
- **[Classification and change labels](classification.md)** — response, progression, interval labels.
- **[PyTorch and MONAI](../reference/torch.md)** — `PairedPatchDataset` in full.
- **[Build and split a cohort](cohorts.md)** — splitting a longitudinal cohort without leaking a subject.
