# Longitudinal

A sample is a subject, and a subject has visits. This page is what follows from
that: timepoints, registration between them, change labels, and joining objects
across visits.

## Timepoints

Declared once on the sample, in acquisition order:

```python
w.add_timepoint("tp0", label="baseline", days_from_baseline=0,
                date="2026-02-03", study_uid="pseudo:study-a")
w.add_timepoint("tp1", index=1, label="fu1", days_from_baseline=92)
```

Indices are dense and start at zero, and `days_from_baseline` must not
decrease. A grid names its timepoint; images and annotations inherit it:

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
s.is_longitudinal                         # True

view = s.at("tp1")
view.images                               # only tp1
view.annotations
```

```
$ medh5 timeline case.medh5
```

## Registration between visits

Two visits are two frames of reference. A transform relates them, and it is an
object in the file:

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

## Change

A label that describes a *difference* is a classification naming both visits:

```python
w.add_classification("response", {"progressive_disease": 1.0},
                     scope="sample", timepoints=["tp0", "tp1"],
                     schemes=["RECIST 1.1"])
```

```python
c = s.annotations["response"]
c.timepoints        # ("tp0", "tp1") — a statement about the interval
c.labels
```

Because both visits are in one file, a change label has a referent. Split
across two files it would be a claim about a filename.

## Tracking objects across visits

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

Then the join is a method:

```python
tracking = s.tracks("lesion")

for instance_id, track in tracking.items():
    track.volumes                        # {"tp0": 812.4, "tp1": 1140.0} mm^3
    track.relative_change("tp0", "tp1")  # 0.403
    track.at("tp1")                      # the Observation, or None
```

```
$ medh5 track case.medh5 --class lesion
```

### Three states, not two

The reason this is not a dictionary lookup:

| State | Meaning |
|---|---|
| `present` | the object was found at this visit |
| `resolved` | the class **was examined** at this visit and the object was not found |
| `unexamined` | nobody looked for that class at this visit |

```python
tracking.state_at(7, "tp1")     # "resolved"
tracking.states(7)              # {"tp0": "present", "tp1": "resolved"}
tracking.is_new(7)
tracking.is_resolved(7)
tracking.is_persistent(7)
tracking.unexamined()           # {timepoint: instance ids nobody looked for}
tracking.coverage               # {timepoint: class ids examined there}
```

A lesion that responded to treatment and a lesion nobody segmented at the
follow-up look identical if you only check whether the id is present. They are
completely different findings, and the state is derived from
`annotated_class_ids` — from what the annotator committed to — rather than from
absence.

### Class conflicts

```python
tracking.class_conflicts()      # {instance_id: (class_id_at_tp0, class_id_at_tp1)}
```

One object with two classes across visits is either a reclassification worth
knowing about or a mistake worth finding. The validator raises `W909` for it,
sample-scoped, because the costly case is a disagreement *between* two visits'
annotations rather than within one.

## Paired sampling for training

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
    annotation="lesions",
)
```

`align="transform"` means the second patch is centred where the registration
says the first patch's centre went — a +4-voxel shift moves the paired patch by
exactly 4 voxels. `align="none"` takes the same index in both, which is right
only when the visits are already resampled onto a common grid.

`dataset.report` says how many pairs were aligned by a transform and how many
had none available, so a silent fallback cannot look like a successful
alignment.

## Cohort-level

A sample is one subject, so assigning whole files to partitions cannot leak a
patient between train and test. That is the property the whole design buys, and
[Cohorts](cohorts.md) is where it is used.
