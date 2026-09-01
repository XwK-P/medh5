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
    annotation="lesions_tp0",   # an id that exists; omit it to pick per visit
)
```

`annotation` names an annotation **in the file**, not a family across visits.
The example above wrote `lesions_tp0` and `lesions_tp1`, so passing `"lesions"`
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

There is no silent fallback — but the failure is deferred. `align="transform"`
on a pair with no transform between its frames raises `MEDH5ValidationError`
from `__getitem__`, which means part way into an epoch rather than at
construction. To find those before you start, resolve them yourself:

```python
import medh5

def grid_of(sample, timepoint, image):
    """The grid the dataset will read at this visit, for this image."""
    return sample.at(timepoint).images[image].grid_id

for path in paths:
    with medh5.open(path) as s:
        for pair in TimepointPairSampler("consecutive").pairs(s):
            first = grid_of(s, pair.first, f"CT_{pair.first}")
            second = grid_of(s, pair.second, f"CT_{pair.second}")
            if s.transform_between(first, second) is not None:
                continue                      # a transform relates those grids
            if s.grids[first].frame_uid == s.grids[second].frame_uid:
                continue                      # same frame, none needed
            print(f"{path}: {first} -> {second} has no transform")
```

Two details decide whether this check is worth running.

**Resolve between the grids, not the timepoints.** A visit may hold a CT grid and
a PET grid on different frames. `transform_between("tp0", "tp1")` searches every
frame of the first visit against every frame of the second and returns the first
path it finds, so a CT registration makes the timepoint-level question answer
"yes" for a PET dataset that has no registration of its own. Ask about the grids
your images are actually on.

**Compare the frames too.** `transform_between` answers `None` both when nothing
relates the grids and when they already share a frame and need nothing. Checking
for `None` alone reports already-aligned visits as unregistered, which is how you
end up registering data that is already aligned.

## Label the change

A label that describes a *difference* names both visits it spans. See
**[Classification and change labels](classification.md)**.

## Related

- **[Registration between visits](registration.md)** — transforms and the frame graph.
- **[Classification and change labels](classification.md)** — response, progression, interval labels.
- **[PyTorch and MONAI](../reference/torch.md)** — `PairedPatchDataset` in full.
- **[Build and split a cohort](cohorts.md)** — splitting a longitudinal cohort without leaking a subject.
