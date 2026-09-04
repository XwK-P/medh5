# Getting started

## Install

```bash
pip install medh5
```

Extras, all optional:

| Extra | For |
|---|---|
| `torch` | `medh5.torch` datasets and samplers |
| `monai` | `medh5.monai` MetaTensor adapter |
| `nifti` | NIfTI import and export (nibabel) |
| `dicom` | DICOM, DICOM SEG and RTSTRUCT reading (pydicom) |
| `dicomseg` | *Writing* DICOM SEG (highdicom) |
| `itk` | Resampling in the converters (SimpleITK) |
| `schema` | JSON Schema validation of `/meta` (jsonschema) |
| `interp` | Cubic displacement-field evaluation (scipy) |

```bash
pip install "medh5[torch,nifti,dicom]"
```

Nothing but `h5py`, `hdf5plugin` and `numpy` is needed to read or write a file.

## Write a sample

```python
import numpy as np
import medh5
from medh5 import LabelClass, LabelSet

labels = LabelSet("demo-v1", version="1.0.0", classes=[
    LabelClass(1, "liver", "Liver", category="organ"),
    LabelClass(2, "spleen", "Spleen", category="organ"),
    LabelClass(3, "lesion", "Lesion", parents=[1], category="lesion"),
])

ct = np.random.default_rng(0).integers(-1000, 1500, (64, 96, 96)).astype(np.int16)
liver = np.zeros(ct.shape, bool); liver[10:40, 20:70, 20:70] = True
lesion = np.zeros(ct.shape, bool); lesion[20:26, 35:45, 35:45] = True

with medh5.create("case_0001.medh5", sample_id="case_0001",
                  subject_id="DEMO-0001") as w:
    w.identity(sex="F", bodypart="abdomen")
    w.label_set(labels)
    w.add_timepoint("tp0", label="baseline", days_from_baseline=0)

    w.add_grid("ct", shape=ct.shape, spacing=(2.0, 0.8, 0.8),
               origin=(-64.0, -38.4, -38.4), timepoint="tp0")

    w.add_image("CT", ct, grid="ct", modality="CT",
                value_type="quantitative", value_units="HU")

    w.add_segmentation("organs", grid="ct",
                       masks={"liver": liver, "lesion": lesion},
                       annotated_classes=["liver", "spleen", "lesion"])
```

Four things happened that are worth naming.

**The grid carries the geometry**, and the image and the segmentation reference
it. They cannot drift apart.

**`annotated_classes` names the spleen** even though there is no spleen mask.
That records "we looked and found none", which is a usable negative example.
Leave it out and the default `"all_given"` records only what you handed over;
pass `annotated_classes="all"` to claim the whole label set, which records every
class in it as examined.

**The encoding was chosen by measurement.** Liver and lesion overlap, so
`add_segmentation` measured the overlap graph and picked an encoding that can
represent it. It returns which one:

<!-- illustrative -->
```python
kind, stats = w.add_segmentation(...)   # ("layers", OverlapStats(...))
```

**The file is written atomically.** It appears complete or not at all; a
crashed writer cannot leave a half-file where a valid one used to be.

## Read it back

```python
with medh5.open("case_0001.medh5") as s:
    s.identity.subject_id                  # "DEMO-0001"
    s.profiles                             # {"core", "seg"}

    ct = s.images["CT"].read(physical=True)          # HU
    patch = s.images["CT"].read((slice(10, 20),) * 3)  # just that block

    organs = s.annotations["organs"]
    organs.kind                            # "layers"
    organs.dense(["liver", "lesion"])      # (2, 64, 96, 96) bool
    organs.labelmap()                      # (64, 96, 96) of class ids
    organs.voxel_counts()                  # {1: 75000, 2: 0, 3: 600}
```

Reads are lazy. `medh5.open` parses the metadata document and nothing else;
slicing an image reads only the chunks that slice touches.

## Look at it from the shell

```
$ medh5 info case_0001.medh5
$ medh5 tree case_0001.medh5
$ medh5 validate case_0001.medh5 --level strict
$ medh5 verify case_0001.medh5
```

`validate` checks the file against the specification and reports stable
diagnostic codes; `verify` checks that every object still matches its digest.

## Train on it

```python
from torch.utils.data import DataLoader
from medh5.torch import PatchDataset, collate, worker_init_fn
from medh5.sampling import PatchSampler

sampler = PatchSampler((32, 32, 32), strategy="balanced",
                       foreground_classes=["liver", "lesion"])
dataset = PatchDataset(["case_0001.medh5"], sampler,
                       images=["CT"], annotations={"organs": ["liver", "lesion"]},
                       samples_per_volume=8)

loader = DataLoader(dataset, batch_size=2, num_workers=4,
                    worker_init_fn=worker_init_fn, collate_fn=collate)
```

Foreground sampling is O(1) in the volume if the file carries a sampling index.
Build one with:

```
$ medh5 index build case_0001.medh5
```

## Where to go next

- **[Concepts](../explanation/data-model.md)** — the model behind the API.
- **[Converters](../reference/converters.md)** — you probably have NIfTI or DICOM, not this.
- **[Training](../reference/torch.md)** — samplers, transforms, MONAI, and the numbers.
- **[Specification](../spec/medh5-1.0.md)** — when you need the normative answer.
