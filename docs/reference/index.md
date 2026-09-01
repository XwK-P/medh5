# Reference

What everything is, rather than how to do anything. For task-shaped answers, see
[the how-to guides](../guides/index.md).

## The package

| | |
|---|---|
| **[Python API](python-api.md)** | Reading, writing, amending. |
| **[Command line](cli.md)** | Every `medh5` command. |
| **[PyTorch and MONAI](torch.md)** | Datasets, samplers, collation, the MetaTensor adapter. |

## The data

| | |
|---|---|
| **[Annotation kinds](annotations.md)** | Thirteen kinds: five voxel encodings, boxes, contours, keypoints, meshes, classification. |
| **[Converters](converters.md)** | NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net v2, and 0.x migration. |
| **[Curation records](curation.md)** | Provenance, quality, agreement, identity, de-identification. |
| **[Storage](storage.md)** | On-disk layout, codec profiles, chunking, integrity, collections. |

## Tables

| | |
|---|---|
| **[Sample document schema](schema.md)** | Every field of `/meta`. |
| **[Diagnostic codes](diagnostic-codes.md)** | All 71 codes `medh5 validate` can report. |
| **[Cohort check codes](cohort-checks.md)** | The `C1xx`–`C5xx` codes `medh5 dataset check` reports. |
| **[Profiles and validation levels](profiles-and-levels.md)** | What to check, and how much. |
| **[Runnable examples](../examples/index.md)** | A complete sample written by following the specification, plus the benchmark scripts. |

The [specification](../spec/medh5-1.0.md) is the normative statement of all of
it. Where this section and the specification disagree, one of them is a bug.
