# How-to guides

Answers to specific questions, assuming you already know roughly what a sample
is. If you do not, [the tutorial](../tutorials/first-sample.md) is twenty
minutes.

## Getting data in

| | |
|---|---|
| **[Import from DICOM](import-dicom.md)** | A directory of studies to one sample per patient, every visit in one file. |
| **[Import from NIfTI and nnU-Net](import-nifti.md)** | Volumes and masks as separate files; nnU-Net v2 datasets. |
| **[Migrate from 0.x](migrate-0x.md)** | The one-way door, and the four decisions it cannot make for you. |

## Working with a cohort

| | |
|---|---|
| **[Build and split a cohort](cohorts.md)** | Manifests, leakage-free splits, streaming statistics, cross-file checks. |
| **[Longitudinal studies](longitudinal.md)** | Timepoints, tracking one lesion across visits, paired sampling. |

## Getting data out

| | |
|---|---|
| **[Export to other formats](export.md)** | NIfTI, DICOM SEG, RTSTRUCT, nnU-Net — and what each will not do. |
| **[De-identify and publish](deidentify.md)** | Find identifiers, apply the shift, record what was done — and what the tool cannot see. |

## Making it fast

| | |
|---|---|
| **[Tune performance](performance.md)** | The sampling index, chunk sizing, codec profiles, workers — in payoff order. |

Each guide ends by pointing at the reference page for the API it used.
