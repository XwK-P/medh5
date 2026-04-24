# CLI reference

After `pip install medh5` a `medh5` entry point is available.

```text
medh5 <command> [subcommand] [options]
python -m medh5.cli ...         # equivalent
```

## Exit codes

| Code | Meaning                                                    |
|------|------------------------------------------------------------|
| 0    | Success.                                                   |
| 1    | Runtime error: `MEDH5Error`, `ValueError`, `ImportError`.  |
| 2    | Usage error: no subcommand, unknown subcommand.            |

Shell idiom:

```bash
medh5 validate sample.medh5 || exit 1
```

## Command groups

Implementation lives under `medh5/cli/` split by group — `inspect`,
`dataset`, `convert`, `review`.

### Single-file inspection

```bash
medh5 info <file>                       # summary
medh5 info <file> --json                # machine-readable
medh5 validate <file>                   # structured validation
medh5 validate <file> --strict --json   # warnings become errors
```

### Batch inspection

```bash
medh5 validate-all <dir>                            # validate every .medh5
medh5 validate-all <dir> --fail-fast --workers 4
medh5 audit <dir>                                   # verify SHA-256 checksums
medh5 recompress <dir|file> --compression max       # rewrite with new preset
medh5 recompress <dir> --out-dir out/ --checksum    # or atomic in-place
```

`recompress` rewrites atomically (tempfile + rename) when `--out-dir` is
not given.

### Dataset management

```bash
medh5 index <dir> -o manifest.json
medh5 index <dir> -o manifest.json --recursive
medh5 split manifest.json --ratios 0.7,0.15,0.15 -o splits/ --seed 42
medh5 split manifest.json --ratios 0.8,0.2 --stratify label --group extra.patient_id -o splits/
medh5 split manifest.json --k-folds 5 -o folds/
medh5 stats <dir|manifest.json> -o stats.json --json --workers 4
```

See [Datasets and statistics](dataset-and-stats.md) for the underlying API.

### Converters

NIfTI:

```bash
medh5 import nifti --image CT ct.nii.gz -o sample.medh5
medh5 import nifti --image CT ct.nii.gz --image PET pet.nii.gz \
      --seg tumor tumor.nii.gz \
      --resample-to CT --interpolator linear \
      -o sample.medh5
medh5 export nifti sample.medh5 -o export/
```

DICOM:

```bash
medh5 import dicom /path/to/series -o sample.medh5
medh5 import dicom /path/to/series -o sample.medh5 \
      --series-uid 1.2.3.4.5 \
      --no-modality-lut
```

nnU-Net v2 (requires `medh5[nifti]`):

```bash
medh5 import nnunetv2 Dataset042_BraTS/ -o medh5_out/
medh5 import nnunetv2 Dataset042_BraTS/ -o medh5_out/ \
      --no-test --compression max --checksum
medh5 export nnunetv2 medh5_out/ -o Dataset042_BraTS_roundtrip/ \
      --dataset-name Dataset042_BraTS --file-ending .nii.gz
```

### Review / QA

```bash
medh5 review set <file> --status reviewed --annotator puyang --notes "ok"
medh5 review get <file>
medh5 review get <file> --json
medh5 review list <dir> --status pending
medh5 review import-seg <file> --name tumor --from edited.nii.gz
medh5 review import-seg <file> --name tumor --from edited.nii.gz --resample --replace
```

Status values: `pending`, `reviewed`, `flagged`, `rejected`.
