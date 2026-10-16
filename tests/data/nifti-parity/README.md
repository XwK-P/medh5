# The NIfTI parity corpus

What medh5 2.1.0's NIfTI converter made of a set of small inputs, recorded
under the released 2.1.0. From 2.2.0 the converter is the engine's, and its
contract is parity with these records (2.2.0 decision E3): every object an
import writes has the digest 2.1.0's had, `/meta` is the same JSON but for the
version its software agent records, the conversion report is the same but for
the paths it names, a refusal is the same exception with the same text, and
an export writes the same header fields and voxel bytes.

| Path | What it is |
|---|---|
| `make_inputs.py` | builds `inputs/` and `cases.json` (run once; the results are committed) |
| `inputs/` | NIfTI files written field by field (both byte orders; NIfTI-1 and NIfTI-2; `.nii`, `.nii.gz`, `.hdr`/`.img`; every element type; scaled; sform, qform, both, neither; 2-D, 3-D, 4-D, 5-D; broken ones), their sidecars, label sets, and the samples (`inputs/samples/`, written by 2.1.0) that exports and `import_seg_nifti` start from |
| `cases.json` | each case: the call (`from_nifti`, `import_seg_nifti`, `to_nifti` or `read_nifti`), its inputs and its options |
| `record.py` | runs every case under 2.1.0 and writes `records/` |
| `records/<id>.json` | what came out of one case |

A record holds the outcome (`ok`, or `refused` with the exception's type,
`code` and text), the report's JSON, and what was written: for an import,
every object's attributes and type, `/meta`, and the root's attributes (but
`created`); for an export, each file beside the target, a NIfTI file by its
stored header fields and a digest of its voxel bytes; for `read_nifti`, the
geometry and a digest of the array. Paths are relative to the corpus
(`{corpus}/inputs/…`) and to the case's own directory (`{out}/…`).

`tests/integrations/test_nifti_parity.py` checks that every case has a record
made by 2.1.0, and runs every case through the converter of the tree it is in,
comparing the result with the record. The comparison leaves out the two things
E3 masks, plus what follows from them: the records' tool versions, the root's
`generator`, and `content_id` (which covers `/meta`, and so the version). Any
other difference a later change makes on purpose is listed by case in its
`ALLOWED` table, with the change that made it.

## Recording again

Only to add cases, and only with what the records were made with: medh5
2.1.0, nibabel 5.4.2 and NumPy 2.5.4 (`record.py` refuses any other version
of the three, since a later nibabel or NumPy could change a header, a
refusal or an input under the same medh5):

```sh
python -m venv /tmp/medh5-2.1.0
/tmp/medh5-2.1.0/bin/pip install "medh5[nifti]==2.1.0" nibabel==5.4.2 numpy==2.5.4 h5py
/tmp/medh5-2.1.0/bin/python tests/data/nifti-parity/make_inputs.py
/tmp/medh5-2.1.0/bin/python tests/data/nifti-parity/record.py
```

`make_inputs.py` rewrites every input; the NIfTI files come out byte for byte
the same, and the samples differ only in their `created` time.
