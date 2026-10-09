# medh5-cli

The native `medh5` command line: inspect, validate, verify, curate, pack and
benchmark MEDH5 samples.  It is a frontend over the
[`medh5`](https://crates.io/crates/medh5) format engine --- the same code the
Python package's `medh5` command runs --- so both produce the same output and
the same exit codes: `0` success, `1` a handled error, `2` a usage error.

```bash
cargo install medh5-cli          # or a binary from the GitHub Release, or Homebrew
medh5 info case.medh5            # grids, images, annotations, coverage
medh5 validate case.medh5 --level strict
medh5 verify case.medh5          # digests and content_id
medh5 conformance run corpus/    # the 153-case conformance corpus
```

HDF5 is linked statically: the binary needs nothing installed. Building it
(`cargo install`) needs a C compiler and CMake, which HDF5's build uses.

**On Windows (MSVC)**, give HDF5's C build `NDEBUG` before installing:

```powershell
$env:CFLAGS_x86_64_pc_windows_msvc = "/DNDEBUG"
cargo install medh5-cli
```

cmake-rs, which builds HDF5, drops CMake's release flags under the Visual
Studio generator, `/DNDEBUG` with them; HDF5 would keep its assertions, and a
damaged file would abort the process instead of being reported. The build stops
and says so when the flag is missing. The release binaries are built with it.

## Converters

`medh5 convert …` (NIfTI, DICOM, DICOM SEG, RTSTRUCT, nnU-Net) and `medh5
migrate` are integrations with Python libraries --- nibabel, pydicom,
highdicom --- and live in the Python package.  The binary hands them to a
Python interpreter that has it installed (`python3`, or `MEDH5_PYTHON`), and
says what to install when there is none:

```bash
pip install "medh5[nifti,dicom]"
medh5 convert from-nifti case.medh5 --image CT=ct.nii.gz
```

The full command reference is at
[medh5.readthedocs.io](https://medh5.readthedocs.io/en/latest/reference/cli/).
