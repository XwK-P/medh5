# Rust crate

The format is implemented once, in Rust. The [`medh5`](https://crates.io/crates/medh5)
crate is that implementation --- the data model, HDF5 I/O, validation, chunked
access, compression, geometry and transforms, annotations, provenance and
integrity --- and the Python package and the `medh5` command line are frontends
over it. All three read and write the same bytes.

```toml
[dependencies]
medh5 = "2"
```

HDF5 and the compression filters are built from source and linked statically
(the `medh5-sys` crate), so a dependent crate needs nothing installed --- only a
C compiler and CMake at build time, which HDF5's build uses.

**For an MSVC target**, give the C build `NDEBUG`:
`CFLAGS_x86_64_pc_windows_msvc=-DNDEBUG` (or the variable for your target), in
the environment or under `[env]` in your `.cargo/config.toml` --- `-D`, which
`cl` takes as it takes `/D`, since Git Bash rewrites `/DNDEBUG` into a path. cmake-rs drops
CMake's release flags under the Visual Studio generator, `/DNDEBUG` with them,
so HDF5 would keep its assertions and abort the process on a damaged file where
every other build reports it. An optimised build whose HDF5 lacks the flag
stops with the variable to set; should HDF5 not be rebuilt once it is set,
`cargo clean --release -p hdf5-metno-src` makes it so.
`MEDH5_SYS_SKIP_NDEBUG_CHECK=1` builds anyway.

## Write and read a sample

```rust
use medh5::array::NdArray;
use medh5::sample::{create, open_sample, GridOptions, ImageOptions};

fn main() -> medh5::Result<()> {
    let path = std::env::temp_dir().join("medh5-readme-case_0001.medh5");

    // A writer validates every call and commits atomically: a reader never
    // sees half a file, and a crash leaves the previous one intact.
    let ct = NdArray::from_vec(&[8, 16, 16], vec![0i16; 8 * 16 * 16])?;
    let mut w = create(&path, Some("case_0001"), Some("DEMO-0001"), "balanced", &[])?;
    w.add_grid("ct", &[8, 16, 16], &[2.0, 0.8, 0.8], GridOptions::default())?;
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default())?;
    let content_id = w.commit(true)?; // the Merkle root over every stored digest

    let sample = open_sample(&path)?;
    assert_eq!(sample.identity()?.subject_id, "DEMO-0001");
    assert_eq!(sample.content_id()?, content_id);
    let voxels = sample.image("CT")?.read(None, false, None)?;
    assert_eq!(voxels.shape(), [8, 16, 16]);
    assert!(sample.verify(None)?.ok());

    // The validator the writer is held to (§15); `strict` would also ask for a
    // de-identification record (W903).
    let report = medh5::validate::validate_file(&path, "integrity", None)?;
    assert!(report.ok(), "{:?}", report.codes());

    std::fs::remove_file(&path)?;
    Ok(())
}
```

That example is the crate's README, which `cargo test` compiles and runs.

## Where things are

The modules follow the specification's sections, so a clause has one obvious
home; the [API documentation](https://docs.rs/medh5) has every item.

| Module | Specification |
|---|---|
| `sample` | §2, §4, §14.4 --- `Sample` (reading), `SampleWriter`, `create` and `amend` |
| `geometry` | §3 --- grids, affines, frames, multiscale |
| `annotations` | §5–§9 --- the voxel and geometric encodings, headers, coverage |
| `transforms` | §10 --- affine, displacement, B-spline and composite transforms |
| `labels` | §5 --- label sets, hierarchies, ontology codes |
| `curation` | §11, §12 --- identity, provenance, quality, timelines, splits, scrubbing |
| `integrity` | §13 --- object digests and `content_id` |
| `storage` | §14 --- codec profiles, chunking, the sampling index |
| `validate`, `codes` | §15 --- the validator and the diagnostic code table |
| `collection` | §2.2 --- `.medh5c` shards |
| `dataset`, `sampling` | cohort manifests, splits and statistics; patch sampling |
| `conformance` | the conformance corpus |
| `version` | [1.1](../spec/medh5-1.1.md) §2 --- which versions are read fully, read as a projection, or refused |
| `clinical` | [1.1](../spec/medh5-1.1.md) §3–§10 --- the clinical profile: records, columns, checks, selection, augmentation |
| `companion` | [task-cache-1](../spec/task-cache-1.md) --- task manifests, source pins, preflight row views, feature caches |

Errors are one `medh5::Error`; when a defect has a §15.2 code, the error carries
the code the validator reports for the same defect.

## The command line

[`medh5-cli`](https://crates.io/crates/medh5-cli) is the `medh5` binary, and a
library (`medh5_cli::run`) for embedding the command line in another program;
see [Command line](cli.md).
