# medh5

The MEDH5 format engine.  A `.medh5` file is **one subject at one or more
timepoints**, with every image, annotation, transform and curation record about
them, in one self-describing HDF5 container.  This crate is the canonical
implementation of formats 1.0 and 1.1 ([the specification], and [1.1] for the
optional clinical profile): the data model, HDF5 I/O, validation, chunked
access, compression, geometry and transforms, annotations, provenance,
integrity --- and, for 1.1, clinical events on a subject clock, strict
prospective selection, and the [task and cache contract] training builds on.

It has two siblings, and all three read and write the same bytes:

- the **Python package** (`pip install medh5`), a thin layer over this crate
  that adds the NumPy, PyTorch, MONAI and converter integrations;
- the **`medh5` command line** ([`medh5-cli`]), a native binary over this crate.

HDF5 and C-Blosc2 are built from source and linked statically (see
[`medh5-sys`]), so a dependent crate needs no HDF5 installation.

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

## Where things are

| Module | Specification |
|---|---|
| [`sample`] | §2, §4, §14.4 --- reading (`Sample`) and writing (`SampleWriter`, `create`, `amend`) |
| [`geometry`] | §3 --- grids, affines, frames, multiscale |
| [`annotations`] | §5--§9 --- the voxel and geometric encodings, headers, coverage |
| [`transforms`] | §10 --- affine, displacement, B-spline and composite transforms |
| [`labels`] | §5 --- label sets, hierarchies and ontology codes |
| [`curation`] | §11, §12 --- identity, provenance, quality, timelines, splits, scrubbing |
| [`integrity`] | §13 --- per-object digests and `content_id` |
| [`storage`] | §14 --- codec profiles, chunking, the sampling index |
| [`validate`] | §15 --- the validator and its diagnostic codes ([`codes`]) |
| [`collection`] | §2.2 --- packing samples into a `.medh5c` shard |
| [`dataset`], [`sampling`] | cohort manifests, splits, statistics; patch sampling |
| [`conformance`] | the conformance corpus third-party implementations run |
| [`version`] | 1.1 §2 --- which versions are read fully, read as a projection, or refused |
| [`clinical`] | 1.1 §3--§10 --- the clinical profile: records, columns, checks, selection, augmentation |
| [`companion`] | task-cache-1 --- task manifests, source pins, preflight row views, feature caches |

Errors are one [`Error`] type whose diagnostic code, when it has one, is the
§15.2 code the validator reports for the same defect.

[the specification]: https://medh5.readthedocs.io/en/latest/spec/medh5-1.0/
[1.1]: https://medh5.readthedocs.io/en/latest/spec/medh5-1.1/
[task and cache contract]: https://medh5.readthedocs.io/en/latest/spec/task-cache-1/
[`version`]: https://docs.rs/medh5/latest/medh5/version/
[`clinical`]: https://docs.rs/medh5/latest/medh5/clinical/
[`companion`]: https://docs.rs/medh5/latest/medh5/companion/
[`medh5-cli`]: https://crates.io/crates/medh5-cli
[`medh5-sys`]: https://crates.io/crates/medh5-sys
[`sample`]: https://docs.rs/medh5/latest/medh5/sample/
[`geometry`]: https://docs.rs/medh5/latest/medh5/geometry/
[`annotations`]: https://docs.rs/medh5/latest/medh5/annotations/
[`transforms`]: https://docs.rs/medh5/latest/medh5/transforms/
[`labels`]: https://docs.rs/medh5/latest/medh5/labels/
[`curation`]: https://docs.rs/medh5/latest/medh5/curation/
[`integrity`]: https://docs.rs/medh5/latest/medh5/integrity/
[`storage`]: https://docs.rs/medh5/latest/medh5/storage/
[`validate`]: https://docs.rs/medh5/latest/medh5/validate/
[`codes`]: https://docs.rs/medh5/latest/medh5/codes/
[`collection`]: https://docs.rs/medh5/latest/medh5/collection/
[`dataset`]: https://docs.rs/medh5/latest/medh5/dataset/
[`sampling`]: https://docs.rs/medh5/latest/medh5/sampling/
[`conformance`]: https://docs.rs/medh5/latest/medh5/conformance/
[`Error`]: https://docs.rs/medh5/latest/medh5/enum.Error.html
