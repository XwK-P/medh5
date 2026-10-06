# medh5

The MEDH5 format engine.  A `.medh5` file is **one subject at one or more
timepoints**, with every image, annotation, transform and curation record about
them, in one self-describing HDF5 container.  This crate is the canonical
implementation of format 1.0 (`docs/spec/medh5-1.0.md`): the data model, HDF5
I/O, validation, chunked access, compression, geometry and transforms,
annotations, provenance and integrity.

The Python package (`pip install medh5`) and the `medh5` command line are
frontends over this crate; all three read and write the same bytes.

```rust,no_run
let sample = medh5::sample::open_sample(std::path::Path::new("case.medh5"))?;
println!("{}", sample.summary()?);
# Ok::<(), medh5::Error>(())
```
