//! Hostile files that are not conformance cases: regression fixtures for the
//! denial-of-service bugs 2.0.2 fixed, which `.github/scripts/damaged.py
//! --structural` holds every binary to report within a time and a memory
//! bound.
//!
//! Each is a valid sample with one structure planted in it --- a chain of
//! groups deeper than [`MAX_DEPTH`], a dataset naming a filter nobody
//! registered, a Zstandard frame that decodes to far more than its chunk, a
//! Blosc2 chunk that claims more than its dataset's chunk --- written by the
//! engine's own writer and the HDF5 calls its unit tests make.  Random damage
//! (`damaged.py`'s other mode) rarely produces these shapes.  They are not in
//! the corpus: the limits they cross are this implementation's, not the
//! specification's.

use std::path::{Path, PathBuf};

use hdf5::filters::Filter;

use super::build::{base, mutate, SHAPE};
use crate::array::{DType, NdArray};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::data::{self, Layout};
use crate::h5::ops::{self, MAX_DEPTH};
use crate::h5sys::{h5d, h5p, h5s, h5t, h5z};
use crate::{Error, Result};

/// Where each fixture plants its structure.
const GROUP: &str = "x_robustness";

/// A digest no dataset has, so that verifying the dataset reads it.
fn any_digest() -> AttrValue {
    AttrValue::Str(format!("sha256:{}", "0".repeat(64)))
}

/// What plants one fixture's structure in a valid sample.
type Plant = fn(&hdf5::Group) -> Result<()>;

/// Write each fixture into `dir`, and return their paths.
#[doc(hidden)]
pub fn write(dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let fixtures: [(&str, Plant); 4] = [
        ("deep-groups", deep_groups),
        ("unregistered-filter", unregistered_filter),
        ("zstd-frame-larger-than-its-chunk", zstd_bomb),
        ("blosc2-chunk-larger-than-its-dataset-chunk", blosc2_overclaim),
    ];
    let mut out = Vec::new();
    for (name, plant) in fixtures {
        let path = dir.join(format!("{name}.medh5"));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        base(&path, &SHAPE)?;
        mutate(&path, plant)?;
        out.push(path);
    }
    Ok(out)
}

/// A chain of groups 4,096 deep: HDF5's own walk of it recursed without
/// limit, and 30,000 overflowed the stack.  Opening it refuses it past
/// [`MAX_DEPTH`].
fn deep_groups(root: &hdf5::Group) -> Result<()> {
    let mut group = ops::new_group(root, GROUP)?;
    for _ in 0..64 * MAX_DEPTH {
        group = ops::new_group(&group, "a")?;
    }
    Ok(())
}

/// One chunked `uint8` dataset of `len` bytes with `filters`, in the planted
/// group, carrying a digest: verifying it reads it.
fn planted(root: &hdf5::Group, len: usize, filters: Vec<Filter>) -> Result<hdf5::Dataset> {
    let group = ops::new_group(root, GROUP)?;
    let layout = Layout { chunks: Some(vec![len]), filters };
    let ds = data::create_empty(&group, "data", DType::U8, &[len], &layout)?;
    attrs::write(&ds, "digest", &any_digest())?;
    Ok(ds)
}

/// Store `bytes` as the dataset's first chunk, as if its filters had written
/// them.
fn write_raw_chunk(ds: &hdf5::Dataset, bytes: &[u8]) -> Result<()> {
    let offset = [0u64];
    // SAFETY: under HDF5's lock, for a dataset open for the call; `bytes`
    // is valid for its length.
    let status = crate::h5::locked(|| unsafe {
        h5d::H5Dwrite_chunk(ds.id(), h5p::H5P_DEFAULT, 0, offset.as_ptr(), bytes.len(), bytes.as_ptr().cast())
    });
    if status < 0 {
        return Err(Error::Io(format!("could not write the chunk of {}", ds.name())));
    }
    Ok(())
}

/// A dataset naming filter 40000, which nobody registers --- optional, as
/// h5py names a filter it has not got.  HDF5 searched its plugin directories
/// for a library to run; the engine turns that off, and the read fails.
fn unregistered_filter(root: &hdf5::Group) -> Result<()> {
    let group = ops::new_group(root, GROUP)?;
    let dims = [4096u64];
    // SAFETY: under HDF5's lock; every identifier made here is closed here.
    let id = crate::h5::locked(|| unsafe {
        let dcpl = h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_CREATE);
        h5p::H5Pset_chunk(dcpl, 1, dims.as_ptr());
        h5p::H5Pset_filter(dcpl, 40000, h5z::H5Z_FLAG_OPTIONAL, 0, std::ptr::null());
        let space = h5s::H5Screate_simple(1, dims.as_ptr(), std::ptr::null());
        let id = h5d::H5Dcreate2(
            group.id(),
            c"data".as_ptr(),
            *h5t::H5T_NATIVE_UCHAR,
            space,
            h5p::H5P_DEFAULT,
            dcpl,
            h5p::H5P_DEFAULT,
        );
        h5s::H5Sclose(space);
        h5p::H5Pclose(dcpl);
        if id >= 0 {
            h5d::H5Dclose(id);
        }
        id
    });
    if id < 0 {
        return Err(Error::Io("could not create a dataset naming filter 40000".into()));
    }
    let ds = group.dataset("data")?;
    attrs::write(&ds, "digest", &any_digest())?;
    write_raw_chunk(&ds, &[0x5A; 4096])
}

/// A Zstandard frame of 16,384 run-length blocks of 128 KiB --- 64 KiB that
/// decode to 2 GiB --- with no content size in its header, for a 64 KiB
/// chunk.  The filter grew its buffer, doubling, until the frame ended.
fn zstd_bomb(root: &hdf5::Group) -> Result<()> {
    const BLOCK: u32 = 128 * 1024;
    const BLOCKS: usize = 16_384;
    let ds = planted(root, 64 * 1024, vec![Filter::user(medh5_sys::ZSTD_FILTER_ID as _, &[3])])?;
    // The magic number, then a frame header with no content size and a
    // 128 KiB window; each block is a three-byte header and its one byte.
    let mut frame = 0xFD2F_B528u32.to_le_bytes().to_vec();
    frame.extend([0x00, 0x38]);
    for i in 0..BLOCKS {
        let header = u32::from(i + 1 == BLOCKS) | (1 << 1) | (BLOCK << 3);
        frame.extend(&header.to_le_bytes()[..3]);
        frame.push(0x41);
    }
    write_raw_chunk(&ds, &frame)
}

/// A Blosc2 chunk of 4 MiB of zeros --- a few hundred bytes --- stored as the
/// one chunk of a 4 KiB dataset.  The filter trusted the chunk's header for
/// the size it allocated and decompressed.
fn blosc2_overclaim(root: &hdf5::Group) -> Result<()> {
    let blosc2 = || vec![Filter::user(medh5_sys::BLOSC2_FILTER_ID as _, &[0, 0, 0, 0, 5, 1, 0])];
    let large = 4 << 20;
    let scratch = ops::new_group(root, "x_scratch")?;
    let zeros = NdArray::from_vec(&[large], vec![0f64; large])?.astype(DType::U8);
    let source = data::create(&scratch, "zeros", &zeros, &Layout { chunks: Some(vec![large]), filters: blosc2() })?;
    let (_, chunk) = ops::read_raw_chunk(&source, &[0])?;
    drop(source);
    drop(scratch);
    root.unlink("x_scratch")?;
    let ds = planted(root, 4096, blosc2())?;
    write_raw_chunk(&ds, &chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each fixture is validated to a report with an error, never a panic,
    /// and the report names what was planted: the file is refused at open,
    /// or the planted dataset is the one that could not be read.
    #[test]
    fn every_robustness_fixture_validates_to_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let paths = write(dir.path()).unwrap();
        assert_eq!(paths.len(), 4);
        for path in paths {
            let started = std::time::Instant::now();
            let report = crate::validate::validate_file(&path, "strict", None).unwrap();
            let elapsed = started.elapsed();
            let errors: Vec<_> = report.diagnostics.iter().filter(|d| d.severity == "error").collect();
            let name = path.file_stem().unwrap().to_string_lossy();
            assert!(
                errors.iter().any(|d| d.code == "E001" || d.location.contains(GROUP) || d.message.contains(GROUP)),
                "{name}: {:?}",
                report.diagnostics
            );
            assert!(elapsed < std::time::Duration::from_secs(30), "{name}: {elapsed:?}");
        }
    }
}
