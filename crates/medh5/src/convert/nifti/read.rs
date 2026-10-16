//! Finding a NIfTI file's header and voxels: what a path names, the header
//! read and checked, its extensions stepped over, and the voxel bytes read.
//!
//! What a path is follows 2.1.0's reader exactly, because its refusals are
//! part of the converter's contract (2.2.0 decision E3): the same formats are
//! recognised in the same order, from the same first kilobyte, and a file
//! that is not NIfTI is refused with the same words --- naming the format
//! that reader would have opened it as, where it would have.
//!
//! gzip is inflated in pure Rust (`flate2` over `miniz_oxide`), from the
//! file, never further than the bytes the header implies: the first
//! kilobyte to recognise the file, the header and its extensions up to
//! `vox_offset`, then exactly the voxels.  A stream that inflates to more
//! than that is never inflated further, so a small file cannot make the
//! reader allocate or decompress what the header does not ask for.

use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;

use super::header::{Element, Header, Version, VoxOffset, NATIVE_BIG};
use crate::{Error, Result};

/// How much of a file is read to recognise it.
const SNIFF_BYTES: u64 = 1024;

/// The suffixes a NIfTI file may carry after its own: compressed copies.
const COMPRESSED: [&str; 3] = [".gz", ".bz2", ".zst"];

/// Why a path could not be loaded.
#[derive(Debug)]
pub(crate) enum LoadError {
    /// The path does not exist or cannot be examined.
    NotFound,
    /// The path is no image this reader recognises; the text says why.
    NotAnImage(String),
    /// A recognised file is broken.
    Broken(Error),
}

impl From<Error> for LoadError {
    fn from(err: Error) -> Self {
        LoadError::Broken(err)
    }
}

/// What a path holds.
#[derive(Debug)]
pub(crate) enum Loaded {
    /// A NIfTI-1 or NIfTI-2 image, single-file or pair.
    Nifti(Box<Image>),
    /// An image of another format: the name 2.1.0's reader gave its class.
    Other(&'static str),
}

/// A NIfTI image whose header has been read and checked.
#[derive(Debug, Clone)]
pub(crate) struct Image {
    pub header: Header,
    /// The file the voxels are in, spelled as the reader spells paths.
    pub image_name: String,
    /// The same file, as the filesystem names it.
    pub image_path: PathBuf,
    /// The voxels' shape, NIfTI order (`x` fastest).
    pub shape: Vec<i64>,
    /// Where the voxels start.
    pub offset: u64,
    /// The scale the header states, if any.
    pub slope_inter: Option<(f64, f64)>,
}

/// Voxels in the header's byte order: as read from the file, in Fortran
/// order and NIfTI's shape; as [`read_nifti`](super::read_nifti) returns
/// them, in C order and the returned shape.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Voxels {
    pub element: Element,
    /// Whether multi-byte elements are stored big-end first.
    pub big_endian: bool,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// A path as the reader spells it: `~` expanded, `.` and repeated
/// separators dropped, and `/` on every platform.
pub(crate) fn spelled(path: &Path) -> String {
    let text = path.to_string_lossy().into_owned();
    let expanded = match text.strip_prefix('~') {
        Some(rest) if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') => match home() {
            Some(home) => format!("{home}{rest}"),
            None => text.clone(),
        },
        _ => text.clone(),
    };
    crate::pyval::py_path(Path::new(&expanded))
}

/// The file `wanted` (`name`, or its companion) is, for the filesystem:
/// `wanted` itself, or, where `path` is not UTF-8 (a Unix name in another
/// encoding, which `name` spells lossily), `path`'s own bytes with the ASCII
/// suffix `wanted` changes, so the file opened is the one named.
fn fs_path(path: &Path, name: &str, wanted: &str) -> PathBuf {
    #[cfg(unix)]
    if path.to_str().is_none() {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};
        let bytes = path.as_os_str().as_bytes();
        let mut out = match (bytes.strip_prefix(b"~"), home()) {
            (Some(rest), Some(home)) if rest.is_empty() || rest.starts_with(b"/") => [home.as_bytes(), rest].concat(),
            _ => bytes.to_vec(),
        };
        let common = name.bytes().zip(wanted.bytes()).take_while(|(a, b)| a == b).count();
        out.truncate(out.len().saturating_sub(name.len() - common));
        out.extend_from_slice(&wanted.as_bytes()[common..]);
        return PathBuf::from(std::ffi::OsString::from_vec(out));
    }
    #[cfg(not(unix))]
    let _ = (path, name);
    PathBuf::from(wanted)
}

fn home() -> Option<String> {
    let key = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
    std::env::var(key).ok().filter(|h| !h.is_empty())
}

/// `(root, ext, addext)`: the suffix of `addexts` the name ends with (in any
/// case), and the extension before it.  The extension is found in the whole
/// path, as 2.1.0's reader found it.
fn split_ext<'a>(name: &'a str, addexts: &[&str]) -> (&'a str, &'a str, &'a str) {
    let lower = name.to_ascii_lowercase();
    let mut base = name;
    let mut addext = "";
    for ext in addexts {
        if lower.ends_with(&ext.to_ascii_lowercase()) {
            let at = name.len() - ext.len();
            base = &name[..at];
            addext = &name[at..];
            break;
        }
    }
    match base.rfind('.') {
        Some(at) if !base.trim_matches('.').is_empty() => (&base[..at], &base[at..], addext),
        _ => (base, "", addext),
    }
}

/// The name of the file of `kind` (`.hdr` or `.img`) beside `ext`'s, its
/// case following the extension's.
fn companion(root: &str, ext: &str, addext: &str, kind: &str) -> String {
    let kind = if ext == ext.to_ascii_uppercase() && ext != ext.to_ascii_lowercase() {
        kind.to_ascii_uppercase()
    } else {
        kind.to_string()
    };
    format!("{root}{kind}{addext}")
}

/// Whether `name` is opened through gzip: its last suffix, in any case.
fn is_gzip(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".gz")
}

fn compression(name: &str) -> Option<&'static str> {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".bz2") {
        Some("bzip2")
    } else if lower.ends_with(".zst") {
        Some("ztsd")
    } else {
        None
    }
}

/// A file as a byte stream, inflated when its name ends in `.gz`.
fn open(path: &Path, name: &str) -> io::Result<Box<dyn Read>> {
    let file = File::open(path)?;
    if is_gzip(name) {
        Ok(Box::new(MultiGzDecoder::new(BufReader::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Read up to `limit` bytes; short only at the end of the stream.
fn read_up_to(stream: &mut dyn Read, limit: u64) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    stream.take(limit).read_to_end(&mut out)?;
    Ok(out)
}

/// Read and discard up to `limit` bytes; how many there were.
fn skip(stream: &mut dyn Read, limit: u64) -> io::Result<u64> {
    io::copy(&mut stream.take(limit), &mut io::sink())
}

/// The first kilobyte of `name`, or `None` where it cannot be read.
fn sniff(path: &Path, name: &str) -> Option<Vec<u8>> {
    let mut stream = open(path, name).ok()?;
    read_up_to(&mut *stream, SNIFF_BYTES).ok()
}

fn nifti1_magic(block: &[u8]) -> bool {
    block.len() >= 348 && matches!(super::header::strip_nuls(&block[344..348]), b"ni1" | b"n+1")
}

fn sizeof_either(block: &[u8], size: usize) -> bool {
    if block.len() < size {
        return false;
    }
    let raw: [u8; 4] = block[..4].try_into().expect("four bytes");
    let size = size as i32;
    i32::from_le_bytes(raw) == size || i32::from_be_bytes(raw) == size
}

/// The image classes 2.1.0's reader tried, in its order, by the extensions
/// each takes and the compressed suffixes it sees through.
enum Candidate {
    Nifti1Pair,
    Nifti1Image,
    Nifti2Pair,
    Cifti2Image,
    Nifti2Image,
    Analyze,
    Minc1,
    Minc2,
    Mgh,
    Parrec,
    Gifti,
    Afni,
}

impl Candidate {
    const ALL: [Candidate; 12] = [
        Candidate::Nifti1Pair,
        Candidate::Nifti1Image,
        Candidate::Nifti2Pair,
        Candidate::Cifti2Image,
        Candidate::Nifti2Image,
        Candidate::Analyze,
        Candidate::Minc1,
        Candidate::Minc2,
        Candidate::Mgh,
        Candidate::Parrec,
        Candidate::Gifti,
        Candidate::Afni,
    ];

    fn exts(&self) -> &'static [&'static str] {
        match self {
            Candidate::Nifti1Pair | Candidate::Nifti2Pair | Candidate::Analyze => &[".img", ".hdr"],
            Candidate::Nifti1Image | Candidate::Cifti2Image | Candidate::Nifti2Image => &[".nii"],
            Candidate::Minc1 | Candidate::Minc2 => &[".mnc"],
            Candidate::Mgh => &[".mgh", ".mgz"],
            Candidate::Parrec => &[".rec", ".par"],
            Candidate::Gifti => &[".gii"],
            Candidate::Afni => &[".brik", ".head"],
        }
    }

    fn compressed(&self) -> &'static [&'static str] {
        match self {
            Candidate::Cifti2Image | Candidate::Minc2 | Candidate::Mgh | Candidate::Parrec => &[],
            Candidate::Gifti => &[".gz", ".bz2"],
            Candidate::Afni => &[".gz", ".bz2", ".Z", ".zst"],
            _ => &COMPRESSED,
        }
    }

    fn pair(&self) -> bool {
        matches!(self, Candidate::Nifti1Pair | Candidate::Nifti2Pair | Candidate::Analyze)
    }

    fn name(&self) -> &'static str {
        match self {
            Candidate::Nifti1Pair => "Nifti1Pair",
            Candidate::Nifti1Image => "Nifti1Image",
            Candidate::Nifti2Pair => "Nifti2Pair",
            Candidate::Cifti2Image => "Cifti2Image",
            Candidate::Nifti2Image => "Nifti2Image",
            Candidate::Analyze => "Spm2AnalyzeImage",
            Candidate::Minc1 => "Minc1Image",
            Candidate::Minc2 => "Minc2Image",
            Candidate::Mgh => "MGHImage",
            Candidate::Parrec => "PARRECImage",
            Candidate::Gifti => "GiftiImage",
            Candidate::Afni => "AFNIImage",
        }
    }
}

/// Recognise and load what `path` names: a NIfTI image's checked header, or
/// the format it is instead.
pub(crate) fn load(path: &Path) -> std::result::Result<Loaded, LoadError> {
    let name = spelled(path);
    let fs = |wanted: &str| fs_path(path, &name, wanted);
    let meta = std::fs::metadata(fs(&name)).map_err(|_| LoadError::NotFound)?;
    if meta.len() == 0 {
        return Err(LoadError::NotAnImage(format!("Empty file: '{name}'")));
    }
    if let Some(format) = compression(&name) {
        // 2.1.0 read these through Python's codecs; this reader inflates
        // gzip only.
        return Err(LoadError::NotAnImage(match signature(&fs(&name), &name) {
            Some(message) => message,
            None => format!("{name} is {format}-compressed, and only gzip is read; decompress it first"),
        }));
    }
    for candidate in Candidate::ALL {
        let (root, ext, addext) = split_ext(&name, candidate.compressed());
        if !candidate.exts().contains(&ext.to_ascii_lowercase().as_str()) {
            continue;
        }
        let meta_name = if candidate.pair() { companion(root, ext, addext, ".hdr") } else { name.clone() };
        let image_name = if candidate.pair() { companion(root, ext, addext, ".img") } else { name.clone() };
        let block = || sniff(&fs(&meta_name), &meta_name).unwrap_or_default();
        let found = match candidate {
            Candidate::Nifti1Pair | Candidate::Nifti1Image => nifti1_magic(&block()),
            Candidate::Nifti2Pair | Candidate::Nifti2Image => sizeof_either(&block(), 540),
            Candidate::Cifti2Image => {
                let block = block();
                if !sizeof_either(&block, 540) {
                    false
                } else {
                    // CIFTI-2 is told by its intent, read from a header
                    // checked as any NIfTI-2 header is.
                    let mut header = Header::decode(&block[..540], Version::Two)?;
                    header.check_fix()?;
                    (3000..3100).contains(&header.intent_code)
                }
            }
            Candidate::Analyze => sizeof_either(&block(), 348),
            Candidate::Minc1 => block().starts_with(b"CDF\x01"),
            Candidate::Minc2 => block().starts_with(b"\x89HDF"),
            Candidate::Mgh | Candidate::Parrec | Candidate::Gifti | Candidate::Afni => true,
        };
        if !found {
            continue;
        }
        return match candidate {
            Candidate::Nifti1Pair | Candidate::Nifti1Image => {
                let files = (fs(&meta_name), meta_name, fs(&image_name), image_name);
                Ok(Loaded::Nifti(Box::new(load_nifti(files, Version::One, !candidate.pair())?)))
            }
            Candidate::Nifti2Pair | Candidate::Nifti2Image => {
                let files = (fs(&meta_name), meta_name, fs(&image_name), image_name);
                Ok(Loaded::Nifti(Box::new(load_nifti(files, Version::Two, !candidate.pair())?)))
            }
            other => Ok(Loaded::Other(other.name())),
        };
    }
    Err(LoadError::NotAnImage(
        signature(&fs(&name), &name).unwrap_or_else(|| format!("Cannot work out file type of \"{name}\"")),
    ))
}

/// Why a compressed file's first bytes are not what its suffix says, if
/// they are not.
fn signature(path: &Path, name: &str) -> Option<String> {
    let lower = name.to_ascii_lowercase();
    let (magic, format): (&[u8], &str) = if lower.ends_with(".gz") {
        (b"\x1f\x8b", "gzip")
    } else if lower.ends_with(".bz2") {
        (b"BZh", "bzip2")
    } else if lower.ends_with(".zst") {
        (b"\x28\xb5\x2f\xfd", "ztsd")
    } else {
        return None;
    };
    let mut head = Vec::new();
    match File::open(path).and_then(|f| f.take(magic.len() as u64).read_to_end(&mut head)) {
        Err(_) => Some(format!("Could not read file: {name}")),
        Ok(_) if head.starts_with(magic) => None,
        Ok(_) => Some(format!("File {name} is not a {format} file")),
    }
}

/// Read and check a NIfTI header, step over its extensions, and work out
/// where and what its voxels are.
/// `files` is the header's path and name, then the voxels'.
fn load_nifti(files: (PathBuf, String, PathBuf, String), version: Version, single: bool) -> Result<Image> {
    let (meta_path, meta_name, image_path, image_name) = files;
    let mut stream = open(&meta_path, &meta_name).map_err(|e| open_error(&e, &meta_name))?;
    let block = read_up_to(&mut *stream, version.size() as u64).map_err(gzip_error)?;
    let mut header = Header::decode(&block, version)?;
    header.check_fix()?;
    skip_extensions(&mut *stream, &header, single)?;
    let shape = header.data_shape()?;
    let offset = match header.vox_offset {
        VoxOffset::Float(v) => data_offset(v)?,
        // A negative offset reads from the start, as seeking before it does.
        VoxOffset::Int(v) => v.max(0) as u64,
    };
    let slope_inter = header.slope_inter()?;
    Ok(Image { header, image_name, image_path, shape, offset, slope_inter })
}

/// `int(vox_offset)`.
fn data_offset(value: f64) -> Result<u64> {
    if value.is_nan() {
        return Err(Error::Value("cannot convert float NaN to integer".into()));
    }
    if value.is_infinite() {
        return Err(Error::Value("cannot convert float infinity to integer".into()));
    }
    // A negative offset reads from the start, as seeking before it does.
    Ok(value.trunc().max(0.0) as u64)
}

/// Step over the header extensions, checking each is whole: up to
/// `vox_offset` in a single file, to the end of a header file of its own.
fn skip_extensions(stream: &mut dyn Read, header: &Header, single: bool) -> Result<()> {
    let flag = read_up_to(stream, 4).map_err(gzip_error)?;
    if flag.len() < 4 || flag[0] == 0 {
        return Ok(());
    }
    let mut size: f64 = if single { header.vox_offset.as_f64() - (header.version.size() + 4) as f64 } else { -1.0 };
    let swap = header.big_endian != NATIVE_BIG;
    while size >= 16.0 || size < 0.0 {
        let head = read_up_to(stream, 8).map_err(gzip_error)?;
        if head.is_empty() && size < 0.0 {
            break;
        }
        if head.len() != 8 {
            return Err(Error::Value("failed to read extension header".into()));
        }
        let word = |at: usize| {
            let raw: [u8; 4] = head[at..at + 4].try_into().expect("four bytes");
            let v = i32::from_ne_bytes(raw);
            i64::from(if swap { v.swap_bytes() } else { v })
        };
        let esize = word(0);
        let wanted = esize - 8;
        // nibabel read a negative count as "the rest of the file", which can
        // never be one, and then refused the extension: refused here without
        // inflating the rest, which a small gzip stream can make enormous.
        if wanted < 0 {
            return Err(Error::Value("failed to read extension content".into()));
        }
        let got = skip(stream, wanted as u64).map_err(gzip_error)? as i64;
        if got != wanted {
            return Err(Error::Value("failed to read extension content".into()));
        }
        size -= esize as f64;
    }
    Ok(())
}

/// A file that would not open, as Python words it on every platform: a
/// missing one is `[Errno 2] No such file or directory`, which Windows
/// words otherwise.
fn open_error(err: &io::Error, name: &str) -> Error {
    if err.kind() == io::ErrorKind::NotFound {
        Error::Io(format!("[Errno 2] No such file or directory: {}", crate::json::repr_str(name)))
    } else {
        crate::error::os_error(err, name)
    }
}

/// A failure to inflate, as 2.1.0's reader reported one.
fn gzip_error(err: io::Error) -> Error {
    if err.kind() == io::ErrorKind::UnexpectedEof {
        Error::Io("Compressed file ended before the end-of-stream marker was reached".into())
    } else {
        Error::Io(err.to_string())
    }
}

impl Image {
    /// The voxel type.
    pub fn element(&self) -> Element {
        self.header.element().expect("a checked header names a known datatype")
    }

    /// Read the voxels: exactly the bytes the header implies, from
    /// `vox_offset`, and never more.
    pub fn read_voxels(&self) -> Result<Voxels> {
        let element = self.element();
        let big_endian = self.header.big_endian;
        let empty = |shape: Vec<usize>| Voxels { element, big_endian, shape, bytes: Vec::new() };
        let compressed = is_gzip(&self.image_name);
        if self.shape.is_empty() {
            // A shape of no dimensions is numpy's scalar: one element.  Where
            // numpy maps it (an uncompressed file holding the element), it is
            // read, and 2.1.0's `ascontiguousarray` made it `(1,)`; elsewhere
            // nibabel read nothing, and an empty read is `(0,)`.
            let mut file = File::open(&self.image_path).map_err(|e| open_error(&e, &self.image_name))?;
            let item = element.itemsize() as u64;
            let length = file.metadata().map(|m| m.len()).unwrap_or(0);
            if compressed || length < self.offset.saturating_add(item) {
                return Ok(empty(vec![0]));
            }
            file.seek(SeekFrom::Start(self.offset))?;
            let mut bytes = vec![0; element.itemsize()];
            file.read_exact(&mut bytes)?;
            return Ok(Voxels { element, big_endian, shape: vec![1], bytes });
        }
        // NIfTI-2's seven 64-bit dimensions can name more bytes than even
        // an i128 counts: that array is one no machine holds.
        let n_bytes = self
            .shape
            .iter()
            .try_fold(element.itemsize() as i128, |acc, v| acc.checked_mul(i128::from(*v)))
            .ok_or_else(too_big)?;
        if n_bytes < 0 {
            return Err(Error::Value("negative count".into()));
        }
        // Nor can any machine hold more bytes than an address counts, which
        // numpy refuses for any array: refused before a byte is read.
        if n_bytes > isize::MAX as i128 {
            return Err(too_big());
        }
        let file = File::open(&self.image_path).map_err(|e| open_error(&e, &self.image_name))?;
        if n_bytes == 0 {
            // A file mapped from a non-zero offset keeps its declared shape,
            // when numpy can make an array of it; read, or when numpy refuses
            // the shape, an empty array is `(0,)`.
            let mapped = !compressed
                && self.offset > 0
                && numpy_holds(&self.shape, element.itemsize())
                && file.metadata().map(|m| m.len() >= self.offset).unwrap_or(false);
            return Ok(empty(if mapped { self.dims()? } else { vec![0] }));
        }
        let wanted = u64::try_from(n_bytes).unwrap_or(u64::MAX);
        let bytes = if is_gzip(&self.image_name) {
            let mut stream = MultiGzDecoder::new(BufReader::new(file));
            skip(&mut stream, self.offset).map_err(gzip_error)?;
            read_capped(&mut stream, wanted, None).map_err(gzip_error)?
        } else {
            let mut file = file;
            let length = file.metadata().map(|m| m.len()).unwrap_or(0);
            file.seek(SeekFrom::Start(self.offset))?;
            let available = length.saturating_sub(self.offset);
            read_capped(&mut BufReader::new(file), wanted, Some(available))?
        };
        if bytes.len() as u64 != wanted {
            return Err(Error::Io(format!(
                "Expected {n_bytes} bytes, got {} bytes from {}\n - could the file be damaged?",
                bytes.len(),
                self.image_name
            )));
        }
        Ok(Voxels { element, big_endian, shape: self.dims()?, bytes })
    }

    /// The shape as an array's, refused where a dimension is not one.
    fn dims(&self) -> Result<Vec<usize>> {
        self.shape.iter().map(|v| usize::try_from(*v).map_err(|_| too_big())).collect()
    }
}

/// Whether numpy makes an array of `shape`: no extent negative, and the
/// non-zero extents' bytes counted in an `npy_intp` (an empty array's other
/// extents still count, so `(2^63 - 1, 3, 0)` is too big).
fn numpy_holds(shape: &[i64], itemsize: usize) -> bool {
    let limit = isize::MAX as i64;
    shape.iter().all(|v| *v >= 0)
        && shape
            .iter()
            .filter(|v| **v != 0)
            .try_fold(itemsize as i64, |acc, v| acc.checked_mul(*v).filter(|n| *n <= limit))
            .is_some()
}

/// numpy's refusal of an array larger than the address space.
fn too_big() -> Error {
    Error::Value("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.".into())
}

/// Read `wanted` bytes or to the end, reserving no more than `available`
/// (or a modest start, for a stream of unknown length) up front.
fn read_capped(stream: &mut dyn Read, wanted: u64, available: Option<u64>) -> io::Result<Vec<u8>> {
    const START: u64 = 64 << 20;
    let reserve = wanted.min(available.unwrap_or(START)).min(usize::MAX as u64) as usize;
    let mut out = Vec::new();
    // Up front only where the memory is there; reading grows it either way.
    let _ = out.try_reserve_exact(reserve);
    stream.take(wanted).read_to_end(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::write::GzEncoder;
    use flate2::Compression;

    use super::*;

    fn header(version: Version, shape: &[i64]) -> Header {
        let mut h = Header::new(version);
        h.dim[0] = shape.len() as i64;
        for (k, v) in shape.iter().enumerate() {
            h.dim[k + 1] = *v;
        }
        h.datatype = 4;
        h.bitpix = 16;
        h.vox_offset = VoxOffset::Float(if version == Version::One { 352.0 } else { 544.0 });
        h
    }

    fn single(h: &Header, voxels: &[u8]) -> Vec<u8> {
        let mut out = h.encode();
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(voxels);
        out
    }

    fn gzip(bytes: &[u8]) -> Vec<u8> {
        let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
        enc.write_all(bytes).unwrap();
        enc.finish().unwrap()
    }

    fn voxels(n: usize) -> Vec<u8> {
        (0..n as i16).flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn nifti(path: &Path) -> Image {
        match load(path).unwrap() {
            Loaded::Nifti(image) => *image,
            Loaded::Other(name) => panic!("read as {name}"),
        }
    }

    #[test]
    fn s02_2_nifti2_and_hdr_img_read() {
        let dir = tempfile::tempdir().unwrap();
        let data = voxels(24);
        // NIfTI-2, single file, both byte orders.
        for big in [false, true] {
            let mut h = header(Version::Two, &[4, 3, 2]);
            h.big_endian = big;
            let stored: Vec<u8> =
                if big == NATIVE_BIG { data.clone() } else { data.chunks(2).flat_map(|c| [c[1], c[0]]).collect() };
            let path = dir.path().join(format!("n2-{big}.nii"));
            std::fs::write(&path, single(&h, &stored)).unwrap();
            let image = nifti(&path);
            assert_eq!(image.header.version, Version::Two);
            let read = image.read_voxels().unwrap();
            assert_eq!((read.big_endian, read.shape.clone(), read.bytes), (big, vec![4, 3, 2], stored));
        }
        // A NIfTI-1 pair, named by either file and in upper case.
        for (version, magic) in [(Version::One, *b"ni1\0"), (Version::Two, *b"ni2\0")] {
            let mut h = header(version, &[4, 3, 2]);
            h.magic = magic;
            h.vox_offset = VoxOffset::Float(0.0);
            let stem = format!("pair{}", version.size());
            std::fs::write(dir.path().join(format!("{stem}.hdr")), h.encode()).unwrap();
            std::fs::write(dir.path().join(format!("{stem}.img")), &data).unwrap();
            for name in [format!("{stem}.hdr"), format!("{stem}.img")] {
                let image = nifti(&dir.path().join(name));
                assert_eq!(image.header.version, version);
                assert_eq!(image.read_voxels().unwrap().bytes, data);
            }
        }
        std::fs::write(dir.path().join("UP.HDR"), {
            let mut h = header(Version::One, &[4, 3, 2]);
            h.magic = *b"ni1\0";
            h.vox_offset = VoxOffset::Float(0.0);
            h.encode()
        })
        .unwrap();
        std::fs::write(dir.path().join("UP.IMG"), &data).unwrap();
        assert_eq!(nifti(&dir.path().join("UP.HDR")).read_voxels().unwrap().bytes, data);
        // And compressed.
        let path = dir.path().join("one.nii.gz");
        std::fs::write(&path, gzip(&single(&header(Version::One, &[4, 3, 2]), &data))).unwrap();
        assert_eq!(nifti(&path).read_voxels().unwrap().bytes, data);
    }

    #[test]
    fn s02_2_gzip_bomb_is_refused() {
        // A 2x2x2 header followed by 256 MiB of zeros: the header asks for
        // 16 bytes, and nothing past them is inflated.
        let dir = tempfile::tempdir().unwrap();
        let h = header(Version::One, &[2, 2, 2]);
        let mut enc = GzEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&single(&h, &[])).unwrap();
        let zeros = vec![0u8; 1 << 20];
        for _ in 0..256 {
            enc.write_all(&zeros).unwrap();
        }
        let bomb = enc.finish().unwrap();
        assert!(bomb.len() < 1 << 20, "the bomb is small: {}", bomb.len());
        let path = dir.path().join("bomb.nii.gz");
        std::fs::write(&path, &bomb).unwrap();
        let started = std::time::Instant::now();
        let read = nifti(&path).read_voxels().unwrap();
        assert_eq!(read.bytes, vec![0u8; 16]);
        assert!(started.elapsed().as_secs() < 5, "nothing past the voxels is inflated");

        // A header that asks for far more than the stream holds is refused
        // by what the stream holds, not by allocating what it asks for.
        let h = header(Version::One, &[32767, 32767, 32767]);
        let path = dir.path().join("claims.nii.gz");
        std::fs::write(&path, gzip(&single(&h, &[1, 2, 3, 4]))).unwrap();
        let err = nifti(&path).read_voxels().unwrap_err();
        assert!(err.message().starts_with("Expected 70362301923326 bytes, got 4 bytes from"), "{err}");
    }

    #[test]
    fn s02_2_a_count_no_integer_holds_is_refused() {
        // 2^62 x 2^62 x 16 voxels of two bytes overflow an i128; checked,
        // the count is refused rather than wrapped (to zero, an empty read).
        let dir = tempfile::tempdir().unwrap();
        let h = header(Version::Two, &[1 << 62, 1 << 62, 16]);
        let path = dir.path().join("huge.nii");
        std::fs::write(&path, single(&h, &[1, 2, 3, 4])).unwrap();
        let err = nifti(&path).read_voxels().unwrap_err();
        assert!(matches!(&err, Error::Value(m) if m.starts_with("array is too big")), "{err:?}");
        // Empty, mapped, and a shape numpy refuses (a negative extent, or
        // non-zero extents past an npy_intp): nibabel fell back to reading,
        // and an empty read is (0,), as 2.1.0 gave.
        for dims in [[0, -3, 2], [i64::MAX, 3, 0], [1 << 62, 4, 0], [i64::MAX, 1, 0]] {
            let h = header(Version::Two, &dims);
            let path = dir.path().join("empty.nii");
            std::fs::write(&path, single(&h, &[])).unwrap();
            assert_eq!(nifti(&path).read_voxels().unwrap().shape, vec![0], "{dims:?}");
        }
        // One numpy makes keeps its shape.
        let h = header(Version::Two, &[1 << 40, 4, 0]);
        let path = dir.path().join("empty.nii");
        std::fs::write(&path, single(&h, &[])).unwrap();
        assert_eq!(nifti(&path).read_voxels().unwrap().shape, vec![1 << 40, 4, 0]);
    }

    #[test]
    fn s02_2_a_shape_of_no_dimensions_is_one_element_where_numpy_maps_it() {
        // dim[0] = -1 slices no dimensions: numpy's scalar.  Mapped, from a
        // .nii holding it, its one element is read, as (1,); compressed, or
        // from a file too short to map, nibabel read none, as (0,).
        let dir = tempfile::tempdir().unwrap();
        // Big-endian: dim[0] = -1 is no little-endian header's, so the byte
        // order is guessed as big.
        let mut h = header(Version::One, &[4, 3, 2]);
        h.dim[0] = -1;
        h.big_endian = true;
        let data = voxels(2);
        let path = dir.path().join("scalar.nii");
        std::fs::write(&path, single(&h, &data)).unwrap();
        let read = nifti(&path).read_voxels().unwrap();
        assert_eq!((read.shape, read.bytes), (vec![1], data[..2].to_vec()));
        std::fs::write(&path, single(&h, &data[..1])).unwrap();
        assert_eq!(nifti(&path).read_voxels().unwrap().shape, vec![0]);
        let gz = dir.path().join("scalar.nii.gz");
        std::fs::write(&gz, gzip(&single(&h, &data))).unwrap();
        assert_eq!(nifti(&gz).read_voxels().unwrap().shape, vec![0]);
    }

    #[test]
    fn s02_2_more_bytes_than_an_address_counts_are_refused_unread() {
        // 2^31 x 2^31 two-byte voxels: 2^63 bytes, past isize::MAX, though
        // an i128 counts them.
        let dir = tempfile::tempdir().unwrap();
        let h = header(Version::Two, &[1 << 31, 1 << 31]);
        let path = dir.path().join("huge.nii");
        std::fs::write(&path, single(&h, &[0; 64])).unwrap();
        let err = nifti(&path).read_voxels().unwrap_err();
        assert!(matches!(&err, Error::Value(m) if m.starts_with("array is too big")), "{err:?}");
    }

    // Linux only: macOS refuses a file name that is not UTF-8.
    #[cfg(target_os = "linux")]
    #[test]
    fn s02_2_a_name_in_another_encoding_opens_its_own_files() {
        // "caf\xe9": Latin-1, no UTF-8. The name is spelled lossily; the
        // files opened, the companion included, are the ones named.
        use std::os::unix::ffi::OsStringExt;
        let dir = tempfile::tempdir().unwrap();
        let named = |suffix: &str| {
            dir.path().join(std::ffi::OsString::from_vec([b"caf\xe9".as_slice(), suffix.as_bytes()].concat()))
        };
        let data = voxels(24);
        std::fs::write(named(".nii"), single(&header(Version::One, &[4, 3, 2]), &data)).unwrap();
        let image = nifti(&named(".nii"));
        assert!(image.image_name.ends_with("caf\u{fffd}.nii"), "{}", image.image_name);
        assert_eq!(image.read_voxels().unwrap().bytes, data);
        let mut h = header(Version::One, &[4, 3, 2]);
        h.magic = *b"ni1\0";
        h.vox_offset = VoxOffset::Float(0.0);
        std::fs::write(named(".hdr"), h.encode()).unwrap();
        std::fs::write(named(".img"), &data).unwrap();
        for suffix in [".hdr", ".img"] {
            assert_eq!(nifti(&named(suffix)).read_voxels().unwrap().bytes, data, "{suffix}");
        }
    }

    #[test]
    fn s02_2_an_extension_shorter_than_its_header_is_refused_unread() {
        // A gzip'd header file with an extension of size 4 (< 8), then a
        // stream that inflates to 64 MiB and is cut short: refused for the
        // extension, not for the cut, which only reading on would find.
        let dir = tempfile::tempdir().unwrap();
        let mut h = header(Version::One, &[2, 3, 4]);
        h.magic = *b"ni1\0";
        h.vox_offset = VoxOffset::Float(0.0);
        let mut bytes = h.encode();
        bytes.extend_from_slice(&[1, 0, 0, 0]);
        bytes.extend_from_slice(&4i32.to_le_bytes());
        bytes.extend_from_slice(&0i32.to_le_bytes());
        let mut enc = GzEncoder::new(Vec::new(), Compression::best());
        enc.write_all(&bytes).unwrap();
        enc.write_all(&vec![0u8; 64 << 20]).unwrap();
        let whole = enc.finish().unwrap();
        std::fs::write(dir.path().join("x.hdr.gz"), &whole[..whole.len() - 64]).unwrap();
        std::fs::write(dir.path().join("x.img.gz"), gzip(&voxels(48))).unwrap();
        match load(&dir.path().join("x.hdr.gz")) {
            Err(LoadError::Broken(Error::Value(m))) => assert_eq!(m, "failed to read extension content"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn s02_2_a_nifti2_offset_is_read_exactly() {
        // 2^53 + 1 has no f64: the offset is the header's integer, not its
        // nearest double.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("far.nii");
        for big in [false, true] {
            let mut h = header(Version::Two, &[2, 3, 4]);
            h.big_endian = big;
            let mut bytes = single(&h, &voxels(24));
            let far: i64 = (1 << 53) + 1;
            bytes[168..176].copy_from_slice(&if big { far.to_be_bytes() } else { far.to_le_bytes() });
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(nifti(&path).offset, (1 << 53) + 1);
        }
    }

    #[test]
    fn s02_2_truncated_gzip_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let data = voxels(24 * 100);
        let whole = gzip(&single(&header(Version::One, &[40, 30, 2]), &data));
        // Cut inside the header: no file this reader recognises.
        let path = dir.path().join("head.nii.gz");
        std::fs::write(&path, &whole[..20]).unwrap();
        match load(&path) {
            Err(LoadError::NotAnImage(m)) => assert!(m.starts_with("Cannot work out file type of"), "{m}"),
            other => panic!("{other:?}"),
        }
        // Cut inside the voxels: the header reads, the voxels do not.
        let path = dir.path().join("body.nii.gz");
        std::fs::write(&path, &whole[..whole.len() - 40]).unwrap();
        let err = nifti(&path).read_voxels().unwrap_err();
        assert!(matches!(err, Error::Io(_)), "{err:?}");
    }

    #[test]
    fn s02_2_other_formats_are_named_and_unknown_ones_refused() {
        let dir = tempfile::tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = dir.path().join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        };
        assert!(matches!(load(&write("x.mgh", b"anything")), Ok(Loaded::Other("MGHImage"))));
        assert!(matches!(load(&write("x.mnc", b"CDF\x01rest")), Ok(Loaded::Other("Minc1Image"))));
        let analyze = {
            let mut h = header(Version::One, &[2, 2, 2]);
            h.magic = [0; 4];
            h.encode()
        };
        write("a.img", &[0; 16]);
        assert!(matches!(load(&write("a.hdr", &analyze)), Ok(Loaded::Other("Spm2AnalyzeImage"))));
        match load(&write("x.nii.gz", b"not gzip at all")) {
            Err(LoadError::NotAnImage(m)) => assert!(m.ends_with("x.nii.gz is not a gzip file"), "{m}"),
            other => panic!("{other:?}"),
        }
        match load(&write("x.nii.bz2", b"BZh9")) {
            Err(LoadError::NotAnImage(m)) => assert!(m.contains("only gzip is read"), "{m}"),
            other => panic!("{other:?}"),
        }
        match load(&write("e.nii", b"")) {
            Err(LoadError::NotAnImage(m)) => assert!(m.starts_with("Empty file: '"), "{m}"),
            other => panic!("{other:?}"),
        }
        assert!(matches!(load(&dir.path().join("missing.nii")), Err(LoadError::NotFound)));
    }

    #[test]
    fn s02_2_extensions_are_stepped_over_and_checked() {
        let dir = tempfile::tempdir().unwrap();
        let data = voxels(8);
        let mut h = header(Version::One, &[2, 2, 2]);
        h.vox_offset = VoxOffset::Float(352.0 + 32.0);
        let mut file = h.encode();
        file.extend_from_slice(&[1, 0, 0, 0]);
        file.extend_from_slice(&32i32.to_ne_bytes());
        file.extend_from_slice(&6i32.to_ne_bytes());
        file.extend_from_slice(&[b'x'; 24]);
        file.extend_from_slice(&data);
        let path = dir.path().join("ext.nii");
        std::fs::write(&path, &file).unwrap();
        assert_eq!(nifti(&path).read_voxels().unwrap().bytes, data);
        // An extension longer than the file.
        file[352 + 4..352 + 8].copy_from_slice(&4000i32.to_ne_bytes());
        h.vox_offset = VoxOffset::Float(352.0 + 4096.0);
        file[..348].copy_from_slice(&h.encode());
        std::fs::write(&path, &file).unwrap();
        match load(&path) {
            Err(LoadError::Broken(e)) => assert_eq!(e.message(), "failed to read extension content"),
            other => panic!("{other:?}"),
        }
    }
}
