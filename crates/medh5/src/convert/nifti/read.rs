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
use std::path::Path;

use flate2::read::MultiGzDecoder;

use super::header::{Element, Header, Version, NATIVE_BIG};
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
fn open(name: &str) -> io::Result<Box<dyn Read>> {
    let file = File::open(name)?;
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
fn sniff(name: &str) -> Option<Vec<u8>> {
    let mut stream = open(name).ok()?;
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
    let meta = std::fs::metadata(&name).map_err(|_| LoadError::NotFound)?;
    if meta.len() == 0 {
        return Err(LoadError::NotAnImage(format!("Empty file: '{name}'")));
    }
    if let Some(format) = compression(&name) {
        // 2.1.0 read these through Python's codecs; this reader inflates
        // gzip only.
        return Err(LoadError::NotAnImage(match signature(&name) {
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
        let block = || sniff(&meta_name).unwrap_or_default();
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
                Ok(Loaded::Nifti(Box::new(load_nifti(&meta_name, image_name, Version::One, !candidate.pair())?)))
            }
            Candidate::Nifti2Pair | Candidate::Nifti2Image => {
                Ok(Loaded::Nifti(Box::new(load_nifti(&meta_name, image_name, Version::Two, !candidate.pair())?)))
            }
            other => Ok(Loaded::Other(other.name())),
        };
    }
    Err(LoadError::NotAnImage(signature(&name).unwrap_or_else(|| format!("Cannot work out file type of \"{name}\""))))
}

/// Why a compressed file's first bytes are not what its suffix says, if
/// they are not.
fn signature(name: &str) -> Option<String> {
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
    match File::open(name).and_then(|f| f.take(magic.len() as u64).read_to_end(&mut head)) {
        Err(_) => Some(format!("Could not read file: {name}")),
        Ok(_) if head.starts_with(magic) => None,
        Ok(_) => Some(format!("File {name} is not a {format} file")),
    }
}

/// Read and check a NIfTI header, step over its extensions, and work out
/// where and what its voxels are.
fn load_nifti(meta_name: &str, image_name: String, version: Version, single: bool) -> Result<Image> {
    let mut stream = open(meta_name).map_err(|e| open_error(&e, meta_name))?;
    let block = read_up_to(&mut *stream, version.size() as u64).map_err(gzip_error)?;
    let mut header = Header::decode(&block, version)?;
    header.check_fix()?;
    skip_extensions(&mut *stream, &header, single)?;
    let shape = header.data_shape()?;
    let offset = match version {
        Version::One => data_offset(header.vox_offset)?,
        Version::Two => exact_offset(&block, header.big_endian),
    };
    let slope_inter = header.slope_inter()?;
    Ok(Image { header, image_name, shape, offset, slope_inter })
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

/// A NIfTI-2 `vox_offset`, an integer, read from the header's bytes: the
/// header's `f64` holds every offset exactly only up to 2^53.
fn exact_offset(block: &[u8], big_endian: bool) -> u64 {
    let raw: [u8; 8] = block[168..176].try_into().expect("a NIfTI-2 header holds its offset");
    let value = if big_endian { i64::from_be_bytes(raw) } else { i64::from_le_bytes(raw) };
    // A negative offset reads from the start, as seeking before it does.
    value.max(0) as u64
}

/// Step over the header extensions, checking each is whole: up to
/// `vox_offset` in a single file, to the end of a header file of its own.
fn skip_extensions(stream: &mut dyn Read, header: &Header, single: bool) -> Result<()> {
    let flag = read_up_to(stream, 4).map_err(gzip_error)?;
    if flag.len() < 4 || flag[0] == 0 {
        return Ok(());
    }
    let mut size: f64 = if single { header.vox_offset - (header.version.size() + 4) as f64 } else { -1.0 };
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
        if self.shape.is_empty() {
            return Ok(empty(vec![0]));
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
        let compressed = is_gzip(&self.image_name);
        let file = File::open(&self.image_name).map_err(|e| open_error(&e, &self.image_name))?;
        if n_bytes == 0 {
            // A file mapped from a non-zero offset keeps its declared shape;
            // read, an empty array is `(0,)`.
            let mapped =
                !compressed && self.offset > 0 && file.metadata().map(|m| m.len() >= self.offset).unwrap_or(false);
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

/// numpy's refusal of an array larger than the address space.
fn too_big() -> Error {
    Error::Value("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size.".into())
}

/// Read `wanted` bytes or to the end, reserving no more than `available`
/// (or a modest start, for a stream of unknown length) up front.
fn read_capped(stream: &mut dyn Read, wanted: u64, available: Option<u64>) -> io::Result<Vec<u8>> {
    const START: u64 = 64 << 20;
    let reserve = wanted.min(available.unwrap_or(START)).min(usize::MAX as u64) as usize;
    let mut out = Vec::with_capacity(reserve);
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
        h.vox_offset = if version == Version::One { 352.0 } else { 544.0 };
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
            h.vox_offset = 0.0;
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
            h.vox_offset = 0.0;
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
        // A zero dimension beside a negative one: empty, but no array's shape.
        let h = header(Version::Two, &[0, -3, 2]);
        let path = dir.path().join("negative.nii");
        std::fs::write(&path, single(&h, &[])).unwrap();
        let err = nifti(&path).read_voxels().unwrap_err();
        assert!(matches!(&err, Error::Value(m) if m.starts_with("array is too big")), "{err:?}");
    }

    #[test]
    fn s02_2_an_extension_shorter_than_its_header_is_refused_unread() {
        // A gzip'd header file with an extension of size 4 (< 8), then a
        // stream that inflates to 64 MiB and is cut short: refused for the
        // extension, not for the cut, which only reading on would find.
        let dir = tempfile::tempdir().unwrap();
        let mut h = header(Version::One, &[2, 3, 4]);
        h.magic = *b"ni1\0";
        h.vox_offset = 0.0;
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
        h.vox_offset = 352.0 + 32.0;
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
        h.vox_offset = 352.0 + 4096.0;
        file[..348].copy_from_slice(&h.encode());
        std::fs::write(&path, &file).unwrap();
        match load(&path) {
            Err(LoadError::Broken(e)) => assert_eq!(e.message(), "failed to read extension content"),
            other => panic!("{other:?}"),
        }
    }
}
