//! The NIfTI-1 (348-byte) and NIfTI-2 (540-byte) headers, field by field, in
//! either byte order.
//!
//! The codec is hand-written so that it reads every header exactly as
//! 2.1.0's reader did: the byte order is guessed from `dim[0]` (falling back
//! on `sizeof_hdr`), and the header is checked and repaired by the same
//! battery, with the same wording, refusing what that battery refused
//! ([`Header::check_fix`]).  A float a NIfTI-1 header stores in 32 bits is
//! held here as the `f64` it converts to exactly, so nothing is rounded on
//! the way in, and [`Header::encode`] writes it back to the same bits.

use crate::{Error, Result};

/// The size of a NIfTI-1 header.
pub const NIFTI1_SIZE: usize = 348;
/// The size of a NIfTI-2 header.
pub const NIFTI2_SIZE: usize = 540;
/// Where a single-file NIfTI-1 image's voxels may start, at the earliest.
pub const NIFTI1_SINGLE_OFFSET: f64 = 352.0;
/// Where a single-file NIfTI-2 image's voxels may start, at the earliest.
pub const NIFTI2_SINGLE_OFFSET: f64 = 544.0;
/// NIfTI-2's `eol_check`: what an end-of-line conversion would corrupt.
pub const EOL_CHECK: [u8; 4] = [13, 10, 26, 10];

/// Which header layout.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// NIfTI-1, 348 bytes.
    One,
    /// NIfTI-2, 540 bytes.
    Two,
}

impl Version {
    /// The header's size in bytes.
    pub fn size(self) -> usize {
        match self {
            Version::One => NIFTI1_SIZE,
            Version::Two => NIFTI2_SIZE,
        }
    }

    /// The magic of a header and its voxels in one file.
    pub fn single_magic(self) -> &'static [u8] {
        match self {
            Version::One => b"n+1",
            Version::Two => b"n+2",
        }
    }

    /// The magic of a header in a file of its own.
    pub fn pair_magic(self) -> &'static [u8] {
        match self {
            Version::One => b"ni1",
            Version::Two => b"ni2",
        }
    }

    fn single_offset(self) -> f64 {
        match self {
            Version::One => NIFTI1_SINGLE_OFFSET,
            Version::Two => NIFTI2_SINGLE_OFFSET,
        }
    }
}

/// A voxel's type, as `datatype` names it.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
    /// Two `f32`: real, imaginary.
    C64,
    /// Two `f64`: real, imaginary.
    C128,
    /// Three `u8`: R, G, B.
    Rgb24,
    /// Four `u8`: R, G, B, A.
    Rgba32,
}

/// What a `datatype` code is: an element, a code NIfTI defines for a type
/// with no size here (`none`, `binary`, `all`, the 128- and 256-bit floats),
/// or nothing NIfTI defines.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Datatype {
    Element(Element),
    Unsupported,
    Unknown,
}

impl Element {
    /// The element of a `datatype` code.
    pub fn from_code(code: i64) -> Datatype {
        use Element::*;
        let element = match code {
            2 => U8,
            4 => I16,
            8 => I32,
            16 => F32,
            32 => C64,
            64 => F64,
            128 => Rgb24,
            256 => I8,
            512 => U16,
            768 => U32,
            1024 => I64,
            1280 => U64,
            1792 => C128,
            2304 => Rgba32,
            0 | 1 | 255 | 1536 | 2048 => return Datatype::Unsupported,
            _ => return Datatype::Unknown,
        };
        Datatype::Element(element)
    }

    /// The `datatype` code.
    pub fn code(self) -> i64 {
        use Element::*;
        match self {
            U8 => 2,
            I16 => 4,
            I32 => 8,
            F32 => 16,
            C64 => 32,
            F64 => 64,
            Rgb24 => 128,
            I8 => 256,
            U16 => 512,
            U32 => 768,
            I64 => 1024,
            U64 => 1280,
            C128 => 1792,
            Rgba32 => 2304,
        }
    }

    /// Bytes per voxel.
    pub fn itemsize(self) -> usize {
        use Element::*;
        match self {
            U8 | I8 => 1,
            I16 | U16 => 2,
            I32 | U32 | F32 => 4,
            I64 | U64 | F64 | C64 => 8,
            C128 => 16,
            Rgb24 => 3,
            Rgba32 => 4,
        }
    }

    /// The bytes each byte-order swap reverses: the whole element, a
    /// complex number's halves, or nothing for bytes.
    pub fn swap_unit(self) -> usize {
        use Element::*;
        match self {
            C64 => 4,
            C128 => 8,
            Rgb24 | Rgba32 => 1,
            other => other.itemsize(),
        }
    }

    /// NumPy's name for the type in native byte order (`str(dtype)`).
    pub fn numpy_name(self) -> &'static str {
        use Element::*;
        match self {
            U8 => "uint8",
            I8 => "int8",
            I16 => "int16",
            U16 => "uint16",
            I32 => "int32",
            U32 => "uint32",
            I64 => "int64",
            U64 => "uint64",
            F32 => "float32",
            F64 => "float64",
            C64 => "complex64",
            C128 => "complex128",
            Rgb24 => "[('R', 'u1'), ('G', 'u1'), ('B', 'u1')]",
            Rgba32 => "[('R', 'u1'), ('G', 'u1'), ('B', 'u1'), ('A', 'u1')]",
        }
    }

    /// NumPy's type code without the byte order (`i2`, `f8`, `V3`, ...).
    pub fn numpy_code(self) -> &'static str {
        use Element::*;
        match self {
            U8 => "u1",
            I8 => "i1",
            I16 => "i2",
            U16 => "u2",
            I32 => "i4",
            U32 => "u4",
            I64 => "i8",
            U64 => "u8",
            F32 => "f4",
            F64 => "f8",
            C64 => "c8",
            C128 => "c16",
            Rgb24 => "V3",
            Rgba32 => "V4",
        }
    }

    /// Whether the byte order applies to it at all.
    pub fn has_byte_order(self) -> bool {
        self.swap_unit() > 1
    }
}

/// One header, every field held at the widest type either version stores it
/// in.  Fields only NIfTI-1 has (the Analyze leftovers, `glmax`, `glmin`)
/// are zero, and ignored, in a NIfTI-2 header, as `eol_check` and
/// `unused_str` are in a NIfTI-1 one.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub version: Version,
    /// The byte order the header is stored in.
    pub big_endian: bool,
    pub sizeof_hdr: i64,
    pub data_type: [u8; 10],
    pub db_name: [u8; 18],
    pub extents: i64,
    pub session_error: i64,
    pub regular: u8,
    pub dim_info: u8,
    pub dim: [i64; 8],
    pub intent_p1: f64,
    pub intent_p2: f64,
    pub intent_p3: f64,
    pub intent_code: i64,
    pub datatype: i64,
    pub bitpix: i64,
    pub slice_start: i64,
    pub pixdim: [f64; 8],
    pub vox_offset: f64,
    /// NIfTI-2's `vox_offset` as stored, an integer `vox_offset` rounds past
    /// 2^53; `encode` writes it while `vox_offset` still says the same.
    /// 0 for NIfTI-1, whose offset is a float.
    pub vox_offset_exact: i64,
    pub scl_slope: f64,
    pub scl_inter: f64,
    pub slice_end: i64,
    pub slice_code: i64,
    pub xyzt_units: i64,
    pub cal_max: f64,
    pub cal_min: f64,
    pub slice_duration: f64,
    pub toffset: f64,
    pub glmax: i64,
    pub glmin: i64,
    pub descrip: [u8; 80],
    pub aux_file: [u8; 24],
    pub qform_code: i64,
    pub sform_code: i64,
    pub quatern_b: f64,
    pub quatern_c: f64,
    pub quatern_d: f64,
    pub qoffset_x: f64,
    pub qoffset_y: f64,
    pub qoffset_z: f64,
    pub srow_x: [f64; 4],
    pub srow_y: [f64; 4],
    pub srow_z: [f64; 4],
    pub intent_name: [u8; 16],
    /// NIfTI-1's four bytes or NIfTI-2's first four (the rest is
    /// `eol_check`).
    pub magic: [u8; 4],
    pub eol_check: [u8; 4],
    pub unused_str: [u8; 15],
}

/// Reads the fields of one byte order from a block.
struct Fields<'a> {
    block: &'a [u8],
    big: bool,
}

impl Fields<'_> {
    fn bytes<const N: usize>(&self, at: usize) -> [u8; N] {
        let mut out = [0u8; N];
        out.copy_from_slice(&self.block[at..at + N]);
        out
    }

    fn i16(&self, at: usize) -> i64 {
        let b = self.bytes::<2>(at);
        i64::from(if self.big { i16::from_be_bytes(b) } else { i16::from_le_bytes(b) })
    }

    fn i32(&self, at: usize) -> i64 {
        let b = self.bytes::<4>(at);
        i64::from(if self.big { i32::from_be_bytes(b) } else { i32::from_le_bytes(b) })
    }

    fn i64(&self, at: usize) -> i64 {
        let b = self.bytes::<8>(at);
        if self.big {
            i64::from_be_bytes(b)
        } else {
            i64::from_le_bytes(b)
        }
    }

    fn f32(&self, at: usize) -> f64 {
        let b = self.bytes::<4>(at);
        f64::from(if self.big { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) })
    }

    fn f64(&self, at: usize) -> f64 {
        let b = self.bytes::<8>(at);
        if self.big {
            f64::from_be_bytes(b)
        } else {
            f64::from_le_bytes(b)
        }
    }

    fn i16s<const N: usize>(&self, at: usize) -> [i64; N] {
        std::array::from_fn(|k| self.i16(at + 2 * k))
    }

    fn i64s<const N: usize>(&self, at: usize) -> [i64; N] {
        std::array::from_fn(|k| self.i64(at + 8 * k))
    }

    fn f32s<const N: usize>(&self, at: usize) -> [f64; N] {
        std::array::from_fn(|k| self.f32(at + 4 * k))
    }

    fn f64s<const N: usize>(&self, at: usize) -> [f64; N] {
        std::array::from_fn(|k| self.f64(at + 8 * k))
    }
}

/// Writes the fields of one byte order into a block.
struct Writer {
    block: Vec<u8>,
    big: bool,
}

impl Writer {
    fn put(&mut self, at: usize, bytes: &[u8]) {
        self.block[at..at + bytes.len()].copy_from_slice(bytes);
    }

    fn i16(&mut self, at: usize, v: i64) {
        let v = v as i16;
        self.put(at, &if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn i32(&mut self, at: usize, v: i64) {
        let v = v as i32;
        self.put(at, &if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn i64(&mut self, at: usize, v: i64) {
        self.put(at, &if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn f32(&mut self, at: usize, v: f64) {
        let v = v as f32;
        self.put(at, &if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }

    fn f64(&mut self, at: usize, v: f64) {
        self.put(at, &if self.big { v.to_be_bytes() } else { v.to_le_bytes() });
    }
}

/// Whether this machine stores numbers big-end first.
pub const NATIVE_BIG: bool = cfg!(target_endian = "big");

/// `dim[0]` and `sizeof_hdr` read in one byte order.
fn probe(block: &[u8], version: Version, big: bool) -> (i64, i64) {
    let f = Fields { block, big };
    match version {
        Version::One => (f.i16(40), f.i32(0)),
        Version::Two => (f.i64(16), f.i32(0)),
    }
}

impl Header {
    /// A header of `version` in this machine's byte order, every field as
    /// a freshly built one has it: one voxel, unit voxel sizes, `float32`.
    pub fn new(version: Version) -> Header {
        let mut dim = [1i64; 8];
        dim[0] = 0;
        let mut pixdim = [1.0; 8];
        pixdim[0] = 1.0;
        let mut magic = [0u8; 4];
        magic[..3].copy_from_slice(version.single_magic());
        Header {
            version,
            big_endian: NATIVE_BIG,
            sizeof_hdr: version.size() as i64,
            data_type: [0; 10],
            db_name: [0; 18],
            extents: 0,
            session_error: 0,
            regular: 0,
            dim_info: 0,
            dim,
            intent_p1: 0.0,
            intent_p2: 0.0,
            intent_p3: 0.0,
            intent_code: 0,
            datatype: 16,
            bitpix: 32,
            slice_start: 0,
            pixdim,
            vox_offset: 0.0,
            vox_offset_exact: 0,
            scl_slope: f64::NAN,
            scl_inter: f64::NAN,
            slice_end: 0,
            slice_code: 0,
            xyzt_units: 0,
            cal_max: 0.0,
            cal_min: 0.0,
            slice_duration: 0.0,
            toffset: 0.0,
            glmax: 0,
            glmin: 0,
            descrip: [0; 80],
            aux_file: [0; 24],
            qform_code: 0,
            sform_code: 0,
            quatern_b: 0.0,
            quatern_c: 0.0,
            quatern_d: 0.0,
            qoffset_x: 0.0,
            qoffset_y: 0.0,
            qoffset_z: 0.0,
            srow_x: [0.0; 4],
            srow_y: [0.0; 4],
            srow_z: [0.0; 4],
            intent_name: [0; 16],
            magic,
            eol_check: if version == Version::Two { EOL_CHECK } else { [0; 4] },
            unused_str: [0; 15],
        }
    }

    /// The byte order a block is stored in, guessed as 2.1.0 guessed it:
    /// `dim[0]` from 1 to 7 is this machine's order, 0 is settled by
    /// `sizeof_hdr`, and anything else is the other order.
    fn guess_big_endian(block: &[u8], version: Version) -> bool {
        let (dim0, _) = probe(block, version, NATIVE_BIG);
        match dim0 {
            0 => {
                let (_, swapped) = probe(block, version, !NATIVE_BIG);
                if swapped == version.size() as i64 {
                    !NATIVE_BIG
                } else {
                    NATIVE_BIG
                }
            }
            1..=7 => NATIVE_BIG,
            _ => !NATIVE_BIG,
        }
    }

    /// Decode `block`, exactly `version.size()` bytes, in the byte order
    /// 2.1.0 would have guessed for it (a block of another size is refused,
    /// not probed).  Nothing is
    /// checked: see [`check_fix`](Self::check_fix).
    pub fn decode(block: &[u8], version: Version) -> Result<Header> {
        if block.len() != version.size() {
            return Err(Error::Value("Binary block is wrong size".into()));
        }
        let big = Header::guess_big_endian(block, version);
        Ok(Header::decode_as(block, version, big))
    }

    /// Decode `block` in the given byte order.
    fn decode_as(block: &[u8], version: Version, big: bool) -> Header {
        let f = Fields { block, big };
        let mut h = Header::new(version);
        h.big_endian = big;
        h.sizeof_hdr = f.i32(0);
        match version {
            Version::One => {
                h.data_type = f.bytes(4);
                h.db_name = f.bytes(14);
                h.extents = f.i32(32);
                h.session_error = f.i16(36);
                h.regular = block[38];
                h.dim_info = block[39];
                h.dim = f.i16s(40);
                h.intent_p1 = f.f32(56);
                h.intent_p2 = f.f32(60);
                h.intent_p3 = f.f32(64);
                h.intent_code = f.i16(68);
                h.datatype = f.i16(70);
                h.bitpix = f.i16(72);
                h.slice_start = f.i16(74);
                h.pixdim = f.f32s(76);
                h.vox_offset = f.f32(108);
                h.scl_slope = f.f32(112);
                h.scl_inter = f.f32(116);
                h.slice_end = f.i16(120);
                h.slice_code = i64::from(block[122]);
                h.xyzt_units = i64::from(block[123]);
                h.cal_max = f.f32(124);
                h.cal_min = f.f32(128);
                h.slice_duration = f.f32(132);
                h.toffset = f.f32(136);
                h.glmax = f.i32(140);
                h.glmin = f.i32(144);
                h.descrip = f.bytes(148);
                h.aux_file = f.bytes(228);
                h.qform_code = f.i16(252);
                h.sform_code = f.i16(254);
                h.quatern_b = f.f32(256);
                h.quatern_c = f.f32(260);
                h.quatern_d = f.f32(264);
                h.qoffset_x = f.f32(268);
                h.qoffset_y = f.f32(272);
                h.qoffset_z = f.f32(276);
                h.srow_x = f.f32s(280);
                h.srow_y = f.f32s(296);
                h.srow_z = f.f32s(312);
                h.intent_name = f.bytes(328);
                h.magic = f.bytes(344);
                h.eol_check = [0; 4];
            }
            Version::Two => {
                h.magic = f.bytes(4);
                h.eol_check = f.bytes(8);
                h.datatype = f.i16(12);
                h.bitpix = f.i16(14);
                h.dim = f.i64s(16);
                h.intent_p1 = f.f64(80);
                h.intent_p2 = f.f64(88);
                h.intent_p3 = f.f64(96);
                h.pixdim = f.f64s(104);
                h.vox_offset_exact = f.i64(168);
                h.vox_offset = h.vox_offset_exact as f64;
                h.scl_slope = f.f64(176);
                h.scl_inter = f.f64(184);
                h.cal_max = f.f64(192);
                h.cal_min = f.f64(200);
                h.slice_duration = f.f64(208);
                h.toffset = f.f64(216);
                h.slice_start = f.i64(224);
                h.slice_end = f.i64(232);
                h.descrip = f.bytes(240);
                h.aux_file = f.bytes(320);
                h.qform_code = f.i32(344);
                h.sform_code = f.i32(348);
                h.quatern_b = f.f64(352);
                h.quatern_c = f.f64(360);
                h.quatern_d = f.f64(368);
                h.qoffset_x = f.f64(376);
                h.qoffset_y = f.f64(384);
                h.qoffset_z = f.f64(392);
                h.srow_x = f.f64s(400);
                h.srow_y = f.f64s(432);
                h.srow_z = f.f64s(464);
                h.slice_code = f.i32(496);
                h.xyzt_units = f.i32(500);
                h.intent_code = f.i32(504);
                h.intent_name = f.bytes(508);
                h.dim_info = block[524];
                h.unused_str = f.bytes(525);
                h.data_type = [0; 10];
                h.db_name = [0; 18];
                h.extents = 0;
                h.session_error = 0;
                h.regular = 0;
                h.glmax = 0;
                h.glmin = 0;
            }
        }
        h
    }

    /// The header as stored: `version.size()` bytes in its byte order.
    /// Each field is cast to its stored type, as NumPy assigns it.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer { block: vec![0u8; self.version.size()], big: self.big_endian };
        w.i32(0, self.sizeof_hdr);
        match self.version {
            Version::One => {
                w.put(4, &self.data_type);
                w.put(14, &self.db_name);
                w.i32(32, self.extents);
                w.i16(36, self.session_error);
                w.block[38] = self.regular;
                w.block[39] = self.dim_info;
                for (k, v) in self.dim.iter().enumerate() {
                    w.i16(40 + 2 * k, *v);
                }
                w.f32(56, self.intent_p1);
                w.f32(60, self.intent_p2);
                w.f32(64, self.intent_p3);
                w.i16(68, self.intent_code);
                w.i16(70, self.datatype);
                w.i16(72, self.bitpix);
                w.i16(74, self.slice_start);
                for (k, v) in self.pixdim.iter().enumerate() {
                    w.f32(76 + 4 * k, *v);
                }
                w.f32(108, self.vox_offset);
                w.f32(112, self.scl_slope);
                w.f32(116, self.scl_inter);
                w.i16(120, self.slice_end);
                w.block[122] = self.slice_code as u8;
                w.block[123] = self.xyzt_units as u8;
                w.f32(124, self.cal_max);
                w.f32(128, self.cal_min);
                w.f32(132, self.slice_duration);
                w.f32(136, self.toffset);
                w.i32(140, self.glmax);
                w.i32(144, self.glmin);
                w.put(148, &self.descrip);
                w.put(228, &self.aux_file);
                w.i16(252, self.qform_code);
                w.i16(254, self.sform_code);
                w.f32(256, self.quatern_b);
                w.f32(260, self.quatern_c);
                w.f32(264, self.quatern_d);
                w.f32(268, self.qoffset_x);
                w.f32(272, self.qoffset_y);
                w.f32(276, self.qoffset_z);
                for (row, at) in [(&self.srow_x, 280), (&self.srow_y, 296), (&self.srow_z, 312)] {
                    for (k, v) in row.iter().enumerate() {
                        w.f32(at + 4 * k, *v);
                    }
                }
                w.put(328, &self.intent_name);
                w.put(344, &self.magic);
            }
            Version::Two => {
                w.put(4, &self.magic);
                w.put(8, &self.eol_check);
                w.i16(12, self.datatype);
                w.i16(14, self.bitpix);
                for (k, v) in self.dim.iter().enumerate() {
                    w.i64(16 + 8 * k, *v);
                }
                w.f64(80, self.intent_p1);
                w.f64(88, self.intent_p2);
                w.f64(96, self.intent_p3);
                for (k, v) in self.pixdim.iter().enumerate() {
                    w.f64(104 + 8 * k, *v);
                }
                w.i64(168, self.exact_offset());
                w.f64(176, self.scl_slope);
                w.f64(184, self.scl_inter);
                w.f64(192, self.cal_max);
                w.f64(200, self.cal_min);
                w.f64(208, self.slice_duration);
                w.f64(216, self.toffset);
                w.i64(224, self.slice_start);
                w.i64(232, self.slice_end);
                w.put(240, &self.descrip);
                w.put(320, &self.aux_file);
                w.i32(344, self.qform_code);
                w.i32(348, self.sform_code);
                w.f64(352, self.quatern_b);
                w.f64(360, self.quatern_c);
                w.f64(368, self.quatern_d);
                w.f64(376, self.qoffset_x);
                w.f64(384, self.qoffset_y);
                w.f64(392, self.qoffset_z);
                for (row, at) in [(&self.srow_x, 400), (&self.srow_y, 432), (&self.srow_z, 464)] {
                    for (k, v) in row.iter().enumerate() {
                        w.f64(at + 8 * k, *v);
                    }
                }
                w.i32(496, self.slice_code);
                w.i32(500, self.xyzt_units);
                w.i32(504, self.intent_code);
                w.put(508, &self.intent_name);
                w.block[524] = self.dim_info;
                w.put(525, &self.unused_str);
            }
        }
        w.block
    }

    /// The magic as NumPy reads a byte string: trailing NULs dropped.
    pub fn magic_str(&self) -> &[u8] {
        strip_nuls(&self.magic)
    }

    /// Check the header and repair what can be repaired, as 2.1.0's reader
    /// did on every load: a wrong `sizeof_hdr` or `bitpix`, a zero or
    /// negative voxel size, a `qfac` other than ±1, an unknown xform code and
    /// an all-zero `eol_check` are fixed; an unknown or unsized `datatype`,
    /// an unknown magic, a single file's voxels inside its header and a
    /// corrupted `eol_check` are refused, the first in this order.
    pub fn check_fix(&mut self) -> Result<()> {
        let mut problems: Vec<String> = Vec::new();
        // sizeof_hdr: level 30, fixed.
        self.sizeof_hdr = self.version.size() as i64;
        // datatype: level 40.
        let datatype = Element::from_code(self.datatype);
        match datatype {
            Datatype::Unknown => problems.push(format!("data code {} not recognized", self.datatype)),
            Datatype::Unsupported => problems.push(format!("data code {} not supported", self.datatype)),
            Datatype::Element(_) => {}
        }
        // bitpix: level 10, fixed when the datatype is known.
        match datatype {
            Datatype::Element(e) => self.bitpix = (e.itemsize() * 8) as i64,
            Datatype::Unsupported => self.bitpix = 0,
            Datatype::Unknown => {}
        }
        // pixdim[1..4]: zero becomes 1 (level 30), negative its absolute
        // value (35).
        for v in &mut self.pixdim[1..4] {
            if *v == 0.0 {
                *v = 1.0;
            }
        }
        if self.pixdim[1..4].iter().any(|v| *v < 0.0) {
            for v in &mut self.pixdim[1..4] {
                *v = v.abs();
            }
        }
        // qfac: level 20, set to 1.
        if !(self.pixdim[0] == 1.0 || self.pixdim[0] == -1.0) {
            self.pixdim[0] = 1.0;
        }
        // magic: level 45.
        let magic = self.magic_str().to_vec();
        if magic != self.version.pair_magic() && magic != self.version.single_magic() {
            problems.push(format!("magic string {} is not valid", crate::json::repr_str(&latin1(&magic))));
        }
        // vox_offset: level 40 for a single file's voxels inside its header.
        if self.vox_offset != 0.0
            && magic == self.version.single_magic()
            && self.vox_offset < self.version.single_offset()
        {
            problems.push(format!("vox offset {} too low for single file nifti1", self.vox_offset.trunc() as i64));
        }
        // qform_code and sform_code: level 30, set to 0.
        for code in [&mut self.qform_code, &mut self.sform_code] {
            if !(0..=5).contains(code) {
                *code = 0;
            }
        }
        // eol_check (NIfTI-2): all zero is fixed (20); anything else but the
        // right bytes is refused (40).
        if self.version == Version::Two && self.eol_check != EOL_CHECK {
            if self.eol_check != [0; 4] {
                problems.push("EOL check not 0 or 13, 10, 26, 10; data may be corrupted by EOL conversion".into());
            }
            self.eol_check = EOL_CHECK;
        }
        match problems.into_iter().next() {
            Some(first) => Err(Error::Value(first)),
            None => Ok(()),
        }
    }

    /// NIfTI-2's integer `vox_offset`: the stored one, unless `vox_offset`
    /// has since been set to another value.
    pub fn exact_offset(&self) -> i64 {
        if self.vox_offset_exact as f64 == self.vox_offset {
            self.vox_offset_exact
        } else {
            self.vox_offset as i64
        }
    }

    /// The element type, if the `datatype` names one this reads.
    pub fn element(&self) -> Option<Element> {
        match Element::from_code(self.datatype) {
            Datatype::Element(e) => Some(e),
            _ => None,
        }
    }

    /// `dim[1..=dim[0]]`, sliced as Python slices: `dim[0]` past 7 stops at
    /// the last, a negative one counts from the end, and 0 is the shape
    /// `(0,)`.
    pub fn raw_shape(&self) -> Vec<i64> {
        if self.dim[0] == 0 {
            return vec![0];
        }
        py_slice(&self.dim, 1, self.ndim_stop()).to_vec()
    }

    /// The voxel array's shape, with the FreeSurfer conventions for long
    /// vectors (`(-1, 1, 1)` and `glmin`) and the order-7 icosahedron.
    pub fn data_shape(&self) -> Result<Vec<i64>> {
        let mut shape = self.raw_shape();
        if shape.len() >= 3 && shape[..3] == [-1, 1, 1] {
            if self.version == Version::Two {
                return Err(Error::Value("no field of name glmin".into()));
            }
            if self.glmin == 0 {
                return Err(Error::Value("-1 in dim[1] but 0 in glmin; inconsistent freesurfer type header?".into()));
            }
            shape[0] = self.glmin;
        } else if shape.len() >= 3 && shape[..3] == [27307, 1, 6] {
            shape[..3].copy_from_slice(&[163842, 1, 1]);
        }
        Ok(shape)
    }

    /// `pixdim[1..=dim[0]]`, or `(1.0,)` for `dim[0] = 0`.
    pub fn zooms(&self) -> Vec<f64> {
        if self.dim[0] == 0 {
            return vec![1.0];
        }
        py_slice(&self.pixdim, 1, self.ndim_stop()).to_vec()
    }

    /// `dim[0] + 1`, the end of the slices above. A NIfTI-2 `dim[0]` is 64
    /// bits, so `i64::MAX` saturates rather than overflow; any end past 8
    /// slices to the last element, as Python's unbounded integers do.
    fn ndim_stop(&self) -> i64 {
        self.dim[0].saturating_add(1)
    }

    /// `scl_slope` and `scl_inter` as a scale, or `None` where the slope says
    /// "unscaled" (0 or not finite).
    pub fn slope_inter(&self) -> Result<Option<(f64, f64)>> {
        let (slope, inter) = (self.scl_slope, self.scl_inter);
        if slope == 0.0 || !slope.is_finite() {
            return Ok(None);
        }
        if !inter.is_finite() {
            return Err(Error::Value(format!("Valid slope but invalid intercept {}", crate::json::py_float(inter))));
        }
        Ok(Some((slope, inter)))
    }

    /// The spatial and temporal unit names, or `None` where either code is
    /// one NIfTI does not define.
    pub fn xyzt_units(&self) -> Option<(&'static str, &'static str)> {
        let xyz = self.xyzt_units.rem_euclid(8);
        let t = self.xyzt_units - xyz;
        Some((unit_label(xyz)?, unit_label(t)?))
    }

    /// The sform as a 4x4 matrix, row-major.
    pub fn sform(&self) -> [[f64; 4]; 4] {
        [self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]]
    }

    /// The affine NIfTI readers fall back on without a qform or sform: the
    /// voxel sizes on the diagonal, x flipped, and the volume's centre at the
    /// origin.
    pub fn base_affine(&self) -> [[f64; 4]; 4] {
        let dims = py_slice(&self.dim, 1, self.ndim_stop());
        let pixdims = py_slice(&self.pixdim, 1, self.ndim_stop());
        let (mut shape, mut zooms) = ([1.0f64; 3], [1.0f64; 3]);
        for k in 0..dims.len().min(3) {
            shape[k] = dims[k] as f64;
            zooms[k] = pixdims[k];
        }
        zooms[0] *= -1.0;
        let mut out = identity();
        for k in 0..3 {
            out[k][k] = zooms[k];
            let origin = (shape[k] - 1.0) / 2.0;
            out[k][3] = -origin * zooms[k];
        }
        out
    }

    /// The quaternion's `(w, x, y, z)` as the reader building the qform at
    /// load fills `w` in: zero within three float epsilons of the stored
    /// precision, refused below it.
    pub fn qform_quaternion(&self) -> Result<[f64; 4]> {
        let (b, c, d) = (self.quatern_b, self.quatern_c, self.quatern_d);
        let threshold = match self.version {
            Version::One => f64::from(f32::EPSILON) * 3.0,
            Version::Two => f64::EPSILON * 3.0,
        };
        let w2 = 1.0 - (b * b + c * c + d * d);
        let w = if w2.abs() < threshold {
            0.0
        } else if w2 < 0.0 {
            return Err(Error::Value(format!("w2 should be positive, but is {}", format_e(w2))));
        } else {
            w2.sqrt()
        };
        Ok([w, b, c, d])
    }

    /// The text of a byte-string field: up to its first NUL, as UTF-8 with
    /// replacement.
    pub fn text(field: &[u8]) -> String {
        let end = field.iter().position(|b| *b == 0).unwrap_or(field.len());
        String::from_utf8_lossy(&field[..end]).into_owned()
    }
}

/// NumPy's view of a fixed-length byte string: trailing NULs dropped.
pub fn strip_nuls(bytes: &[u8]) -> &[u8] {
    let end = bytes.iter().rposition(|b| *b != 0).map_or(0, |p| p + 1);
    &bytes[..end]
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|b| char::from(*b)).collect()
}

fn unit_label(code: i64) -> Option<&'static str> {
    Some(match code {
        0 => "unknown",
        1 => "meter",
        2 => "mm",
        3 => "micron",
        8 => "sec",
        16 => "msec",
        24 => "usec",
        32 => "hz",
        40 => "ppm",
        48 => "rads",
        _ => return None,
    })
}

/// `values[start:stop]` as Python slices it.
pub(crate) fn py_slice<T>(values: &[T], start: i64, stop: i64) -> &[T] {
    let n = values.len() as i64;
    let clamp = |v: i64| -> usize {
        let v = if v < 0 { (v + n).max(0) } else { v.min(n) };
        v as usize
    };
    let (a, b) = (clamp(start), clamp(stop));
    if a >= b {
        &values[0..0]
    } else {
        &values[a..b]
    }
}

pub(crate) fn identity() -> [[f64; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (k, row) in out.iter_mut().enumerate() {
        row[k] = 1.0;
    }
    out
}

/// `format(value, "e")`: six decimals and a two-digit exponent at least.
pub(crate) fn format_e(value: f64) -> String {
    if !value.is_finite() {
        return crate::json::py_float(value);
    }
    let text = format!("{value:.6e}");
    let (mantissa, exponent) = text.split_once('e').expect("an exponent");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    format!("{mantissa}e{}{:02}", if exponent < 0 { '-' } else { '+' }, exponent.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(version: Version, big: bool) -> Header {
        let mut h = Header::new(version);
        h.big_endian = big;
        h.dim = [3, 4, 3, 2, 1, 1, 1, 1];
        h.datatype = 4;
        h.bitpix = 16;
        h.pixdim = [1.0, 0.75, 0.875, 2.0, 1.0, 1.0, 1.0, 1.0];
        h.vox_offset = version.single_offset();
        if version == Version::Two {
            h.vox_offset_exact = h.vox_offset as i64;
        }
        h.scl_slope = 2.0;
        h.scl_inter = -1024.0;
        h.xyzt_units = 10;
        h.descrip[..6].copy_from_slice(b"parity");
        h.sform_code = 1;
        h.srow_x = [0.75, 0.0, 0.0, -10.0];
        h.srow_y = [0.0, 0.875, 0.0, -20.0];
        h.srow_z = [0.0, 0.0, 2.0, 5.0];
        h.glmax = if version == Version::One { 7 } else { 0 };
        h
    }

    #[test]
    fn s02_2_headers_round_trip_in_both_versions_and_byte_orders() {
        for version in [Version::One, Version::Two] {
            for big in [false, true] {
                let h = sample(version, big);
                let block = h.encode();
                assert_eq!(block.len(), version.size());
                let back = Header::decode(&block, version).unwrap();
                assert_eq!(back.big_endian, big, "{version:?}");
                // NaN is not equal to itself; this header has none.
                assert_eq!(back, h, "{version:?} big={big}");
                assert_eq!(back.encode(), block);
            }
        }
    }

    #[test]
    fn s02_2_a_nifti2_offset_survives_decode_and_encode() {
        // 2^53 + 1 has no f64: kept as the integer it is, and written back.
        let mut h = sample(Version::Two, false);
        h.vox_offset_exact = (1 << 53) + 1;
        h.vox_offset = h.vox_offset_exact as f64;
        let block = h.encode();
        let back = Header::decode(&block, Version::Two).unwrap();
        assert_eq!(back.exact_offset(), (1 << 53) + 1);
        assert_eq!(back.encode(), block);
        // Set to another value, the float is what is meant.
        let mut moved = back.clone();
        moved.vox_offset = 1024.0;
        assert_eq!(Header::decode(&moved.encode(), Version::Two).unwrap().exact_offset(), 1024);
    }

    #[test]
    fn s02_2_the_byte_order_is_guessed_from_dim0_then_sizeof_hdr() {
        let mut h = sample(Version::One, !NATIVE_BIG);
        h.dim[0] = 0;
        let block = h.encode();
        assert_eq!(Header::guess_big_endian(&block, Version::One), !NATIVE_BIG);
        h.sizeof_hdr = 0;
        assert_eq!(Header::guess_big_endian(&h.encode(), Version::One), NATIVE_BIG);
        h.dim[0] = 9;
        h.big_endian = NATIVE_BIG;
        assert_eq!(Header::guess_big_endian(&h.encode(), Version::One), !NATIVE_BIG);
    }

    #[test]
    fn s02_2_check_fix_repairs_and_refuses_as_2_1_0_did() {
        let mut h = sample(Version::One, false);
        h.sizeof_hdr = 0;
        h.bitpix = 3;
        h.pixdim[0] = 0.0;
        h.pixdim[1] = -0.8;
        h.pixdim[2] = 0.0;
        h.qform_code = 9;
        h.check_fix().unwrap();
        assert_eq!((h.sizeof_hdr, h.bitpix, h.qform_code), (348, 16, 0));
        assert_eq!(&h.pixdim[..3], &[1.0, 0.8, 1.0]);

        let mut h = sample(Version::One, false);
        h.datatype = 77;
        h.magic = *b"abc\0";
        assert_eq!(h.check_fix().unwrap_err().message(), "data code 77 not recognized");
        let mut h = sample(Version::One, false);
        h.datatype = 1536;
        assert_eq!(h.check_fix().unwrap_err().message(), "data code 1536 not supported");
        let mut h = sample(Version::One, false);
        h.magic = *b"ab\0\0";
        assert_eq!(h.check_fix().unwrap_err().message(), "magic string 'ab' is not valid");
        let mut h = sample(Version::One, false);
        h.vox_offset = 100.0;
        assert_eq!(h.check_fix().unwrap_err().message(), "vox offset 100 too low for single file nifti1");
        let mut h = sample(Version::Two, false);
        h.eol_check = [0; 4];
        h.check_fix().unwrap();
        assert_eq!(h.eol_check, EOL_CHECK);
        h.eol_check = [13, 13, 26, 10];
        assert!(h.check_fix().unwrap_err().message().starts_with("EOL check not 0"));
    }

    #[test]
    fn s02_2_shapes_slopes_units_and_the_quaternion() {
        let mut h = sample(Version::One, false);
        assert_eq!(h.data_shape().unwrap(), vec![4, 3, 2]);
        h.dim[0] = 0;
        assert_eq!(h.raw_shape(), vec![0]);
        h.dim[0] = i64::MAX;
        assert_eq!(h.raw_shape(), h.dim[1..].to_vec());
        assert_eq!(h.zooms(), h.pixdim[1..].to_vec());
        assert_eq!(h.base_affine()[0][0], -h.pixdim[1]);
        h.dim = [3, -1, 1, 1, 1, 1, 1, 1];
        h.glmin = 5;
        assert_eq!(h.data_shape().unwrap(), vec![5, 1, 1]);
        h.scl_inter = f64::INFINITY;
        assert_eq!(h.slope_inter().unwrap_err().message(), "Valid slope but invalid intercept inf");
        h.scl_slope = f64::NAN;
        assert_eq!(h.slope_inter().unwrap(), None);
        assert_eq!(h.xyzt_units(), Some(("mm", "sec")));
        h.xyzt_units = 7;
        assert_eq!(h.xyzt_units(), None);
        h.quatern_b = 0.6;
        h.quatern_c = 0.6;
        h.quatern_d = 0.6;
        assert_eq!(h.qform_quaternion().unwrap_err().message(), "w2 should be positive, but is -8.000000e-02");
        assert_eq!(format_e(-0.9200001), "-9.200001e-01");
    }
}
