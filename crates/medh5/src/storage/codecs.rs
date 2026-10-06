//! Codec profiles (spec §14.2).
//!
//! A file's datasets need not share a codec.  Four named profiles cover the
//! real trade space:
//!
//! | Profile | Codec | Intended use |
//! |---|---|---|
//! | `training` | Blosc2 lz4 L1 | hot dataloader path |
//! | `balanced` | Blosc2 zstd L3 | general use (default) |
//! | `archive` | Blosc2 zstd L9 | cold storage and distribution |
//! | `portable` | gzip L4 | readers without the Blosc2 plugin |
//!
//! `portable` exists because Blosc2 needs a plugin on the *reader*; a
//! `portable` file opens in stock h5py, MATLAB, R and `h5dump`.

use hdf5::filters::Filter;

use crate::h5::data::Layout;
use crate::json::{repr_list, repr_str};
use crate::{Error, Result};

/// Below this raw size a dataset is stored contiguous and uncompressed.
pub const COMPRESS_MIN_BYTES: usize = 64 * 1024;
/// At or above this raw size a dataset is 'bulk' and W902 applies.
pub const BULK_MIN_BYTES: usize = 1024 * 1024;
/// The HDF Group's id for the Blosc2 filter.
pub const BLOSC2_FILTER_ID: i32 = 32026;
/// The HDF Group's id for the Blosc (v1) filter.
pub const BLOSC_FILTER_ID: i32 = 32001;
/// HDF5's own filters: deflate, shuffle, fletcher32, szip, n-bit, scale-offset.
pub const BUILTIN_FILTER_IDS: [i32; 6] = [1, 2, 3, 4, 5, 6];
/// The default profile.
pub const DEFAULT_PROFILE: &str = "balanced";

/// What a dataset holds, which decides its codec within a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Image,
    Label,
    Aux,
}

/// One codec setting.
#[derive(Debug, Clone, PartialEq)]
pub struct Codec {
    pub name: &'static str,
    /// `(cname, clevel, shuffle)` for Blosc2.
    pub blosc2: Option<(&'static str, u32, &'static str)>,
    pub gzip_level: Option<u8>,
    pub shuffle: bool,
}

impl Codec {
    /// The HDF5 filter pipeline, in h5py's order (shuffle before compression).
    pub fn filters(&self) -> Vec<Filter> {
        if let Some((cname, clevel, shuffle)) = self.blosc2 {
            let compcode = match cname {
                "blosclz" => 0,
                "lz4" => 1,
                "lz4hc" => 2,
                "zlib" => 4,
                "zstd" => 5,
                _ => 0,
            };
            let mode = match shuffle {
                "shuffle" => 1,
                "bitshuffle" => 2,
                _ => 0,
            };
            return vec![Filter::user(BLOSC2_FILTER_ID as _, &[0, 0, 0, 0, clevel, mode, compcode])];
        }
        if let Some(level) = self.gzip_level {
            let mut out = Vec::new();
            if self.shuffle {
                out.push(Filter::shuffle());
            }
            out.push(Filter::deflate(level));
            return out;
        }
        Vec::new()
    }
}

/// A named pairing of codecs for image data and for label/field data.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecProfile {
    pub name: &'static str,
    pub image: Codec,
    pub label: Codec,
    pub description: &'static str,
}

impl CodecProfile {
    /// The codec for a role: images get `image`, everything else `label`.
    pub fn codec(&self, role: Role) -> &Codec {
        if role == Role::Image {
            &self.image
        } else {
            &self.label
        }
    }
}

const fn blosc2(name: &'static str, cname: &'static str, clevel: u32, shuffle: &'static str) -> Codec {
    Codec { name, blosc2: Some((cname, clevel, shuffle)), gzip_level: None, shuffle: true }
}

const fn gzip(level: u8) -> Codec {
    Codec { name: "gzip:4", blosc2: None, gzip_level: Some(level), shuffle: true }
}

/// The four profiles, in name order.
pub fn profiles() -> Vec<CodecProfile> {
    vec![
        CodecProfile {
            name: "archive",
            image: blosc2("blosc2:zstd:9:bitshuffle", "zstd", 9, "bitshuffle"),
            label: blosc2("blosc2:zstd:9:bitshuffle", "zstd", 9, "bitshuffle"),
            description: "smallest on disk; cold storage and distribution",
        },
        CodecProfile {
            name: "balanced",
            image: blosc2("blosc2:zstd:3:shuffle", "zstd", 3, "shuffle"),
            label: blosc2("blosc2:zstd:3:bitshuffle", "zstd", 3, "bitshuffle"),
            description: "general use; the default",
        },
        CodecProfile { name: "portable", image: gzip(4), label: gzip(4), description: "readable without hdf5plugin" },
        CodecProfile {
            name: "training",
            image: blosc2("blosc2:lz4:1:shuffle", "lz4", 1, "shuffle"),
            label: blosc2("blosc2:lz4:1:shuffle", "lz4", 1, "shuffle"),
            description: "fastest decompression; hot dataloader path",
        },
    ]
}

/// Profile names, sorted.
pub fn profile_names() -> Vec<&'static str> {
    profiles().iter().map(|p| p.name).collect()
}

/// Resolve a profile name.  `None` gives the default (`balanced`).
pub fn resolve_profile(name: Option<&str>) -> Result<CodecProfile> {
    let wanted = name.unwrap_or(DEFAULT_PROFILE);
    profiles().into_iter().find(|p| p.name == wanted).ok_or_else(|| {
        Error::invalid(format!(
            "unknown codec profile {}; expected one of {}",
            repr_str(wanted),
            profile_names().join(", ")
        ))
    })
}

/// The layout one dataset gets under a profile.
///
/// Datasets below [`COMPRESS_MIN_BYTES`] (and empty ones) are stored
/// contiguous: chunking and a filter pipeline cost more than they save at that
/// size.  `chunks = None` asks for h5py's guessed chunk shape.
pub fn dataset_layout(
    shape: &[usize],
    itemsize: usize,
    profile: &CodecProfile,
    role: Role,
    chunks: Option<Vec<usize>>,
) -> Layout {
    let nbytes: usize = if shape.is_empty() { 0 } else { shape.iter().product::<usize>() * itemsize };
    if nbytes < COMPRESS_MIN_BYTES || shape.contains(&0) {
        return Layout::contiguous();
    }
    let chunks = chunks.unwrap_or_else(|| super::chunking::guess_chunk(shape, itemsize));
    Layout { chunks: Some(chunks), filters: profile.codec(role).filters() }
}

const BLOSC_CNAMES: [&str; 6] = ["blosclz", "lz4", "lz4hc", "snappy", "zlib", "zstd"];
const BLOSC_SHUFFLE: [&str; 3] = ["noshuffle", "shuffle", "bitshuffle"];

fn describe_blosc(values: &[u32]) -> String {
    if values.len() < 7 {
        return "blosc2".into();
    }
    let (clevel, shuffle, cname) = (values[4], values[5], values[6]);
    let cname = BLOSC_CNAMES.get(cname as usize).map(|s| s.to_string()).unwrap_or_else(|| cname.to_string());
    let shuffle = BLOSC_SHUFFLE.get(shuffle as usize).map(|s| s.to_string()).unwrap_or_else(|| shuffle.to_string());
    format!("blosc2:{cname}:{clevel}+{shuffle}")
}

/// Describe a dataset's actual HDF5 filter pipeline, for `medh5 info`.
///
/// The codec used is discoverable from the file itself (spec §14.2); nothing
/// records the profile name, because a profile is a writer convenience and a
/// file may mix codecs.
pub fn describe_filters(ds: &hdf5::Dataset) -> Result<String> {
    if crate::h5::data::chunks(ds).is_none() {
        return Ok("contiguous".into());
    }
    let pipeline = crate::h5::data::filters(ds)?;
    let named_compression = pipeline.iter().find_map(|(id, values)| match id {
        1 => Some(format!("gzip:{}", values.first().copied().unwrap_or(0))),
        32000 => Some("lzf".to_string()),
        4 => Some("szip".to_string()),
        _ => None,
    });
    let has_shuffle = pipeline.iter().any(|(id, _)| *id == 2);
    let mut parts: Vec<String> = Vec::new();
    let mut named = false;
    for (id, values) in &pipeline {
        if *id == BLOSC2_FILTER_ID {
            parts.push(describe_blosc(values));
        } else if *id == BLOSC_FILTER_ID {
            let d = describe_blosc(values);
            parts.push(format!("blosc:{}", d.split_once(':').map(|x| x.1).unwrap_or("")));
        } else if !named {
            if let Some(c) = &named_compression {
                named = true;
                parts.push(c.clone());
            }
        }
    }
    if has_shuffle {
        parts.push("shuffle".into());
    }
    Ok(if parts.is_empty() { "chunked".into() } else { parts.join("+") })
}

/// `portable` when every dataset under `root` needs only HDF5's own filters,
/// else `balanced`.
///
/// An amend has to infer what a file was written for: amending a `portable`
/// file must not add Blosc2 datasets its intended readers cannot open.
pub fn profile_family(root: &hdf5::Group) -> Result<&'static str> {
    let mut needs_plugin = false;
    crate::h5::ops::visit(root, &mut |_, node| {
        if let crate::h5::ops::Node::Dataset(ds) = node {
            if crate::h5::data::chunks(ds).is_some() {
                for (id, _) in crate::h5::data::filters(ds)? {
                    if !BUILTIN_FILTER_IDS.contains(&id) {
                        needs_plugin = true;
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    })?;
    Ok(if needs_plugin { "balanced" } else { "portable" })
}

/// Whether a dataset is large enough for the W902 warning.
pub fn is_bulk(ds: &hdf5::Dataset) -> bool {
    crate::h5::data::nbytes(ds).map(|n| n >= BULK_MIN_BYTES).unwrap_or(false)
}

/// `repr_list` of the profile names, for messages.
pub fn profiles_repr() -> String {
    repr_list(&profile_names())
}
