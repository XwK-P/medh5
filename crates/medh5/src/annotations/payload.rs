//! The intermediate an encoder produces before anything touches HDF5.
//!
//! Keeping encoders pure --- arrays in, arrays out --- is what lets the
//! transcoding matrix be tested without a file, and what lets the writer
//! decide chunking and codecs in one place instead of in every encoder.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use ndarray::ArrayD;
use serde_json::{json, Value};

use crate::array::{Index, NdArray, Slice};
use crate::h5::attrs::AttrValue;
use crate::h5::data;
use crate::json::repr_int_tuple;
use crate::labels::check_class_id;
use crate::{Error, Result};

/// One dataset of a payload: numbers, or strings.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadData {
    Array(NdArray),
    Strings(Vec<String>),
}

impl PayloadData {
    pub fn shape(&self) -> Vec<usize> {
        match self {
            PayloadData::Array(a) => a.shape(),
            PayloadData::Strings(s) => vec![s.len()],
        }
    }

    pub fn nbytes(&self) -> usize {
        match self {
            PayloadData::Array(a) => a.nbytes(),
            // An object array of references, as NumPy counts it.
            PayloadData::Strings(s) => s.len() * 8,
        }
    }

    pub fn dtype_str(&self) -> String {
        match self {
            PayloadData::Array(a) => a.dtype().numpy_str().to_string(),
            PayloadData::Strings(_) => "|O".to_string(),
        }
    }

    pub fn as_array(&self) -> Option<&NdArray> {
        match self {
            PayloadData::Array(a) => Some(a),
            PayloadData::Strings(_) => None,
        }
    }
}

impl From<NdArray> for PayloadData {
    fn from(a: NdArray) -> Self {
        PayloadData::Array(a)
    }
}

/// Datasets and kind-specific attributes for one encoded annotation.
#[derive(Debug, Clone, PartialEq)]
pub struct Payload {
    pub kind: String,
    pub datasets: IndexMap<String, PayloadData>,
    pub attrs: Vec<(String, AttrValue)>,
    /// Leading axes of `data` that must get chunk extent 1 (spec §14.1).
    pub stacked_axes: usize,
    pub class_ids: Vec<i64>,
}

impl Payload {
    /// An empty payload of `kind`.
    pub fn new(kind: &str) -> Payload {
        Payload {
            kind: kind.into(),
            datasets: IndexMap::new(),
            attrs: Vec::new(),
            stacked_axes: 0,
            class_ids: Vec::new(),
        }
    }

    /// The `data` dataset.
    pub fn data(&self) -> Result<&NdArray> {
        self.array("data")
    }

    /// A numeric dataset by name.
    pub fn array(&self, name: &str) -> Result<&NdArray> {
        match self.datasets.get(name) {
            Some(PayloadData::Array(a)) => Ok(a),
            Some(PayloadData::Strings(_)) => Err(Error::Type(format!("{name:?} holds strings"))),
            None => Err(Error::Key(crate::json::repr_str(name))),
        }
    }

    /// A kind-specific attribute.
    pub fn attr(&self, name: &str) -> Option<&AttrValue> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v)
    }

    /// Set (or replace) a kind-specific attribute.
    pub fn set_attr(&mut self, name: &str, value: AttrValue) {
        if let Some(slot) = self.attrs.iter_mut().find(|(k, _)| k == name) {
            slot.1 = value;
        } else {
            self.attrs.push((name.into(), value));
        }
    }

    /// Total bytes of every dataset.
    pub fn nbytes(&self) -> usize {
        self.datasets.values().map(PayloadData::nbytes).sum()
    }

    /// Shapes and dtypes, for reports.
    pub fn describe(&self) -> Value {
        let datasets: serde_json::Map<String, Value> = self
            .datasets
            .iter()
            .map(|(name, d)| (name.clone(), json!({"shape": d.shape(), "dtype": d.dtype_str()})))
            .collect();
        json!({"kind": self.kind, "datasets": datasets, "nbytes": self.nbytes()})
    }
}

/// Class id -> boolean occupancy over the grid's spatial shape, sorted by id.
pub type Masks = BTreeMap<i64, ArrayD<bool>>;

/// Check every mask has one agreed shape and a writable class id.
pub fn normalize_masks(masks: &Masks, spatial_shape: Option<&[usize]>) -> Result<Vec<usize>> {
    let mut shape: Option<Vec<usize>> = spatial_shape.map(<[usize]>::to_vec);
    for (class_id, mask) in masks {
        match &shape {
            None => shape = Some(mask.shape().to_vec()),
            Some(s) if mask.shape() != s.as_slice() => {
                return Err(Error::coded(
                    "E405",
                    format!(
                        "mask for class {class_id} has shape {}, expected {}",
                        repr_int_tuple(mask.shape()),
                        repr_int_tuple(s)
                    ),
                ))
            }
            _ => {}
        }
        check_class_id(*class_id)?;
    }
    shape.ok_or_else(|| Error::coded("E410", "no masks were supplied"))
}

/// Read budget for a scan that only needs a yes/no or a count.
pub const SLAB_BYTES: usize = 8 * 1024 * 1024;

/// Selections that read `ds` in slabs within [`SLAB_BYTES`], with `prefix`
/// fixing leading indices.  A row over the budget descends an axis instead.
pub fn slabs(shape: &[usize], itemsize: usize, prefix: &[usize]) -> Vec<Vec<Index>> {
    let rest = &shape[prefix.len()..];
    let mut out = Vec::new();
    if rest.is_empty() || rest[0] == 0 {
        return out;
    }
    let per_row = rest[1..].iter().product::<usize>() * itemsize;
    if per_row > SLAB_BYTES && rest.len() > 1 {
        for index in 0..rest[0] {
            let mut p = prefix.to_vec();
            p.push(index);
            out.extend(slabs(shape, itemsize, &p));
        }
        return out;
    }
    let step = (SLAB_BYTES / per_row.max(1)).min(rest[0]).max(1);
    let mut start = 0;
    while start < rest[0] {
        let mut sel: Vec<Index> = prefix.iter().map(|i| Index::At(*i as i64)).collect();
        sel.push(Index::Slice(Slice::new(start as i64, (start + step) as i64)));
        out.push(sel);
        start += step;
    }
    out
}

/// Whether any element of a dataset equals `value`, scanning in slabs.
pub fn contains_value(ds: &hdf5::Dataset, value: i64) -> Result<bool> {
    let dtype = data::dtype(ds)?;
    for sel in slabs(&ds.shape(), dtype.itemsize(), &[]) {
        let block = data::read_region(ds, &sel)?;
        if block.cast::<i64>().iter().any(|v| *v == value) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The number of nonzero elements of a dataset, in slabs.
pub fn count_nonzero(ds: &hdf5::Dataset) -> Result<u64> {
    let dtype = data::dtype(ds)?;
    let mut total = 0u64;
    for sel in slabs(&ds.shape(), dtype.itemsize(), &[]) {
        let block = data::read_region(ds, &sel)?;
        total += block.nonzero_mask().iter().filter(|v| **v).count() as u64;
    }
    Ok(total)
}

/// How many voxels hold each value up to `ceiling`, in slabs.
pub fn value_counts(ds: &hdf5::Dataset, ceiling: i64) -> Result<BTreeMap<i64, u64>> {
    let dtype = data::dtype(ds)?;
    let mut totals = vec![0u64; (ceiling.max(0) + 1) as usize];
    for sel in slabs(&ds.shape(), dtype.itemsize(), &[]) {
        let block = data::read_region(ds, &sel)?;
        for v in block.cast::<i64>().iter() {
            if *v >= 0 && *v <= ceiling {
                totals[*v as usize] += 1;
            }
        }
    }
    Ok(totals.into_iter().enumerate().filter(|(_, c)| *c > 0).map(|(v, c)| (v as i64, c)).collect())
}

/// Per-plane population counts of a packed `uint64` bitmask: `(P, 64)`.
pub fn popcounts(ds: &hdf5::Dataset) -> Result<Vec<[u64; 64]>> {
    let shape = ds.shape();
    let planes = shape.first().copied().unwrap_or(0);
    let mut out = vec![[0u64; 64]; planes];
    for (plane, counts) in out.iter_mut().enumerate() {
        for sel in slabs(&shape, 8, &[plane]) {
            let block = data::read_region(ds, &sel)?;
            for word in block.cast::<u64>().iter() {
                let mut w = *word;
                while w != 0 {
                    let bit = w.trailing_zeros() as usize;
                    counts[bit] += 1;
                    w &= w - 1;
                }
            }
        }
    }
    Ok(out)
}

/// `np.packbits` of a flat boolean sequence: MSB-first within each byte.
pub fn packbits(bits: impl IntoIterator<Item = bool>) -> Vec<u8> {
    let mut out = Vec::new();
    let mut byte = 0u8;
    let mut n = 0;
    for bit in bits {
        byte = (byte << 1) | u8::from(bit);
        n += 1;
        if n == 8 {
            out.push(byte);
            byte = 0;
            n = 0;
        }
    }
    if n > 0 {
        out.push(byte << (8 - n));
    }
    out
}

/// `np.unpackbits(packed)[:n]`, MSB-first.
pub fn unpackbits(packed: &[u8], n: usize) -> Vec<bool> {
    let mut out = Vec::with_capacity(n);
    for byte in packed {
        for k in (0..8).rev() {
            if out.len() == n {
                return out;
            }
            out.push((byte >> k) & 1 == 1);
        }
    }
    out.resize(n, false);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_pack_like_numpy() {
        // np.packbits([1,0,1,1,0,0,0,0,1]) == [176, 128]
        let bits = [true, false, true, true, false, false, false, false, true];
        assert_eq!(packbits(bits), vec![176, 128]);
        assert_eq!(unpackbits(&[176, 128], 9), bits.to_vec());
    }
}
