//! Datasets of any element type: create, read whole, read a region.
//!
//! Regions are read **in one call** (spec §14.5): `d[(k, *roi)]`, never
//! `d[k][roi]`, which materialises the whole sub-array first.

use hdf5::filters::Filter;
use hdf5::types::{TypeDescriptor as TD, VarLenAscii, VarLenUnicode};
use hdf5::{Hyperslab, SliceOrIndex};
use ndarray::{ArrayD, IxDyn};

use super::attrs::{bytes_to_array, numeric_dtype, read_fixed_strings};
use crate::array::{DType, Index, NdArray};
use crate::{with_array, with_dtype, Error, Result};

/// How a new dataset is laid out on disk.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    /// Chunk shape; `None` stores the dataset contiguous.
    pub chunks: Option<Vec<usize>>,
    /// The filter pipeline, applied in order (chunked datasets only).
    pub filters: Vec<Filter>,
}

impl Layout {
    /// Contiguous, unfiltered.
    pub fn contiguous() -> Layout {
        Layout::default()
    }
}

/// What a dataset stores.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Numbers or booleans.
    Numeric(DType),
    /// Strings (variable- or fixed-length).
    Strings,
    /// Something else (compound, reference, ...).
    Other(String),
}

/// The element kind of a dataset.
pub fn kind(ds: &hdf5::Dataset) -> Result<Kind> {
    let td = ds.dtype()?.to_descriptor();
    Ok(match td {
        Ok(TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_)) => Kind::Strings,
        Ok(td) => match numeric_dtype(&td) {
            Some(d) => Kind::Numeric(d),
            None => Kind::Other(format!("{td}")),
        },
        Err(e) => Kind::Other(e.to_string()),
    })
}

/// The numeric dtype of a dataset, or an error for strings and others.
pub fn dtype(ds: &hdf5::Dataset) -> Result<DType> {
    match kind(ds)? {
        Kind::Numeric(d) => Ok(d),
        Kind::Strings => Err(Error::Type(format!("{} holds strings, not numbers", ds.name()))),
        Kind::Other(t) => Err(Error::Type(format!("{} has an unsupported datatype {t}", ds.name()))),
    }
}

/// Whether a dataset holds strings.
pub fn is_strings(ds: &hdf5::Dataset) -> bool {
    matches!(kind(ds), Ok(Kind::Strings))
}

fn is_plain_enum(ds: &hdf5::Dataset) -> bool {
    matches!(ds.dtype().and_then(|t| t.to_descriptor()), Ok(TD::Enum(_)))
}

/// Read a whole numeric dataset.
pub fn read(ds: &hdf5::Dataset) -> Result<NdArray> {
    let dtype = dtype(ds)?;
    if is_plain_enum(ds) {
        return read_enum(ds, dtype);
    }
    Ok(with_dtype!(dtype, T => NdArray::from(ds.read_dyn::<T>()?)))
}

fn read_enum(ds: &hdf5::Dataset, dtype: DType) -> Result<NdArray> {
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let shape = ds.shape();
    let n: usize = shape.iter().product::<usize>().max(1);
    let mut buf = vec![0u8; n * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let status =
            h5d::H5Dread(ds.id(), mtype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not read {}", ds.name())));
        }
        Ok(())
    })?;
    bytes_to_array(&buf, dtype, &shape)
}

/// Whether a dataset is stored as an HDF5 enum (booleans, as h5py writes them).
pub fn is_enum(ds: &hdf5::Dataset) -> bool {
    is_plain_enum(ds)
}

/// Read a hyperslab of an enum dataset as its base integer (or `bool`).
fn read_enum_region(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
) -> Result<NdArray> {
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    let start: Vec<u64> = axes.iter().map(|a| a.0 as u64).collect();
    let count: Vec<u64> = axes.iter().map(|a| a.1 as u64).collect();
    let stride: Vec<u64> = axes.iter().map(|a| a.2 as u64).collect();
    let n: usize = count.iter().product::<u64>() as usize;
    let mut buf = vec![0u8; n.max(1) * dtype.itemsize()];
    super::locked(|| unsafe {
        let ftype = h5d::H5Dget_type(ds.id());
        let mtype = h5t::H5Tget_native_type(ftype, h5t::H5T_direction_t::H5T_DIR_ASCEND);
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = h5s::H5Sselect_hyperslab(
            fspace,
            h5s::H5S_seloper_t::H5S_SELECT_SET,
            start.as_ptr(),
            stride.as_ptr(),
            count.as_ptr(),
            std::ptr::null(),
        );
        let mspace = h5s::H5Screate_simple(count.len() as i32, count.as_ptr(), std::ptr::null());
        if status >= 0 {
            status = h5d::H5Dread(ds.id(), mtype, mspace, fspace, h5p::H5P_DEFAULT, buf.as_mut_ptr().cast());
        }
        h5s::H5Sclose(mspace);
        h5s::H5Sclose(fspace);
        h5t::H5Tclose(mtype);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not read a region of {}", ds.name())));
        }
        Ok(())
    })?;
    let shape: Vec<usize> = count.iter().map(|c| *c as usize).collect();
    bytes_to_array(&buf, dtype, &shape)?.reshape(out_shape)
}

/// Resolve a selection against a shape: per-axis `(start, count, step)` and
/// whether the axis is kept in the output.
fn resolve(shape: &[usize], index: &[Index]) -> Result<Vec<(usize, usize, usize, bool)>> {
    if index.len() > shape.len() {
        return Err(Error::Index(format!(
            "too many indices: {} for a {}-dimensional dataset",
            index.len(),
            shape.len()
        )));
    }
    let mut out = Vec::with_capacity(shape.len());
    for (axis, n) in shape.iter().enumerate() {
        match index.get(axis).copied().unwrap_or(Index::Slice(crate::array::Slice::full())) {
            Index::Slice(s) => {
                let (start, stop, step) = s.resolve(*n)?;
                out.push((start, (stop - start).div_ceil(step), step, true));
            }
            Index::At(i) => {
                let k = if i < 0 { i + *n as i64 } else { i };
                if k < 0 || k >= *n as i64 {
                    return Err(Error::Index(format!("index {i} is out of range for axis {axis} with size {n}")));
                }
                out.push((k as usize, 1, 1, false));
            }
        }
    }
    Ok(out)
}

/// Read a region of a numeric dataset in one call.
pub fn read_region(ds: &hdf5::Dataset, index: &[Index]) -> Result<NdArray> {
    let dtype = dtype(ds)?;
    let shape = ds.shape();
    let axes = resolve(&shape, index)?;
    let out_shape: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
    if axes.iter().all(|(start, _count, step, kept)| *kept && *start == 0 && *step == 1)
        && axes.iter().zip(&shape).all(|(a, n)| a.1 == *n)
    {
        return read(ds);
    }
    if axes.iter().any(|a| a.1 == 0) {
        return Ok(NdArray::zeros(dtype, &out_shape));
    }
    if is_plain_enum(ds) {
        return read_enum_region(ds, dtype, &axes, &out_shape);
    }
    let selection: Vec<SliceOrIndex> = axes
        .iter()
        .map(|(start, count, step, kept)| {
            if *kept {
                SliceOrIndex::SliceCount { start: *start, step: *step, count: *count, block: 1 }
            } else {
                SliceOrIndex::Index(*start)
            }
        })
        .collect();
    let hyper = Hyperslab::from(selection);
    Ok(with_dtype!(dtype, T => {
        let a: ArrayD<T> = ds.read_slice::<T, _, IxDyn>(hyper)?;
        NdArray::from(a.into_shape_with_order(IxDyn(&out_shape))?)
    }))
}

/// Slice an in-memory array with the same semantics as [`read_region`].
pub fn slice_array(array: &NdArray, index: &[Index]) -> Result<NdArray> {
    let shape = array.shape();
    let axes = resolve(&shape, index)?;
    with_array!(array, a => {
        let mut view = a.view();
        // Apply slices first (keeping every axis), then drop indexed axes from
        // the highest down so positions stay valid.
        for (axis, (start, count, step, _)) in axes.iter().enumerate() {
            let stop = start + count.saturating_sub(1) * step + usize::from(*count > 0);
            let stop = if *count == 0 { *start } else { stop };
            view.slice_axis_inplace(
                ndarray::Axis(axis),
                ndarray::Slice::new(*start as isize, Some(stop as isize), *step as isize),
            );
        }
        let mut owned = view.to_owned();
        for (axis, (_, _, _, kept)) in axes.iter().enumerate().rev() {
            if !kept {
                owned = owned.index_axis_move(ndarray::Axis(axis), 0);
            }
        }
        Ok(NdArray::from(owned))
    })
}

/// Read a string dataset (any shape), flattened in C order.
pub fn read_strings(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    let td = ds.dtype()?.to_descriptor()?;
    Ok(match td {
        TD::VarLenUnicode => ds.read_raw::<VarLenUnicode>()?.into_iter().map(|s| s.as_str().to_string()).collect(),
        TD::VarLenAscii => ds.read_raw::<VarLenAscii>()?.into_iter().map(|s| s.as_str().to_string()).collect(),
        TD::FixedAscii(_) | TD::FixedUnicode(_) => {
            read_fixed_strings(ds.id(), ds.size().max(usize::from(ds.is_scalar())), &td, false)?
        }
        other => return Err(Error::Type(format!("{} holds {other}, not strings", ds.name()))),
    })
}

/// Read a scalar string dataset (`/meta`).
pub fn read_scalar_string(ds: &hdf5::Dataset) -> Result<String> {
    let mut values = read_strings(ds)?;
    if values.is_empty() {
        return Err(Error::Value(format!("{} holds no string", ds.name())));
    }
    Ok(values.swap_remove(0))
}

/// Create a numeric dataset holding `data`.
pub fn create(group: &hdf5::Group, name: &str, data: &NdArray, layout: &Layout) -> Result<hdf5::Dataset> {
    let chunks = layout.chunks.clone();
    with_array!(data, a => {
        let mut builder = group.new_dataset_builder().obj_track_times(false).with_data(a.view());
        if let Some(c) = &chunks {
            builder = builder.chunk(c.clone()).set_filters(&layout.filters);
        }
        Ok(builder.create(name)?)
    })
}

/// Create an empty numeric dataset to fill region by region.
pub fn create_empty(
    group: &hdf5::Group,
    name: &str,
    dtype: DType,
    shape: &[usize],
    layout: &Layout,
) -> Result<hdf5::Dataset> {
    let chunks = layout.chunks.clone();
    with_dtype!(dtype, T => {
        let mut builder = group.new_dataset::<T>().obj_track_times(false).shape(shape.to_vec());
        if let Some(c) = &chunks {
            builder = builder.chunk(c.clone()).set_filters(&layout.filters);
        }
        Ok(builder.create(name)?)
    })
}

/// Write `data` into a region of an existing dataset (slices only).
pub fn write_region(ds: &hdf5::Dataset, data: &NdArray, starts: &[usize]) -> Result<()> {
    let shape = data.shape();
    let selection: Vec<SliceOrIndex> = starts
        .iter()
        .zip(&shape)
        .map(|(s, n)| SliceOrIndex::SliceCount { start: *s, step: 1, count: *n, block: 1 })
        .collect();
    if shape.iter().any(|n| *n == 0) {
        return Ok(());
    }
    let hyper = Hyperslab::from(selection);
    with_array!(data, a => ds.write_slice(a.view(), hyper)?);
    Ok(())
}

/// Create a 1-D variable-length UTF-8 string dataset.
pub fn create_strings(group: &hdf5::Group, name: &str, values: &[String]) -> Result<hdf5::Dataset> {
    let parsed: Vec<VarLenUnicode> = values
        .iter()
        .map(|s| s.parse::<VarLenUnicode>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Error::Value(format!("{e}")))?;
    let arr = ndarray::Array1::from(parsed);
    Ok(group.new_dataset_builder().obj_track_times(false).with_data(arr.view()).create(name)?)
}

/// Create a scalar variable-length UTF-8 string dataset.
pub fn create_scalar_string(group: &hdf5::Group, name: &str, value: &str) -> Result<hdf5::Dataset> {
    let parsed: VarLenUnicode = value.parse().map_err(|e| Error::Value(format!("{e}")))?;
    let ds = group.new_dataset::<VarLenUnicode>().obj_track_times(false).shape(()).create(name)?;
    ds.write_scalar(&parsed)?;
    Ok(ds)
}

/// The dataset's chunk shape, or `None` when contiguous.
pub fn chunks(ds: &hdf5::Dataset) -> Option<Vec<usize>> {
    ds.chunk()
}

/// The dataset's filter pipeline, as `(id, client data)` pairs.
pub fn filters(ds: &hdf5::Dataset) -> Result<Vec<(i32, Vec<u32>)>> {
    use crate::h5sys::{h5d, h5p};
    super::locked(|| unsafe {
        let plist = h5d::H5Dget_create_plist(ds.id());
        if plist < 0 {
            return Err(Error::Io(format!("could not read the creation properties of {}", ds.name())));
        }
        let n = h5p::H5Pget_nfilters(plist);
        let mut out = Vec::new();
        for i in 0..n.max(0) {
            let mut flags: u32 = 0;
            let mut nelmts: usize = 32;
            let mut values = vec![0u32; 32];
            let mut name = vec![0i8; 64];
            let mut config: u32 = 0;
            let id = h5p::H5Pget_filter2(
                plist,
                i as u32,
                &mut flags,
                &mut nelmts,
                values.as_mut_ptr(),
                name.len(),
                name.as_mut_ptr().cast(),
                &mut config,
            );
            values.truncate(nelmts.min(32));
            out.push((id, values));
        }
        h5p::H5Pclose(plist);
        Ok(out)
    })
}

/// The number of bytes the dataset occupies when decompressed.
pub fn nbytes(ds: &hdf5::Dataset) -> Result<usize> {
    let itemsize = match kind(ds)? {
        Kind::Numeric(d) => d.itemsize(),
        _ => ds.dtype()?.size(),
    };
    let n: usize = ds.shape().iter().product();
    Ok(n * itemsize)
}
