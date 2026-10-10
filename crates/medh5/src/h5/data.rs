//! Datasets of any element type: create, read whole, read a region.
//!
//! Regions are read **in one call** (spec §14.5): `d[(k, *roi)]`, never
//! `d[k][roi]`, which materialises the whole sub-array first.

use hdf5::filters::Filter;
use hdf5::types::{TypeDescriptor as TD, VarLenAscii, VarLenUnicode};
use hdf5::{Hyperslab, SliceOrIndex};
use ndarray::{ArrayD, IxDyn};

use super::attrs::{bytes_to_array, lossy, numeric_dtype, read_fixed_strings};
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
    super::alive(ds)?;
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
    super::alive(ds)?;
    match kind(ds)? {
        Kind::Numeric(d) => Ok(d),
        Kind::Strings => Err(Error::Type(format!("{} holds strings, not numbers", ds.name()))),
        Kind::Other(t) => Err(Error::Type(format!("{} has an unsupported datatype {t}", ds.name()))),
    }
}

/// Whether `ds` is stored in exactly the type a fresh write of `dtype`
/// creates: that type, not committed.  An enumeration's names, a committed
/// type and another byte order are what a rebuild from `dtype` would lose.
pub fn stored_as_written(ds: &hdf5::Dataset, dtype: DType) -> Result<bool> {
    use crate::h5sys::{h5d, h5t};
    let committed = super::locked(|| unsafe {
        let tid = h5d::H5Dget_type(ds.id());
        let committed = tid >= 0 && h5t::H5Tcommitted(tid) > 0;
        if tid >= 0 {
            h5t::H5Tclose(tid);
        }
        committed
    });
    if committed {
        return Ok(false);
    }
    let stored = ds.dtype()?;
    Ok(with_dtype!(dtype, T => stored == hdf5::Datatype::from_type::<T>()?))
}

/// Whether a dataset holds strings.
pub fn is_strings(ds: &hdf5::Dataset) -> bool {
    matches!(kind(ds), Ok(Kind::Strings))
}

fn is_plain_enum(ds: &hdf5::Dataset) -> bool {
    matches!(ds.dtype().and_then(|t| t.to_descriptor()), Ok(TD::Enum(_)))
}

/// Reads at least this large reserve their buffer fallibly first.
const PROBE_BYTES: usize = 64 * 1024 * 1024;

/// The machine's physical memory, where the platform states it.
fn physical_memory() -> Option<usize> {
    #[cfg(unix)]
    {
        // SAFETY: `sysconf` reads a system constant and has no preconditions.
        let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
        if pages > 0 && size > 0 {
            return usize::try_from(pages).ok()?.checked_mul(usize::try_from(size).ok()?);
        }
    }
    None
}

/// Refuse a read of `extents` elements of `itemsize` bytes that no buffer
/// can hold, rather than let the allocation fail.
///
/// A file can declare a dataset of any extent and store none of it --- an
/// unallocated chunk reads as the fill value --- so a 10 KiB file asked the
/// allocator for 2**63 bytes, which panicked (`capacity overflow`), or for
/// 9 TB, which aborted the process (N07 of the 2.0 re-audit).  The size is
/// computed with checked arithmetic, a buffer larger than the machine's
/// memory is refused outright --- where the system overcommits (macOS, say)
/// reserving one succeeds, and filling it is what kills the process --- and
/// a large one is reserved fallibly, so such a read is an error the caller
/// reports.
pub(crate) fn ensure_allocatable(ds: &hdf5::Dataset, extents: &[usize], itemsize: usize) -> Result<()> {
    ensure_holdable(&ds.name(), extents, itemsize)
}

/// [`ensure_allocatable`] for anything HDF5 reads into a buffer, named
/// `what` --- an attribute, say.
pub(crate) fn ensure_holdable(what: &str, extents: &[usize], itemsize: usize) -> Result<()> {
    if holdable(extents, itemsize) {
        return Ok(());
    }
    Err(Error::Io(format!(
        "{what} cannot be read: {} elements of {itemsize} bytes are more than this process can hold",
        extents.iter().map(usize::to_string).collect::<Vec<_>>().join(" x ")
    )))
}

/// Whether a buffer of `extents` elements of `itemsize` bytes can be held.
pub(crate) fn holdable(extents: &[usize], itemsize: usize) -> bool {
    if extents.contains(&0) {
        return true;
    }
    let bytes = extents.iter().try_fold(itemsize, |n, e| n.checked_mul(*e)).filter(|b| *b <= isize::MAX as usize);
    match bytes {
        Some(b) if b >= PROBE_BYTES => {
            physical_memory().is_none_or(|memory| b <= memory) && Vec::<u8>::new().try_reserve_exact(b).is_ok()
        }
        Some(_) => true,
        None => false,
    }
}

/// Read a whole numeric dataset.
pub fn read(ds: &hdf5::Dataset) -> Result<NdArray> {
    super::alive(ds)?;
    let dtype = dtype(ds)?;
    ensure_allocatable(ds, &ds.shape(), dtype.itemsize())?;
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
pub(crate) fn resolve(shape: &[usize], index: &[Index]) -> Result<Vec<(usize, usize, usize, bool)>> {
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
    super::alive(ds)?;
    let dtype = dtype(ds)?;
    let shape = ds.shape();
    let axes = resolve(&shape, index)?;
    let out_shape: Vec<usize> = axes.iter().filter(|a| a.3).map(|a| a.1).collect();
    ensure_allocatable(ds, &axes.iter().map(|a| a.1).collect::<Vec<_>>(), dtype.itemsize())?;
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
    if let Some(window) = super::window::read_window(ds, dtype, &axes, &out_shape)? {
        return Ok(window);
    }
    read_hyperslab(ds, dtype, &axes, &out_shape)
}

/// [`read_region`] through HDF5's filter pipeline: whole chunks, decompressed
/// under HDF5's lock.
pub(crate) fn read_hyperslab(
    ds: &hdf5::Dataset,
    dtype: DType,
    axes: &[(usize, usize, usize, bool)],
    out_shape: &[usize],
) -> Result<NdArray> {
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
        NdArray::from(a.into_shape_with_order(IxDyn(out_shape))?)
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

/// What one decoded string costs beyond its text: the `String` itself, and
/// what the allocator spends on the smallest allocation it makes for the text.
pub(crate) const STRING_COST: usize = std::mem::size_of::<String>() + 16;

/// The stored width of a fixed-length string type; `None` for a
/// variable-length one, and an error for anything else.
fn string_width(ds: &hdf5::Dataset, td: &TD) -> Result<Option<usize>> {
    match td {
        TD::VarLenUnicode | TD::VarLenAscii => Ok(None),
        TD::FixedAscii(width) | TD::FixedUnicode(width) => Ok(Some((*width).max(1))),
        other => Err(Error::Type(format!("{} holds {other}, not strings", ds.name()))),
    }
}

/// Bytes of text reading rows `[start, start + count)` of a
/// variable-length string dataset allocates, measured before anything is
/// (`H5Dvlen_get_buf_size`).  Many elements may name one stored string, so
/// the text a read allocates is not bounded by the file's size.
///
/// A scalar holds one string, which the file stores: it is not measured
/// (HDF5 iterates a scalar selection without coordinates, which the
/// measurement dereferences).
fn vlen_text_bytes(ds: &hdf5::Dataset, rows: Option<(usize, usize)>) -> Result<usize> {
    use crate::h5sys::{h5d, h5s, h5t};
    let shape = ds.shape();
    if shape.is_empty() {
        return Ok(0);
    }
    super::locked(|| unsafe {
        let stored = h5d::H5Dget_type(ds.id());
        // The measurement reads as a read would, into memory's type.
        let ftype = h5t::H5Tget_native_type(stored, h5t::H5T_direction_t::H5T_DIR_DEFAULT);
        h5t::H5Tclose(stored);
        if ftype < 0 {
            return Err(Error::Io(format!("could not measure the strings of {}", ds.name())));
        }
        let fspace = h5d::H5Dget_space(ds.id());
        let mut status = 0;
        if let Some((start, count)) = rows {
            let mut first = vec![0u64; shape.len()];
            let mut counts: Vec<u64> = shape.iter().map(|n| *n as u64).collect();
            first[0] = start as u64;
            counts[0] = count as u64;
            status = h5s::H5Sselect_hyperslab(
                fspace,
                h5s::H5S_seloper_t::H5S_SELECT_SET,
                first.as_ptr(),
                std::ptr::null(),
                counts.as_ptr(),
                std::ptr::null(),
            );
        }
        let mut size: u64 = 0;
        if status >= 0 {
            status = h5d::H5Dvlen_get_buf_size(ds.id(), ftype, fspace, &mut size);
        }
        h5s::H5Sclose(fspace);
        h5t::H5Tclose(ftype);
        if status < 0 {
            return Err(Error::Io(format!("could not measure the strings of {}", ds.name())));
        }
        Ok(usize::try_from(size).unwrap_or(usize::MAX))
    })
}

/// The memory reading `n` strings costs once decoded: one `String` each and
/// its text, held twice while it is decoded --- as HDF5 read it and as text.
/// That is `text` bytes for variable-length strings (and a pointer each), and
/// the width for fixed-length ones.
fn decoded_bytes(n: usize, width: Option<usize>, text: usize) -> Option<usize> {
    let per = STRING_COST.checked_add(width.map_or(std::mem::size_of::<VarLenUnicode>(), |w| w.saturating_mul(2)))?;
    n.checked_mul(per)?.checked_add(text.checked_mul(2)?)
}

/// Read rows `[start, start + count)` of a string dataset along its first
/// axis, every other axis whole, flattened in C order.  The caller has sized
/// the read.
fn read_string_rows(ds: &hdf5::Dataset, td: &TD, start: usize, count: usize) -> Result<Vec<String>> {
    let shape = ds.shape();
    let selection: Vec<SliceOrIndex> = shape
        .iter()
        .enumerate()
        .map(|(axis, n)| {
            let (first, many) = if axis == 0 { (start, count) } else { (0, *n) };
            SliceOrIndex::SliceCount { start: first, step: 1, count: many, block: 1 }
        })
        .collect();
    let hyper = Hyperslab::from(selection);
    let mut out = Vec::new();
    let n: usize = count.saturating_mul(shape[1..].iter().product::<usize>());
    out.try_reserve_exact(n)
        .map_err(|_| Error::Io(format!("{n} strings of {} are more than this process can hold", ds.name())))?;
    match td {
        TD::VarLenUnicode => {
            out.extend(ds.read_slice::<VarLenUnicode, _, IxDyn>(hyper)?.iter().map(|s| lossy(s.as_bytes())))
        }
        TD::VarLenAscii => {
            out.extend(ds.read_slice::<VarLenAscii, _, IxDyn>(hyper)?.iter().map(|s| lossy(s.as_bytes())))
        }
        _ => out.extend(super::attrs::read_fixed_string_rows(ds, &shape, start, count)?),
    }
    Ok(out)
}

/// Read a string dataset slab by slab along its first axis, handing each
/// slab's strings to `each` in C order, so that about `budget` bytes of
/// decoded strings are held at a time --- what a digest needs, where it
/// held every string of the dataset at once.
///
/// A string decoded costs a `String` beyond its text, so a few kilobytes of
/// one-byte strings, compressed, held hundreds of megabytes once read
/// (F10 of the round-4 audit).  A slab is sized by what it decodes to:
/// variable-length text is measured before it is read, and a slab that
/// measures over the budget is halved.  A single row that no buffer can hold
/// is refused, as any read is ([`ensure_allocatable`]).
pub fn for_each_string_slab(
    ds: &hdf5::Dataset,
    budget: usize,
    each: &mut dyn FnMut(Vec<String>) -> Result<()>,
) -> Result<()> {
    super::alive(ds)?;
    let td = ds.dtype()?.to_descriptor()?;
    let width = string_width(ds, &td)?;
    let shape = ds.shape();
    if shape.is_empty() {
        return each(read_strings(ds)?);
    }
    if shape.contains(&0) {
        return Ok(());
    }
    // A declared extent no machine holds is refused before it is measured,
    // as a whole read refuses it: a slab at a time, an 11 KiB file declaring
    // 2**58 strings was a read without end (N12 of the 2.0 re-audit).
    ensure_allocatable(ds, &shape, width.unwrap_or(std::mem::size_of::<VarLenUnicode>()))?;
    let row: usize = shape[1..].iter().try_fold(1usize, |n, e| n.checked_mul(*e)).unwrap_or(usize::MAX);
    let per_row = decoded_bytes(row, width, 0).unwrap_or(usize::MAX).max(1);
    let mut step = (budget / per_row).clamp(1, shape[0]);
    let mut start = 0;
    while start < shape[0] {
        let count = step.min(shape[0] - start);
        let text = match width {
            None => vlen_text_bytes(ds, Some((start, count)))?,
            Some(_) => 0,
        };
        let need = decoded_bytes(count.saturating_mul(row), width, text);
        if count > 1 && need.is_none_or(|b| b > budget) {
            step = count / 2;
            continue;
        }
        if !need.is_some_and(|b| holdable(&[b], 1)) {
            return Err(Error::Io(format!(
                "{} cannot be read: row {start} decodes to more than this process can hold",
                ds.name()
            )));
        }
        each(read_string_rows(ds, &td, start, count)?)?;
        start += count;
    }
    Ok(())
}

/// Read a string dataset (any shape), flattened in C order.
///
/// Sized by what it decodes to, before anything is read: a fixed-length
/// string type declares any width and a dataspace any extent, and an 11 KiB
/// file of 2**59 16-byte strings panicked the validator on a capacity
/// overflow, where a 1 GiB scalar string aborted a capped process (N12 of the
/// 2.0 re-audit).  Each string read costs a `String` beyond its stored bytes,
/// and variable-length text is measured first, since many elements can name
/// one stored string (F10 of the round-4 audit).
pub fn read_strings(ds: &hdf5::Dataset) -> Result<Vec<String>> {
    super::alive(ds)?;
    let td = ds.dtype()?.to_descriptor()?;
    let width = string_width(ds, &td)?;
    let extents = if ds.is_scalar() { vec![1] } else { ds.shape() };
    let n = extents.iter().try_fold(1usize, |n, e| n.checked_mul(*e));
    // The element count is checked against the stored width first, as it
    // always was, so that a declared extent no machine holds is refused
    // before it is measured.
    ensure_allocatable(ds, &extents, width.unwrap_or(std::mem::size_of::<VarLenUnicode>()))?;
    let text = match width {
        None if n.is_some_and(|n| n > 0) => vlen_text_bytes(ds, None)?,
        _ => 0,
    };
    if !n.and_then(|n| decoded_bytes(n, width, text)).is_some_and(|b| holdable(&[b], 1)) {
        return Err(Error::Io(format!(
            "{} cannot be read: {} strings decode to more than this process can hold",
            ds.name(),
            extents.iter().map(usize::to_string).collect::<Vec<_>>().join(" x ")
        )));
    }
    Ok(match td {
        TD::VarLenUnicode if ds.is_scalar() => {
            ds.read_raw::<VarLenUnicode>()?.iter().map(|s| lossy(s.as_bytes())).collect()
        }
        TD::VarLenAscii if ds.is_scalar() => {
            ds.read_raw::<VarLenAscii>()?.iter().map(|s| lossy(s.as_bytes())).collect()
        }
        TD::FixedAscii(_) | TD::FixedUnicode(_) if ds.is_scalar() => read_fixed_strings(ds.id(), 1, &td, false)?,
        _ if extents.contains(&0) => Vec::new(),
        _ => read_string_rows(ds, &td, 0, extents[0])?,
    })
}

/// Read a scalar string dataset (`/meta`, a table's descriptor, a cache's
/// manifest).
///
/// One string, so a dataset of any other shape is refused rather than read
/// whole and cut to its first element (N12 of the 2.0 re-audit).
pub fn read_scalar_string(ds: &hdf5::Dataset) -> Result<String> {
    if !ds.is_scalar() {
        return Err(Error::Value(format!(
            "{} is a dataset of shape {}, not one string",
            ds.name(),
            crate::json::repr_int_tuple(&ds.shape())
        )));
    }
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
    if shape.contains(&0) {
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
        // h5py reads variable-length strings as an object array: one
        // pointer per element, as NumPy counts it.
        Kind::Strings if matches!(ds.dtype()?.to_descriptor(), Ok(TD::VarLenUnicode | TD::VarLenAscii)) => 8,
        _ => ds.dtype()?.size(),
    };
    let n: usize = ds.shape().iter().product();
    Ok(n * itemsize)
}

/// A fixed-length string type: NUL-padded, `width` bytes, ASCII or UTF-8 ---
/// what h5py creates for NumPy's `S` dtype.
fn fixed_string_type(width: usize, ascii: bool) -> Result<crate::h5sys::h5i::hid_t> {
    use crate::h5sys::h5t;
    // SAFETY: plain HDF5 type construction; the caller closes the type.
    let tid = unsafe { h5t::H5Tcopy(*h5t::H5T_C_S1) };
    if tid < 0 {
        return Err(Error::Io("could not create a string type".into()));
    }
    let cset = if ascii { h5t::H5T_cset_t::H5T_CSET_ASCII } else { h5t::H5T_cset_t::H5T_CSET_UTF8 };
    // SAFETY: `tid` is a valid, writable string type.
    let ok = unsafe {
        h5t::H5Tset_size(tid, width.max(1)) >= 0
            && h5t::H5Tset_strpad(tid, h5t::H5T_str_t::H5T_STR_NULLPAD) >= 0
            && h5t::H5Tset_cset(tid, cset) >= 0
    };
    if !ok {
        // SAFETY: `tid` is valid.
        unsafe { h5t::H5Tclose(tid) };
        return Err(Error::Io("could not configure a string type".into()));
    }
    Ok(tid)
}

/// What [`write_fixed_strings`] and [`recreate_fixed_strings`] hand HDF5 must
/// be what it reads: both write through `H5S_ALL`, which takes one
/// `width`-byte element from the buffer for every element of the dataset,
/// whatever length the buffer has.  A shorter buffer is a read past its end.
fn check_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize) -> Result<()> {
    super::alive(ds)?;
    if !is_strings(ds) {
        return Err(Error::Type(format!("{} does not hold strings", ds.name())));
    }
    if width == 0 {
        return Err(Error::Value("a fixed-length string is at least one byte wide".into()));
    }
    // Counted as `read_strings` counts them: a scalar holds one.
    let n = ds.size().max(usize::from(ds.is_scalar()));
    if values.len() != n {
        return Err(Error::Value(format!(
            "{} holds {n} string(s) and {} were given: every element is rewritten",
            ds.name(),
            values.len()
        )));
    }
    if n.checked_mul(width).is_none() {
        return Err(Error::Value(format!("{n} strings of {width} bytes do not fit in memory")));
    }
    Ok(())
}

fn fixed_bytes(values: &[String], width: usize) -> Vec<u8> {
    let mut buf = vec![0u8; values.len() * width.max(1)];
    for (i, value) in values.iter().enumerate() {
        let bytes = value.as_bytes();
        let n = bytes.len().min(width);
        buf[i * width..i * width + n].copy_from_slice(&bytes[..n]);
    }
    buf
}

/// Overwrite a fixed-length string dataset whose width fits `values`, one
/// value per element of the dataset.
pub fn write_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize, ascii: bool) -> Result<()> {
    use crate::h5sys::{h5d, h5p, h5s, h5t};
    check_fixed_strings(ds, values, width)?;
    let buf = fixed_bytes(values, width);
    super::locked(|| -> Result<()> {
        let mtype = fixed_string_type(width, ascii)?;
        // SAFETY: `buf` holds one `width`-byte element per selected value.
        let status =
            unsafe { h5d::H5Dwrite(ds.id(), mtype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_ptr().cast()) };
        // SAFETY: `mtype` was created above.
        unsafe { h5t::H5Tclose(mtype) };
        if status < 0 {
            return Err(Error::Io(format!("could not rewrite {}", ds.name())));
        }
        Ok(())
    })
}

/// Recreate a fixed-length string dataset wider, keeping its creation
/// property list (filters, chunking, fill value) and its attributes.
///
/// Built under a temporary name and moved into place, so a pipeline HDF5
/// cannot rebuild for the new type leaves the original untouched.  One value
/// per element of the dataset, as [`write_fixed_strings`].
pub fn recreate_fixed_strings(ds: &hdf5::Dataset, values: &[String], width: usize, ascii: bool) -> Result<()> {
    use crate::h5sys::{h5d, h5i, h5l, h5p, h5s, h5t};
    use std::ffi::CString;
    check_fixed_strings(ds, values, width)?;
    let full = ds.name();
    let (parent_path, name) = match full.rsplit_once('/') {
        Some((p, n)) => (if p.is_empty() { "/".to_string() } else { p.to_string() }, n.to_string()),
        None => ("/".to_string(), full.clone()),
    };
    let file = ds.file()?;
    let parent = file.group(&parent_path)?;
    let suffix: String = {
        let mut rng = crate::rng::Rng::from_entropy();
        format!("{:08x}", rng.next_u32())
    };
    let staging = format!("{name}.scrub-{suffix}");
    let chunks = ds.chunk();
    let buf = fixed_bytes(values, width);
    let created = super::locked(|| -> Result<h5i::hid_t> {
        let ftype = fixed_string_type(width, ascii)?;
        // SAFETY: valid ids from the open dataset; every id is closed below.
        unsafe {
            let dcpl = h5d::H5Dget_create_plist(ds.id());
            if let Some(c) = &chunks {
                let dims: Vec<u64> = c.iter().map(|v| *v as u64).collect();
                h5p::H5Pset_chunk(dcpl, dims.len() as i32, dims.as_ptr());
            }
            let space = h5d::H5Dget_space(ds.id());
            let cname = CString::new(staging.as_str()).map_err(|e| Error::Value(e.to_string()))?;
            let id =
                h5d::H5Dcreate2(parent.id(), cname.as_ptr(), ftype, space, h5p::H5P_DEFAULT, dcpl, h5p::H5P_DEFAULT);
            h5s::H5Sclose(space);
            h5p::H5Pclose(dcpl);
            if id < 0 {
                h5t::H5Tclose(ftype);
                return Err(Error::Io(format!("could not recreate {full} with a wider string type")));
            }
            let status = h5d::H5Dwrite(id, ftype, h5s::H5S_ALL, h5s::H5S_ALL, h5p::H5P_DEFAULT, buf.as_ptr().cast());
            h5t::H5Tclose(ftype);
            if status < 0 {
                h5d::H5Dclose(id);
                return Err(Error::Io(format!("could not write the widened {full}")));
            }
            Ok(id)
        }
    })?;
    // SAFETY: `created` is a dataset id we own; closing it releases it.
    super::locked(|| unsafe { h5d::H5Dclose(created) });
    let replacement = parent.dataset(&staging)?;
    for key in super::attrs::names(ds)? {
        super::attrs::copy_raw(ds, &replacement, &key)?;
    }
    drop(replacement);
    super::ops::unlink(&parent, &name)?;
    super::locked(|| -> Result<()> {
        let from = CString::new(staging.as_str()).map_err(|e| Error::Value(e.to_string()))?;
        let to = CString::new(name.as_str()).map_err(|e| Error::Value(e.to_string()))?;
        // SAFETY: both names are valid C strings in the same group.
        let status = unsafe {
            h5l::H5Lmove(parent.id(), from.as_ptr(), parent.id(), to.as_ptr(), h5p::H5P_DEFAULT, h5p::H5P_DEFAULT)
        };
        if status < 0 {
            return Err(Error::Io(format!("could not move the widened {full} into place")));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed(n: usize) -> (tempfile::TempDir, hdf5::File, hdf5::Dataset) {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
        let ds = file.new_dataset::<hdf5::types::FixedAscii<8>>().shape([n]).create("s").unwrap();
        (dir, file, ds)
    }

    /// B01: both writers hand HDF5 a buffer through `H5S_ALL`, which reads one
    /// element per element of the dataset whatever the buffer holds.  Fewer
    /// values read past the buffer (none at all was a segfault); more were
    /// silently dropped.
    #[test]
    fn b01_fixed_strings_take_one_value_per_element() {
        let (_dir, _file, ds) = fixed(3);
        let values: Vec<String> = ["a", "bb", "ccc", "d"].iter().map(|s| s.to_string()).collect();
        for wrong in [&values[..0], &values[..1], &values[..]] {
            assert!(write_fixed_strings(&ds, wrong, 8, true).is_err(), "{} values", wrong.len());
            assert!(recreate_fixed_strings(&ds, wrong, 16, true).is_err(), "{} values", wrong.len());
        }
        assert!(write_fixed_strings(&ds, &values[..3], 0, true).is_err(), "a zero width");
        write_fixed_strings(&ds, &values[..3], 8, true).unwrap();
        assert_eq!(read_strings(&ds).unwrap(), values[..3]);
    }

    /// B01: the element count is checked against a dataset of strings only.
    #[test]
    fn b01_only_strings_are_rewritten_as_strings() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("n.h5")).unwrap();
        let ds = file.new_dataset::<u8>().shape([1]).create("n").unwrap();
        let err = write_fixed_strings(&ds, &["x".to_string()], 8, true).unwrap_err();
        assert!(err.to_string().contains("does not hold strings"), "{err}");
    }

    /// A read is refused before anything is reserved when it overflows or
    /// exceeds the machine's memory: where the system overcommits, a
    /// reservation of terabytes succeeds and filling it killed the process
    /// (N07 of the 2.0 re-audit, on macOS).
    #[test]
    fn n07_a_read_no_machine_can_hold_is_refused_before_it_is_reserved() {
        assert!(holdable(&[4, 4], 8));
        assert!(holdable(&[0, usize::MAX], 8), "empty, whatever the other extents");
        assert!(!holdable(&[usize::MAX, 2], 8), "overflows");
        assert!(!holdable(&[1 << 62], 2), "beyond isize::MAX");
        if let Some(memory) = physical_memory() {
            assert!(!holdable(&[memory, 2], 1), "twice the machine's memory");
        }
    }

    /// N12: a string read is sized as a numeric one.  Its buffer was
    /// `vec![0; width * n]`, unchecked: 2**59 strings of 16 bytes, declared
    /// and never stored, panicked on a capacity overflow.
    #[test]
    fn n12_a_string_read_no_machine_can_hold_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("big.h5")).unwrap();
        let fixed = file
            .new_dataset::<hdf5::types::FixedAscii<16>>()
            .shape([1usize << 59])
            .chunk([1024])
            .create("fixed")
            .unwrap();
        let vlen = file
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([1usize << 61])
            .chunk([1024])
            .create("vlen")
            .unwrap();
        for ds in [&fixed, &vlen] {
            let err = read_strings(ds).unwrap_err();
            assert!(err.to_string().contains("more than this process can hold"), "{err}");
            // A slab at a time too: refused before the first, not read
            // without end.
            let err = for_each_string_slab(ds, 1 << 20, &mut |_| Ok(())).unwrap_err();
            assert!(err.to_string().contains("more than this process can hold"), "{err}");
        }
        let small = file.new_dataset::<hdf5::types::FixedAscii<8>>().shape([2]).create("small").unwrap();
        let values = ["a".to_string(), "bc".to_string()];
        write_fixed_strings(&small, &values, 8, true).unwrap();
        assert_eq!(read_strings(&small).unwrap(), values);
        // One string is a scalar: a vector is refused, not cut to its first.
        let err = read_scalar_string(&small).unwrap_err();
        assert!(err.to_string().contains("not one string"), "{err}");
    }

    /// F10: a string read is held a slab at a time, each slab within its
    /// budget by what it decodes to, and the slabs are the whole dataset in
    /// C order.  A dataset of one-byte strings decoded to a `String` each,
    /// all at once: a 17 KB file took 687 MB to digest.
    #[test]
    fn f10_s13_1_strings_are_digested_a_bounded_slab_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("s.h5")).unwrap();
        let n = 20_000usize;
        let fixed = file.new_dataset::<hdf5::types::FixedAscii<1>>().shape([n / 4, 4]).create("fixed").unwrap();
        let values: Vec<String> = (0..n).map(|i| ((b'a' + (i % 26) as u8) as char).to_string()).collect();
        write_fixed_strings(&fixed, &values, 1, true).unwrap();
        let long: Vec<String> = (0..64).map(|i| format!("{i:04}").repeat(1000)).collect();
        let vlen = create_strings(&file.as_group().unwrap(), "vlen", &long).unwrap();
        for (ds, budget, whole) in [(&fixed, 64 * 1024, &values), (&vlen, 20_000, &long)] {
            let mut slabs = Vec::new();
            for_each_string_slab(ds, budget, &mut |slab| {
                slabs.push(slab);
                Ok(())
            })
            .unwrap();
            assert!(slabs.len() > 1, "{}: one slab of {}", ds.name(), slabs[0].len());
            for slab in &slabs {
                let text: usize = slab.iter().map(String::len).sum();
                let width = if ds.name() == "/fixed" { Some(1) } else { None };
                let held = decoded_bytes(slab.len(), width, if width.is_some() { 0 } else { text }).unwrap();
                assert!(held <= budget || slab.len() == ds.shape()[1..].iter().product::<usize>(), "{held} > {budget}");
            }
            assert_eq!(&slabs.concat(), whole);
            assert_eq!(&read_strings(ds).unwrap(), whole);
            let streamed = crate::integrity::digest::dataset_digest_streamed(ds, "x", "sha256", budget).unwrap();
            let at_once = crate::integrity::digest::strings_digest("x", whole, &ds.shape(), "sha256").unwrap();
            assert_eq!(streamed, at_once);
        }
        // A scalar string reads whole, unmeasured: it holds one string.
        let scalar = create_scalar_string(&file.as_group().unwrap(), "scalar", "{\"a\": 1}").unwrap();
        assert_eq!(read_strings(&scalar).unwrap(), vec!["{\"a\": 1}".to_string()]);
        assert_eq!(vlen_text_bytes(&scalar, None).unwrap(), 0);
        assert!(vlen_text_bytes(&vlen, None).unwrap() >= 64 * 4000);
    }

    /// F10: a whole read is budgeted by what it decodes to, not by its stored
    /// bytes alone: many elements may name one stored string.
    #[test]
    fn f10_a_whole_read_counts_what_it_decodes_to() {
        assert_eq!(decoded_bytes(10, Some(1), 0), Some(10 * (STRING_COST + 2)));
        assert_eq!(decoded_bytes(10, None, 100), Some(10 * (STRING_COST + 8) + 200));
        assert_eq!(decoded_bytes(usize::MAX, Some(1), 0), None);
        assert_eq!(decoded_bytes(1, None, usize::MAX), None);
    }
}
