//! The column encoding of the clinical tables (1.1 §4).
//!
//! One primitive layout, no HDF5 compound types and no group per row:
//!
//! | Logical column | HDF5 |
//! |---|---|
//! | integer, float | dataset `(N,)`, one fixed-width dtype |
//! | UTF-8 string | group `<column>/` with `data: uint8[B]`, `offsets: uint64[N+1]` |
//! | nullable | optional `valid/<column>: uint8[N]`, values 0 or 1 |
//!
//! A required column never has a mask; an optional column without one is
//! entirely valid; an omitted optional column is entirely null.  A null cell
//! holds zero (numbers) or no bytes (strings), so a value deleted by nulling
//! it cannot survive in the file.  Every valid float is finite.
//!
//! [`read_table`] reads a table *as stored* and records every structural
//! problem it meets with the diagnostic code the validator reports for it,
//! rather than stopping at the first: the validator wants them all, and the
//! typed reader refuses a table that has any.
//!
//! A UTF-8 column may be read **deferred** ([`read_table_deferring`]): its
//! offsets are read and checked against the byte buffer's stored length, and
//! the bytes stay in the file.  Document text is read that way --- by the
//! reader, by the validator and by a task's preflight --- so opening a sample
//! never decompresses a report.  A deferred column's cells are read one at a
//! time through a [`TextColumn`], each checked as UTF-8 when it is read, and
//! the validator checks every cell by streaming the buffer in bounded slabs
//! ([`TextColumn::scan`]).

use std::collections::BTreeSet;

use indexmap::IndexMap;

use crate::array::{DType, Index, NdArray, Slice};
use crate::h5::data::{self, Kind};
use crate::h5::ops;
use crate::storage::codecs::{dataset_layout, CodecProfile, Role};
use crate::{Error, Result};

use super::model::VALID;

/// How a logical column is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Utf8,
    I64,
    U64,
    F64,
}

impl ColumnType {
    /// The dtype a numeric column is stored in.
    pub fn dtype(self) -> Option<DType> {
        match self {
            ColumnType::Utf8 => None,
            ColumnType::I64 => Some(DType::I64),
            ColumnType::U64 => Some(DType::U64),
            ColumnType::F64 => Some(DType::F64),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ColumnType::Utf8 => "UTF-8",
            ColumnType::I64 => "int64",
            ColumnType::U64 => "uint64",
            ColumnType::F64 => "float64",
        }
    }
}

/// One column a table defines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnSpec {
    pub name: &'static str,
    pub ty: ColumnType,
    /// Non-null in every row, and so never masked.
    pub required: bool,
}

const fn col(name: &'static str, ty: ColumnType, required: bool) -> ColumnSpec {
    ColumnSpec { name, ty, required }
}

use ColumnType::{Utf8, F64, I64, U64};

/// `clinical/events` (§5.1), in the order the specification lists them.
pub const EVENT_COLUMNS: [ColumnSpec; 22] = [
    col("event_id", Utf8, true),
    col("record_id", Utf8, true),
    col("kind", Utf8, true),
    col("temporal_type", Utf8, true),
    col("effective_start_lo_us", I64, false),
    col("effective_start_hi_us", I64, false),
    col("effective_end_lo_us", I64, false),
    col("effective_end_hi_us", I64, false),
    col("available_lo_us", I64, false),
    col("available_hi_us", I64, false),
    col("status", Utf8, true),
    col("timepoint_id", Utf8, false),
    col("encounter_id", Utf8, false),
    col("code_system", Utf8, false),
    col("code", Utf8, false),
    col("code_version", Utf8, false),
    col("value_num", F64, false),
    col("value_comparator", Utf8, false),
    col("unit", Utf8, false),
    col("value_text", Utf8, false),
    col("missing_reason", Utf8, false),
    col("prov", Utf8, false),
];

/// `clinical/documents` (§6).
pub const DOCUMENT_COLUMNS: [ColumnSpec; 5] = [
    col("document_id", Utf8, true),
    col("media_type", Utf8, true),
    col("text", Utf8, true),
    col("language", Utf8, false),
    col("source_type", Utf8, false),
];

/// `clinical/links` (§7).
pub const LINK_COLUMNS: [ColumnSpec; 9] = [
    col("source_type", Utf8, true),
    col("source_id", Utf8, true),
    col("relation", Utf8, true),
    col("target_type", Utf8, true),
    col("target_id", Utf8, true),
    col("source_start", U64, false),
    col("source_end", U64, false),
    col("target_annotation_id", Utf8, false),
    col("asserted_by_event_id", Utf8, false),
];

/// The columns of a table, by its name under `clinical/`.
pub fn columns_of(table: &str) -> &'static [ColumnSpec] {
    match table {
        super::model::EVENTS => &EVENT_COLUMNS,
        super::model::DOCUMENTS => &DOCUMENT_COLUMNS,
        _ => &LINK_COLUMNS,
    }
}

/// A column's cells as stored.
#[derive(Debug, Clone, PartialEq)]
pub enum Values {
    Utf8 {
        data: Vec<u8>,
        offsets: Vec<u64>,
    },
    /// A UTF-8 column read by its offsets alone: the bytes stay in the file,
    /// and are read a cell at a time through a [`TextColumn`].
    Deferred {
        offsets: Vec<u64>,
    },
    I64(Vec<i64>),
    U64(Vec<u64>),
    F64(Vec<f64>),
}

/// One stored column and its validity mask.
#[derive(Debug, Clone, PartialEq)]
pub struct Column {
    pub spec: ColumnSpec,
    pub values: Values,
    /// `valid/<column>`; `None` means every cell is valid.
    pub mask: Option<Vec<u8>>,
}

impl Column {
    pub fn rows(&self) -> usize {
        match &self.values {
            Values::Utf8 { offsets, .. } | Values::Deferred { offsets } => offsets.len().saturating_sub(1),
            Values::I64(v) => v.len(),
            Values::U64(v) => v.len(),
            Values::F64(v) => v.len(),
        }
    }

    /// Whether row `i` holds a value.
    pub fn is_valid(&self, i: usize) -> bool {
        self.mask.as_ref().is_none_or(|m| m.get(i).copied() == Some(1))
    }

    /// Row `i`'s byte range in a string column's buffer (`None` when null).
    pub fn span(&self, i: usize) -> Option<(u64, u64)> {
        match &self.values {
            Values::Utf8 { offsets, .. } | Values::Deferred { offsets } if self.is_valid(i) => {
                Some((*offsets.get(i)?, *offsets.get(i + 1)?))
            }
            _ => None,
        }
    }

    /// Row `i`'s bytes, for a string column read whole (`None` for a deferred
    /// one, whose bytes are in the file).
    pub fn bytes(&self, i: usize) -> Option<&[u8]> {
        match &self.values {
            Values::Utf8 { data, offsets } if self.is_valid(i) => {
                let (a, b) = (*offsets.get(i)? as usize, *offsets.get(i + 1)? as usize);
                data.get(a..b)
            }
            _ => None,
        }
    }

    /// Row `i` as text (`None` when null; invalid UTF-8 is replaced).
    pub fn text(&self, i: usize) -> Option<String> {
        self.bytes(i).map(|b| String::from_utf8_lossy(b).into_owned())
    }

    pub fn i64(&self, i: usize) -> Option<i64> {
        match &self.values {
            Values::I64(v) if self.is_valid(i) => v.get(i).copied(),
            _ => None,
        }
    }

    pub fn u64(&self, i: usize) -> Option<u64> {
        match &self.values {
            Values::U64(v) if self.is_valid(i) => v.get(i).copied(),
            _ => None,
        }
    }

    pub fn f64(&self, i: usize) -> Option<f64> {
        match &self.values {
            Values::F64(v) if self.is_valid(i) => v.get(i).copied(),
            _ => None,
        }
    }
}

/// One structural finding: the code the validator reports for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub code: &'static str,
    pub location: String,
    pub message: String,
}

impl Problem {
    pub fn new(code: &'static str, location: impl Into<String>, message: impl Into<String>) -> Problem {
        Problem { code, location: location.into(), message: message.into() }
    }
}

/// A table as stored: the columns that could be read, its row count, and every
/// structural problem met reading it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RawTable {
    pub name: String,
    pub rows: usize,
    pub columns: IndexMap<String, Column>,
    pub problems: Vec<Problem>,
}

impl RawTable {
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    pub fn text(&self, column: &str, i: usize) -> Option<String> {
        self.column(column).and_then(|c| c.text(i))
    }

    pub fn i64(&self, column: &str, i: usize) -> Option<i64> {
        self.column(column).and_then(|c| c.i64(i))
    }

    pub fn u64(&self, column: &str, i: usize) -> Option<u64> {
        self.column(column).and_then(|c| c.u64(i))
    }

    pub fn f64(&self, column: &str, i: usize) -> Option<f64> {
        self.column(column).and_then(|c| c.f64(i))
    }

    /// Whether any problem was an error (anything but W913).
    pub fn is_sound(&self) -> bool {
        self.problems.iter().all(|p| p.code.starts_with('W'))
    }
}

fn read_vec<T: crate::array::Element>(ds: &hdf5::Dataset) -> Result<Vec<T>> {
    Ok(data::read(ds)?.cast::<T>().iter().copied().collect())
}

/// Whether a dataset's numbers are stored big-endian.  1.1 §4 stores a
/// numeric column little-endian; HDF5 converts one stored the other way on
/// read, so its values read correctly, and it is still another dtype than the
/// profile's (C11 of the 2.0 audit).
fn big_endian(ds: &hdf5::Dataset) -> bool {
    ds.dtype().is_ok_and(|t| t.size() > 1 && t.byte_order() == hdf5::datatype::ByteOrder::BigEndian)
}

/// What a dataset holds, when it is a 1-D numeric dataset.
fn numeric_1d(ds: &hdf5::Dataset) -> Option<DType> {
    match data::kind(ds) {
        Ok(Kind::Numeric(d)) if ds.ndim() == 1 => Some(d),
        _ => None,
    }
}

/// Read `clinical/<table>`.
///
/// `projection` reports members this engine does not define as W913 (a later
/// minor may define them) rather than E804.
pub fn read_table(group: &hdf5::Group, table: &str, specs: &[ColumnSpec], projection: bool) -> Result<RawTable> {
    read_table_deferring(group, table, specs, projection, &[])
}

/// [`read_table`], with the UTF-8 columns named in `deferred` read by their
/// offsets alone ([`Values::Deferred`]): every structural rule but the
/// cells' UTF-8 is checked, and no byte of their buffers is read.
pub fn read_table_deferring(
    group: &hdf5::Group,
    table: &str,
    specs: &[ColumnSpec],
    projection: bool,
    deferred: &[&str],
) -> Result<RawTable> {
    let base = format!("/clinical/{table}");
    let mut out = RawTable { name: table.to_string(), ..Default::default() };
    let members = ops::members(group)?;
    for name in &members {
        if name != VALID && !specs.iter().any(|s| s.name == name) {
            let (code, why) = if projection {
                ("W913", "is not defined by the clinical profile this engine implements; ignored in this projection")
            } else {
                ("E804", "is not a column the clinical profile defines")
            };
            out.problems.push(Problem::new(
                code,
                format!("{base}/{name}"),
                format!("{} {why}", crate::json::repr_str(name)),
            ));
        }
    }
    // The table's row count is the first required column's.
    let mut rows: Option<usize> = None;
    for spec in specs {
        let location = format!("{base}/{}", spec.name);
        if !ops::exists(group, spec.name) {
            if spec.required {
                out.problems.push(Problem::new("E804", location, format!("required column `{}` is absent", spec.name)));
            }
            continue;
        }
        let defer = deferred.contains(&spec.name);
        let values = match read_values(group, spec, &location, defer, projection, &mut out.problems)? {
            Some(v) => v,
            None => continue,
        };
        let column = Column { spec: *spec, values, mask: None };
        let n = column.rows();
        match rows {
            None => rows = Some(n),
            Some(expected) if expected != n => {
                out.problems.push(Problem::new(
                    "E805",
                    location,
                    format!("column `{}` has {n} rows; the table's other columns have {expected}", spec.name),
                ));
                continue;
            }
            _ => {}
        }
        out.columns.insert(spec.name.to_string(), column);
    }
    out.rows = rows.unwrap_or(0);
    read_masks(group, &base, specs, &mut out, projection)?;
    check_null_cells(&base, &mut out);
    check_finite(&base, &mut out);
    Ok(out)
}

fn read_values(
    group: &hdf5::Group,
    spec: &ColumnSpec,
    location: &str,
    defer: bool,
    projection: bool,
    problems: &mut Vec<Problem>,
) -> Result<Option<Values>> {
    match spec.ty {
        ColumnType::Utf8 => {
            let Some(column) = ops::child_group(group, spec.name) else {
                problems.push(Problem::new(
                    "E805",
                    location,
                    format!("`{}` is a UTF-8 column: a group holding `data` and `offsets`", spec.name),
                ));
                return Ok(None);
            };
            // A member this engine does not define is a later minor's under a
            // projection (1.1 §2.2), as at the table's level; it was E804
            // whatever the version (C06 of the 2.0 audit).
            for member in ops::members(&column)? {
                if member != "data" && member != "offsets" {
                    let (code, why) = if projection {
                        (
                            "W913",
                            "is not defined by the clinical profile this engine implements; ignored in this projection",
                        )
                    } else {
                        ("E804", "is not a member of a UTF-8 column, which holds only `data` and `offsets`")
                    };
                    problems.push(Problem::new(
                        code,
                        format!("{location}/{member}"),
                        format!("{} {why}", crate::json::repr_str(&member)),
                    ));
                }
            }
            let (Some(data_ds), Some(offsets_ds)) =
                (ops::child_dataset(&column, "data"), ops::child_dataset(&column, "offsets"))
            else {
                problems.push(Problem::new(
                    "E806",
                    location,
                    "a UTF-8 column needs both `data` and `offsets` datasets",
                ));
                return Ok(None);
            };
            if numeric_1d(&data_ds) != Some(DType::U8) {
                problems.push(Problem::new("E805", format!("{location}/data"), "`data` must be a 1-D uint8 dataset"));
                return Ok(None);
            }
            if numeric_1d(&offsets_ds) != Some(DType::U64) {
                problems.push(Problem::new(
                    "E805",
                    format!("{location}/offsets"),
                    "`offsets` must be a 1-D uint64 dataset",
                ));
                return Ok(None);
            }
            // The scalar columns' rule holds for offsets too (C11 of the 2.0
            // re-audit): a reader may take the buffer as stored.
            if big_endian(&offsets_ds) {
                problems.push(Problem::new(
                    "E805",
                    format!("{location}/offsets"),
                    "`offsets` is stored big-endian; §4 stores a UTF-8 column's offsets little-endian",
                ));
                return Ok(None);
            }
            // The buffer's length is its shape: checking the offsets reads no byte of it.
            let n_bytes = data_ds.shape().first().copied().unwrap_or(0) as u64;
            let offsets: Vec<u64> = read_vec(&offsets_ds)?;
            if offsets.is_empty() {
                problems.push(Problem::new("E806", format!("{location}/offsets"), "`offsets` must hold N + 1 entries"));
                return Ok(None);
            }
            let mut sound = offsets[0] == 0 && *offsets.last().unwrap_or(&0) == n_bytes;
            sound &= offsets.windows(2).all(|w| w[0] <= w[1]);
            if !sound {
                problems.push(Problem::new(
                    "E806",
                    format!("{location}/offsets"),
                    format!(
                        "offsets must start at 0, never decrease and end at the byte length {n_bytes} (they run {} .. {})",
                        offsets[0],
                        offsets.last().copied().unwrap_or(0)
                    ),
                ));
                return Ok(None);
            }
            if defer {
                return Ok(Some(Values::Deferred { offsets }));
            }
            let data: Vec<u8> = read_vec(&data_ds)?;
            for (i, w) in offsets.windows(2).enumerate() {
                if std::str::from_utf8(&data[w[0] as usize..w[1] as usize]).is_err() {
                    problems.push(Problem::new(
                        "E806",
                        format!("{location}#row={i}"),
                        format!("row {i} of `{}` is not valid UTF-8", spec.name),
                    ));
                    return Ok(None);
                }
            }
            Ok(Some(Values::Utf8 { data, offsets }))
        }
        numeric => {
            let Some(ds) = ops::child_dataset(group, spec.name) else {
                problems.push(Problem::new(
                    "E805",
                    location,
                    format!("`{}` is a {} column: a 1-D dataset", spec.name, numeric.name()),
                ));
                return Ok(None);
            };
            let expected = numeric.dtype().expect("numeric");
            if numeric_1d(&ds) != Some(expected) {
                let found = match data::kind(&ds) {
                    Ok(Kind::Numeric(d)) => format!("{}-D {}", ds.ndim(), d.name()),
                    Ok(Kind::Strings) => "strings".to_string(),
                    Ok(Kind::Other(t)) => t,
                    Err(e) => e.to_string(),
                };
                problems.push(Problem::new(
                    "E805",
                    location,
                    format!("`{}` must be a 1-D {} dataset, not {found}", spec.name, expected.name()),
                ));
                return Ok(None);
            }
            if big_endian(&ds) {
                problems.push(Problem::new(
                    "E805",
                    location,
                    format!("`{}` is stored big-endian; §4 stores a numeric column little-endian", spec.name),
                ));
                return Ok(None);
            }
            Ok(Some(match numeric {
                ColumnType::I64 => Values::I64(read_vec(&ds)?),
                ColumnType::U64 => Values::U64(read_vec(&ds)?),
                _ => Values::F64(read_vec(&ds)?),
            }))
        }
    }
}

fn read_masks(
    group: &hdf5::Group,
    base: &str,
    specs: &[ColumnSpec],
    out: &mut RawTable,
    projection: bool,
) -> Result<()> {
    if !ops::exists(group, VALID) {
        return Ok(());
    }
    let Some(valid) = ops::child_group(group, VALID) else {
        out.problems.push(Problem::new("E807", format!("{base}/{VALID}"), "`valid` must be a group of masks"));
        return Ok(());
    };
    for name in ops::members(&valid)? {
        let location = format!("{base}/{VALID}/{name}");
        let Some(spec) = specs.iter().find(|s| s.name == name) else {
            let code = if projection { "W913" } else { "E807" };
            out.problems.push(Problem::new(
                code,
                location,
                format!("a mask for `{name}`, which is not a column of this table"),
            ));
            continue;
        };
        if spec.required {
            out.problems.push(Problem::new(
                "E807",
                location,
                format!("`{name}` is a required column and is never null, so it has no mask"),
            ));
            continue;
        }
        let Some(column) = out.columns.get_mut(&name) else {
            if ops::exists(group, &name) {
                // The column exists but could not be read; that is reported.
                continue;
            }
            out.problems.push(Problem::new(
                "E807",
                location,
                format!("a mask for `{name}`, which is absent: an omitted optional column is already all null"),
            ));
            continue;
        };
        let Some(ds) = ops::child_dataset(&valid, &name).filter(|d| numeric_1d(d) == Some(DType::U8)) else {
            out.problems.push(Problem::new("E807", location, "a mask is a 1-D uint8 dataset"));
            continue;
        };
        let mask: Vec<u8> = read_vec(&ds)?;
        if mask.len() != out.rows {
            out.problems.push(Problem::new(
                "E807",
                location,
                format!("the mask has {} entries for {} rows", mask.len(), out.rows),
            ));
            continue;
        }
        if let Some(bad) = mask.iter().position(|v| *v > 1) {
            out.problems.push(Problem::new(
                "E807",
                location,
                format!("mask values are exactly 0 or 1; row {bad} holds {}", mask[bad]),
            ));
            continue;
        }
        column.mask = Some(mask);
    }
    Ok(())
}

/// A null cell holds zero or no bytes: nothing deleted by nulling survives.
fn check_null_cells(base: &str, out: &mut RawTable) {
    for (name, column) in &out.columns {
        let Some(mask) = &column.mask else { continue };
        let leaking =
            mask.iter().enumerate().filter(|(_, v)| **v == 0).map(|(i, _)| i).find(|i| match &column.values {
                Values::Utf8 { offsets, .. } | Values::Deferred { offsets } => offsets[*i] != offsets[*i + 1],
                Values::I64(v) => v[*i] != 0,
                Values::U64(v) => v[*i] != 0,
                Values::F64(v) => v[*i].to_bits() != 0,
            });
        if let Some(i) = leaking {
            out.problems.push(Problem::new(
                "E807",
                format!("{base}/{name}#row={i}"),
                format!("row {i} of `{name}` is null but its cell is not empty (a null number is 0, a null string has no bytes)"),
            ));
        }
    }
}

/// Every valid float is finite (§4); a null one is zero (above).
fn check_finite(base: &str, out: &mut RawTable) {
    for (name, column) in &out.columns {
        let Values::F64(values) = &column.values else { continue };
        if let Some(i) = (0..values.len()).find(|i| column.is_valid(*i) && !values[*i].is_finite()) {
            out.problems.push(Problem::new(
                "E808",
                format!("{base}/{name}#row={i}"),
                format!("row {i} of `{name}` is {}; clinical values are finite", values[i]),
            ));
        }
    }
}

// -- writing ----------------------------------------------------------------------------------

/// Writes one table's columns into a group, under a codec profile.
pub struct TableWriter<'a> {
    group: hdf5::Group,
    profile: &'a CodecProfile,
    rows: usize,
    masks: Option<hdf5::Group>,
}

impl<'a> TableWriter<'a> {
    pub fn new(group: hdf5::Group, profile: &'a CodecProfile, rows: usize) -> TableWriter<'a> {
        TableWriter { group, profile, rows, masks: None }
    }

    fn create(&self, group: &hdf5::Group, name: &str, array: NdArray) -> Result<()> {
        let layout = dataset_layout(&array.shape(), array.dtype().itemsize(), self.profile, Role::Aux, None);
        data::create(group, name, &array, &layout)?;
        Ok(())
    }

    /// Write `valid/<name>` when some cells are null; the caller has already
    /// decided the column is written at all.
    fn mask(&mut self, spec: &ColumnSpec, valid: &[bool]) -> Result<()> {
        if valid.iter().all(|v| *v) {
            return Ok(());
        }
        if spec.required {
            return Err(Error::coded("E807", format!("`{}` is required and cannot hold a null", spec.name)));
        }
        if self.masks.is_none() {
            self.masks = Some(self.group.create_group(VALID)?);
        }
        let values: Vec<u8> = valid.iter().map(|v| u8::from(*v)).collect();
        let masks = self.masks.clone().expect("created above");
        self.create(&masks, spec.name, NdArray::from_vec(&[values.len()], values)?)
    }

    fn check_len(&self, spec: &ColumnSpec, n: usize) -> Result<()> {
        if n != self.rows {
            return Err(Error::coded("E805", format!("column `{}` has {n} cells for {} rows", spec.name, self.rows)));
        }
        Ok(())
    }

    /// A UTF-8 column; omitted when optional and entirely null.
    pub fn utf8(&mut self, spec: &ColumnSpec, cells: &[Option<&str>]) -> Result<()> {
        self.check_len(spec, cells.len())?;
        if !spec.required && cells.iter().all(Option::is_none) {
            return Ok(());
        }
        let mut data: Vec<u8> = Vec::new();
        let mut offsets: Vec<u64> = Vec::with_capacity(cells.len() + 1);
        offsets.push(0);
        for cell in cells {
            if let Some(text) = cell {
                data.extend_from_slice(text.as_bytes());
            }
            offsets.push(data.len() as u64);
        }
        let column = self.group.create_group(spec.name)?;
        self.create(&column, "data", NdArray::from_vec(&[data.len()], data)?)?;
        self.create(&column, "offsets", NdArray::from_vec(&[offsets.len()], offsets)?)?;
        let valid: Vec<bool> = cells.iter().map(Option::is_some).collect();
        self.mask(spec, &valid)
    }

    fn numeric<T: crate::array::Element>(&mut self, spec: &ColumnSpec, cells: &[Option<T>]) -> Result<()> {
        self.check_len(spec, cells.len())?;
        if !spec.required && cells.iter().all(Option::is_none) {
            return Ok(());
        }
        let values: Vec<T> = cells.iter().map(|c| c.unwrap_or_default()).collect();
        self.create(&self.group, spec.name, NdArray::from_vec(&[values.len()], values)?)?;
        let valid: Vec<bool> = cells.iter().map(Option::is_some).collect();
        self.mask(spec, &valid)
    }

    pub fn i64(&mut self, spec: &ColumnSpec, cells: &[Option<i64>]) -> Result<()> {
        self.numeric(spec, cells)
    }

    pub fn u64(&mut self, spec: &ColumnSpec, cells: &[Option<u64>]) -> Result<()> {
        self.numeric(spec, cells)
    }

    pub fn f64(&mut self, spec: &ColumnSpec, cells: &[Option<f64>]) -> Result<()> {
        if let Some(bad) = cells.iter().flatten().find(|v| !v.is_finite()) {
            return Err(Error::coded("E808", format!("`{}` holds {bad}; clinical values are finite", spec.name)));
        }
        self.numeric(spec, cells)
    }
}

// -- deferred text --------------------------------------------------------------------------

/// How many bytes of a deferred column [`TextColumn::scan`] holds at once.
pub const SCAN_BYTES: usize = 1 << 20;

/// A deferred UTF-8 column: its offsets in memory, its bytes in the file.
///
/// The buffer's dataset stays open, so neighbouring cells share HDF5's chunk
/// cache; a cell is checked as UTF-8 when it is read (E806), so no invalid
/// text is ever returned, though nothing was decompressed to open the table.
#[derive(Debug, Clone)]
pub struct TextColumn {
    data: hdf5::Dataset,
    offsets: Vec<u64>,
    /// `/clinical/<table>/<column>`, for messages.
    location: String,
}

/// What [`TextColumn::scan`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Scan {
    /// The first row that is not valid UTF-8.
    pub invalid_row: Option<usize>,
    /// The probed byte positions that fall inside a character rather than
    /// before one.
    pub inside_character: BTreeSet<u64>,
}

/// Continue checking a UTF-8 stream with `piece`, where the previous piece
/// left `carry` (a character cut by a slab boundary); leave in `carry` the
/// character `piece` ends inside of, if any.  `false` on invalid UTF-8.
fn continue_utf8(carry: &mut Vec<u8>, piece: &[u8]) -> bool {
    let joined;
    let bytes: &[u8] = if carry.is_empty() {
        piece
    } else {
        joined = [carry.as_slice(), piece].concat();
        &joined
    };
    match std::str::from_utf8(bytes) {
        Ok(_) => {
            carry.clear();
            true
        }
        Err(e) if e.error_len().is_none() => {
            *carry = bytes[e.valid_up_to()..].to_vec();
            true
        }
        Err(_) => false,
    }
}

impl TextColumn {
    /// The column `name` of a table group, with the offsets
    /// [`read_table_deferring`] read and checked for it.
    pub fn open(table: &hdf5::Group, name: &str, offsets: Vec<u64>, location: impl Into<String>) -> Result<TextColumn> {
        let data = table.group(name)?.dataset("data")?;
        Ok(TextColumn { data, offsets, location: location.into() })
    }

    pub fn rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Row `i`'s byte range in the buffer.
    pub fn span(&self, i: usize) -> Option<(u64, u64)> {
        Some((*self.offsets.get(i)?, *self.offsets.get(i + 1)?))
    }

    fn read(&self, a: u64, b: u64) -> Result<Vec<u8>> {
        if a == b {
            return Ok(Vec::new());
        }
        let block = data::read_region(&self.data, &[Index::Slice(Slice::new(a as i64, b as i64))])?;
        Ok(block.cast::<u8>().iter().copied().collect())
    }

    fn invalid(&self, i: usize) -> Error {
        Error::coded("E806", format!("{}#row={i}: row {i} is not valid UTF-8", self.location))
    }

    /// Row `i`, read from the file now: its bytes, and no others.
    pub fn cell(&self, i: usize) -> Result<String> {
        let (a, b) =
            self.span(i).ok_or_else(|| Error::Index(format!("row {i} is outside a column of {} rows", self.rows())))?;
        String::from_utf8(self.read(a, b)?).map_err(|_| self.invalid(i))
    }

    /// Every row, in one read: what an export of the whole table needs.
    pub fn cells(&self) -> Result<Vec<String>> {
        let bytes = self.read(0, self.offsets.last().copied().unwrap_or(0))?;
        self.offsets
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                std::str::from_utf8(&bytes[w[0] as usize..w[1] as usize])
                    .map(str::to_string)
                    .map_err(|_| self.invalid(i))
            })
            .collect()
    }

    /// Stream the whole buffer through about `slab` bytes of memory: check
    /// that every row is UTF-8 on its own, and find which `probes` (absolute
    /// byte positions in the buffer, in any order) fall inside a character.
    /// Stops at the first invalid row.
    pub fn scan(&self, probes: &[u64], slab: usize) -> Result<Scan> {
        let mut out = Scan::default();
        let total = self.offsets.last().copied().unwrap_or(0);
        let mut probes: Vec<u64> = probes.iter().copied().filter(|p| *p < total).collect();
        probes.sort_unstable();
        probes.dedup();
        let (mut next_probe, mut row, mut at) = (0usize, 0usize, 0u64);
        let mut carry: Vec<u8> = Vec::new();
        let slab = slab.max(4) as u64;
        while at < total {
            let end = (at + slab).min(total);
            let block = self.read(at, end)?;
            while next_probe < probes.len() && probes[next_probe] < end {
                let p = probes[next_probe];
                if block[(p - at) as usize] & 0xC0 == 0x80 {
                    out.inside_character.insert(p);
                }
                next_probe += 1;
            }
            let mut pos = at;
            while pos < end {
                // The row holding byte `pos`: an empty row holds none.
                while self.offsets[row + 1] <= pos {
                    row += 1;
                }
                let row_end = self.offsets[row + 1];
                let stop = row_end.min(end);
                let valid = continue_utf8(&mut carry, &block[(pos - at) as usize..(stop - at) as usize]);
                // A row may continue into the next slab, but not end inside a character.
                if !valid || (stop == row_end && !carry.is_empty()) {
                    out.invalid_row = Some(row);
                    return Ok(out);
                }
                pos = stop;
            }
            at = end;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::codecs::resolve_profile;

    /// A one-column table of `cells`, its buffer chunked and compressed.
    fn table(dir: &std::path::Path, cells: &[Option<&str>]) -> (hdf5::File, hdf5::Group) {
        let file = hdf5::File::create(dir.join("t.h5")).unwrap();
        let group = file.create_group("documents").unwrap();
        let profile = resolve_profile(Some("training")).unwrap();
        let spec = DOCUMENT_COLUMNS[2]; // `text`, required
        let mut w = TableWriter::new(group.clone(), &profile, cells.len());
        w.utf8(&spec, cells).unwrap();
        (file, group)
    }

    fn offsets(group: &hdf5::Group) -> Vec<u64> {
        let raw = read_table_deferring(group, "documents", &DOCUMENT_COLUMNS[2..3], false, &["text"]).unwrap();
        match &raw.column("text").unwrap().values {
            Values::Deferred { offsets } => offsets.clone(),
            other => panic!("not deferred: {other:?}"),
        }
    }

    #[test]
    fn s4_a_deferred_column_reads_offsets_and_no_text() {
        let dir = tempfile::tempdir().unwrap();
        let (_file, group) = table(dir.path(), &[Some("first"), Some(""), Some("結節 14 mm")]);
        let offsets = offsets(&group);
        assert_eq!(offsets, [0, 5, 5, 5 + "結節 14 mm".len() as u64]);
        let column = TextColumn::open(&group, "text", offsets, "/clinical/documents/text").unwrap();
        assert_eq!(column.cell(2).unwrap(), "結節 14 mm");
        assert_eq!(column.cell(1).unwrap(), "");
        assert_eq!(column.cells().unwrap(), ["first", "", "結節 14 mm"]);
    }

    #[test]
    fn s4_offsets_are_checked_against_the_stored_length_without_reading_it() {
        let dir = tempfile::tempdir().unwrap();
        let (_file, group) = table(dir.path(), &[Some("abc"), Some("de")]);
        let offsets_ds = group.group("text").unwrap().dataset("offsets").unwrap();
        let mut bad = vec![0u64, 3, 9];
        offsets_ds.write(&bad).unwrap();
        let raw = read_table_deferring(&group, "documents", &DOCUMENT_COLUMNS[2..3], false, &["text"]).unwrap();
        assert_eq!(raw.problems.iter().map(|p| p.code).collect::<Vec<_>>(), ["E806"]);
        bad[2] = 5;
        offsets_ds.write(&bad).unwrap();
        let raw = read_table_deferring(&group, "documents", &DOCUMENT_COLUMNS[2..3], false, &["text"]).unwrap();
        assert!(raw.problems.is_empty());
    }

    #[test]
    fn s4_a_scan_holds_one_slab_and_checks_every_row() {
        let dir = tempfile::tempdir().unwrap();
        // Multibyte characters cut by every slab boundary, an empty row, and a
        // row longer than a slab.
        let rows = ["añb", "", "€€€€€€", "x", "日本語のテキスト"];
        let cells: Vec<Option<&str>> = rows.iter().map(|r| Some(*r)).collect();
        let (_file, group) = table(dir.path(), &cells);
        let column = TextColumn::open(&group, "text", offsets(&group), "/clinical/documents/text").unwrap();
        // 'ñ' is bytes 1..3 of row 0: byte 2 is inside it, byte 1 and 3 are not.
        for slab in [4, 5, 7, 1 << 20] {
            let scan = column.scan(&[1, 2, 3, 4, 99_999], slab).unwrap();
            assert_eq!(scan.invalid_row, None, "slab {slab}");
            assert_eq!(scan.inside_character.into_iter().collect::<Vec<_>>(), [2], "slab {slab}");
        }
    }

    #[test]
    fn s4_a_scan_finds_the_first_invalid_row_wherever_the_slab_ends() {
        let dir = tempfile::tempdir().unwrap();
        let (_file, group) = table(dir.path(), &[Some("ok"), Some("€uro"), Some("tail")]);
        let data = group.group("text").unwrap().dataset("data").unwrap();
        let mut bytes: Vec<u8> = data.read_raw().unwrap();
        // Truncate the euro sign's last byte into the next row: row 1 now ends
        // inside a character.
        bytes[2 + 2] = b'X';
        data.write(&bytes).unwrap();
        let offsets = offsets(&group);
        let column = TextColumn::open(&group, "text", offsets, "/clinical/documents/text").unwrap();
        for slab in [4, 5, 1 << 20] {
            assert_eq!(column.scan(&[], slab).unwrap().invalid_row, Some(1), "slab {slab}");
        }
        assert_eq!(column.cell(0).unwrap(), "ok");
        assert_eq!(column.cell(2).unwrap(), "tail");
        let err = column.cell(1).unwrap_err();
        assert_eq!(err.code(), Some("E806"));
    }

    #[test]
    fn s4_each_row_is_utf8_on_its_own() {
        // "ok€x": a valid stream, but offsets that cut the euro sign between
        // rows 1 and 2 make row 1 end inside a character.
        let dir = tempfile::tempdir().unwrap();
        let (_file, group) = table(dir.path(), &[Some("ok"), Some("€x")]);
        let column = TextColumn::open(&group, "text", vec![0, 2, 4, 6], "/clinical/documents/text").unwrap();
        for slab in [3, 4, 1 << 20] {
            assert_eq!(column.scan(&[], slab).unwrap().invalid_row, Some(1), "slab {slab}");
        }
    }
}
