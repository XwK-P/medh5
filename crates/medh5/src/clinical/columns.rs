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
    Utf8 { data: Vec<u8>, offsets: Vec<u64> },
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
            Values::Utf8 { offsets, .. } => offsets.len().saturating_sub(1),
            Values::I64(v) => v.len(),
            Values::U64(v) => v.len(),
            Values::F64(v) => v.len(),
        }
    }

    /// Whether row `i` holds a value.
    pub fn is_valid(&self, i: usize) -> bool {
        self.mask.as_ref().is_none_or(|m| m.get(i).copied() == Some(1))
    }

    /// Row `i`'s bytes, for a string column.
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
        let values = match read_values(group, spec, &location, &mut out.problems)? {
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
            for member in ops::members(&column)? {
                if member != "data" && member != "offsets" {
                    problems.push(Problem::new(
                        "E804",
                        format!("{location}/{member}"),
                        "a UTF-8 column holds only `data` and `offsets`",
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
            let data: Vec<u8> = read_vec(&data_ds)?;
            let offsets: Vec<u64> = read_vec(&offsets_ds)?;
            if offsets.is_empty() {
                problems.push(Problem::new("E806", format!("{location}/offsets"), "`offsets` must hold N + 1 entries"));
                return Ok(None);
            }
            let mut sound = offsets[0] == 0 && *offsets.last().unwrap_or(&0) == data.len() as u64;
            sound &= offsets.windows(2).all(|w| w[0] <= w[1]);
            if !sound {
                problems.push(Problem::new(
                    "E806",
                    format!("{location}/offsets"),
                    format!(
                        "offsets must start at 0, never decrease and end at the byte length {} (they run {} .. {})",
                        data.len(),
                        offsets[0],
                        offsets.last().copied().unwrap_or(0)
                    ),
                ));
                return Ok(None);
            }
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
                Values::Utf8 { offsets, .. } => offsets[*i] != offsets[*i + 1],
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

/// Read row `i` of a stored UTF-8 column without reading the whole column:
/// two offsets, then the bytes between them.  For document text, which the
/// training path reads only for the documents it selected (task-and-cache
/// contract §6).
pub fn read_cell(column: &hdf5::Group, offsets: &[u64], i: usize) -> Result<String> {
    let (Some(a), Some(b)) = (offsets.get(i), offsets.get(i + 1)) else {
        return Err(Error::Index(format!("row {i} is outside a column of {} rows", offsets.len().saturating_sub(1))));
    };
    if a == b {
        return Ok(String::new());
    }
    let data = column.dataset("data")?;
    let block = data::read_region(&data, &[Index::Slice(Slice::new(*a as i64, *b as i64))])?;
    let bytes: Vec<u8> = block.cast::<u8>().iter().copied().collect();
    String::from_utf8(bytes)
        .map_err(|_| Error::coded("E806", format!("row {i} of {} is not valid UTF-8", column.name())))
}
