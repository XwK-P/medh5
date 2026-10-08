//! A task's preflight as columns: what crosses into Python for a cohort of
//! any size.
//!
//! Rows do not carry their inputs; they index their subject's merged history,
//! which crosses once.  Strings cross *packed* --- one `bytes` buffer, its
//! `int32` offsets and a validity mask, the clinical tables' own encoding
//! (1.1 §4); a column that is entirely null crosses as `None` --- and numbers
//! as NumPy arrays, so a preflight of millions of selected events is a few
//! arrays rather than millions of Python objects.  `medh5/task.py` reads the
//! columns lazily: a row, an event or a selection becomes a Python object only
//! when it is asked for.
//!
//! The columns are built as the engine hands each subject over
//! ([`preflight_each`]), and the subject's records dropped: the cohort is
//! never held as records, only as these columns, which then move into NumPy
//! without a copy.

use std::collections::BTreeMap;
use std::path::Path;

use ndarray::Array2;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use serde_json::Value;

use medh5::clinical::model::{Bounds, Event, Link, COMPARATORS, EVENT_KINDS, STATUSES, TEMPORAL_TYPES};
use medh5::clinical::select::POLICIES;
use medh5::companion::task::TaskManifest;
use medh5::companion::view::{preflight_each, RowView, SlotFill, SubjectHistory, TargetLabel};
use medh5::companion::Finding;

use crate::convert::json_to_py;
use crate::errors::R;

/// Row statuses, in the order their codes count.
pub const ROW_STATUSES: [&str; 4] = ["eligible", "uncertifiable", "excluded", "error"];
/// Selection statuses.
pub const SELECTION_STATUSES: [&str; 3] = ["certified", "uncertifiable", "provable"];
/// Target statuses.
pub const TARGET_STATUSES: [&str; 5] = ["positive", "negative", "censored", "prevalent", "none"];
/// How a slot's region was centred.
pub const ROIS: [&str; 3] = ["center", "eligible_instances", "center_fallback"];

/// A packed UTF-8 column being built.
struct Packed {
    data: Vec<u8>,
    offsets: Vec<i32>,
    valid: Vec<bool>,
}

impl Packed {
    fn new() -> Packed {
        Packed { data: Vec::new(), offsets: vec![0], valid: Vec::new() }
    }

    fn push(&mut self, value: Option<&str>) {
        if let Some(v) = value {
            self.data.extend_from_slice(v.as_bytes());
        }
        self.offsets.push(i32::try_from(self.data.len()).expect("a packed column under 2 GiB"));
        self.valid.push(value.is_some());
    }

    fn of<'a>(values: impl IntoIterator<Item = Option<&'a str>>) -> Packed {
        let mut out = Packed::new();
        for v in values {
            out.push(v);
        }
        out
    }

    /// `(bytes, offsets, valid)`; `valid` is `None` when every cell is, and the
    /// whole column `None` when no cell is.
    fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        if !self.valid.is_empty() && self.valid.iter().all(|v| !*v) {
            return Ok(py.None().into_bound(py));
        }
        let valid: Bound<'_, PyAny> = if self.valid.iter().all(|v| *v) {
            py.None().into_bound(py)
        } else {
            PyArray1::from_vec(py, self.valid).into_any()
        };
        Ok(PyTuple::new(
            py,
            [PyBytes::new(py, &self.data).into_any(), PyArray1::from_vec(py, self.offsets).into_any(), valid],
        )?
        .into_any())
    }
}

/// A vocabulary column: each value's index in `vocabulary`, `-1` for null.
fn codes<'a>(values: impl IntoIterator<Item = Option<&'a str>>, vocabulary: &[&str]) -> Vec<i8> {
    values.into_iter().map(|v| v.map_or(-1, |v| code(v, vocabulary))).collect()
}

fn code(value: &str, vocabulary: &[&str]) -> i8 {
    vocabulary.iter().position(|w| *w == value).map_or(-1, |i| i as i8)
}

fn i32s(values: impl IntoIterator<Item = usize>) -> Vec<i32> {
    values.into_iter().map(|v| v as i32).collect()
}

/// One column of a table, compact until it crosses.
enum Column {
    Text(Packed),
    I8(Vec<i8>),
    I32(Vec<i32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    /// `(n, 2)` bounds, flat.
    Pairs(Vec<i64>),
}

impl Column {
    fn bounds(values: impl Iterator<Item = Option<Bounds>>) -> (Column, Column) {
        let (mut flat, mut known) = (Vec::new(), Vec::new());
        for b in values {
            flat.extend(b.map_or([0, 0], |b| [b.lo, b.hi]));
            known.push(b.is_some());
        }
        (Column::Pairs(flat), Column::Bool(known))
    }

    fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        Ok(match self {
            Column::Text(p) => p.into_py(py)?,
            Column::I8(v) => PyArray1::from_vec(py, v).into_any(),
            Column::I32(v) => PyArray1::from_vec(py, v).into_any(),
            Column::F64(v) => PyArray1::from_vec(py, v).into_any(),
            Column::Bool(v) => PyArray1::from_vec(py, v).into_any(),
            Column::Pairs(v) => pairs(py, v),
        })
    }
}

/// `(n, 2)` from flat pairs, moved rather than copied.
fn pairs(py: Python<'_>, flat: Vec<i64>) -> Bound<'_, PyAny> {
    let n = flat.len() / 2;
    numpy::PyArray::from_owned_array(py, Array2::from_shape_vec((n, 2), flat).expect("pairs")).into_any()
}

/// Named columns of `n` rows.
struct Table {
    n: usize,
    columns: Vec<(String, Column)>,
}

impl Table {
    fn new(n: usize) -> Table {
        Table { n, columns: Vec::new() }
    }

    fn add(&mut self, name: impl Into<String>, column: Column) {
        self.columns.push((name.into(), column));
    }

    fn into_py(self, py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("n", self.n)?;
        for (name, column) in self.columns {
            out.set_item(name, column.into_py(py)?)?;
        }
        Ok(out)
    }
}

fn event_table(events: &[Event], fragments: &[usize]) -> Table {
    let mut t = Table::new(events.len());
    let text = |f: fn(&Event) -> Option<&str>| Column::Text(Packed::of(events.iter().map(f)));
    t.add("event_id", text(|e| Some(e.event_id.as_str())));
    t.add("record_id", text(|e| Some(e.record_id.as_str())));
    for (name, f) in [
        ("timepoint_id", (|e: &Event| e.timepoint_id.as_deref()) as fn(&Event) -> Option<&str>),
        ("encounter_id", |e| e.encounter_id.as_deref()),
        ("code_system", |e| e.code_system.as_deref()),
        ("code", |e| e.code.as_deref()),
        ("code_version", |e| e.code_version.as_deref()),
        ("unit", |e| e.unit.as_deref()),
        ("value_text", |e| e.value_text.as_deref()),
        ("missing_reason", |e| e.missing_reason.as_deref()),
        ("prov", |e| e.prov.as_deref()),
    ] {
        t.add(name, text(f));
    }
    // Vocabulary columns cross twice: as text, which a record is rebuilt
    // from exactly, and as codes into the vocabulary, which a batch reads.
    for (name, vocabulary, f) in [
        ("kind", &EVENT_KINDS[..], (|e: &Event| Some(e.kind.as_str())) as fn(&Event) -> Option<&str>),
        ("temporal_type", &TEMPORAL_TYPES[..], |e| Some(e.temporal_type.as_str())),
        ("status", &STATUSES[..], |e| Some(e.status.as_str())),
        ("value_comparator", &COMPARATORS[..], |e| e.value_comparator.as_deref()),
    ] {
        t.add(name, text(f));
        t.add(format!("{name}_code"), Column::I8(codes(events.iter().map(f), vocabulary)));
    }
    for (name, f) in [
        ("effective_start", (|e: &Event| e.effective_start) as fn(&Event) -> Option<Bounds>),
        ("effective_end", |e| e.effective_end),
        ("available", |e| e.available),
    ] {
        let (values, known) = Column::bounds(events.iter().map(f));
        t.add(name, values);
        t.add(format!("{name}_known"), known);
    }
    t.add("value_num", Column::F64(events.iter().map(|e| e.value_num.unwrap_or(0.0)).collect()));
    t.add("value_num_valid", Column::Bool(events.iter().map(|e| e.value_num.is_some()).collect()));
    t.add("fragment", Column::I32(i32s(fragments.iter().copied())));
    t
}

fn link_table(links: &[(usize, Link)]) -> Table {
    let mut t = Table::new(links.len());
    t.add("fragment", Column::I32(i32s(links.iter().map(|(f, _)| *f))));
    let text = |f: fn(&Link) -> Option<&str>| Column::Text(Packed::of(links.iter().map(|(_, l)| f(l))));
    for (name, f) in [
        ("source_type", (|l: &Link| Some(l.source_type.as_str())) as fn(&Link) -> Option<&str>),
        ("source_id", |l| Some(l.source_id.as_str())),
        ("relation", |l| Some(l.relation.as_str())),
        ("target_type", |l| Some(l.target_type.as_str())),
        ("target_id", |l| Some(l.target_id.as_str())),
        ("target_annotation_id", |l| l.target_annotation_id.as_deref()),
        ("asserted_by_event_id", |l| l.asserted_by_event_id.as_deref()),
    ] {
        t.add(name, text(f));
    }
    let spans = links.iter().map(|(_, l)| l.source_span.map(|(a, b)| Bounds { lo: a as i64, hi: b as i64 }));
    let (values, known) = Column::bounds(spans);
    t.add("source_span", values);
    t.add("source_span_valid", known);
    t
}

/// One subject's history, compact: its records are dropped once this is built.
struct SubjectColumns {
    subject_id: String,
    partition: Option<String>,
    sources: Vec<Value>,
    events: Table,
    links: Table,
    documents: Table,
    payloads: Table,
}

/// One row, compact: its selection as indices.
struct RowColumns {
    row_id: String,
    subject_id: String,
    partition: Option<String>,
    fingerprint: String,
    subject: i32,
    cutoff_us: i64,
    status: i8,
    reasons: Vec<String>,
    selection: Option<SelectionColumns>,
    slots: Vec<SlotFill>,
    target: TargetLabel,
}

#[derive(Default)]
struct SelectionColumns {
    status: i8,
    policy: i8,
    index: Vec<i32>,
    order: Vec<i64>,
    order_known: Vec<bool>,
    tie_group: Vec<i32>,
    plan: Vec<bool>,
    links: Vec<i32>,
    payloads: Vec<i32>,
    uncertain: Vec<String>,
    excluded: BTreeMap<String, usize>,
}

fn row_columns(view: RowView, payload_index: &BTreeMap<(usize, String, String), i32>) -> RowColumns {
    let selection = view.selection.map(|s| {
        let mut out = SelectionColumns {
            status: code(&s.status, &SELECTION_STATUSES),
            policy: code(&s.policy, &POLICIES),
            links: i32s(s.links.iter().copied()),
            payloads: s.payloads.iter().map(|p| payload_index[p]).collect(),
            uncertain: s.uncertain_records,
            excluded: s.excluded,
            ..Default::default()
        };
        for e in &s.events {
            out.index.push(e.index as i32);
            out.order.extend(e.order.map_or([0, 0], |b| [b.lo, b.hi]));
            out.order_known.push(e.order.is_some());
            out.tie_group.push(e.tie_group as i32);
            out.plan.push(e.plan);
        }
        out
    });
    RowColumns {
        row_id: view.row_id,
        subject_id: view.subject_id,
        partition: view.partition,
        fingerprint: view.fingerprint,
        subject: view.subject.map_or(-1, |s| s as i32),
        cutoff_us: view.cutoff_us,
        status: code(&view.status, &ROW_STATUSES),
        reasons: view.reasons,
        selection,
        slots: view.slots,
        target: view.target,
    }
}

/// The preflight's columns, built a subject at a time.
struct Builder {
    slot_names: Vec<String>,
    subjects: Vec<SubjectColumns>,
    rows: Vec<Option<RowColumns>>,
}

impl Builder {
    /// Take one subject: compact its history and its rows' views, and let
    /// the records go.
    fn subject(&mut self, history: SubjectHistory, rows: Vec<(usize, RowView)>) {
        // The subject's payload table: every payload any of its rows admits.
        let mut payloads: BTreeMap<(usize, String, String), i32> = BTreeMap::new();
        for p in rows.iter().flat_map(|(_, v)| v.selection.iter().flat_map(|s| s.payloads.iter())) {
            payloads.entry(p.clone()).or_insert(0);
        }
        for (n, v) in payloads.values_mut().enumerate() {
            *v = n as i32;
        }
        let mut table = Table::new(payloads.len());
        table.add("fragment", Column::I32(i32s(payloads.keys().map(|p| p.0))));
        table.add("kind", Column::Text(Packed::of(payloads.keys().map(|p| Some(p.1.as_str())))));
        table.add("id", Column::Text(Packed::of(payloads.keys().map(|p| Some(p.2.as_str())))));
        let owned = &history.documents;
        let mut documents = Table::new(owned.len());
        documents.add("event", Column::I32(i32s(owned.iter().map(|d| d.0))));
        documents.add("fragment", Column::I32(i32s(owned.iter().map(|d| d.1))));
        documents.add("document_id", Column::Text(Packed::of(owned.iter().map(|d| Some(d.2.as_str())))));
        self.subjects.push(SubjectColumns {
            subject_id: history.subject_id.clone(),
            partition: history.partition.clone(),
            sources: history.sources.iter().map(|s| s.to_json()).collect(),
            events: event_table(&history.events, &history.event_fragments),
            links: link_table(&history.links),
            documents,
            payloads: table,
        });
        for (r, view) in rows {
            self.rows[r] = Some(row_columns(view, &payloads));
        }
    }

    fn into_py<'py>(
        self,
        py: Python<'py>,
        findings: &[Finding],
        fingerprints: (&str, &str),
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        out.set_item("task_fingerprint", fingerprints.0)?;
        out.set_item("manifest_fingerprint", fingerprints.1)?;
        let found: Vec<Bound<'_, PyAny>> =
            findings.iter().map(|f| json_to_py(py, &f.to_json())).collect::<PyResult<_>>()?;
        out.set_item("findings", PyList::new(py, found)?)?;
        let subjects = PyList::empty(py);
        for s in self.subjects {
            let d = PyDict::new(py);
            d.set_item("subject_id", s.subject_id)?;
            d.set_item("partition", s.partition)?;
            let sources: Vec<Bound<'_, PyAny>> =
                s.sources.iter().map(|v| json_to_py(py, v)).collect::<PyResult<_>>()?;
            d.set_item("sources", PyList::new(py, sources)?)?;
            d.set_item("events", s.events.into_py(py)?)?;
            d.set_item("links", s.links.into_py(py)?)?;
            d.set_item("documents", s.documents.into_py(py)?)?;
            d.set_item("payloads", s.payloads.into_py(py)?)?;
            subjects.append(d)?;
        }
        out.set_item("subjects", subjects)?;
        let rows: Vec<RowColumns> = self.rows.into_iter().map(|r| r.expect("every row")).collect();
        out.set_item("rows", rows_to_py(py, rows, &self.slot_names)?)?;
        Ok(out)
    }
}

/// Per-row arrays as offsets into one flat array, each row's taken from it.
fn flat<T>(rows: &mut [RowColumns], take: fn(&mut SelectionColumns) -> Vec<T>) -> (Vec<i64>, Vec<T>) {
    let mut offsets = vec![0i64];
    let mut values = Vec::new();
    for r in rows.iter_mut() {
        if let Some(s) = r.selection.as_mut() {
            values.extend(take(s));
        }
        offsets.push(values.len() as i64);
    }
    (offsets, values)
}

fn strings_csr<'py, 'a>(py: Python<'py>, rows: impl Iterator<Item = &'a [String]>) -> PyResult<Bound<'py, PyDict>> {
    let mut offsets = vec![0i64];
    let mut ids = Packed::new();
    for row in rows {
        for id in row {
            ids.push(Some(id));
        }
        offsets.push(ids.valid.len() as i64);
    }
    let out = PyDict::new(py);
    out.set_item("offsets", PyArray1::from_vec(py, offsets))?;
    // An empty column is no ids, not an all-null one.
    let ids = if ids.valid.is_empty() {
        (PyBytes::new(py, b""), PyArray1::from_vec(py, vec![0i32]), py.None()).into_pyobject(py)?.into_any()
    } else {
        ids.into_py(py)?
    };
    out.set_item("ids", ids)?;
    Ok(out)
}

fn rows_to_py<'py>(py: Python<'py>, mut rows: Vec<RowColumns>, slot_names: &[String]) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let n = rows.len();
    out.set_item("n", n)?;
    for (key, f) in [
        ("row_id", (|r: &RowColumns| Some(r.row_id.as_str())) as fn(&RowColumns) -> Option<&str>),
        ("subject_id", |r| Some(r.subject_id.as_str())),
        ("partition", |r| r.partition.as_deref()),
        ("fingerprint", |r| Some(r.fingerprint.as_str())),
    ] {
        out.set_item(key, Packed::of(rows.iter().map(f)).into_py(py)?)?;
    }
    out.set_item("subject", PyArray1::from_vec(py, rows.iter().map(|r| r.subject).collect()))?;
    out.set_item("cutoff_us", PyArray1::from_vec(py, rows.iter().map(|r| r.cutoff_us).collect()))?;
    out.set_item("status", PyArray1::from_vec(py, rows.iter().map(|r| r.status).collect()))?;
    let empty = PyTuple::empty(py);
    let tuple_of = |items: &[String]| -> PyResult<Bound<'py, PyTuple>> {
        if items.is_empty() {
            Ok(empty.clone())
        } else {
            PyTuple::new(py, items)
        }
    };
    let reasons: Vec<Bound<'_, PyTuple>> = rows.iter().map(|r| tuple_of(&r.reasons)).collect::<PyResult<_>>()?;
    out.set_item("reasons", PyList::new(py, reasons)?)?;

    // Selections: flat arrays, each row an offset range into them.
    let selection = |f: fn(&SelectionColumns) -> i8| -> Vec<i8> {
        rows.iter().map(|r| r.selection.as_ref().map_or(-1, f)).collect()
    };
    out.set_item("selection", PyArray1::from_vec(py, rows.iter().map(|r| r.selection.is_some()).collect()))?;
    out.set_item("selection_status", PyArray1::from_vec(py, selection(|s| s.status)))?;
    out.set_item("policy", PyArray1::from_vec(py, selection(|s| s.policy)))?;
    let uncertain: Vec<Bound<'_, PyTuple>> = rows
        .iter()
        .map(|r| r.selection.as_ref().map_or(Ok(empty.clone()), |s| tuple_of(&s.uncertain)))
        .collect::<PyResult<_>>()?;
    out.set_item("uncertain_records", PyList::new(py, uncertain)?)?;
    let mut keys: Vec<String> =
        rows.iter().flat_map(|r| r.selection.iter().flat_map(|s| s.excluded.keys().cloned())).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut counts = vec![0i64; n * keys.len()];
    for (i, r) in rows.iter().enumerate() {
        for (k, v) in r.selection.iter().flat_map(|s| s.excluded.iter()) {
            let j = keys.binary_search(k).expect("collected");
            counts[i * keys.len() + j] = *v as i64;
        }
    }
    out.set_item("excluded_keys", PyTuple::new(py, &keys)?)?;
    let counts = Array2::from_shape_vec((n, keys.len()), counts).expect("a count per key");
    out.set_item("excluded", numpy::PyArray::from_owned_array(py, counts))?;
    let events = PyDict::new(py);
    let (offsets, index) = flat(&mut rows, |s| std::mem::take(&mut s.index));
    events.set_item("offsets", PyArray1::from_vec(py, offsets))?;
    events.set_item("index", PyArray1::from_vec(py, index))?;
    events.set_item("order", pairs(py, flat(&mut rows, |s| std::mem::take(&mut s.order)).1))?;
    events
        .set_item("order_known", PyArray1::from_vec(py, flat(&mut rows, |s| std::mem::take(&mut s.order_known)).1))?;
    events.set_item("tie_group", PyArray1::from_vec(py, flat(&mut rows, |s| std::mem::take(&mut s.tie_group)).1))?;
    events.set_item("plan", PyArray1::from_vec(py, flat(&mut rows, |s| std::mem::take(&mut s.plan)).1))?;
    out.set_item("events", events)?;
    for (key, take) in [
        ("links", (|s: &mut SelectionColumns| std::mem::take(&mut s.links)) as fn(&mut SelectionColumns) -> Vec<i32>),
        ("payloads", |s| std::mem::take(&mut s.payloads)),
    ] {
        let (offsets, index) = flat(&mut rows, take);
        let d = PyDict::new(py);
        d.set_item("offsets", PyArray1::from_vec(py, offsets))?;
        d.set_item("index", PyArray1::from_vec(py, index))?;
        out.set_item(key, d)?;
    }

    // Rows the preflight did not fill slots for (an error, a manifest with
    // findings) have none, rather than empty ones.
    out.set_item("slotted", PyArray1::from_vec(py, rows.iter().map(|r| !r.slots.is_empty()).collect()))?;
    out.set_item("slots", slots_to_py(py, slot_names, &rows)?)?;
    let target = PyDict::new(py);
    let statuses = rows.iter().map(|r| code(&r.target.status, &TARGET_STATUSES)).collect();
    target.set_item("status", PyArray1::from_vec(py, statuses))?;
    target
        .set_item("value", PyArray1::from_vec(py, rows.iter().map(|r| r.target.value.unwrap_or(f64::NAN)).collect()))?;
    target.set_item("event_id", Packed::of(rows.iter().map(|r| r.target.event_id.as_deref())).into_py(py)?)?;
    target.set_item("reason", Packed::of(rows.iter().map(|r| r.target.reason.as_deref())).into_py(py)?)?;
    out.set_item("target", target)?;
    Ok(out)
}

fn slots_to_py<'py>(py: Python<'py>, names: &[String], rows: &[RowColumns]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    let none: &[String] = &[];
    for (k, name) in names.iter().enumerate() {
        let fills: Vec<Option<&SlotFill>> = rows.iter().map(|r| r.slots.get(k).filter(|f| &f.slot == name)).collect();
        let slot = PyDict::new(py);
        slot.set_item("name", name)?;
        let fragments = fills.iter().map(|f| f.and_then(|f| f.fragment).map_or(-1, |v| v as i32)).collect();
        slot.set_item("fragment", PyArray1::from_vec(py, fragments))?;
        for (key, f) in [
            ("image_id", (|f: &SlotFill| f.image_id.as_deref()) as fn(&SlotFill) -> Option<&str>),
            ("grid_id", |f| f.grid_id.as_deref()),
            ("event_id", |f| f.event_id.as_deref()),
        ] {
            slot.set_item(key, Packed::of(fills.iter().map(|x| x.and_then(f))).into_py(py)?)?;
        }
        let ndim = fills.iter().filter_map(|f| f.and_then(|f| f.center.as_ref()).map(Vec::len)).max().unwrap_or(0);
        let mut center = Vec::with_capacity(rows.len() * ndim);
        let mut center_ndim = Vec::with_capacity(rows.len());
        for f in &fills {
            let c = f.and_then(|f| f.center.as_ref());
            center_ndim.push(c.map_or(-1, |c| c.len() as i8));
            for i in 0..ndim {
                center.push(c.and_then(|c| c.get(i)).copied().unwrap_or(0));
            }
        }
        let array = Array2::from_shape_vec((rows.len(), ndim), center).expect("ndim per row");
        slot.set_item("center", numpy::PyArray::from_owned_array(py, array))?;
        slot.set_item("center_ndim", PyArray1::from_vec(py, center_ndim))?;
        slot.set_item("roi", PyArray1::from_vec(py, codes(fills.iter().map(|f| f.map(|f| f.roi.as_str())), &ROIS)))?;
        let annotations = fills.iter().map(|f| f.map_or(none, |f| f.annotations.as_slice()));
        slot.set_item("annotations", strings_csr(py, annotations)?)?;
        let labels = fills.iter().map(|f| f.map_or(none, |f| f.label_annotations.as_slice()));
        slot.set_item("label_annotations", strings_csr(py, labels)?)?;
        out.append(slot)?;
    }
    Ok(out)
}

/// Preflight `manifest` and return its columns: the engine runs without the
/// GIL, compacting each subject as it is handed over.
pub fn preflight_to_py<'py>(
    py: Python<'py>,
    manifest: TaskManifest,
    base: Option<&Path>,
    deep: bool,
) -> R<Bound<'py, PyDict>> {
    let slot_names: Vec<String> = manifest.slots.iter().map(|s| s.name.clone()).collect();
    let base = base.map(Path::to_path_buf);
    let (builder, done) = py.detach(move || -> medh5::Result<_> {
        let mut builder = Builder { slot_names, subjects: Vec::new(), rows: Vec::new() };
        builder.rows.resize_with(manifest.rows.len(), || None);
        let done = preflight_each(&manifest, base.as_deref(), deep, &mut |history, rows| {
            builder.subject(history, rows);
            Ok(())
        })?;
        for (slot, blank) in builder.rows.iter_mut().zip(&done.unclaimed) {
            if slot.is_none() {
                *slot = blank.clone().map(|view| row_columns(view, &BTreeMap::new()));
            }
        }
        Ok((builder, done))
    })?;
    let fingerprints = (done.task_fingerprint.as_str(), done.manifest_fingerprint.as_str());
    Ok(builder.into_py(py, &done.findings, fingerprints)?)
}

/// The vocabularies the columns' codes index.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    for (name, values) in [
        ("ROW_STATUSES", &ROW_STATUSES[..]),
        ("SELECTION_STATUSES", &SELECTION_STATUSES[..]),
        ("TARGET_STATUSES", &TARGET_STATUSES[..]),
        ("ROIS", &ROIS[..]),
    ] {
        m.add(name, PyTuple::new(py, values.iter().copied())?)?;
    }
    Ok(())
}
