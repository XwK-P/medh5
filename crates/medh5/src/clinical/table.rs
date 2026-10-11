//! The three clinical tables as logical records, and back (1.1 §3--§7).
//!
//! Writing sorts rows so that one logical content has one byte form: events by
//! known effective start and then id, with unknown times last (§4), documents
//! by id, links by their fields.  Physical order carries no meaning --- it is
//! never chronology and never an identifier --- so a reader that wants an
//! order asks for one ([`select`](super::select)).

use std::collections::HashMap;

use super::columns::{
    read_table_deferring, Column, RawTable, TableWriter, TextColumn, Values, DOCUMENT_COLUMNS, EVENT_COLUMNS,
    LINK_COLUMNS,
};
use super::model::{
    Bounds, ClinicalRecords, Descriptor, Document, Event, Link, DESCRIPTOR, DOCUMENTS, EVENTS, GROUP, LINKS, SCHEMA,
};
use crate::h5::data;
use crate::h5::ops;
use crate::json::repr_str;
use crate::storage::codecs::CodecProfile;
use crate::{Error, Result};

/// One column's cells, the column looked up once: absent is all null.
#[derive(Clone, Copy)]
struct Cells<'a>(Option<&'a Column>);

impl<'a> Cells<'a> {
    fn of(table: &'a RawTable, name: &str) -> Cells<'a> {
        Cells(table.column(name))
    }

    fn text(self, i: usize) -> Option<String> {
        self.0.and_then(|c| c.text(i))
    }

    fn i64(self, i: usize) -> Option<i64> {
        self.0.and_then(|c| c.i64(i))
    }

    fn u64(self, i: usize) -> Option<u64> {
        self.0.and_then(|c| c.u64(i))
    }

    fn f64(self, i: usize) -> Option<f64> {
        self.0.and_then(|c| c.f64(i))
    }
}

fn bounds(lo: Cells<'_>, hi: Cells<'_>, i: usize) -> Option<Bounds> {
    match (lo.i64(i), hi.i64(i)) {
        (Some(lo), Some(hi)) => Some(Bounds { lo, hi }),
        _ => None,
    }
}

/// The event rows of a sound `events` table.
pub fn events_of(table: &RawTable) -> Vec<Event> {
    let c = |name: &str| Cells::of(table, name);
    let (event_id, record_id, kind, temporal_type, status) =
        (c("event_id"), c("record_id"), c("kind"), c("temporal_type"), c("status"));
    let (start_lo, start_hi, end_lo, end_hi, available_lo, available_hi) = (
        c("effective_start_lo_us"),
        c("effective_start_hi_us"),
        c("effective_end_lo_us"),
        c("effective_end_hi_us"),
        c("available_lo_us"),
        c("available_hi_us"),
    );
    let (timepoint_id, encounter_id, code_system, code, code_version) =
        (c("timepoint_id"), c("encounter_id"), c("code_system"), c("code"), c("code_version"));
    let (value_num, value_comparator, unit, value_text, missing_reason, prov) =
        (c("value_num"), c("value_comparator"), c("unit"), c("value_text"), c("missing_reason"), c("prov"));
    (0..table.rows)
        .map(|i| Event {
            event_id: event_id.text(i).unwrap_or_default(),
            record_id: record_id.text(i).unwrap_or_default(),
            kind: kind.text(i).unwrap_or_default(),
            temporal_type: temporal_type.text(i).unwrap_or_default(),
            effective_start: bounds(start_lo, start_hi, i),
            effective_end: bounds(end_lo, end_hi, i),
            available: bounds(available_lo, available_hi, i),
            status: status.text(i).unwrap_or_default(),
            timepoint_id: timepoint_id.text(i),
            encounter_id: encounter_id.text(i),
            code_system: code_system.text(i),
            code: code.text(i),
            code_version: code_version.text(i),
            value_num: value_num.f64(i),
            value_comparator: value_comparator.text(i),
            unit: unit.text(i),
            value_text: value_text.text(i),
            missing_reason: missing_reason.text(i),
            prov: prov.text(i),
        })
        .collect()
}

/// The document rows of a sound `documents` table: text included when the
/// table was read whole, empty when its `text` was deferred (read that
/// through a [`TextColumn`]).
pub fn documents_of(table: &RawTable) -> Vec<Document> {
    let c = |name: &str| Cells::of(table, name);
    let (document_id, media_type, text, language, source_type) =
        (c("document_id"), c("media_type"), c("text"), c("language"), c("source_type"));
    (0..table.rows)
        .map(|i| Document {
            document_id: document_id.text(i).unwrap_or_default(),
            media_type: media_type.text(i).unwrap_or_default(),
            text: text.text(i).unwrap_or_default(),
            language: language.text(i),
            source_type: source_type.text(i),
        })
        .collect()
}

/// The link rows of a sound `links` table.
pub fn links_of(table: &RawTable) -> Vec<Link> {
    let c = |name: &str| Cells::of(table, name);
    let (source_type, source_id, relation, target_type, target_id) =
        (c("source_type"), c("source_id"), c("relation"), c("target_type"), c("target_id"));
    let (start, end, target_annotation_id, asserted_by_event_id) =
        (c("source_start"), c("source_end"), c("target_annotation_id"), c("asserted_by_event_id"));
    (0..table.rows)
        .map(|i| Link {
            source_type: source_type.text(i).unwrap_or_default(),
            source_id: source_id.text(i).unwrap_or_default(),
            relation: relation.text(i).unwrap_or_default(),
            target_type: target_type.text(i).unwrap_or_default(),
            target_id: target_id.text(i).unwrap_or_default(),
            source_span: match (start.u64(i), end.u64(i)) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            },
            target_annotation_id: target_annotation_id.text(i),
            asserted_by_event_id: asserted_by_event_id.text(i),
        })
        .collect()
}

/// The order events are stored in: known effective start, then id; unknown
/// times last (§4).
pub fn storage_order(events: &[Event]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..events.len()).collect();
    order.sort_by(|a, b| {
        let key =
            |e: &Event| (e.effective_start.is_none(), e.effective_start.map(|s| (s.lo, s.hi)), e.event_id.clone());
        key(&events[*a]).cmp(&key(&events[*b]))
    });
    order
}

/// Write `clinical/` under `root` from logical records.
///
/// The records are written as given; the rules they must satisfy are the
/// validator's, which the writer's commit runs over the finished file.
pub fn write(root: &hdf5::Group, records: &ClinicalRecords, profile: &CodecProfile) -> Result<()> {
    if ops::exists(root, GROUP) {
        return Err(Error::invalid(format!("{} already exists; it is replaced, never merged", repr_str(GROUP))));
    }
    let group = root.create_group(GROUP)?;
    data::create_scalar_string(&group, DESCRIPTOR, &records.descriptor.dumps())?;
    write_events(&group.create_group(EVENTS)?, &records.events, profile)?;
    if !records.documents.is_empty() {
        write_documents(&group.create_group(DOCUMENTS)?, &records.documents, profile)?;
    }
    if !records.links.is_empty() {
        write_links(&group.create_group(LINKS)?, &records.links, profile)?;
    }
    Ok(())
}

fn write_events(group: &hdf5::Group, events: &[Event], profile: &CodecProfile) -> Result<()> {
    let rows: Vec<&Event> = storage_order(events).into_iter().map(|i| &events[i]).collect();
    let mut w = TableWriter::new(group.clone(), profile, rows.len());
    let text = |f: fn(&Event) -> Option<&str>| -> Vec<Option<&str>> { rows.iter().map(|e| f(e)).collect() };
    let lo =
        |f: fn(&Event) -> Option<Bounds>| -> Vec<Option<i64>> { rows.iter().map(|e| f(e).map(|b| b.lo)).collect() };
    let hi =
        |f: fn(&Event) -> Option<Bounds>| -> Vec<Option<i64>> { rows.iter().map(|e| f(e).map(|b| b.hi)).collect() };
    for spec in &EVENT_COLUMNS {
        match spec.name {
            "event_id" => w.utf8(spec, &text(|e| Some(e.event_id.as_str())))?,
            "record_id" => w.utf8(spec, &text(|e| Some(e.record_id.as_str())))?,
            "kind" => w.utf8(spec, &text(|e| Some(e.kind.as_str())))?,
            "temporal_type" => w.utf8(spec, &text(|e| Some(e.temporal_type.as_str())))?,
            "effective_start_lo_us" => w.i64(spec, &lo(|e| e.effective_start))?,
            "effective_start_hi_us" => w.i64(spec, &hi(|e| e.effective_start))?,
            "effective_end_lo_us" => w.i64(spec, &lo(|e| e.effective_end))?,
            "effective_end_hi_us" => w.i64(spec, &hi(|e| e.effective_end))?,
            "available_lo_us" => w.i64(spec, &lo(|e| e.available))?,
            "available_hi_us" => w.i64(spec, &hi(|e| e.available))?,
            "status" => w.utf8(spec, &text(|e| Some(e.status.as_str())))?,
            "timepoint_id" => w.utf8(spec, &text(|e| e.timepoint_id.as_deref()))?,
            "encounter_id" => w.utf8(spec, &text(|e| e.encounter_id.as_deref()))?,
            "code_system" => w.utf8(spec, &text(|e| e.code_system.as_deref()))?,
            "code" => w.utf8(spec, &text(|e| e.code.as_deref()))?,
            "code_version" => w.utf8(spec, &text(|e| e.code_version.as_deref()))?,
            "value_num" => w.f64(spec, &rows.iter().map(|e| e.value_num).collect::<Vec<_>>())?,
            "value_comparator" => w.utf8(spec, &text(|e| e.value_comparator.as_deref()))?,
            "unit" => w.utf8(spec, &text(|e| e.unit.as_deref()))?,
            "value_text" => w.utf8(spec, &text(|e| e.value_text.as_deref()))?,
            "missing_reason" => w.utf8(spec, &text(|e| e.missing_reason.as_deref()))?,
            "prov" => w.utf8(spec, &text(|e| e.prov.as_deref()))?,
            other => unreachable!("event column {other} has no writer"),
        }
    }
    Ok(())
}

fn write_documents(group: &hdf5::Group, documents: &[Document], profile: &CodecProfile) -> Result<()> {
    let mut rows: Vec<&Document> = documents.iter().collect();
    rows.sort_by(|a, b| a.document_id.cmp(&b.document_id));
    let mut w = TableWriter::new(group.clone(), profile, rows.len());
    let text = |f: fn(&Document) -> Option<&str>| -> Vec<Option<&str>> { rows.iter().map(|d| f(d)).collect() };
    for spec in &DOCUMENT_COLUMNS {
        match spec.name {
            "document_id" => w.utf8(spec, &text(|d| Some(d.document_id.as_str())))?,
            "media_type" => w.utf8(spec, &text(|d| Some(d.media_type.as_str())))?,
            "text" => w.utf8(spec, &text(|d| Some(d.text.as_str())))?,
            "language" => w.utf8(spec, &text(|d| d.language.as_deref()))?,
            "source_type" => w.utf8(spec, &text(|d| d.source_type.as_deref()))?,
            other => unreachable!("document column {other} has no writer"),
        }
    }
    Ok(())
}

fn write_links(group: &hdf5::Group, links: &[Link], profile: &CodecProfile) -> Result<()> {
    let mut rows: Vec<&Link> = links.iter().collect();
    rows.sort();
    let mut w = TableWriter::new(group.clone(), profile, rows.len());
    let text = |f: fn(&Link) -> Option<&str>| -> Vec<Option<&str>> { rows.iter().map(|l| f(l)).collect() };
    for spec in &LINK_COLUMNS {
        match spec.name {
            "source_type" => w.utf8(spec, &text(|l| Some(l.source_type.as_str())))?,
            "source_id" => w.utf8(spec, &text(|l| Some(l.source_id.as_str())))?,
            "relation" => w.utf8(spec, &text(|l| Some(l.relation.as_str())))?,
            "target_type" => w.utf8(spec, &text(|l| Some(l.target_type.as_str())))?,
            "target_id" => w.utf8(spec, &text(|l| Some(l.target_id.as_str())))?,
            "source_start" => w.u64(spec, &rows.iter().map(|l| l.source_span.map(|s| s.0)).collect::<Vec<_>>())?,
            "source_end" => w.u64(spec, &rows.iter().map(|l| l.source_span.map(|s| s.1)).collect::<Vec<_>>())?,
            "target_annotation_id" => w.utf8(spec, &text(|l| l.target_annotation_id.as_deref()))?,
            "asserted_by_event_id" => w.utf8(spec, &text(|l| l.asserted_by_event_id.as_deref()))?,
            other => unreachable!("link column {other} has no writer"),
        }
    }
    Ok(())
}

/// Whether `root` holds a `clinical/meta` this engine recognises as a
/// `medh5.clinical/1` descriptor --- what makes a `clinical` group the
/// profile's rather than somebody's 1.0 extension (1.1 §3, §10).
pub fn recognised(root: &hdf5::Group) -> bool {
    let Some(group) = ops::child_group(root, GROUP) else { return false };
    let Some(meta) = ops::child_dataset(&group, DESCRIPTOR) else { return false };
    let Ok(text) = data::read_scalar_string(&meta) else { return false };
    match crate::json::loads(&text) {
        Ok(value) => value.get("schema").and_then(serde_json::Value::as_str) == Some(SCHEMA),
        Err(_) => false,
    }
}

/// A document's metadata; its text is read on demand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentInfo {
    pub document_id: String,
    pub media_type: String,
    pub language: Option<String>,
    pub source_type: Option<String>,
    /// The text's length in UTF-8 bytes --- what a span is measured in.
    pub n_bytes: u64,
    row: usize,
}

/// A sample's documents: their metadata and their text's offsets.
///
/// No byte of text is read to open them; a document's text is read --- and
/// checked as UTF-8 --- when it is asked for, so reading one report
/// decompresses no other, and needs neither the events nor the links.
#[derive(Debug, Default)]
pub struct Documents {
    infos: Vec<DocumentInfo>,
    by_id: HashMap<String, usize>,
    text: Option<TextColumn>,
}

impl Documents {
    /// The documents under `root`, and nothing else of the profile; `None`
    /// when `root` has no `clinical` group, empty without a documents table.
    pub fn open(root: &hdf5::Group, projection: bool) -> Result<Option<Documents>> {
        let Some(group) = ops::child_group(root, GROUP) else { return Ok(None) };
        let table = match ops::child_group(&group, DOCUMENTS) {
            None => None,
            Some(g) => {
                let table = read_table_deferring(&g, DOCUMENTS, &DOCUMENT_COLUMNS, projection, &["text"])?;
                refuse(&table)?;
                Some(table)
            }
        };
        Documents::from_table(root, table.as_ref()).map(Some)
    }

    /// From a documents table read with its `text` deferred.
    fn from_table(root: &hdf5::Group, table: Option<&RawTable>) -> Result<Documents> {
        let Some(table) = table else { return Ok(Documents::default()) };
        refuse(table)?;
        let column = table.column("text");
        let text = match column.map(|c| &c.values) {
            Some(Values::Deferred { offsets }) => {
                let g = root.group(&format!("{GROUP}/{DOCUMENTS}"))?;
                Some(TextColumn::open(&g, "text", offsets.clone(), "/clinical/documents/text")?)
            }
            Some(_) => return Err(Error::invalid("document text is read on demand: defer the `text` column")),
            None => None,
        };
        let c = |name: &str| Cells::of(table, name);
        let (document_id, media_type, language, source_type) =
            (c("document_id"), c("media_type"), c("language"), c("source_type"));
        let infos: Vec<DocumentInfo> = (0..table.rows)
            .map(|i| DocumentInfo {
                document_id: document_id.text(i).unwrap_or_default(),
                media_type: media_type.text(i).unwrap_or_default(),
                language: language.text(i),
                source_type: source_type.text(i),
                n_bytes: column.and_then(|c| c.span(i)).map_or(0, |(a, b)| b - a),
                row: i,
            })
            .collect();
        let by_id = infos.iter().enumerate().map(|(i, d)| (d.document_id.clone(), i)).collect();
        Ok(Documents { infos, by_id, text })
    }

    pub fn infos(&self) -> &[DocumentInfo] {
        &self.infos
    }

    pub fn len(&self) -> usize {
        self.infos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    pub fn info(&self, document_id: &str) -> Option<&DocumentInfo> {
        self.by_id.get(document_id).map(|i| &self.infos[*i])
    }

    /// One document's text, read from the file now: its bytes and no other
    /// document's, checked as UTF-8 (E806).
    pub fn text(&self, document_id: &str) -> Result<String> {
        let info = self.info(document_id).ok_or_else(|| {
            crate::sample::reader::missing_key(
                "document",
                document_id,
                self.infos.iter().map(|d| d.document_id.clone()),
            )
        })?;
        match &self.text {
            Some(column) => column.cell(info.row),
            None => Ok(String::new()),
        }
    }

    /// One document, text included.
    pub fn document(&self, document_id: &str) -> Result<Document> {
        let text = self.text(document_id)?;
        let info = self.info(document_id).expect("found above");
        Ok(Document {
            document_id: info.document_id.clone(),
            media_type: info.media_type.clone(),
            text,
            language: info.language.clone(),
            source_type: info.source_type.clone(),
        })
    }

    /// Every document, text included, in one read: what an export needs.
    pub fn all(&self) -> Result<Vec<Document>> {
        let mut texts = match &self.text {
            Some(column) => column.cells()?,
            None => Vec::new(),
        };
        Ok(self
            .infos
            .iter()
            .map(|d| Document {
                document_id: d.document_id.clone(),
                media_type: d.media_type.clone(),
                text: texts.get_mut(d.row).map(std::mem::take).unwrap_or_default(),
                language: d.language.clone(),
                source_type: d.source_type.clone(),
            })
            .collect())
    }
}

/// The clinical profile of one sample, read for use.
///
/// Events and links are read whole --- they are small and selection needs all
/// of them.  Documents are read by their metadata and their text's offsets
/// ([`Documents`]): no byte of text is read to open the profile, so selecting
/// at a cutoff never decompresses a report.
#[derive(Debug)]
pub struct Clinical {
    pub descriptor: Descriptor,
    pub events: Vec<Event>,
    pub links: Vec<Link>,
    /// Read from a higher minor, as the supported projection (1.1 §2.2).
    pub projection: bool,
    documents: Documents,
    by_event: HashMap<String, usize>,
}

fn refuse(table: &RawTable) -> Result<()> {
    match table.problems.iter().find(|p| !p.code.starts_with('W')) {
        None => Ok(()),
        Some(p) => Err(Error::coded(
            p.code,
            format!(
                "{}: {} (the clinical tables are malformed; `medh5 validate` lists every problem)",
                p.location, p.message
            ),
        )),
    }
}

impl Clinical {
    /// Read the profile under `root`; `None` when `root` has no `clinical`
    /// group.  A malformed table is an error naming its first problem.
    pub fn open(root: &hdf5::Group, projection: bool) -> Result<Option<Clinical>> {
        let Some(group) = ops::child_group(root, GROUP) else { return Ok(None) };
        let meta =
            ops::child_dataset(&group, DESCRIPTOR).ok_or_else(|| Error::coded("E801", "`clinical/meta` is absent"))?;
        let descriptor = Descriptor::loads(&data::read_scalar_string(&meta)?)?;
        let read = |name: &str, specs, deferred: &[&str]| -> Result<Option<RawTable>> {
            match ops::child_group(&group, name) {
                None => Ok(None),
                Some(g) => {
                    let table = read_table_deferring(&g, name, specs, projection, deferred)?;
                    refuse(&table)?;
                    Ok(Some(table))
                }
            }
        };
        let events = read(EVENTS, &EVENT_COLUMNS[..], &[])?
            .ok_or_else(|| Error::coded("E804", "`clinical/events` is absent"))?;
        let links = read(LINKS, &LINK_COLUMNS[..], &[])?;
        // Documents: metadata and offsets now, text on demand.
        let documents = read(DOCUMENTS, &DOCUMENT_COLUMNS[..], &["text"])?;
        Clinical::from_tables(root, descriptor, &events, documents.as_ref(), links.as_ref(), projection).map(Some)
    }

    /// The profile from tables already read --- by [`Clinical::open`], or by
    /// the validator ([`clinical_checked`](crate::validate::clinical_checked)),
    /// so a source that is checked and then used is read once.  The tables
    /// must be sound; a documents table's `text` must have been deferred.
    pub fn from_tables(
        root: &hdf5::Group,
        descriptor: Descriptor,
        events: &RawTable,
        documents: Option<&RawTable>,
        links: Option<&RawTable>,
        projection: bool,
    ) -> Result<Clinical> {
        for table in [Some(events), links].into_iter().flatten() {
            refuse(table)?;
        }
        let documents = Documents::from_table(root, documents)?;
        let events = events_of(events);
        let links = links.map(links_of).unwrap_or_default();
        let by_event = events.iter().enumerate().map(|(i, e)| (e.event_id.clone(), i)).collect();
        Ok(Clinical { descriptor, events, links, projection, documents, by_event })
    }

    pub fn event(&self, event_id: &str) -> Option<&Event> {
        self.by_event.get(event_id).map(|i| &self.events[*i])
    }

    /// The documents' metadata (no text).
    pub fn documents(&self) -> &[DocumentInfo] {
        self.documents.infos()
    }

    pub fn document_info(&self, document_id: &str) -> Option<&DocumentInfo> {
        self.documents.info(document_id)
    }

    /// One document's text, read from the file now: its bytes and no other
    /// document's, checked as UTF-8 (E806).
    pub fn text(&self, document_id: &str) -> Result<String> {
        self.documents.text(document_id)
    }

    /// One document, text included.
    pub fn document(&self, document_id: &str) -> Result<Document> {
        self.documents.document(document_id)
    }

    /// Everything, as logical records: every document's text, in one read.
    pub fn records(&self) -> Result<ClinicalRecords> {
        Ok(ClinicalRecords {
            descriptor: self.descriptor.clone(),
            events: self.events.clone(),
            documents: self.documents.all()?,
            links: self.links.clone(),
        })
    }

    /// Links leaving `(kind, id)`.
    pub fn links_from<'a>(&'a self, source_type: &'a str, source_id: &'a str) -> impl Iterator<Item = &'a Link> + 'a {
        self.links.iter().filter(move |l| l.source_type == source_type && l.source_id == source_id)
    }

    /// Links arriving at `(kind, id)`.
    pub fn links_to<'a>(&'a self, target_type: &'a str, target_id: &'a str) -> impl Iterator<Item = &'a Link> + 'a {
        self.links.iter().filter(move |l| l.target_type == target_type && l.target_id == target_id)
    }

    /// A compact description, for `medh5 info`.
    pub fn summary(&self) -> serde_json::Value {
        let mut kinds: std::collections::BTreeMap<&str, usize> = Default::default();
        for e in &self.events {
            *kinds.entry(e.kind.as_str()).or_default() += 1;
        }
        let records: std::collections::BTreeSet<&str> = self.events.iter().map(|e| e.record_id.as_str()).collect();
        serde_json::json!({
            "clock": self.descriptor.clock.to_json(),
            "events": self.events.len(),
            "records": records.len(),
            "kinds": kinds,
            "documents": self.documents.len(),
            "links": self.links.len(),
            "unknown_availability": self.events.iter().filter(|e| e.available.is_none()).count(),
        })
    }
}
