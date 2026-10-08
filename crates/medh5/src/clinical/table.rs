//! The three clinical tables as logical records, and back (1.1 §3--§7).
//!
//! Writing sorts rows so that one logical content has one byte form: events by
//! known effective start and then id, with unknown times last (§4), documents
//! by id, links by their fields.  Physical order carries no meaning --- it is
//! never chronology and never an identifier --- so a reader that wants an
//! order asks for one ([`select`](super::select)).

use std::collections::HashMap;

use super::columns::{read_cell, read_table, RawTable, TableWriter, DOCUMENT_COLUMNS, EVENT_COLUMNS, LINK_COLUMNS};
use super::model::{
    Bounds, ClinicalRecords, Descriptor, Document, Event, Link, DESCRIPTOR, DOCUMENTS, EVENTS, GROUP, LINKS, SCHEMA,
};
use crate::h5::data;
use crate::h5::ops;
use crate::json::repr_str;
use crate::storage::codecs::CodecProfile;
use crate::{Error, Result};

fn bounds(table: &RawTable, i: usize, lo: &str, hi: &str) -> Option<Bounds> {
    match (table.i64(lo, i), table.i64(hi, i)) {
        (Some(lo), Some(hi)) => Some(Bounds { lo, hi }),
        _ => None,
    }
}

/// The event rows of a sound `events` table.
pub fn events_of(table: &RawTable) -> Vec<Event> {
    (0..table.rows)
        .map(|i| {
            let text = |c: &str| table.text(c, i);
            Event {
                event_id: text("event_id").unwrap_or_default(),
                record_id: text("record_id").unwrap_or_default(),
                kind: text("kind").unwrap_or_default(),
                temporal_type: text("temporal_type").unwrap_or_default(),
                effective_start: bounds(table, i, "effective_start_lo_us", "effective_start_hi_us"),
                effective_end: bounds(table, i, "effective_end_lo_us", "effective_end_hi_us"),
                available: bounds(table, i, "available_lo_us", "available_hi_us"),
                status: text("status").unwrap_or_default(),
                timepoint_id: text("timepoint_id"),
                encounter_id: text("encounter_id"),
                code_system: text("code_system"),
                code: text("code"),
                code_version: text("code_version"),
                value_num: table.f64("value_num", i),
                value_comparator: text("value_comparator"),
                unit: text("unit"),
                value_text: text("value_text"),
                missing_reason: text("missing_reason"),
                prov: text("prov"),
            }
        })
        .collect()
}

/// The document rows of a sound `documents` table, text included.
pub fn documents_of(table: &RawTable) -> Vec<Document> {
    (0..table.rows)
        .map(|i| Document {
            document_id: table.text("document_id", i).unwrap_or_default(),
            media_type: table.text("media_type", i).unwrap_or_default(),
            text: table.text("text", i).unwrap_or_default(),
            language: table.text("language", i),
            source_type: table.text("source_type", i),
        })
        .collect()
}

/// The link rows of a sound `links` table.
pub fn links_of(table: &RawTable) -> Vec<Link> {
    (0..table.rows)
        .map(|i| Link {
            source_type: table.text("source_type", i).unwrap_or_default(),
            source_id: table.text("source_id", i).unwrap_or_default(),
            relation: table.text("relation", i).unwrap_or_default(),
            target_type: table.text("target_type", i).unwrap_or_default(),
            target_id: table.text("target_id", i).unwrap_or_default(),
            source_span: match (table.u64("source_start", i), table.u64("source_end", i)) {
                (Some(a), Some(b)) => Some((a, b)),
                _ => None,
            },
            target_annotation_id: table.text("target_annotation_id", i),
            asserted_by_event_id: table.text("asserted_by_event_id", i),
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

/// The clinical profile of one sample, read for use.
///
/// Events and links are read whole --- they are small and selection needs all
/// of them; document text is read one document at a time, when asked for,
/// so selecting at a cutoff never decompresses a report it does not use.
#[derive(Debug)]
pub struct Clinical {
    pub descriptor: Descriptor,
    pub events: Vec<Event>,
    pub links: Vec<Link>,
    pub documents: Vec<DocumentInfo>,
    /// Read from a higher minor, as the supported projection (1.1 §2.2).
    pub projection: bool,
    text: Option<(hdf5::Group, Vec<u64>)>,
    by_event: HashMap<String, usize>,
    by_document: HashMap<String, usize>,
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
        let read = |name: &str, specs| -> Result<Option<RawTable>> {
            match ops::child_group(&group, name) {
                None => Ok(None),
                Some(g) => {
                    let table = read_table(&g, name, specs, projection)?;
                    refuse(&table)?;
                    Ok(Some(table))
                }
            }
        };
        let events =
            read(EVENTS, &EVENT_COLUMNS[..])?.ok_or_else(|| Error::coded("E804", "`clinical/events` is absent"))?;
        let links = read(LINKS, &LINK_COLUMNS[..])?;
        // Documents: metadata now, text on demand.
        let mut documents = Vec::new();
        let mut text = None;
        if let Some(g) = ops::child_group(&group, DOCUMENTS) {
            let table = read_table(&g, DOCUMENTS, &DOCUMENT_COLUMNS, projection)?;
            refuse(&table)?;
            if let Some(column) = table.column("text") {
                if let super::columns::Values::Utf8 { offsets, .. } = &column.values {
                    text = Some((g.group("text")?, offsets.clone()));
                }
            }
            for i in 0..table.rows {
                let n_bytes = table.column("text").and_then(|c| c.bytes(i)).map_or(0, |b| b.len() as u64);
                documents.push(DocumentInfo {
                    document_id: table.text("document_id", i).unwrap_or_default(),
                    media_type: table.text("media_type", i).unwrap_or_default(),
                    language: table.text("language", i),
                    source_type: table.text("source_type", i),
                    n_bytes,
                    row: i,
                });
            }
        }
        let events = events_of(&events);
        let links = links.as_ref().map(links_of).unwrap_or_default();
        let by_event = events.iter().enumerate().map(|(i, e)| (e.event_id.clone(), i)).collect();
        let by_document = documents.iter().enumerate().map(|(i, d)| (d.document_id.clone(), i)).collect();
        Ok(Some(Clinical { descriptor, events, links, documents, projection, text, by_event, by_document }))
    }

    pub fn event(&self, event_id: &str) -> Option<&Event> {
        self.by_event.get(event_id).map(|i| &self.events[*i])
    }

    pub fn document_info(&self, document_id: &str) -> Option<&DocumentInfo> {
        self.by_document.get(document_id).map(|i| &self.documents[*i])
    }

    /// One document's text, read from the file now.
    pub fn text(&self, document_id: &str) -> Result<String> {
        let info = self.document_info(document_id).ok_or_else(|| {
            crate::sample::reader::missing_key(
                "document",
                document_id,
                self.documents.iter().map(|d| d.document_id.clone()),
            )
        })?;
        match &self.text {
            Some((column, offsets)) => read_cell(column, offsets, info.row),
            None => Ok(String::new()),
        }
    }

    /// One document, text included.
    pub fn document(&self, document_id: &str) -> Result<Document> {
        let text = self.text(document_id)?;
        let info = self.document_info(document_id).expect("found above");
        Ok(Document {
            document_id: info.document_id.clone(),
            media_type: info.media_type.clone(),
            text,
            language: info.language.clone(),
            source_type: info.source_type.clone(),
        })
    }

    /// Everything, as logical records (every document's text read).
    pub fn records(&self) -> Result<ClinicalRecords> {
        Ok(ClinicalRecords {
            descriptor: self.descriptor.clone(),
            events: self.events.clone(),
            documents: self.documents.iter().map(|d| self.document(&d.document_id)).collect::<Result<_>>()?,
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
