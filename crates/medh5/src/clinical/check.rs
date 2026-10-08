//! The record rules (1.1 §3, §5--§7): what a writer checks as rows arrive and
//! what the validator checks over a file.
//!
//! [`check_event`], [`check_document`] and [`check_link`] need only the row;
//! [`check_records`] needs all of them and the sample they live in --- ids
//! unique, every reference resolved, every document owned, every revision
//! chain one chain.  Both report [`Finding`]s carrying the §15.2 code the
//! validator emits, so a writer's refusal and a validator's report name a
//! defect identically.
//!
//! Of a document's text the rules need only its length and where its
//! characters begin (a span's ends, §7.1): [`DocumentTexts`].  Records in
//! memory answer from their text; the validator answers from one bounded
//! scan of the stored buffer, never holding the text
//! ([`check_records_with`]).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::model::{
    ClinicalRecords, Descriptor, Document, Event, Link, ASSESSMENT_SYSTEM, COMPARATORS, ENDPOINT_TYPES, EVENT_KINDS,
    LESION_PRESENCE, LESION_VALUES, MEDIA_TYPES, RELATIONS, STATUSES, TEMPORAL_TYPES,
};
use crate::h5::attrs;
use crate::h5::ops;
use crate::ids::is_valid_id;
use crate::json::{repr_list, repr_str};
use crate::Result;

/// One rule violation, with the code the validator reports for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub code: &'static str,
    pub location: String,
    pub message: String,
}

fn finding(code: &'static str, location: &str, message: impl Into<String>) -> Finding {
    Finding { code, location: location.to_string(), message: message.into() }
}

/// What the rules need to know about the sample around the clinical tables.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampleContext {
    pub timepoints: BTreeSet<String>,
    /// grid id -> its timepoint, with §3.7's single implicit timepoint resolved.
    pub grids: BTreeMap<String, Option<String>>,
    /// image id -> its grid id.
    pub images: BTreeMap<String, Option<String>>,
    /// annotation id -> the instance ids it carries, when it carries any.
    pub annotations: BTreeMap<String, Option<BTreeSet<u64>>>,
    pub transforms: BTreeSet<String>,
    pub activities: BTreeSet<String>,
    /// Whether `/meta` carries a de-identification record (1.0 §11.4).
    pub deidentified: bool,
}

impl SampleContext {
    /// Read the context from a sample root; `document` is its parsed `/meta`.
    pub fn from_root(root: &hdf5::Group, document: Option<&crate::document::SampleDocument>) -> Result<SampleContext> {
        let mut ctx = SampleContext::default();
        if let Some(doc) = document {
            ctx.timepoints = doc.timepoints.ids().into_iter().collect();
            ctx.activities = doc.provenance.activities().map(|a| a.id.clone()).collect();
            ctx.deidentified = doc.deidentification.is_some();
        }
        let only = if ctx.timepoints.len() == 1 { ctx.timepoints.iter().next().cloned() } else { None };
        if let Some(node) = ops::child_group(root, "grids") {
            for name in ops::members(&node)? {
                if let Some(g) = ops::child_group(&node, &name) {
                    let tp = attrs::get_str(&g, "timepoint")?.filter(|t| !t.is_empty()).or_else(|| only.clone());
                    ctx.grids.insert(name, tp);
                }
            }
        }
        if let Some(node) = ops::child_group(root, "images") {
            for name in ops::members(&node)? {
                let grid = match ops::node_kind(&node, &name) {
                    Some(ops::NodeKind::Group) => {
                        let g = node.group(&name)?;
                        attrs::get_str(&g, "grid")?
                    }
                    Some(ops::NodeKind::Dataset) => {
                        let d = node.dataset(&name)?;
                        attrs::get_str(&d, "grid")?
                    }
                    _ => continue,
                };
                ctx.images.insert(name, grid);
            }
        }
        if let Some(node) = ops::child_group(root, "annotations") {
            for name in ops::members(&node)? {
                let Some(g) = ops::child_group(&node, &name) else { continue };
                let ids = match ops::child_dataset(&g, "instance_ids") {
                    Some(ds) => crate::h5::data::read(&ds)
                        .ok()
                        .map(|a| a.cast::<u64>().iter().copied().collect::<BTreeSet<u64>>()),
                    None => None,
                };
                ctx.annotations.insert(name, ids);
            }
        }
        if let Some(node) = ops::child_group(root, "transforms") {
            ctx.transforms = ops::members(&node)?.into_iter().collect();
        }
        Ok(ctx)
    }

    /// The timepoint of an image, through its grid.
    pub fn image_timepoint(&self, image_id: &str) -> Option<&str> {
        let grid = self.images.get(image_id)?.as_deref()?;
        self.grids.get(grid)?.as_deref()
    }

    /// Every instance id any annotation carries.
    pub fn instances(&self) -> BTreeSet<u64> {
        self.annotations.values().flatten().flat_map(|ids| ids.iter().copied()).collect()
    }
}

/// An instance id's canonical decimal form: no sign, no leading zero (§7).
pub fn parse_instance(text: &str) -> Option<u64> {
    let canonical =
        !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) && (text == "0" || !text.starts_with('0'));
    if canonical {
        text.parse().ok()
    } else {
        None
    }
}

fn nonempty(value: &Option<String>, field: &str, location: &str, out: &mut Vec<Finding>) {
    if value.as_deref() == Some("") {
        out.push(finding(
            "E809",
            location,
            format!("`{field}` is empty; a valid identifier never is (null it instead)"),
        ));
    }
}

fn in_vocabulary(value: &str, field: &str, vocabulary: &[&str], location: &str, out: &mut Vec<Finding>) {
    if !vocabulary.contains(&value) {
        out.push(finding(
            "E810",
            location,
            format!("`{field}` is {}, not one of {}", repr_str(value), repr_list(vocabulary)),
        ));
    }
}

/// The rules one event row must satisfy on its own (§5).
pub fn check_event(e: &Event, location: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    for (field, value) in [("event_id", &e.event_id), ("record_id", &e.record_id)] {
        if !is_valid_id(value) {
            out.push(finding(
                "E809",
                location,
                format!("`{field}` {} must match [A-Za-z0-9_.-]{{1,128}}", repr_str(value)),
            ));
        }
    }
    for (field, value) in [
        ("timepoint_id", &e.timepoint_id),
        ("encounter_id", &e.encounter_id),
        ("code_system", &e.code_system),
        ("code", &e.code),
        ("code_version", &e.code_version),
        ("unit", &e.unit),
        ("missing_reason", &e.missing_reason),
        ("prov", &e.prov),
    ] {
        nonempty(value, field, location, &mut out);
    }
    in_vocabulary(&e.kind, "kind", &EVENT_KINDS, location, &mut out);
    in_vocabulary(&e.temporal_type, "temporal_type", &TEMPORAL_TYPES, location, &mut out);
    in_vocabulary(&e.status, "status", &STATUSES, location, &mut out);
    if let Some(c) = &e.value_comparator {
        in_vocabulary(c, "value_comparator", &COMPARATORS, location, &mut out);
    }
    // §5.2: every pair whole and ordered; the pairs a temporal type allows.
    for (field, bounds) in
        [("effective_start", &e.effective_start), ("effective_end", &e.effective_end), ("available", &e.available)]
    {
        if let Some(b) = bounds {
            if b.lo > b.hi {
                out.push(finding("E811", location, format!("`{field}` has lo {} > hi {}", b.lo, b.hi)));
            }
        }
    }
    match e.temporal_type.as_str() {
        "point" => {
            if e.effective_start.is_none() {
                out.push(finding("E811", location, "a `point` event needs effective start bounds"));
            }
            if e.effective_end.is_some() {
                out.push(finding("E811", location, "a `point` event has no end bounds; an interval does"));
            }
        }
        "interval" => match (&e.effective_start, &e.effective_end) {
            (None, _) => out.push(finding("E811", location, "an `interval` event needs effective start bounds")),
            (Some(s), Some(end)) if s.lo > end.hi => out.push(finding(
                "E811",
                location,
                format!("the interval ends (at most {}) before it can start (at least {})", end.hi, s.lo),
            )),
            _ => {}
        },
        "static" | "unknown" if e.effective_start.is_some() || e.effective_end.is_some() => {
            out.push(finding(
                "E811",
                location,
                format!("a `{}` event has no clinical start or end bounds", e.temporal_type),
            ));
        }
        _ => {}
    }
    if let Some(v) = e.value_num {
        if !v.is_finite() {
            out.push(finding("E808", location, format!("`value_num` is {v}; clinical values are finite")));
        }
    }
    // §5.1: values, codes and the reasons values are missing.
    if e.code_system.is_some() != e.code.is_some() {
        out.push(finding("E812", location, "`code_system` and `code` are both valid or both null"));
    }
    if e.value_num.is_none() && (e.value_comparator.is_some() || e.unit.is_some()) {
        out.push(finding("E812", location, "`value_comparator` and `unit` describe `value_num`, which is null"));
    }
    if e.value_num.is_some() && e.value_text.is_some() {
        out.push(finding("E812", location, "`value_num` and `value_text` cannot both be valid"));
    }
    if e.missing_reason.is_some() && (e.value_num.is_some() || e.value_text.is_some()) {
        out.push(finding("E812", location, "`missing_reason` says a value is absent, but the row carries one"));
    }
    if e.kind == "observation" && (e.value_num.is_some() || e.value_text.is_some()) && e.code.is_none() {
        out.push(finding(
            "E812",
            location,
            "an observation with a value names what was measured (`code_system` and `code`; a local concept will do)",
        ));
    }
    out
}

/// The rules one document row must satisfy on its own (§6).
pub fn check_document(d: &Document, location: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    if !is_valid_id(&d.document_id) {
        out.push(finding(
            "E809",
            location,
            format!("`document_id` {} must match [A-Za-z0-9_.-]{{1,128}}", repr_str(&d.document_id)),
        ));
    }
    in_vocabulary(&d.media_type, "media_type", &MEDIA_TYPES, location, &mut out);
    nonempty(&d.language, "language", location, &mut out);
    nonempty(&d.source_type, "source_type", location, &mut out);
    out
}

/// The rules one link row must satisfy on its own (§7).
pub fn check_link(l: &Link, location: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    in_vocabulary(&l.source_type, "source_type", &ENDPOINT_TYPES, location, &mut out);
    in_vocabulary(&l.target_type, "target_type", &ENDPOINT_TYPES, location, &mut out);
    in_vocabulary(&l.relation, "relation", &RELATIONS, location, &mut out);
    for (field, value) in [("source_id", &l.source_id), ("target_id", &l.target_id)] {
        if value.is_empty() {
            out.push(finding("E809", location, format!("`{field}` is empty")));
        }
    }
    nonempty(&l.target_annotation_id, "target_annotation_id", location, &mut out);
    nonempty(&l.asserted_by_event_id, "asserted_by_event_id", location, &mut out);
    if let Some((start, end)) = l.source_span {
        if l.source_type != "document" {
            out.push(finding("E814", location, "a text span is valid only on a `document` source"));
        }
        if start > end {
            out.push(finding("E814", location, format!("the span [{start}, {end}) ends before it starts")));
        }
    }
    if l.target_annotation_id.is_some() && l.target_type != "instance" {
        out.push(finding("E814", location, "`target_annotation_id` qualifies an `instance` target only"));
    }
    if l.relation == "supersedes" && (l.source_type != "event" || l.target_type != "event") {
        out.push(finding("E816", location, "`supersedes` relates two event versions of one record"));
    }
    out
}

/// The clock rules a descriptor's schema cannot state (§3).
pub fn check_descriptor(descriptor: &Descriptor, ctx: &SampleContext, location: &str) -> Vec<Finding> {
    let mut out = Vec::new();
    let clock = &descriptor.clock;
    if clock.reference == "shifted_utc" && !ctx.deidentified {
        out.push(finding(
            "E802",
            location,
            "a `shifted_utc` clock is shifted by the de-identification `/meta` declares, and `/meta` declares none",
        ));
    }
    if clock.reference == "relative" && clock.origin_description.as_deref().is_none_or(str::is_empty) {
        out.push(finding("E802", location, "a `relative` clock documents its origin in `origin_description`"));
    }
    out
}

/// Where the validator reports row `i` of a table: by id when it has one.
pub fn row_location(base: &str, id: Option<&str>, i: usize) -> String {
    match id {
        Some(id) if !id.is_empty() => format!("{base}#{id}"),
        _ => format!("{base}#row={i}"),
    }
}

/// What the span rule needs of each document's text (§7.1).
pub trait DocumentTexts {
    /// The document's length in UTF-8 bytes; `None` for no such document.
    fn n_bytes(&self, document_id: &str) -> Option<u64>;
    /// Whether byte `at`, with `0 < at < n_bytes`, begins a character.
    fn starts_character(&self, document_id: &str, at: u64) -> bool;
}

/// Texts held in memory, by document id.
impl DocumentTexts for HashMap<&str, &str> {
    fn n_bytes(&self, document_id: &str) -> Option<u64> {
        self.get(document_id).map(|t| t.len() as u64)
    }

    fn starts_character(&self, document_id: &str, at: u64) -> bool {
        self.get(document_id).is_some_and(|t| t.is_char_boundary(at as usize))
    }
}

/// Every rule across the rows and the sample (§5--§7), the documents' text
/// read from the records; the descriptor's own rules are [`check_descriptor`].
pub fn check_records(records: &ClinicalRecords, ctx: &SampleContext, base: &str) -> Vec<Finding> {
    let texts: HashMap<&str, &str> =
        records.documents.iter().map(|d| (d.document_id.as_str(), d.text.as_str())).collect();
    check_records_with(records, &texts, ctx, base)
}

/// [`check_records`], with what the rules need of the documents' text from
/// `texts`: the records' own `text` is not read.
pub fn check_records_with(
    records: &ClinicalRecords,
    texts: &dyn DocumentTexts,
    ctx: &SampleContext,
    base: &str,
) -> Vec<Finding> {
    let mut out = Vec::new();
    let event_loc = |i: usize| row_location(&format!("{base}/events"), Some(&records.events[i].event_id), i);
    let document_loc =
        |i: usize| row_location(&format!("{base}/documents"), Some(&records.documents[i].document_id), i);
    let link_loc = |i: usize| format!("{base}/links#row={i}");

    let mut events: HashMap<&str, &Event> = HashMap::new();
    for (i, e) in records.events.iter().enumerate() {
        out.extend(check_event(e, &event_loc(i)));
        if events.insert(e.event_id.as_str(), e).is_some() {
            out.push(finding("E809", &event_loc(i), format!("`event_id` {} is not unique", repr_str(&e.event_id))));
        }
        if let Some(tp) = &e.timepoint_id {
            if !tp.is_empty() && !ctx.timepoints.contains(tp) {
                out.push(finding(
                    "E813",
                    &event_loc(i),
                    format!("`timepoint_id` {} is not a declared timepoint", repr_str(tp)),
                ));
            }
        }
        if let Some(p) = &e.prov {
            if !p.is_empty() && !ctx.activities.contains(p) {
                out.push(finding(
                    "E813",
                    &event_loc(i),
                    format!("`prov` {} is not an activity in the provenance graph", repr_str(p)),
                ));
            }
        }
        if e.value_num.is_some() && e.unit.is_none() {
            out.push(finding(
                "W914",
                &event_loc(i),
                "a numeric value carries no unit (`1` is dimensionless); converters never invent one",
            ));
        }
    }
    let mut documents: HashMap<&str, &Document> = HashMap::new();
    for (i, d) in records.documents.iter().enumerate() {
        out.extend(check_document(d, &document_loc(i)));
        if documents.insert(d.document_id.as_str(), d).is_some() {
            out.push(finding(
                "E809",
                &document_loc(i),
                format!("`document_id` {} is not unique", repr_str(&d.document_id)),
            ));
        }
    }
    let instances = ctx.instances();
    let resolves = |kind: &str, id: &str| -> bool {
        match kind {
            "event" => events.contains_key(id),
            "document" => documents.contains_key(id),
            "image" => ctx.images.contains_key(id),
            "grid" => ctx.grids.contains_key(id),
            "annotation" => ctx.annotations.contains_key(id),
            "transform" => ctx.transforms.contains(id),
            "timepoint" => ctx.timepoints.contains(id),
            "instance" => parse_instance(id).is_some_and(|n| instances.contains(&n)),
            _ => true, // an unknown type is E810, already reported
        }
    };
    for (i, l) in records.links.iter().enumerate() {
        let loc = link_loc(i);
        out.extend(check_link(l, &loc));
        for (role, kind, id) in [("source", &l.source_type, &l.source_id), ("target", &l.target_type, &l.target_id)] {
            if !id.is_empty() && !resolves(kind, id) {
                let why = if kind == "instance" && parse_instance(id).is_none() {
                    " (an instance is named by its id in canonical decimal)"
                } else {
                    ""
                };
                out.push(finding(
                    "E813",
                    &loc,
                    format!("{role} {kind} {} does not resolve in this sample{why}", repr_str(id)),
                ));
            }
        }
        if let Some(by) = &l.asserted_by_event_id {
            if !by.is_empty() && !events.contains_key(by.as_str()) {
                out.push(finding(
                    "E813",
                    &loc,
                    format!("`asserted_by_event_id` {} is not an event here", repr_str(by)),
                ));
            }
        }
        if let Some(ann) = &l.target_annotation_id {
            match ctx.annotations.get(ann) {
                None => out.push(finding(
                    "E813",
                    &loc,
                    format!("`target_annotation_id` {} is not an annotation", repr_str(ann)),
                )),
                Some(ids) => {
                    let has =
                        parse_instance(&l.target_id).is_some_and(|n| ids.as_ref().is_some_and(|s| s.contains(&n)));
                    if l.target_type == "instance" && !has {
                        out.push(finding(
                            "E814",
                            &loc,
                            format!(
                                "annotation {} does not contain instance {}",
                                repr_str(ann),
                                repr_str(&l.target_id)
                            ),
                        ));
                    }
                }
            }
        }
        if let (Some((start, end)), "document") = (l.source_span, l.source_type.as_str()) {
            if let (true, Some(len)) = (documents.contains_key(l.source_id.as_str()), texts.n_bytes(&l.source_id)) {
                let aligned = |p: u64| p == 0 || p >= len || texts.starts_character(&l.source_id, p);
                if end > len || start > end {
                    out.push(finding(
                        "E814",
                        &loc,
                        format!("span [{start}, {end}) is outside document {} of {len} bytes", repr_str(&l.source_id)),
                    ));
                } else if !aligned(start) || !aligned(end) {
                    out.push(finding(
                        "E814",
                        &loc,
                        format!(
                            "span [{start}, {end}) splits a UTF-8 character of document {}",
                            repr_str(&l.source_id)
                        ),
                    ));
                }
            }
        }
        if l.source_type == "event" && l.relation == "describes" && l.target_type == "image" {
            if let Some(e) = events.get(l.source_id.as_str()) {
                if let (Some(tp), Some(actual)) = (&e.timepoint_id, ctx.image_timepoint(&l.target_id)) {
                    if e.kind == "imaging" && tp != actual {
                        out.push(finding(
                            "E814",
                            &loc,
                            format!(
                                "imaging event {} names timepoint {}, but image {} is on a grid of timepoint {}",
                                repr_str(&e.event_id),
                                repr_str(tp),
                                repr_str(&l.target_id),
                                repr_str(actual)
                            ),
                        ));
                    }
                }
            }
        }
    }
    check_ownership(records, &events, base, &mut out);
    check_chains(records, &events, base, &mut out);
    check_assessments(records, base, &mut out);
    out
}

/// §6: every document is owned by exactly one `document` event, through
/// `describes`.
fn check_ownership(records: &ClinicalRecords, events: &HashMap<&str, &Event>, base: &str, out: &mut Vec<Finding>) {
    let mut owners: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for l in &records.links {
        if l.relation == "describes" && l.source_type == "event" && l.target_type == "document" {
            if let Some(e) = events.get(l.source_id.as_str()) {
                if e.kind == "document" {
                    owners.entry(l.target_id.as_str()).or_default().insert(l.source_id.as_str());
                }
            }
        }
    }
    for (i, d) in records.documents.iter().enumerate() {
        let found = owners.get(d.document_id.as_str()).map(BTreeSet::len).unwrap_or(0);
        if found != 1 {
            let named: Vec<&str> =
                owners.get(d.document_id.as_str()).map(|s| s.iter().copied().collect()).unwrap_or_default();
            out.push(finding(
                "E815",
                &row_location(&format!("{base}/documents"), Some(&d.document_id), i),
                if found == 0 {
                    format!(
                        "document {} is described by no `document` event; the event owns its timing, status and provenance",
                        repr_str(&d.document_id)
                    )
                } else {
                    format!("document {} is owned by {found} document events ({})", repr_str(&d.document_id), named.join(", "))
                },
            ));
        }
    }
}

/// §5.2, §7: the versions of one record form one acyclic `supersedes` chain,
/// which known availability never contradicts.
fn check_chains(records: &ClinicalRecords, events: &HashMap<&str, &Event>, base: &str, out: &mut Vec<Finding>) {
    let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();
    for (i, l) in records.links.iter().enumerate() {
        if l.relation != "supersedes" || l.source_type != "event" || l.target_type != "event" {
            continue;
        }
        let loc = format!("{base}/links#row={i}");
        let (Some(new), Some(old)) = (events.get(l.source_id.as_str()), events.get(l.target_id.as_str())) else {
            continue; // E813, reported above
        };
        if new.event_id == old.event_id {
            out.push(finding("E816", &loc, format!("event {} supersedes itself", repr_str(&new.event_id))));
            continue;
        }
        if new.record_id != old.record_id {
            out.push(finding(
                "E816",
                &loc,
                format!(
                    "{} (record {}) cannot supersede {} (record {}): revisions share a record",
                    repr_str(&new.event_id),
                    repr_str(&new.record_id),
                    repr_str(&old.event_id),
                    repr_str(&old.record_id)
                ),
            ));
            continue;
        }
        if let (Some(a_new), Some(a_old)) = (new.available, old.available) {
            if a_new.hi < a_old.lo {
                out.push(finding(
                    "E816",
                    &loc,
                    format!(
                        "{} was available by {} but supersedes {}, available no earlier than {}",
                        repr_str(&new.event_id),
                        a_new.hi,
                        repr_str(&old.event_id),
                        a_old.lo
                    ),
                ));
            }
        }
        successors.entry(old.event_id.as_str()).or_default().push(new.event_id.as_str());
        predecessors.entry(new.event_id.as_str()).or_default().push(old.event_id.as_str());
    }
    let loc = format!("{base}/links");
    for (old, news) in &successors {
        if news.len() > 1 {
            out.push(finding(
                "E816",
                &loc,
                format!(
                    "event {} is superseded by {} versions ({}): the chain branches",
                    repr_str(old),
                    news.len(),
                    news.join(", ")
                ),
            ));
        }
    }
    for (new, olds) in &predecessors {
        if olds.len() > 1 {
            out.push(finding(
                "E816",
                &loc,
                format!(
                    "event {} supersedes {} versions ({}): chains merge",
                    repr_str(new),
                    olds.len(),
                    olds.join(", ")
                ),
            ));
        }
    }
    // A cycle: walking back from a version returns to it.
    let mut cyclic: BTreeSet<&str> = BTreeSet::new();
    for start in predecessors.keys() {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut at = *start;
        while let Some(olds) = predecessors.get(at) {
            if !seen.insert(at) {
                cyclic.insert(at);
                break;
            }
            at = olds[0];
        }
    }
    if let Some(at) = cyclic.iter().next() {
        out.push(finding("E816", &loc, format!("the `supersedes` links through {} form a cycle", repr_str(at))));
    }
    // One chain per record: exactly one version that supersedes nothing.
    let mut by_record: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for e in &records.events {
        by_record.entry(e.record_id.as_str()).or_default().push(e.event_id.as_str());
    }
    for (record, versions) in by_record {
        if versions.len() < 2 || !cyclic.is_empty() {
            continue;
        }
        let roots = versions.iter().filter(|v| !predecessors.contains_key(**v)).count();
        if roots != 1 {
            out.push(finding(
                "E816",
                &format!("{base}/events"),
                format!(
                    "record {} has {} versions ({}) that `supersedes` links do not order into one chain; \
                     independent claims take distinct record ids",
                    repr_str(record),
                    versions.len(),
                    versions.join(", ")
                ),
            ));
        }
    }
}

/// §7: a lesion assessment names its timepoint, links its instance and says
/// one of the assessment values.
fn check_assessments(records: &ClinicalRecords, base: &str, out: &mut Vec<Finding>) {
    for (i, e) in records.events.iter().enumerate() {
        if !(e.code_system.as_deref() == Some(ASSESSMENT_SYSTEM) && e.code.as_deref() == Some(LESION_PRESENCE)) {
            continue;
        }
        let loc = row_location(&format!("{base}/events"), Some(&e.event_id), i);
        if e.kind != "assessment" {
            out.push(finding(
                "E817",
                &loc,
                format!("a lesion-presence assessment has kind `assessment`, not {}", repr_str(&e.kind)),
            ));
        }
        match &e.value_text {
            Some(v) if LESION_VALUES.contains(&v.as_str()) => {}
            other => out.push(finding(
                "E817",
                &loc,
                format!(
                    "a lesion-presence assessment says one of {}, not {}",
                    repr_list(&LESION_VALUES),
                    other.as_deref().map(repr_str).unwrap_or_else(|| "nothing".into())
                ),
            )),
        }
        if e.timepoint_id.is_none() {
            out.push(finding("E817", &loc, "a lesion assessment names the imaging timepoint it assessed"));
        }
        let linked = records.links.iter().any(|l| {
            l.relation == "assesses"
                && l.source_type == "event"
                && l.source_id == e.event_id
                && l.target_type == "instance"
        });
        if !linked {
            out.push(finding(
                "E817",
                &loc,
                "a lesion assessment links the instance it assessed by `assesses` (a disappearance links the known instance)",
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::model::{Bounds, Clock, Descriptor};

    fn event(id: &str, record: &str) -> Event {
        Event {
            event_id: id.into(),
            record_id: record.into(),
            kind: "observation".into(),
            temporal_type: "point".into(),
            effective_start: Some(Bounds::exact(0)),
            status: "final".into(),
            ..Default::default()
        }
    }

    fn codes(findings: &[Finding]) -> Vec<&'static str> {
        let mut c: Vec<&str> = findings.iter().map(|f| f.code).collect();
        c.sort_unstable();
        c.dedup();
        c
    }

    #[test]
    fn s5_2_bounds_follow_the_temporal_type() {
        let mut e = event("e1", "r1");
        assert!(check_event(&e, "x").is_empty());
        e.effective_end = Some(Bounds::exact(5));
        assert_eq!(codes(&check_event(&e, "x")), ["E811"]);
        e.temporal_type = "interval".into();
        assert!(check_event(&e, "x").is_empty());
        e.effective_end = Some(Bounds::new(-10, -5));
        assert_eq!(codes(&check_event(&e, "x")), ["E811"]);
        let mut s = event("s", "s");
        s.temporal_type = "static".into();
        assert_eq!(codes(&check_event(&s, "x")), ["E811"]);
        s.effective_start = None;
        assert!(check_event(&s, "x").is_empty());
        let mut inverted = event("i", "i");
        inverted.available = Some(Bounds::new(5, 1));
        assert_eq!(codes(&check_event(&inverted, "x")), ["E811"]);
    }

    #[test]
    fn s5_1_values_codes_and_missing_reasons() {
        let mut e = event("e1", "r1");
        e.value_num = Some(5.0);
        assert_eq!(codes(&check_event(&e, "x")), ["E812"]); // an observation value names its concept
        e.code_system = Some("LOINC".into());
        e.code = Some("2160-0".into());
        e.unit = Some("mg/dL".into());
        e.value_comparator = Some("lt".into());
        assert!(check_event(&e, "x").is_empty());
        e.value_text = Some("high".into());
        assert_eq!(codes(&check_event(&e, "x")), ["E812"]);
        e.value_text = None;
        e.value_num = None;
        assert_eq!(codes(&check_event(&e, "x")), ["E812"]); // a unit without a value
        e.unit = None;
        e.value_comparator = None;
        e.missing_reason = Some("not_performed".into());
        assert!(check_event(&e, "x").is_empty());
        e.value_num = Some(f64::NAN);
        e.unit = Some("1".into());
        assert_eq!(codes(&check_event(&e, "x")), ["E808", "E812"]);
    }

    fn records(events: Vec<Event>, links: Vec<Link>) -> ClinicalRecords {
        ClinicalRecords {
            descriptor: Descriptor::new(Clock::relative("c", "baseline")),
            events,
            documents: Vec::new(),
            links,
        }
    }

    #[test]
    fn s7_revisions_form_one_chain() {
        let mut v1 = event("v1", "r");
        v1.available = Some(Bounds::exact(10));
        let mut v2 = event("v2", "r");
        v2.available = Some(Bounds::exact(20));
        let ctx = SampleContext::default();
        let chained =
            records(vec![v1.clone(), v2.clone()], vec![Link::new(("event", "v2"), "supersedes", ("event", "v1"))]);
        assert!(check_records(&chained, &ctx, "/clinical").is_empty());
        let unordered = records(vec![v1.clone(), v2.clone()], vec![]);
        assert_eq!(codes(&check_records(&unordered, &ctx, "/clinical")), ["E816"]);
        let backwards =
            records(vec![v1.clone(), v2.clone()], vec![Link::new(("event", "v1"), "supersedes", ("event", "v2"))]);
        assert_eq!(codes(&check_records(&backwards, &ctx, "/clinical")), ["E816"]);
        let v3 = event("v3", "r");
        let branching = records(
            vec![v1, v2, v3],
            vec![
                Link::new(("event", "v2"), "supersedes", ("event", "v1")),
                Link::new(("event", "v3"), "supersedes", ("event", "v1")),
            ],
        );
        assert_eq!(codes(&check_records(&branching, &ctx, "/clinical")), ["E816"]);
    }

    #[test]
    fn s7_instances_are_canonical_decimals() {
        assert_eq!(parse_instance("7"), Some(7));
        assert_eq!(parse_instance("0"), Some(0));
        assert_eq!(parse_instance("07"), None);
        assert_eq!(parse_instance("+7"), None);
        assert_eq!(parse_instance(""), None);
    }
}
